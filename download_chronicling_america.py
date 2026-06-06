#!/usr/bin/env python3
"""
Chronicling America OCR Text Downloader
========================================

Downloads the full OCR text corpus from the Library of Congress
Chronicling America collection (https://chroniclingamerica.loc.gov).

Organized by year, newspaper LCCN, and issue date.
Fully resumable — tracks progress in a JSON state file.

Usage:
    python download_chronicling_america.py [options]

    # Download everything (will take months)
    python download_chronicling_america.py

    # Download a specific year range
    python download_chronicling_america.py --year-start 1880 --year-end 1890

    # Download only newspapers from a specific state
    python download_chronicling_america.py --state "New York"

    # Use more workers for the index phase (text download is always single-threaded per LOC policy)
    python download_chronicling_america.py --output-dir /mnt/data/chronam

API reference: https://chroniclingamerica.loc.gov/about/api/
Rate limit: ~20 requests/minute. This script enforces 3.1s between requests.

NOTE (2026): The LOC API changed significantly.
  - /newspapers.json now returns only 25 results in 'content.results' (was full list in 'newspapers')
  - /lccn/{lccn}.json is now 403
  - /lccn/{lccn}/{date}/ed-1/seq-N/ocr.txt is now 403
  New approach:
  - Enumerate issues per year via loc.gov/collections/chronicling-america/?dl=issue&dates=YEAR/YEAR
  - Fetch issue page list via loc.gov/item/{lccn}/{date}/ed-1/?fo=json
  - Download Alto XML from tile.loc.gov and extract OCR text from it
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOC_COLLECTION_URL = "https://www.loc.gov/collections/chronicling-america/"
LOC_ITEM_URL = "https://www.loc.gov/item/"
REQUEST_DELAY = 3.1  # seconds between requests (LOC allows ~20/min)
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 5
RETRY_BACKOFF_FACTOR = 2.0
USER_AGENT = "ChronAmDownloader/1.0 (Historical Research; github.com/chronam-pipeline)"

ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v2#"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("chronam")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        log.warning("Force quit.")
        sys.exit(1)
    log.info("Shutdown requested — finishing current download then saving state...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# Progress state
# ---------------------------------------------------------------------------


@dataclass
class DownloadStats:
    total_pages_downloaded: int = 0
    total_pages_skipped: int = 0
    total_pages_failed: int = 0
    total_bytes: int = 0
    total_issues_processed: int = 0
    total_titles_processed: int = 0
    errors: int = 0
    last_updated: str = ""


@dataclass
class ProgressState:
    years_completed: list = field(default_factory=list)
    current_year: Optional[int] = None
    current_year_issues_completed: list = field(default_factory=list)
    stats: dict = field(default_factory=lambda: asdict(DownloadStats()))

    def mark_page_downloaded(self, nbytes: int):
        self.stats["total_pages_downloaded"] += 1
        self.stats["total_bytes"] += nbytes

    def mark_page_skipped(self):
        self.stats["total_pages_skipped"] += 1

    def mark_page_failed(self):
        self.stats["total_pages_failed"] += 1
        self.stats["errors"] += 1

    def mark_issue_done(self, issue_key: str):
        if issue_key not in self.current_year_issues_completed:
            self.current_year_issues_completed.append(issue_key)
        self.stats["total_issues_processed"] += 1

    def mark_year_done(self, year: int):
        if year not in self.years_completed:
            self.years_completed.append(year)
        self.current_year = None
        self.current_year_issues_completed = []

    def is_year_done(self, year: int) -> bool:
        return year in self.years_completed

    def is_issue_done(self, issue_key: str) -> bool:
        return issue_key in self.current_year_issues_completed

    def update_timestamp(self):
        self.stats["last_updated"] = datetime.now(timezone.utc).isoformat()


class ProgressTracker:
    """Persistent progress state backed by a JSON file."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load()

    def _load(self) -> ProgressState:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                # Handle old schema that had newspapers_fetched etc.
                allowed = {f.name for f in ProgressState.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in allowed}
                log.info(f"Resumed from {self.state_path}")
                return ProgressState(**filtered)
            except (json.JSONDecodeError, TypeError) as e:
                log.warning(f"Corrupted progress file, starting fresh: {e}")
        return ProgressState()

    def save(self):
        self.state.update_timestamp()
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self.state), f, indent=2)
        tmp.replace(self.state_path)

    def log_stats(self):
        s = self.state.stats
        log.info(
            f"Progress: {s['total_pages_downloaded']} pages downloaded, "
            f"{s['total_pages_skipped']} skipped, "
            f"{s['total_pages_failed']} failed, "
            f"{s['total_bytes'] / 1e9:.2f} GB, "
            f"{s['total_issues_processed']} issues, "
            f"{s['total_titles_processed']} titles, "
            f"years done: {len(self.state.years_completed)}"
        )


# ---------------------------------------------------------------------------
# HTTP client with rate limiting and retries
# ---------------------------------------------------------------------------


class RateLimitedClient:
    """HTTP client that enforces minimum delay between requests."""

    def __init__(self, delay: float = REQUEST_DELAY):
        self.delay = delay
        self._last_request_time = 0.0
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _wait_for_rate_limit(self):
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def get_json(self, url: str) -> Optional[dict]:
        """GET a JSON endpoint. Returns None on failure."""
        self._wait_for_rate_limit()
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            self._last_request_time = time.monotonic()

            if resp.status_code == 429:
                log.warning(f"Rate limited (429) on {url}. Waiting 120s...")
                time.sleep(120)
                return self.get_json(url)  # retry once after wait

            if "captcha" in resp.text.lower()[:500]:
                log.error(
                    "CAPTCHA detected! LOC has throttled us. "
                    "Wait ~1 hour before restarting. Saving state and exiting."
                )
                return None

            if resp.status_code == 404:
                log.debug(f"404: {url}")
                return None

            if resp.status_code == 403:
                log.warning(f"403 Forbidden: {url}")
                return None

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            log.warning(f"Request failed: {url} — {e}")
            return None

    def get_text(self, url: str) -> Optional[str]:
        """GET a plain text endpoint. Returns None on failure."""
        self._wait_for_rate_limit()
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            self._last_request_time = time.monotonic()

            if resp.status_code == 429:
                log.warning(f"Rate limited (429) on {url}. Waiting 120s...")
                time.sleep(120)
                return self.get_text(url)

            if resp.status_code == 404:
                log.debug(f"404 (no content): {url}")
                return None

            if resp.status_code == 403:
                log.debug(f"403 Forbidden: {url}")
                return None

            resp.raise_for_status()
            return resp.text

        except requests.RequestException as e:
            log.warning(f"Request failed: {url} — {e}")
            return None


# ---------------------------------------------------------------------------
# Issue enumeration via LOC collection API
# ---------------------------------------------------------------------------


def fetch_issues_for_year(
    client: RateLimitedClient,
    year: int,
    state_filter: Optional[str],
    cache_path: Path,
) -> Optional[list]:
    """Fetch all newspaper issues for a given year using the LOC collection API.

    Returns a list of dicts with at least 'lccn', 'date', 'url', 'state' keys,
    or None if the fetch failed due to a server error (so the caller can decide
    not to mark the year as complete).
    Results are cached to disk only on success.
    """
    if cache_path.exists():
        log.debug(f"Loading cached issue list for {year} from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    log.info(f"Fetching issue list for {year} from LOC API...")
    issues = []
    page = 1
    per_page = 150  # LOC max
    fetch_failed = False

    while True:
        url = (
            f"{LOC_COLLECTION_URL}?fo=json&dl=issue"
            f"&dates={year}/{year}&count={per_page}&sp={page}"
            f"&at=results,pagination"
        )
        data = client.get_json(url)
        if data is None:
            log.warning(f"Failed to fetch issues for year {year} page {page}")
            fetch_failed = True
            break

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            lccn_list = r.get("number_lccn", [])
            if not lccn_list:
                continue
            lccn = lccn_list[0]

            date = r.get("date", "")
            item_url = r.get("url", "")
            state_list = r.get("location_state", [])
            state = state_list[0] if state_list else ""

            issues.append({
                "lccn": lccn,
                "date": date,
                "url": item_url,
                "state": state,
                "title": r.get("title", ""),
            })

        pagination = data.get("pagination", {})
        total = pagination.get("total", 0)
        fetched = page * per_page
        log.debug(f"  Year {year}: fetched {min(fetched, total)}/{total} issues (page {page})")

        if fetched >= total or not pagination.get("next"):
            break
        page += 1

    if fetch_failed:
        # Return None so the caller knows the fetch was incomplete and should
        # NOT mark this year as done — it will be retried on next run.
        log.warning(
            f"Year {year}: issue list fetch incomplete due to server error — "
            "year will NOT be marked complete and will be retried on restart."
        )
        return None

    log.info(f"Found {len(issues)} issues in {year}.")

    # Apply state filter
    if state_filter:
        before = len(issues)
        issues = [
            i for i in issues
            if i["state"].lower() == state_filter.lower()
        ]
        log.info(
            f"Filtered to {len(issues)}/{before} issues in state: {state_filter}"
        )

    # Only cache when the fetch succeeded so a partial/empty result is never
    # persisted and the next run will re-fetch.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(issues, f)

    return issues


# ---------------------------------------------------------------------------
# Issue page discovery via LOC item API
# ---------------------------------------------------------------------------


def fetch_issue_pages(
    client: RateLimitedClient,
    lccn: str,
    date: str,
    cache_path: Path,
) -> list[dict]:
    """Fetch page metadata for a single newspaper issue.

    Returns list of dicts with 'seq', 'xml_url' for each page.
    Uses the LOC item API: loc.gov/item/{lccn}/{date}/ed-1/?fo=json
    """
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    url = f"{LOC_ITEM_URL}{lccn}/{date}/ed-1/?fo=json"
    data = client.get_json(url)
    if data is None:
        return []

    resources = data.get("resources", [])
    pages = []
    for resource in resources:
        for seq_idx, file_group in enumerate(resource.get("files", []), start=1):
            xml_url = next(
                (f.get("url", "") for f in file_group if f.get("mimetype") == "text/xml"),
                None,
            )
            pages.append({"seq": seq_idx, "xml_url": xml_url or ""})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(pages, f)

    return pages


# ---------------------------------------------------------------------------
# OCR text extraction from Alto XML
# ---------------------------------------------------------------------------


def extract_text_from_alto_xml(xml_content: str) -> str:
    """Extract plain text from Alto XML OCR content.

    Joins all String CONTENT attributes with spaces, with newlines between
    text blocks (TextBlock elements).
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        log.debug(f"XML parse error: {e}")
        return ""

    # Try standard Alto namespace first
    ns = ALTO_NS
    text_blocks = root.findall(".//alto:TextBlock", ns)

    if not text_blocks:
        # Try without namespace (some Alto files use no namespace)
        text_blocks = root.findall(".//TextBlock")
        ns = {}

    lines = []
    for block in text_blocks:
        block_words = []
        if ns:
            strings = block.findall(".//alto:String", ns)
        else:
            strings = block.findall(".//String")
        for s in strings:
            word = s.get("CONTENT", "").strip()
            if word:
                block_words.append(word)
        if block_words:
            lines.append(" ".join(block_words))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------


def download_issue_pages(
    client: RateLimitedClient,
    lccn: str,
    date: str,
    issue_dir: Path,
    tracker: ProgressTracker,
) -> bool:
    """Download all OCR text pages for a single newspaper issue.

    Returns True if completed successfully, False if interrupted.

    Page storage contract
    ---------------------
    * Page has content   → seq-NNN.txt  (non-empty)
    * Page has no OCR    → seq-NNN.txt  (empty file, 0 bytes)
      Both cases are considered "done" and will be skipped on resume.
    * Server/network error → nothing written
      The missing file means the page will be retried on the next run.
    """
    issue_dir.mkdir(parents=True, exist_ok=True)

    # Fetch issue page metadata (Alto XML URLs per page)
    page_meta_cache = issue_dir / "_pages.json"
    pages = fetch_issue_pages(client, lccn, date, page_meta_cache)

    if not pages:
        log.debug(f"No pages for {lccn}/{date}")
        return True

    for page in pages:
        if _shutdown_requested:
            return False

        seq = page.get("seq", 0)
        xml_url = page.get("xml_url", "")

        txt_filename = f"seq-{str(seq).zfill(3)}.txt"
        txt_path = issue_dir / txt_filename

        # Skip if already processed (file exists, even if empty — empty means
        # no OCR content, which is a valid permanent outcome).
        if txt_path.exists():
            tracker.state.mark_page_skipped()
            continue

        if not xml_url:
            # No XML URL available for this sequence — record as empty so we
            # don't attempt it again on future runs.
            log.debug(f"No XML URL for {lccn}/{date}/seq-{seq} — writing empty marker")
            txt_path.touch()
            tracker.state.mark_page_skipped()
            continue

        # Download Alto XML
        xml_content = client.get_text(xml_url)

        if xml_content is None:
            # Server / network error — do NOT write anything so this page will
            # be retried automatically on the next run.
            log.debug(f"Failed to fetch Alto XML for {lccn}/{date}/seq-{seq} — will retry on restart")
            tracker.state.mark_page_failed()
            continue

        # Extract OCR text from Alto XML
        text = extract_text_from_alto_xml(xml_content)

        # Write text atomically (empty string → 0-byte file, which is fine).
        tmp_path = txt_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        tmp_path.rename(txt_path)

        if text.strip():
            tracker.state.mark_page_downloaded(len(text.encode("utf-8")))
            log.debug(f"Downloaded: {lccn}/{date}/seq-{seq} ({len(text)} chars)")
        else:
            log.debug(f"Empty OCR for {lccn}/{date}/seq-{seq} — stored empty marker")
            tracker.state.mark_page_skipped()

    return True


# ---------------------------------------------------------------------------
# Helper: re-open issues/years whose page files are missing (--retry-missing)
# ---------------------------------------------------------------------------


def _reopen_missing(tracker: ProgressTracker, raw_dir: Path) -> None:
    """Scan completed years/issues and re-open any whose page files are absent.

    This recovers data lost when a previous run marked issues/years as done
    despite server errors that prevented actual file writes.

    Rules:
    - Never retry a page that has a file on disk (content or empty marker).
    - Re-open a year if its _issues.json is missing (never fetched / fetch failed).
    - Re-open a year if its _issues.json has 0 issues but no issue subdirectories
      exist either — this is a stale failed-fetch cache; the cache is deleted so
      the next run will re-fetch the real issue list.
    - Re-open an issue (remove from completed list) if any of its expected
      seq-NNN.txt files are absent.
    - Re-open a year (remove from years_completed) if any of its issues are
      re-opened, so the year loop will process those issues again.
    """
    log.info("--retry-missing: scanning for completed issues with missing page files...")
    reopened_issues = 0
    reopened_years: set = set()

    for year in list(tracker.state.years_completed):
        year_dir = raw_dir / str(year)
        issues_cache = year_dir / "_issues.json"

        if not issues_cache.exists():
            # No cache — year had a fetch error and was incorrectly marked complete.
            log.info(f"  Year {year}: no _issues.json — re-opening for re-fetch")
            tracker.state.years_completed.remove(year)
            reopened_years.add(year)
            continue

        with open(issues_cache) as f:
            issues = json.load(f)

        if not issues:
            # _issues.json is empty. Check if there are any real issue subdirs.
            issue_subdirs = [
                p for p in year_dir.iterdir()
                if p.is_dir()
            ]
            if not issue_subdirs:
                # Empty cache with no data on disk → stale failed-fetch cache.
                # Delete it so the next run re-fetches the real issue list.
                log.info(
                    f"  Year {year}: _issues.json has 0 issues and no data on disk — "
                    "deleting stale cache and re-opening for re-fetch"
                )
                issues_cache.unlink()
                tracker.state.years_completed.remove(year)
                reopened_years.add(year)
            else:
                log.debug(
                    f"  Year {year}: 0 cached issues but has {len(issue_subdirs)} "
                    "subdirs — treating as genuine empty year, skipping"
                )
            continue

        for issue in issues:
            lccn = issue.get("lccn", "")
            date = issue.get("date", "")
            if not lccn or not date:
                continue

            issue_dir = year_dir / lccn / date
            pages_cache = issue_dir / "_pages.json"

            if not pages_cache.exists():
                # Issue directory / page cache absent — re-open year if the
                # issue dir itself is entirely missing.
                if not issue_dir.exists():
                    issue_key = f"{lccn}/{date}"
                    if year not in reopened_years:
                        log.info(
                            f"  Year {year}: issue {issue_key} directory missing — "
                            "re-opening year"
                        )
                        if year in tracker.state.years_completed:
                            tracker.state.years_completed.remove(year)
                        reopened_years.add(year)
                continue

            with open(pages_cache) as f:
                pages = json.load(f)

            missing = []
            for pg in pages:
                seq = pg.get("seq", 0)
                txt_path = issue_dir / f"seq-{str(seq).zfill(3)}.txt"
                if not txt_path.exists():
                    missing.append(seq)

            if missing:
                issue_key = f"{lccn}/{date}"
                log.info(
                    f"  Issue {issue_key}: {len(missing)} page(s) missing "
                    f"(seq {missing[:5]}{'...' if len(missing) > 5 else ''}) — re-opening"
                )
                reopened_issues += 1
                # Remove from the in-progress list if present (handles partial years)
                if issue_key in tracker.state.current_year_issues_completed:
                    tracker.state.current_year_issues_completed.remove(issue_key)
                # Re-open the year so the orchestrator processes it again
                if year in tracker.state.years_completed:
                    tracker.state.years_completed.remove(year)
                reopened_years.add(year)

    log.info(
        f"--retry-missing: re-opened {reopened_issues} issue(s) across "
        f"{len(reopened_years)} year(s): {sorted(reopened_years)}"
    )




def run_download(
    output_dir: Path,
    year_start: int,
    year_end: int,
    state_filter: Optional[str],
    retry_missing: bool = False,
    recheck: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    tracker = ProgressTracker(raw_dir / "_progress.json")
    client = RateLimitedClient()

    # ------------------------------------------------------------------
    # --retry-missing: re-open years/issues whose page files are absent
    # ------------------------------------------------------------------
    if retry_missing:
        _reopen_missing(tracker, raw_dir)
        tracker.save()

    # ------------------------------------------------------------------
    # --recheck: delete cached issue lists for completed years so the
    # server is re-queried and any new issues are downloaded.
    # Only the _issues.json cache is removed — existing page files are
    # kept, so pages already downloaded are never re-fetched.
    # ------------------------------------------------------------------
    if recheck:
        log.info(
            "--recheck: refreshing issue lists for all previously completed years "
            "in range %d–%d...", year_start, year_end
        )
        rechecked = 0
        for year in range(year_start, year_end + 1):
            if year not in tracker.state.years_completed:
                continue
            issues_cache = raw_dir / str(year) / "_issues.json"
            if issues_cache.exists():
                issues_cache.unlink()
                log.debug(f"  Deleted cached issue list for {year}")
            # Re-open the year so the loop below processes it
            tracker.state.years_completed.remove(year)
            rechecked += 1
        log.info(f"--recheck: re-opened {rechecked} year(s) for server re-check.")
        tracker.save()

    # ------------------------------------------------------------------
    # Iterate year by year, fetching issues directly from the collection API
    # ------------------------------------------------------------------
    for year in range(year_start, year_end + 1):
        if _shutdown_requested:
            break

        if tracker.state.is_year_done(year):
            log.info(f"Year {year}: already complete, skipping.")
            continue

        log.info(f"=== Processing year {year} ===")
        tracker.state.current_year = year

        # Reset per-year issue tracking if starting a fresh year
        if not tracker.state.current_year_issues_completed:
            tracker.state.current_year_issues_completed = []

        year_dir = raw_dir / str(year)
        year_dir.mkdir(exist_ok=True)

        # Fetch all issues for this year
        issues_cache = year_dir / "_issues.json"
        issues = fetch_issues_for_year(client, year, state_filter, issues_cache)

        if issues is None:
            # Server error — skip this year for now; do NOT mark it done.
            # The missing _issues.json means the next run will re-fetch.
            log.warning(
                f"=== Year {year}: skipped due to fetch error — will retry on restart ==="
            )
            tracker.save()
            continue

        if not issues:
            log.info(f"=== Year {year} complete: no issues found ===")
            tracker.state.mark_year_done(year)
            tracker.save()
            continue

        issues_downloaded = 0

        for issue in sorted(issues, key=lambda i: (i["lccn"], i["date"])):
            if _shutdown_requested:
                break

            lccn = issue["lccn"]
            date = issue["date"]
            issue_key = f"{lccn}/{date}"

            if tracker.state.is_issue_done(issue_key):
                continue

            issue_dir = year_dir / lccn / date
            title = issue.get("title", lccn)
            log.info(f"  Issue: {title} ({lccn}) {date}")

            ok = download_issue_pages(client, lccn, date, issue_dir, tracker)

            if not ok:
                # Interrupted
                tracker.save()
                break

            tracker.state.mark_issue_done(issue_key)
            issues_downloaded += 1

            # Save progress periodically (every 10 issues)
            if tracker.state.stats["total_issues_processed"] % 10 == 0:
                tracker.save()
                tracker.log_stats()

        if not _shutdown_requested:
            tracker.state.mark_year_done(year)
            tracker.save()
            log.info(
                f"=== Year {year} complete: "
                f"{issues_downloaded} issues downloaded ==="
            )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    tracker.save()
    tracker.log_stats()

    if _shutdown_requested:
        log.info("Download paused. Restart to resume from where we left off.")
    else:
        log.info("Download complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download Chronicling America OCR text corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all years
  python download_chronicling_america.py --output-dir ./data

  # Download only 1880s
  python download_chronicling_america.py --year-start 1880 --year-end 1889

  # Download only New York newspapers, 1850-1860
  python download_chronicling_america.py --state "new york" --year-start 1850 --year-end 1860

  # Resume interrupted download
  python download_chronicling_america.py  # just re-run same command

  # Recover pages missing due to past server errors
  python download_chronicling_america.py --retry-missing

  # Re-check server for new issues in already-completed years
  python download_chronicling_america.py --recheck

Notes:
  - Rate limit: ~20 requests/min to LOC API (enforced automatically)
  - Full corpus: ~20M pages, ~60-120 GB of text
  - Full download via API will take months; use --year-start/--year-end to parallelize
  - Ctrl+C saves state gracefully; re-run to resume
  - Each page is written atomically (temp file + rename)
  - Progress tracked in data/raw/_progress.json
  - State filter uses lowercase state name (e.g. "new york", not "New York")
  - Empty pages (no OCR) are stored as 0-byte .txt files and never re-fetched
  - Pages that failed due to server errors leave no file and are retried on restart
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Root output directory (default: ./data)",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=1770,
        help="First year to download (default: 1770)",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=1963,
        help="Last year to download (default: 1963)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help='Filter by US state name, lowercase (e.g. "new york", "california")',
    )
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help=(
            "Re-open issues/years that were marked complete but have missing page "
            "files (e.g. due to past server errors). Pages that already have a file "
            "on disk — including empty files for no-OCR pages — are never re-fetched."
        ),
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help=(
            "Re-query the LOC server for the issue lists of all already-completed "
            "years in the requested range. Useful when new issues may have been added "
            "to the collection since the last download. Already-downloaded page files "
            "are preserved; only missing issues/pages are downloaded."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("chronam").setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("Chronicling America OCR Downloader")
    log.info(f"  Output:     {args.output_dir.resolve()}")
    log.info(f"  Years:      {args.year_start}–{args.year_end}")
    log.info(f"  State:      {args.state or 'ALL'}")
    log.info(f"  Rate limit: {REQUEST_DELAY}s between requests")
    log.info("=" * 60)

    run_download(
        output_dir=args.output_dir,
        year_start=args.year_start,
        year_end=args.year_end,
        state_filter=args.state,
        retry_missing=args.retry_missing,
        recheck=args.recheck,
    )


if __name__ == "__main__":
    main()
