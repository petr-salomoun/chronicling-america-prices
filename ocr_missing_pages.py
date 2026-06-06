#!/usr/bin/env python3
"""
Chronicling America — Local OCR for Missing Pages
==================================================

Companion to download_chronicling_america.py.

Scans the downloaded data directory for pages whose OCR file exists but is
empty (0 bytes), meaning the LOC server had no OCR for that page.  For each
such page it:

  1. Derives the JP2 image URL from the corresponding Alto-XML URL stored in
     _pages.json.
  2. Downloads the JP2 image (without saving it to disk).
  3. Runs Tesseract OCR locally.
  4. Saves the result as  seq-NNN.ocr.txt  next to the existing  seq-NNN.txt.
     (The original empty file is left unchanged so the main downloader never
     re-fetches it.)

Resumability rules (same as the main downloader):
  - If  seq-NNN.ocr.txt  already exists (even if empty), it is skipped — the
    OCR attempt already happened, whether it produced text or not.
  - If the download or OCR fails, nothing is written; the page will be retried
    on the next run.
  - Progress is tracked in  data/raw/_ocr_progress.json  so completed issues
    and years are not re-scanned on restart.

Parallelism:
  - Use --workers N to run N concurrent OCR workers.
  - Each worker owns its own Tesseract API instance (not thread-safe to share).
  - All workers share a single rate-limited HTTP session with a token-bucket
    that enforces the LOC request rate across the whole process.
  - Progress/stats updates are protected by locks; file writes are atomic
    (tmp + rename) so concurrent workers never corrupt each other's output.
  - Work is dispatched at issue granularity: each worker takes one issue at a
    time from a shared queue, processes all its pages, then picks up the next.

Usage:
    # OCR all years with missing pages (single-threaded)
    python ocr_missing_pages.py

    # 8 parallel workers for a year range
    python ocr_missing_pages.py --year-start 1770 --year-end 1800 --workers 8

    # Limit to a specific year range
    python ocr_missing_pages.py --year-start 1913 --year-end 1940

    # Point at a non-default data directory
    python ocr_missing_pages.py --output-dir /mnt/data/chronam

    # Enable debug logging
    python ocr_missing_pages.py --verbose

Requirements:
    pip install pillow tesserocr requests
    tessdata/eng.traineddata  (downloaded automatically if absent)

Notes:
    - OCR speed: ~30s per full newspaper page on CPU.  8 workers yields ~4×
      real-time throughput (CPU-bound, gains depend on core count).
    - The tessdata directory defaults to  ./tessdata/  (relative to the script).
      Set --tessdata-dir to override.
    - Only English OCR is performed (lang=eng).  Multilingual newspapers will
      have degraded quality.
    - LOC rate limit is shared across all workers: total throughput is
      still bounded by ~20 requests/minute from this process.
"""

import argparse
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

# tesserocr must be imported in the main thread because its C extension
# (via cysignals) installs signal handlers, which is only allowed in the
# main thread.  Workers then call PyTessBaseAPI() to create per-thread
# instances (PyTessBaseAPI itself is not thread-safe).
try:
    import tesserocr as _tesserocr_module  # noqa: F401 — validates import at startup
except ImportError:
    _tesserocr_module = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUEST_DELAY = 3.1      # minimum seconds between HTTP requests (shared across workers)
REQUEST_TIMEOUT = 60     # seconds; JP2 images can be large — but don't stall workers too long
MAX_IMAGE_WIDTH = 2550   # scale JP2 to this width before OCR (8.5 in @ 300 dpi)
USER_AGENT = "ChronAmOCR/1.0 (Historical Research; github.com/chronam-pipeline)"

# Retry settings for image downloads
# Keep retries low: failed pages are not marked done and will be retried on
# restart anyway.  Fewer retries means a worker that hits a bad server
# gives up quickly and picks up the next issue rather than blocking for
# minutes on exponential back-off.  3 attempts = max wait 4+8=12 s.
_IMG_MAX_RETRIES = 3
_IMG_RETRY_STATUSES = {429, 500, 502, 503, 504, 525, 526}  # 525/526 = Cloudflare SSL errors
_IMG_BACKOFF_BASE = 4.0   # seconds; doubles each retry (4, 8 for 3 attempts)

TESSDATA_DEFAULT = Path(__file__).with_name("tessdata")
TESSDATA_DOWNLOAD_URL = (
    "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"
)

# ---------------------------------------------------------------------------
# Low-priority helpers
# ---------------------------------------------------------------------------


def _apply_low_priority() -> None:
    """Set the current process and all future threads/children to lowest CPU and I/O priority.

    Uses two mechanisms:
      1. os.nice(19)   — sets CPU scheduling to lowest priority (nice 19).
      2. ionice via subprocess — sets I/O scheduling to idle class (class 3).
         This applies to the whole process group so every thread and C-level
         sub-thread spawned by Tesseract inherits it.

    Both are best-effort: if either fails (e.g. running as a non-root user
    where nice() may fail for large increases), a warning is logged but the
    process continues normally.

    This should be called early in main() before any worker threads start.
    All POSIX threads share the same nice value within a process; Tesseract's
    internal C threads will also inherit the ionice class because ionice is
    per-process on Linux.
    """
    # --- CPU nice ---
    try:
        current = os.nice(0)          # read current nice value
        increment = 19 - current      # we want nice 19
        if increment > 0:
            os.nice(increment)
            log.info(f"CPU priority: set nice to {os.nice(0)}")
        else:
            log.info(f"CPU priority: already at nice {current}, no change")
    except OSError as e:
        log.warning(f"Could not set CPU nice level: {e}")

    # --- I/O scheduling: idle class (ionice -c 3) ---
    # Apply to the whole current process using ionice.
    try:
        pid = os.getpid()
        result = subprocess.run(
            ["ionice", "-c", "3", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            log.info(f"I/O priority: set to idle class (ionice -c 3) for PID {pid}")
        else:
            log.warning(
                f"ionice returned non-zero ({result.returncode}): {result.stderr.strip()}"
            )
    except FileNotFoundError:
        log.warning("ionice not found — I/O priority not adjusted")
    except Exception as e:
        log.warning(f"Could not set I/O priority: {e}")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("chronam_ocr")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        log.warning("Force quit.")
        sys.exit(1)
    log.info("Shutdown requested — finishing current pages then saving state...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# Progress state
# ---------------------------------------------------------------------------


@dataclass
class OcrStats:
    pages_ocr_done: int = 0       # pages successfully OCR'd (even if empty result)
    pages_ocr_text: int = 0       # pages that produced non-empty OCR text
    pages_skipped: int = 0        # already had .ocr.txt or no xml_url
    pages_failed: int = 0         # download or OCR error (will be retried)
    issues_processed: int = 0
    last_updated: str = ""


@dataclass
class OcrProgressState:
    years_completed: list = field(default_factory=list)
    current_year: Optional[int] = None
    current_year_issues_completed: list = field(default_factory=list)
    stats: dict = field(default_factory=lambda: asdict(OcrStats()))

    def is_year_done(self, year: int) -> bool:
        return year in self.years_completed

    def is_issue_done(self, issue_key: str) -> bool:
        return issue_key in self.current_year_issues_completed

    def mark_issue_done(self, issue_key: str):
        if issue_key not in self.current_year_issues_completed:
            self.current_year_issues_completed.append(issue_key)
        self.stats["issues_processed"] += 1

    def mark_year_done(self, year: int):
        if year not in self.years_completed:
            self.years_completed.append(year)
        self.current_year = None
        self.current_year_issues_completed = []

    def update_timestamp(self):
        self.stats["last_updated"] = datetime.now(timezone.utc).isoformat()


class OcrProgressTracker:
    """Persistent progress state backed by a JSON file.

    All public methods are protected by an internal lock so they are safe to
    call from multiple worker threads simultaneously.
    """

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self._lock = threading.Lock()
        self.state = self._load()

    def _load(self) -> OcrProgressState:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                allowed = {f.name for f in OcrProgressState.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in allowed}
                log.info(f"Resumed OCR progress from {self.state_path}")
                return OcrProgressState(**filtered)
            except (json.JSONDecodeError, TypeError) as e:
                log.warning(f"Corrupted OCR progress file, starting fresh: {e}")
        return OcrProgressState()

    def save(self):
        """Write state to disk atomically.  Acquires the internal lock."""
        with self._lock:
            self._save_locked()

    def _save_locked(self):
        """Write state to disk — caller must hold self._lock."""
        self.state.update_timestamp()
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self.state), f, indent=2)
        tmp.replace(self.state_path)

    def mark_issue_done(self, issue_key: str):
        with self._lock:
            self.state.mark_issue_done(issue_key)

    def is_issue_done(self, issue_key: str) -> bool:
        with self._lock:
            return self.state.is_issue_done(issue_key)

    def mark_year_done(self, year: int):
        with self._lock:
            self.state.mark_year_done(year)
            self._save_locked()

    def add_stats(self, **delta):
        """Atomically add delta values to stats counters."""
        with self._lock:
            for key, val in delta.items():
                self.state.stats[key] += val

    def maybe_save_and_log(self, interval: int = 10):
        """Save and log stats if issues_processed is a multiple of *interval*."""
        with self._lock:
            if self.state.stats["issues_processed"] % interval == 0:
                self._save_locked()
                self._log_stats_locked()

    def _log_stats_locked(self):
        s = self.state.stats
        log.info(
            f"OCR progress: {s['pages_ocr_done']} pages OCR'd "
            f"({s['pages_ocr_text']} with text), "
            f"{s['pages_skipped']} skipped, "
            f"{s['pages_failed']} failed, "
            f"{s['issues_processed']} issues, "
            f"years done: {len(self.state.years_completed)}"
        )

    def log_stats(self):
        with self._lock:
            self._log_stats_locked()


# ---------------------------------------------------------------------------
# Shared rate-limited HTTP session
# ---------------------------------------------------------------------------


class RateLimitedSession:
    """Thread-safe requests.Session with a shared token-bucket rate limiter.

    All worker threads share one instance so total HTTP throughput across
    the process respects REQUEST_DELAY between consecutive requests.

    Design: the lock is held only for a very short critical section that
    atomically *reserves* the next request slot (by advancing
    _next_allowed_time).  The actual sleep happens **outside** the lock so
    that other workers can concurrently compute and sleep their own wait
    times rather than queuing behind a single sleeping thread.

    Example with 4 workers all requesting at t=0 and delay=3.1 s:
      - W1 acquires lock, reserves slot at t=0,   releases lock, sleeps 0 s
      - W2 acquires lock, reserves slot at t=3.1,  releases lock, sleeps 3.1 s
      - W3 acquires lock, reserves slot at t=6.2,  releases lock, sleeps 6.2 s
      - W4 acquires lock, reserves slot at t=9.3,  releases lock, sleeps 9.3 s
    All four workers sleep in parallel; no worker blocks another.

    The lock is **not** held during the sleep, the connect/send, or the
    response body transfer — only during the tiny slot-reservation window.
    """

    def __init__(self, delay: float = REQUEST_DELAY):
        self._delay = delay
        self._lock = threading.Lock()
        # Absolute monotonic time at which the next request may be sent.
        # Initialise to 0 so the very first request fires immediately.
        self._next_allowed_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET *url*, blocking until this worker's rate-limit slot arrives.

        Uses stream=True so the caller reads the body outside the lock.
        Pass stream=False to override (e.g. for small metadata responses).
        """
        kwargs.setdefault("stream", True)

        # --- Step 1: atomically reserve a request slot ---
        # The lock is held only for this brief arithmetic; no I/O or sleep.
        with self._lock:
            now = time.monotonic()
            send_at = max(self._next_allowed_time, now)
            self._next_allowed_time = send_at + self._delay

        # --- Step 2: sleep outside the lock until our slot arrives ---
        wait = send_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)

        # --- Step 3: send the request (still outside the lock) ---
        resp = self._session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        # Body is read by the caller outside this method — large JP2 transfers
        # do not affect other workers' ability to reserve their own slots.
        return resp


# ---------------------------------------------------------------------------
# Tessdata bootstrap (run once before starting workers)
# ---------------------------------------------------------------------------


def _ensure_tessdata(tessdata_dir: Path) -> None:
    """Download eng.traineddata if absent.  Must be called before workers start."""
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    traineddata = tessdata_dir / "eng.traineddata"
    if traineddata.exists():
        return
    log.info(
        f"Downloading eng.traineddata to {tessdata_dir} "
        f"(~15 MB, one-time download)..."
    )
    r = requests.get(
        TESSDATA_DOWNLOAD_URL,
        timeout=120,
        headers={"User-Agent": USER_AGENT},
        stream=True,
    )
    r.raise_for_status()
    tmp = traineddata.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    tmp.rename(traineddata)
    log.info("eng.traineddata downloaded.")


# ---------------------------------------------------------------------------
# Per-worker OCR engine factory
# ---------------------------------------------------------------------------


def _make_ocr_engine(tessdata_dir: Path):
    """Create and return a new PyTessBaseAPI for the calling thread.

    Each worker thread must create its own instance — PyTessBaseAPI is not
    thread-safe.  Called once per worker at thread start.

    The tesserocr module is imported at startup in the main thread (so that
    cysignals can install signal handlers there); here we just reference it.
    """
    if _tesserocr_module is None:
        log.error("tesserocr is not installed.  Run:  pip install tesserocr")
        sys.exit(1)

    return _tesserocr_module.PyTessBaseAPI(
        path=str(tessdata_dir) + "/",
        lang="eng",
    )


# ---------------------------------------------------------------------------
# Image download + scaling
# ---------------------------------------------------------------------------


def _xml_url_to_jp2_url(xml_url: str) -> str:
    """Convert a tile.loc.gov Alto-XML URL to the corresponding JP2 URL."""
    if xml_url.lower().endswith(".xml"):
        return xml_url[:-4] + ".jp2"
    base, _, _ = xml_url.rpartition(".")
    return base + ".jp2"


def _download_image(
    jp2_url: str, session: RateLimitedSession
) -> Optional[Image.Image]:
    """Download a JP2 image and return a scaled RGB PIL Image (or None).

    Uses streaming chunked reads to avoid IncompleteRead errors that occur
    when reading large JP2 files (~3 MB) in one shot.  Retries up to
    _IMG_MAX_RETRIES times with exponential backoff for:
      - IncompleteRead / ChunkedEncodingError  (connection dropped mid-body)
      - Transient HTTP errors (429, 500–504, 525, 526)
    """
    from http.client import IncompleteRead as _IncompleteRead
    from urllib3.exceptions import IncompleteRead as _Urllib3IncompleteRead

    for attempt in range(1, _IMG_MAX_RETRIES + 1):
        if _shutdown_requested:
            return None

        try:
            resp = session.get(jp2_url)  # stream=True by default
        except requests.RequestException as e:
            wait = _IMG_BACKOFF_BASE * (2 ** (attempt - 1))
            if attempt < _IMG_MAX_RETRIES:
                log.debug(
                    f"Request failed (attempt {attempt}/{_IMG_MAX_RETRIES}): "
                    f"{jp2_url} — {e}. Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
                continue
            log.warning(f"Request failed: {jp2_url} — {e}")
            return None

        # Permanent failures — do not retry
        if resp.status_code == 404:
            log.debug(f"404: {jp2_url}")
            return None
        if resp.status_code == 403:
            log.debug(f"403: {jp2_url}")
            return None

        # Transient server-side failures — retry with backoff
        if resp.status_code in _IMG_RETRY_STATUSES:
            wait = _IMG_BACKOFF_BASE * (2 ** (attempt - 1))
            if resp.status_code == 429:
                wait = max(wait, 120)
            if attempt < _IMG_MAX_RETRIES:
                log.debug(
                    f"HTTP {resp.status_code} (attempt {attempt}/{_IMG_MAX_RETRIES}): "
                    f"{jp2_url}. Retrying in {wait:.0f}s..."
                )
                resp.close()
                time.sleep(wait)
                continue
            log.warning(f"HTTP {resp.status_code}: {jp2_url}")
            return None

        if not resp.ok:
            log.warning(f"HTTP {resp.status_code}: {jp2_url}")
            return None

        # Read the response body in chunks to avoid IncompleteRead on large JP2s
        buf = bytearray()
        try:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    buf.extend(chunk)
        except (
            requests.exceptions.ChunkedEncodingError,
            _IncompleteRead,
            _Urllib3IncompleteRead,
            ConnectionError,
        ) as e:
            resp.close()
            wait = _IMG_BACKOFF_BASE * (2 ** (attempt - 1))
            if attempt < _IMG_MAX_RETRIES:
                log.debug(
                    f"Incomplete download (attempt {attempt}/{_IMG_MAX_RETRIES}): "
                    f"{jp2_url} — {e}. Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
                continue
            log.warning(f"Request failed: {jp2_url} — {e}")
            return None
        finally:
            resp.close()

        # Decode the image
        try:
            img = Image.open(BytesIO(bytes(buf))).convert("RGB")
        except Exception as e:
            log.warning(f"Cannot open image from {jp2_url}: {e}")
            return None

        w, h = img.size
        if w > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / w
            img = img.resize((MAX_IMAGE_WIDTH, int(h * scale)), Image.LANCZOS)
        return img

    # Exhausted all retries
    log.warning(f"Gave up downloading {jp2_url} after {_IMG_MAX_RETRIES} attempts")
    return None


# ---------------------------------------------------------------------------
# Per-issue OCR worker (called from thread pool)
# ---------------------------------------------------------------------------


def _ocr_one_issue(
    issue_dir: Path,
    tessdata_dir: Path,
    session: RateLimitedSession,
    tracker: OcrProgressTracker,
    worker_id: int,
) -> bool:
    """OCR all empty pages for a single newspaper issue.

    Creates its own Tesseract engine instance on first actual OCR need.

    Return values:
      True  — issue fully processed (all pages either OCR'd or permanently
              skipped); safe to mark as done in the progress tracker.
      False — interrupted by Ctrl-C, OR one or more page downloads/OCR
              attempts failed.  The issue must NOT be marked done so that
              the failed pages are retried on the next run.
              Already-written .ocr.txt files are never re-processed, so
              pages that succeeded in this run are not duplicated.
    """
    pages_cache = issue_dir / "_pages.json"
    if not pages_cache.exists():
        log.debug(f"[W{worker_id}] No _pages.json in {issue_dir}, skipping.")
        return True

    with open(pages_cache) as f:
        pages = json.load(f)

    # Lazily create a per-thread OCR engine only when we find real work to do.
    ocr_api = None

    stats_delta = dict(
        pages_ocr_done=0, pages_ocr_text=0, pages_skipped=0, pages_failed=0
    )
    had_failures = False  # set True if any page download or OCR fails

    for page in pages:
        if _shutdown_requested:
            tracker.add_stats(**stats_delta)
            return False

        seq = page.get("seq", 0)
        xml_url = page.get("xml_url", "")

        empty_txt = issue_dir / f"seq-{str(seq).zfill(3)}.txt"
        ocr_txt = issue_dir / f"seq-{str(seq).zfill(3)}.ocr.txt"

        # Process pages that need local OCR:
        #   1. seq-NNN.txt exists but is empty (0 bytes) — LOC had no OCR text
        #   2. seq-NNN.txt does not exist — download was never attempted or failed
        # Skip pages where seq-NNN.txt has actual content (LOC OCR succeeded).
        if empty_txt.exists() and empty_txt.stat().st_size > 0:
            stats_delta["pages_skipped"] += 1
            continue

        # Already OCR'd (even if the result was empty) — skip.
        if ocr_txt.exists():
            stats_delta["pages_skipped"] += 1
            continue

        if not xml_url:
            log.debug(
                f"[W{worker_id}] No xml_url for {issue_dir.name}/seq-{seq} "
                "— writing empty OCR marker"
            )
            ocr_txt.touch()
            stats_delta["pages_skipped"] += 1
            continue

        # First real work: create per-thread OCR engine
        if ocr_api is None:
            ocr_api = _make_ocr_engine(tessdata_dir)

        jp2_url = _xml_url_to_jp2_url(xml_url)
        log.debug(f"[W{worker_id}] OCR {issue_dir.name}/seq-{seq}")

        img = _download_image(jp2_url, session)
        if img is None:
            log.debug(
                f"[W{worker_id}] Failed to download image for "
                f"{issue_dir.name}/seq-{seq} — will retry on restart"
            )
            stats_delta["pages_failed"] += 1
            had_failures = True
            continue

        try:
            ocr_api.SetImage(img)
            text = ocr_api.GetUTF8Text()
        except Exception as e:
            log.warning(
                f"[W{worker_id}] OCR error on {issue_dir.name}/seq-{seq}: "
                f"{e} — will retry on restart"
            )
            stats_delta["pages_failed"] += 1
            had_failures = True
            continue
        finally:
            img.close()
            del img

        # Write result atomically (empty result → 0-byte marker)
        tmp = ocr_txt.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        tmp.rename(ocr_txt)

        stats_delta["pages_ocr_done"] += 1
        if text.strip():
            stats_delta["pages_ocr_text"] += 1
            log.debug(
                f"[W{worker_id}]   seq-{seq}: {len(text.strip())} chars"
            )
        else:
            log.debug(
                f"[W{worker_id}]   seq-{seq}: empty OCR result — stored empty marker"
            )

    if ocr_api is not None:
        try:
            ocr_api.End()
        except Exception:
            pass

    tracker.add_stats(**stats_delta)
    # Return False when any page failed so the issue is not marked done;
    # it will be retried on the next run (already-written pages are skipped).
    return not had_failures


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_ocr(
    output_dir: Path,
    year_start: int,
    year_end: int,
    tessdata_dir: Path,
    workers: int = 1,
    recheck: bool = False,
):
    raw_dir = output_dir / "raw"
    if not raw_dir.exists():
        log.error(
            f"Data directory {raw_dir} does not exist. Run the downloader first."
        )
        sys.exit(1)

    # Ensure tessdata is present before spawning any threads (avoids a race
    # where multiple threads all try to download eng.traineddata simultaneously).
    _ensure_tessdata(tessdata_dir)

    tracker = OcrProgressTracker(raw_dir / "_ocr_progress.json")
    session = RateLimitedSession(delay=REQUEST_DELAY)

    # ------------------------------------------------------------------
    # --recheck: re-open completed years so newly downloaded pages (that
    # still lack OCR) are picked up. Issue-level done markers are also
    # cleared for the affected years.
    # ------------------------------------------------------------------
    if recheck:
        rechecked = 0
        for year in range(year_start, year_end + 1):
            if year in tracker.state.years_completed:
                tracker.state.years_completed.remove(year)
                rechecked += 1
        # Also clear issue-level done markers so they are re-evaluated
        if rechecked:
            tracker.state.current_year_issues_completed = []
            log.info(
                f"--recheck: re-opened {rechecked} year(s) for re-scanning "
                "newly downloaded pages."
            )
            tracker.save()

    # ------------------------------------------------------------------
    # Collect all pending issues across the requested year range into a
    # shared thread-safe queue.  Years/issues already marked done are
    # filtered out here in the main thread so workers never see them.
    # ------------------------------------------------------------------
    issue_queue: queue.Queue = queue.Queue()
    years_in_queue: list[int] = []   # years whose issues are enqueued
    issue_count_by_year: dict[int, int] = {}

    for year in range(year_start, year_end + 1):
        if tracker.state.is_year_done(year):
            log.info(f"Year {year}: OCR already complete, skipping.")
            continue

        year_dir = raw_dir / str(year)
        if not year_dir.exists():
            log.debug(f"Year {year}: directory absent, skipping.")
            continue

        issues_cache = year_dir / "_issues.json"
        if not issues_cache.exists():
            log.debug(f"Year {year}: no _issues.json, skipping.")
            continue

        with open(issues_cache) as f:
            issues = json.load(f)

        if not issues:
            log.debug(f"Year {year}: 0 issues cached — marking done.")
            tracker.mark_year_done(year)
            continue

        pending_issues = []
        for issue in sorted(issues, key=lambda i: (i["lccn"], i["date"])):
            lccn = issue.get("lccn", "")
            date = issue.get("date", "")
            if not lccn or not date:
                continue
            issue_key = f"{lccn}/{date}"
            if tracker.is_issue_done(issue_key):
                continue
            issue_dir = year_dir / lccn / date
            if not issue_dir.exists():
                continue
            pending_issues.append((year, issue_key, issue_dir))

        if not pending_issues:
            # All issues for this year were already done
            log.info(f"Year {year}: all issues already OCR'd — marking year done.")
            tracker.mark_year_done(year)
            continue

        for item in pending_issues:
            issue_queue.put(item)
        years_in_queue.append(year)
        issue_count_by_year[year] = len(pending_issues)
        log.info(
            f"Year {year}: queued {len(pending_issues)} issues for OCR."
        )

    total_issues = issue_queue.qsize()
    if total_issues == 0:
        log.info("Nothing to do — all issues already OCR'd.")
        tracker.log_stats()
        return

    log.info(
        f"Starting {workers} worker(s) for {total_issues} pending issue(s) "
        f"across {len(years_in_queue)} year(s)..."
    )

    # ------------------------------------------------------------------
    # Track which issues are completed per year so we can mark years done.
    # ------------------------------------------------------------------
    # issues_remaining[year] = count of issues not yet finished by any worker
    issues_remaining: dict[int, int] = dict(issue_count_by_year)
    remaining_lock = threading.Lock()

    def _worker(worker_id: int):
        while not _shutdown_requested:
            try:
                year, issue_key, issue_dir = issue_queue.get(timeout=0.5)
            except queue.Empty:
                break  # queue exhausted — this worker is done

            lccn_date = "/".join(issue_dir.parts[-2:])
            log.info(f"[W{worker_id}] {lccn_date}")

            ok = _ocr_one_issue(issue_dir, tessdata_dir, session, tracker, worker_id)

            if ok:
                tracker.mark_issue_done(issue_key)
                tracker.maybe_save_and_log(interval=10)

                # Check if this was the last issue in its year
                with remaining_lock:
                    issues_remaining[year] -= 1
                    year_done = issues_remaining[year] == 0

                if year_done and not _shutdown_requested:
                    tracker.mark_year_done(year)
                    log.info(f"=== Year {year} OCR complete ===")
            else:
                # Not marking issue done: either interrupted (Ctrl-C) or some
                # pages failed to download/OCR.  On the next run the issue will
                # be re-queued; pages that already have a .ocr.txt file are
                # skipped automatically so only the failures are retried.
                if _shutdown_requested:
                    log.debug(
                        f"[W{worker_id}] Issue {issue_key} interrupted — "
                        "not marking done"
                    )
                else:
                    log.debug(
                        f"[W{worker_id}] Issue {issue_key} had page failures — "
                        "not marking done so failures will be retried on restart"
                    )

            issue_queue.task_done()

    # ------------------------------------------------------------------
    # Launch worker threads
    # ------------------------------------------------------------------
    threads = [
        threading.Thread(target=_worker, args=(i + 1,), daemon=True, name=f"ocr-{i+1}")
        for i in range(workers)
    ]
    for t in threads:
        t.start()

    # Main thread waits, periodically logging progress
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=30)
            if any(t.is_alive() for t in threads):
                tracker.log_stats()
    except KeyboardInterrupt:
        # Signal handler already set _shutdown_requested; just wait for workers
        pass

    # Final save
    tracker.save()
    tracker.log_stats()

    if _shutdown_requested:
        log.info("OCR paused. Restart to resume.")
    else:
        log.info("OCR complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run local Tesseract OCR on Chronicling America pages that have no "
            "LOC-provided OCR text (empty seq-NNN.txt files)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OCR all years, single-threaded (resumes automatically)
  python ocr_missing_pages.py

  # 8 parallel workers over 1770-1800
  python ocr_missing_pages.py --year-start 1770 --year-end 1800 --workers 8

  # OCR only 1913-1940 (where server errors left many empty pages)
  python ocr_missing_pages.py --year-start 1913 --year-end 1940

  # Use a different data directory
  python ocr_missing_pages.py --output-dir /mnt/data/chronam

Notes:
  - Output files are named seq-NNN.ocr.txt alongside the original seq-NNN.txt
  - seq-NNN.txt (original, may be empty) is never modified
  - seq-NNN.ocr.txt existence (even 0 bytes) means OCR was attempted; no retry
  - If download or OCR fails, no file is written and the page retried on restart
  - Progress is tracked in data/raw/_ocr_progress.json
  - eng.traineddata (~15 MB) is downloaded automatically on first run
  - Each worker thread owns its own Tesseract engine instance
  - HTTP requests are rate-limited across all workers (shared token bucket)
  - CPU (nice 19) and I/O (ionice idle) low-priority scheduling is applied by
    default so OCR does not slow down interactive work; use --no-low-priority
    to run at normal scheduling priority
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
        help="First year to process (default: 1770)",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=1963,
        help="Last year to process (default: 1963)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel OCR workers (default: 1). "
            "Each worker downloads and OCRs independently. "
            "HTTP requests are still rate-limited across all workers."
        ),
    )
    parser.add_argument(
        "--tessdata-dir",
        type=Path,
        default=TESSDATA_DEFAULT,
        help=f"Path to tessdata directory (default: {TESSDATA_DEFAULT})",
    )
    parser.add_argument(
        "--low-priority",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Set CPU (nice 19) and I/O (ionice idle) scheduling to lowest priority "
            "so the OCR process does not slow down interactive work. "
            "Applied at startup before any worker threads start so Tesseract's "
            "internal C threads also inherit the priority settings. "
            "Enabled by default; use --no-low-priority to disable."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help=(
            "Re-open all completed years in the requested range so they are "
            "scanned again for pages that still need OCR (e.g. after the "
            "downloader fetched new pages). Pages that already have a .ocr.txt "
            "file are never re-processed."
        ),
    )

    args = parser.parse_args()

    if args.workers < 1:
        log.error("--workers must be >= 1")
        sys.exit(1)

    if args.verbose:
        logging.getLogger("chronam_ocr").setLevel(logging.DEBUG)

    # Apply low-priority scheduling before any worker threads start.
    # This ensures all Python threads AND Tesseract's internal C-level threads
    # inherit the same process-wide nice and ionice settings.
    # Enabled by default; suppressed only with --no-low-priority.
    if args.low_priority:
        _apply_low_priority()

    log.info("=" * 60)
    log.info("Chronicling America — Local OCR for Missing Pages")
    log.info(f"  Output:    {args.output_dir.resolve()}")
    log.info(f"  Years:     {args.year_start}–{args.year_end}")
    log.info(f"  Workers:   {args.workers}")
    log.info(f"  Tessdata:  {args.tessdata_dir}")
    log.info(f"  Low priority: {args.low_priority}")
    log.info("=" * 60)

    run_ocr(
        output_dir=args.output_dir,
        year_start=args.year_start,
        year_end=args.year_end,
        tessdata_dir=args.tessdata_dir,
        workers=args.workers,
        recheck=args.recheck,
    )


if __name__ == "__main__":
    main()
