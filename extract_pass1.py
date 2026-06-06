#!/usr/bin/env python3
"""
Chronicling America — Phase 1: LLM Extraction (extract_pass1.py)
=================================================================

Processes downloaded OCR text files and extracts:
  - People co-mentions  ("t":"P")
  - Prices/monetary values ("t":"$")

using an LLM via the LiteLLM-compatible OpenAI chat-completions API.

DESIGN PRINCIPLES:
  1. Maximise context window usage: pack multiple pages per LLM call using exact
     token counts (tiktoken).  Pages from different issues are bin-packed together.
  2. Text source: prefers compressed OCR from data/pass0/ if present on a
     per-issue basis; falls back to data/raw/ for issues not yet compressed.
  3. Split outputs by record type:
       data/pass1/social/{year}/{lccn}_{date}.jsonl  — people co-mention records
       data/pass1/prices/{year}/{lccn}_{date}.jsonl  — price/monetary records
  4. Sentinel files: data/raw/{y}/{lccn}/{date}/_pass1_done.json records which
     page refs have been processed.  On re-run only missing pages are re-queued.
  5. Restart-safe: JSONL appended directly; sentinel written only after success.
  6. Partial-issue recovery: failed batches leave sentinels untouched → retry.
  7. Empty placeholder: when an issue produces no extractable records (LLM
     returns []), a zero-byte placeholder JSONL file is written in both social/
     and prices/ directories so downstream tooling can distinguish
     "processed-but-empty" from "never processed".  The sentinel prevents
     re-processing on subsequent runs.

Usage:
    python extract_pass1.py [options]

    # Process 1776 only
    python extract_pass1.py --year-start 1776 --year-end 1776

    # Process 1880-1890, verbose
    python extract_pass1.py --year-start 1880 --year-end 1890 --verbose

    # Dry run (shows what would be processed, no LLM calls)
    python extract_pass1.py --year-start 1776 --year-end 1776 --dry-run

    # Use GitHub Copilot as the LLM backend
    python extract_pass1.py --year-start 1776 --year-end 1776 --backend copilot

    # Auto mode: try LiteLLM proxy first, fall back to GitHub Copilot
    python extract_pass1.py --year-start 1880 --year-end 1890 --backend auto

Environment variables:
    LITELLM_PROXY_BASE      Base URL of LiteLLM proxy (default: http://ai-tools.cz.intinfra.com:4004)
    LITELLM_PROXY_API_KEY   API key (also checked as LITELLM_API_KEY)
    CHRONAM_MODEL           Model name (default: gpt-5-mini)
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import tiktoken
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_litellm_base_raw = os.getenv(
    "LITELLM_PROXY_BASE", "http://ai-tools.cz.intinfra.com:4004"
).rstrip("/")
# If LITELLM_PROXY_BASE already includes the /chat/completions path, use as-is.
if _litellm_base_raw.endswith("/chat/completions"):
    LITELLM_ENDPOINT = _litellm_base_raw
    LITELLM_BASE = _litellm_base_raw[: -len("/chat/completions")]
else:
    LITELLM_BASE = _litellm_base_raw
    LITELLM_ENDPOINT = f"{LITELLM_BASE}/chat/completions"
API_KEY = os.getenv("LITELLM_PROXY_API_KEY", os.getenv("LITELLM_API_KEY", ""))
DEFAULT_MODEL = os.getenv("CHRONAM_MODEL", "gpt-5-mini")

# ---------------------------------------------------------------------------
# GitHub Copilot backend
# ---------------------------------------------------------------------------

GITHUB_COPILOT_CONFIG = Path.home() / ".config" / "github-copilot" / "apps.json"
GITHUB_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_DEVICE_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"  # GitHub Copilot for Vim

_copilot_session_token = None
_copilot_session_expires = 0.0
_copilot_api_endpoint: Optional[str] = None

# Global backend mode — set from --backend argument
_llm_backend = "litellm"  # "litellm" | "copilot" | "auto"


def set_llm_backend(backend: str) -> None:
    global _llm_backend
    _llm_backend = backend


def _github_device_registration() -> Optional[str]:
    """Prompt user to register device with GitHub for Copilot access."""
    print("\n" + "=" * 60)
    print("GitHub Copilot Device Registration Required")
    print("=" * 60)

    resp = requests.post(
        GITHUB_DEVICE_CODE_URL,
        data={"client_id": GITHUB_COPILOT_CLIENT_ID, "scope": "copilot"},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    if not resp.ok:
        print(f"Failed to get device code: {resp.status_code}")
        return None

    data = resp.json()
    user_code = data["user_code"]
    device_code = data["device_code"]
    verification_uri = data["verification_uri"]
    interval = data.get("interval", 5)
    expires_in = data.get("expires_in", 900)

    print(f"\n1. Open: {verification_uri}")
    print(f"2. Enter code: {user_code}")
    print(f"\nWaiting for authorization (expires in {expires_in // 60} minutes)...")

    start = time.time()
    while time.time() - start < expires_in:
        time.sleep(interval)
        resp = requests.post(
            GITHUB_DEVICE_TOKEN_URL,
            data={
                "client_id": GITHUB_COPILOT_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        data = resp.json()
        if "access_token" in data:
            token = data["access_token"]
            GITHUB_COPILOT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            config_data: dict = {}
            if GITHUB_COPILOT_CONFIG.exists():
                config_data = json.loads(GITHUB_COPILOT_CONFIG.read_text())
            config_data[f"github.com:{GITHUB_COPILOT_CLIENT_ID}"] = {
                "oauth_token": token,
                "user": "device-flow-user",
                "githubAppId": GITHUB_COPILOT_CLIENT_ID,
            }
            GITHUB_COPILOT_CONFIG.write_text(json.dumps(config_data))
            print("\nAuthorization successful! Token saved.")
            return token
        error = data.get("error")
        if error == "authorization_pending":
            continue
        elif error == "slow_down":
            interval += 5
        elif error == "expired_token":
            print("\nCode expired. Please try again.")
            return None
        elif error == "access_denied":
            print("\nAuthorization denied.")
            return None
        else:
            print(f"\nError: {error}")
            return None

    print("\nTimeout waiting for authorization.")
    return None


def _get_copilot_oauth_token() -> Optional[str]:
    """Get GitHub OAuth token from config or prompt for device registration."""
    if GITHUB_COPILOT_CONFIG.exists():
        try:
            config = json.loads(GITHUB_COPILOT_CONFIG.read_text())
            for value in config.values():
                if "oauth_token" in value:
                    return value["oauth_token"]
        except (json.JSONDecodeError, KeyError):
            pass
    return _github_device_registration()


def _get_copilot_session_token() -> tuple[Optional[str], Optional[str]]:
    """Get (session_token, api_endpoint), refreshing if expired."""
    global _copilot_session_token, _copilot_session_expires, _copilot_api_endpoint

    if _copilot_session_token and time.time() < _copilot_session_expires - 60:
        return _copilot_session_token, _copilot_api_endpoint

    oauth_token = _get_copilot_oauth_token()
    if not oauth_token:
        return None, None

    try:
        resp = requests.get(
            GITHUB_COPILOT_TOKEN_URL,
            headers={
                "Authorization": f"token {oauth_token}",
                "Accept": "application/json",
                "Editor-Plugin-Version": "copilot.vim/1.50.0",
                "Editor-Version": "Neovim/0.10.0",
                "User-Agent": "GithubCopilot/1.50.0",
            },
            timeout=30,
        )
        if not resp.ok:
            print(f"Failed to get Copilot session token: {resp.status_code}")
            return None, None
        data = resp.json()
        _copilot_session_token = data["token"]
        _copilot_session_expires = data.get("expires_at", time.time() + 3600)
        _copilot_api_endpoint = data.get(
            "endpoints", {}
        ).get("api", "https://api.business.githubcopilot.com")
        return _copilot_session_token, _copilot_api_endpoint
    except Exception as e:
        print(f"Error getting Copilot session: {e}")
        return None, None


def _copilot_chat_completion(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    """Call GitHub Copilot chat completions API (non-streaming)."""
    token, endpoint = _get_copilot_session_token()
    if not token or not endpoint:
        return None

    url = f"{endpoint}/chat/completions"
    payload = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Copilot-Integration-Id": "vscode-chat",
                "Editor-Plugin-Version": "copilot-chat/0.22.0",
                "Editor-Version": "vscode/1.95.0",
            },
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        if not resp.ok:
            print(f"Copilot API error: {resp.status_code} {resp.text[:200]}")
            return None
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Copilot API exception: {e}")
        return None


# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------

# Use cl100k_base (GPT-4/3.5-turbo tokenizer) as the best approximation for
# GPT-family models. Falls back gracefully if the model name is unknown.
try:
    _TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")
except KeyError:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using the GPT-4 tokenizer."""
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# Maximum *input* tokens per LLM call.
# System prompt is ~800 tokens; extraction output is compact (4K ceiling).
# At 28 000 input: total context ≈ 28 000 + 800 + 4 096 = ~32 900 tokens —
# fits a 32K window and uses essentially the full capacity.
# Raise to ~120 000 if the deployed model supports a larger context.
MAX_TOKENS_PER_CALL = 28_000

# Max output tokens per call. Extraction output is compact; 4K is generous.
MAX_OUTPUT_TOKENS = 4096

# Temperature 0 → deterministic extraction
TEMPERATURE = 0.0

# Retry settings for LLM API calls
LLM_MAX_RETRIES = 4
LLM_BACKOFF_FACTOR = 2.0
LLM_RETRY_STATUSES = [429, 500, 502, 503, 504]

# LLM request timeout: (connect_timeout, read_timeout) in seconds.
# Large models can take several minutes to generate a response.
LLM_TIMEOUT = (15, 600)  # 15s connect, 10 min read

# Minimum OCR text length (chars) to bother sending to LLM (very short = blank/garbled)
MIN_PAGE_CHARS = 80

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pass1")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    if _shutdown:
        log.warning("Force quit.")
        sys.exit(1)
    log.info("Shutdown requested — will stop after current issue...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise data extraction engine processing OCR text from historical
American newspapers (1770–1963). The OCR is noisy — expect misspellings, broken words, and
garbled characters. Use context to infer correct readings.

Your task: extract ONLY two types of information, discard everything else.

TYPE 1 — PEOPLE CO-MENTIONS (output tag: "P")
Find groups of 2 or more named people mentioned in the SAME article or paragraph context.
For each group output:
- "t": "P"
- "names": array of full names as they appear (fix obvious OCR errors)
- "rel": one short phrase describing their relationship or why they appear together
  (e.g. "married", "plaintiff and defendant", "board members",
  "mentioned in same crime report"). Keep under 15 words.
- "ref": the page sentinel key (provided in the user message header for each page)

Rules:
- A "person" must be a specific named individual, not a generic role ("the mayor")
- Include titles if present (Dr., Gen., Mrs., Rev., Sen., etc.)
- If the SAME person appears with DIFFERENT people in different article contexts within the
  same page, output SEPARATE records for each group
- Do NOT output single-person mentions — minimum 2 people per record
- Advertisements listing multiple business owners count
- Obituaries mentioning surviving family members count

TYPE 2 — PRICES AND MONETARY VALUES (output tag: "$")
Find any mention of a specific price, wage, cost, rent, or monetary value attached to an
identifiable item, commodity, service, property, or wage.
- "t": "$"
- "item": what is being priced, as specifically as stated (include quantity/unit if given,
  e.g. "wheat per bushel", "board per week")
- "price": the price as stated, including currency symbol and original format
- "note": optional, 1–5 words of context if helpful (e.g. "auction", "wholesale")
- "ref": the page sentinel key (provided in the user message header for each page)

Rules:
- Include prices from: articles, advertisements, market reports, auction notices, real
  estate listings, help wanted ads, legal notices
- Include wages, rents, fares, tolls, fees, fines, bounties, rewards
- Do NOT include numbers that are not prices (vote counts, populations, etc.)
- If a market report lists many commodities, output one record per commodity
- Preserve the original price format — do NOT convert or normalize

OUTPUT FORMAT:
Return a JSON array (and nothing else outside the array). Each element is one of:
  {"t":"P","names":[...],"rel":"...","ref":"..."}
  {"t":"$","item":"...","price":"...","note":"...","ref":"..."}

If NO extractable information exists across ALL pages provided, return exactly: []

IMPORTANT:
- The "ref" in each record MUST match the sentinel key shown in the page header
- Do NOT invent ref values
- Do NOT wrap the array in any object — just output the bare JSON array"""


def build_user_message(pages: list[dict]) -> str:
    """Build a user message packing pages from one or more issues.

    Each page dict must have:
        "ref":   sentinel key  (e.g. "sn82016139/1776-07-15/seq-001")
        "text":  OCR text
        "title": newspaper title string
        "lccn":  LCCN string
        "date":  issue date string
    """
    lines: list[str] = []
    current_issue: str = ""
    for page in pages:
        issue_key = f"{page['lccn']}/{page['date']}"
        if issue_key != current_issue:
            current_issue = issue_key
            lines.append(
                f"--- Issue: {page['title']} ({page['lccn']}) date:{page['date']} ---"
            )
            lines.append("")
        lines.append(f"=== PAGE ref:{page['ref']} ===")
        lines.append(page["text"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
        )
        retry = Retry(
            total=LLM_MAX_RETRIES,
            backoff_factor=LLM_BACKOFF_FACTOR,
            status_forcelist=LLM_RETRY_STATUSES,
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def extract(self, user_message: str) -> Optional[list]:
        """Send extraction request to LLM. Returns parsed list or None on failure.

        Routing:
          --backend litellm  → HTTP proxy only
          --backend copilot  → GitHub Copilot only
          --backend auto     → try HTTP proxy first, fall back to Copilot on failure
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
        }

        raw: Optional[str] = None

        use_litellm = _llm_backend in ("litellm", "auto")
        use_copilot = _llm_backend in ("copilot", "auto")

        if use_litellm:
            raw = self._post_litellm(payload)

        if raw is None and use_copilot:
            if _llm_backend == "auto":
                log.info("Primary LLM unavailable, switching to GitHub Copilot...")
            raw = _copilot_chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )

        if raw is None:
            return None
        return self._parse_response(raw)

    def _post_litellm(self, payload: dict) -> Optional[str]:
        """POST payload to the LiteLLM proxy, returns raw content string or None."""
        try:
            resp = self.session.post(
                LITELLM_ENDPOINT, data=json.dumps(payload), timeout=LLM_TIMEOUT
            )
            if resp.status_code == 429:
                log.warning("Rate limited (429). Waiting 60s...")
                time.sleep(60)
                return self._post_litellm(payload)

            if not resp.ok:
                log.warning(
                    f"LLM API error {resp.status_code}: {resp.text[:300]}"
                )
                return None

            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()

        except (requests.RequestException, KeyError, ValueError) as e:
            log.warning(f"LLM call failed: {e}")
            return None

    @staticmethod
    def _parse_response(raw: str) -> Optional[list]:
        """Parse LLM output as JSON array, stripping markdown fences if present."""
        # Strip markdown code fences if present
        cleaned = raw
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            log.warning(f"LLM returned non-list JSON: {type(parsed)}")
            return None
        except json.JSONDecodeError as e:
            log.warning(f"LLM output is not valid JSON: {e}\nRaw: {cleaned[:200]}")
            return None


# ---------------------------------------------------------------------------
# Progress / stats
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    issues_processed: int = 0
    issues_skipped: int = 0
    issues_failed: int = 0
    issues_empty: int = 0  # processed but LLM found no extractable content
    pages_sent: int = 0
    pages_skipped_short: int = 0
    records_extracted: int = 0
    llm_calls: int = 0
    last_updated: str = ""


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def read_sentinel(sentinel_path: Path) -> Optional[set]:
    """Return set of page refs already processed, or None if sentinel absent/corrupt."""
    if not sentinel_path.exists():
        return None
    try:
        with open(sentinel_path) as f:
            data = json.load(f)
        refs = data.get("pages_done", [])
        return set(refs)
    except (json.JSONDecodeError, KeyError):
        return None


class _null_context:
    """No-op context manager used in place of a threading.Lock when running sequentially."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def write_sentinel(sentinel_path: Path, pages_done: set):
    """Atomically write the sentinel file."""
    tmp = sentinel_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(
            {
                "pages_done": sorted(pages_done),
                "updated": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    tmp.replace(sentinel_path)


def append_jsonl(output_path: Path, records: list):
    """Append records to a JSONL file. Creates the file (or touches it) if absent."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def touch_placeholder(output_path: Path):
    """Create an empty placeholder JSONL file marking the issue as processed-but-empty.

    This distinguishes "processed, no records found" from "never processed".
    A no-op if the file already exists (even if empty).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch()


# ---------------------------------------------------------------------------
# Issue discovery
# ---------------------------------------------------------------------------


def _resolve_raw_page_files(date_dir: Path) -> list[Path]:
    """Return one effective text file per sequence number for *date_dir*.

    Priority for each seq-NNN:
      1. seq-NNN.txt  if non-empty  → use the LOC-downloaded OCR (preferred)
      2. seq-NNN.ocr.txt if non-empty → use locally-generated Tesseract OCR (fallback)
      3. Otherwise skip (no usable text for this sequence)

    When BOTH seq-NNN.txt and seq-NNN.ocr.txt are non-empty the downloaded
    file is always preferred — no duplication occurs.

    Returns a sorted list of Paths (each confirmed non-empty).
    """
    seq_stems: set[str] = set()
    for p in date_dir.iterdir():
        name = p.name
        if not name.startswith("seq-"):
            continue
        if name.endswith(".ocr.txt"):
            seq_stems.add(name[: -len(".ocr.txt")])
        elif name.endswith(".txt"):
            seq_stems.add(name[: -len(".txt")])

    result: list[Path] = []
    for seq in sorted(seq_stems):
        loc_ocr = date_dir / f"{seq}.txt"
        local_ocr = date_dir / f"{seq}.ocr.txt"

        has_loc = loc_ocr.exists() and loc_ocr.stat().st_size > 0
        has_local = local_ocr.exists() and local_ocr.stat().st_size > 0

        if has_loc:
            result.append(loc_ocr)
        elif has_local:
            result.append(local_ocr)
        # else: both absent or both empty → skip

    return result


def discover_issues(
    raw_dir: Path,
    year_start: int,
    year_end: int,
    pass0_dir: Optional[Path] = None,
) -> list[dict]:
    """Walk the raw data directory and return all issues to process.

    For each issue, text files are resolved with per-issue fallback logic:
      1. If *pass0_dir* is given and the compressed issue directory exists and
         contains seq-*.txt files, use those (higher quality compressed OCR).
      2. Otherwise fall back to raw OCR in *raw_dir*.

    Sentinel files always live in *raw_dir*.
    """
    issues = []
    for year in range(year_start, year_end + 1):
        year_dir = raw_dir / str(year)
        if not year_dir.exists():
            continue
        for lccn_dir in sorted(year_dir.iterdir()):
            if not lccn_dir.is_dir():
                continue
            lccn = lccn_dir.name
            if lccn.startswith("_"):
                continue
            for date_dir in sorted(lccn_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                date = date_dir.name
                if date.startswith("_"):
                    continue

                # Per-issue text source: prefer pass0 if available for this issue.
                txt_files = []
                text_source = "raw"
                if pass0_dir is not None:
                    p0_issue_dir = pass0_dir / str(year) / lccn / date
                    p0_files = sorted(p0_issue_dir.glob("seq-*.txt"))
                    if p0_files:
                        txt_files = p0_files
                        text_source = "pass0"

                if not txt_files:
                    # Fall back to raw OCR for this issue.
                    # Deduplicate: for each seq-NNN, prefer the downloaded
                    # seq-NNN.txt over the locally-generated seq-NNN.ocr.txt
                    # when both are non-empty.  The glob seq-*.txt would
                    # otherwise match both, causing duplicate processing.
                    txt_files = _resolve_raw_page_files(date_dir)
                    text_source = "raw"

                if not txt_files:
                    continue

                issues.append(
                    {
                        "year": year,
                        "lccn": lccn,
                        "date": date,
                        "dir": date_dir,        # always raw_dir path (for sentinels)
                        "txt_files": txt_files, # from best available source
                        "text_source": text_source,
                    }
                )
    return issues


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _get_title(lccn_dir: Path) -> str:
    """Try to read newspaper title from cached metadata."""
    meta = lccn_dir / "_title_meta.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            return data.get("name", lccn_dir.name)
        except (json.JSONDecodeError, OSError):
            pass
    return lccn_dir.name  # fall back to LCCN string


def collect_pending_pages(issue: dict, stats: Stats) -> list[dict]:
    """Return the list of pages that still need LLM processing for this issue.

    Each returned dict contains:
        ref, text, seq, lccn, date, year, title,
        out_path (Path), sentinel_path (Path), done_refs (set)

    Short/already-done pages are silently skipped; done_refs is updated in-place
    for trivially-skipped (too short) pages but the sentinel is NOT written here —
    that happens only after a successful LLM call in flush_batch().
    """
    lccn = issue["lccn"]
    date = issue["date"]
    year = issue["year"]
    txt_files = issue["txt_files"]

    sentinel_path: Path = issue["dir"] / "_pass1_done.json"
    done_refs: set = read_sentinel(sentinel_path) or set()

    title = _get_title(issue["dir"].parent)

    pending: list[dict] = []
    for txt_path in txt_files:
        # Normalise seq key: seq-001.ocr.txt → stem "seq-001.ocr" → strip ".ocr"
        stem = txt_path.stem  # e.g. "seq-001" or "seq-001.ocr"
        seq = stem[: -len(".ocr")] if stem.endswith(".ocr") else stem
        page_ref = f"{lccn}/{date}/{seq}"

        if page_ref in done_refs:
            continue

        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            log.warning(f"Cannot read {txt_path}: {e}")
            continue

        if len(text) < MIN_PAGE_CHARS:
            log.debug(f"Skipping short page {page_ref} ({len(text)} chars)")
            stats.pages_skipped_short += 1
            done_refs.add(page_ref)  # mark trivially done
            continue

        # Count tokens now so the main loop can bin-pack accurately.
        # Include a small header overhead (~15 tokens for the === PAGE ref:... === line).
        page_tokens = count_tokens(text) + 15

        pending.append(
            {
                "ref": page_ref,
                "text": text,
                "seq": seq,
                "lccn": lccn,
                "date": date,
                "year": year,
                "title": title,
                "tokens": page_tokens,  # pre-computed for bin-packing
                # Output paths split by record type
                "out_social": (
                    issue["pass1_dir"] / "social" / str(year) / f"{lccn}_{date}.jsonl"
                ),
                "out_prices": (
                    issue["pass1_dir"] / "prices" / str(year) / f"{lccn}_{date}.jsonl"
                ),
                # Sentinel lives in the raw dir
                "sentinel_path": sentinel_path,
                "done_refs": done_refs,  # shared mutable set for this issue
            }
        )

    # If any pages were trivially skipped (too short) and there are no more
    # pending pages, flush the sentinel now so we don't re-scan next run.
    if not pending and done_refs:
        write_sentinel(sentinel_path, done_refs)

    return pending


def flush_batch(
    batch: list[dict],
    client: LLMClient,
    stats: Stats,
    dry_run: bool = False,
    sentinel_locks: Optional[dict] = None,
    stats_lock: Optional[threading.Lock] = None,
) -> bool:
    """Send one LLM call for a batch of pages (potentially from multiple issues).

    On success:
      - Appends extracted records to per-issue JSONL files.
      - Updates each issue's sentinel with newly-processed page refs.

    On failure:
      - Returns False; no sentinels or JSONL files are touched for this batch
        so they will be retried on the next run.

    Returns True on success, False on failure.
    """
    if not batch:
        return True

    batch_refs = {p["ref"] for p in batch}

    user_msg = build_user_message(batch)
    token_count = count_tokens(user_msg)

    # Summarise which issues are in this batch for the log line
    issue_keys = list(dict.fromkeys(f"{p['lccn']}/{p['date']}" for p in batch))
    issues_summary = (
        issue_keys[0]
        if len(issue_keys) == 1
        else f"{issue_keys[0]} … +{len(issue_keys) - 1} more"
    )
    log.info(
        f"  LLM call: {issues_summary} | "
        f"{len(batch)} page(s) from {len(issue_keys)} issue(s) | "
        f"{token_count} tokens in"
    )

    def _inc(**kwargs):
        if stats_lock:
            with stats_lock:
                for attr, val in kwargs.items():
                    setattr(stats, attr, getattr(stats, attr) + val)
        else:
            for attr, val in kwargs.items():
                setattr(stats, attr, getattr(stats, attr) + val)

    if dry_run:
        log.info("  [DRY RUN] Would submit to LLM, skipping.")
        # Mark all pages as done so stats are accurate
        _commit_pages(batch, [], stats, dry_run=True, sentinel_locks=sentinel_locks, stats_lock=stats_lock)
        _inc(llm_calls=1)
        return True

    records = client.extract(user_msg)
    _inc(llm_calls=1)

    if records is None:
        log.warning(
            f"  LLM call failed for batch covering {issue_keys}. "
            "Will retry on next run."
        )
        _inc(issues_failed=len(issue_keys))
        return False

    # Validate refs
    validated: list[dict] = []
    for rec in records:
        ref = rec.get("ref", "")
        if ref not in batch_refs:
            log.debug(f"  Discarding record with unexpected ref '{ref}'")
            continue
        validated.append(rec)

    n_people = sum(1 for r in validated if r.get("t") == "P")
    n_prices = sum(1 for r in validated if r.get("t") == "$")
    log.info(
        f"  → {len(validated)} records extracted "
        f"(P:{n_people} people  $:{n_prices} prices"
        + (f"  — {len(records) - len(validated)} discarded bad ref" if len(records) != len(validated) else "")
        + ")"
    )

    _commit_pages(batch, validated, stats, dry_run=False, sentinel_locks=sentinel_locks, stats_lock=stats_lock)
    return True


def _commit_pages(
    batch: list[dict],
    validated_records: list[dict],
    stats: Stats,
    dry_run: bool,
    sentinel_locks: Optional[dict] = None,
    stats_lock: Optional[threading.Lock] = None,
):
    """Write JSONL records and update sentinels for every issue in the batch.

    People co-mention records ("t":"P") go to pass1/social/{year}/{lccn}_{date}.jsonl.
    Price records ("t":"$") go to pass1/prices/{year}/{lccn}_{date}.jsonl.
    Groups pages by sentinel_path so we do exactly one sentinel write per issue.

    When an issue produces no records of a given type, an empty placeholder file
    is created so that downstream tools can distinguish "processed-but-empty" from
    "never processed".
    """
    from collections import defaultdict

    # Key pages by sentinel path (one per issue)
    issue_pages: dict[str, list[dict]] = defaultdict(list)
    for page in batch:
        issue_pages[str(page["sentinel_path"])].append(page)

    # Build a ref → page mapping for quick record dispatch
    ref_to_page: dict[str, dict] = {page["ref"]: page for page in batch}

    # Bucket records by (sentinel_key, record_type)
    social_records: dict[str, list] = defaultdict(list)  # key: sentinel path str
    price_records: dict[str, list] = defaultdict(list)

    for rec in validated_records:
        page = ref_to_page.get(rec.get("ref", ""))
        if page is None:
            continue
        skey = str(page["sentinel_path"])
        if rec.get("t") == "P":
            social_records[skey].append(rec)
        elif rec.get("t") == "$":
            price_records[skey].append(rec)

    # For each issue: write records to split outputs, update done_refs, flush sentinel
    for skey, pages in issue_pages.items():
        sentinel_path: Path = pages[0]["sentinel_path"]
        done_refs: set = pages[0]["done_refs"]

        s_recs = social_records.get(skey, [])
        p_recs = price_records.get(skey, [])

        # Acquire per-sentinel lock when running in parallel to prevent
        # two threads clobbering the same sentinel / JSONL file.
        lock = (sentinel_locks or {}).get(skey)
        ctx = lock if lock is not None else _null_context()
        with ctx:
            if not dry_run:
                if s_recs:
                    append_jsonl(pages[0]["out_social"], s_recs)
                else:
                    # Placeholder: marks this issue as processed-but-empty for social
                    touch_placeholder(pages[0]["out_social"])

                if p_recs:
                    append_jsonl(pages[0]["out_prices"], p_recs)
                else:
                    # Placeholder: marks this issue as processed-but-empty for prices
                    touch_placeholder(pages[0]["out_prices"])

            for page in pages:
                done_refs.add(page["ref"])

            if not dry_run:
                write_sentinel(sentinel_path, done_refs)

        n_records = len(s_recs) + len(p_recs)
        n_pages = len(pages)
        is_empty = not s_recs and not p_recs
        if stats_lock:
            with stats_lock:
                stats.records_extracted += n_records
                stats.pages_sent += n_pages
                stats.issues_processed += 1
                if is_empty:
                    stats.issues_empty += 1
        else:
            stats.records_extracted += n_records
            stats.pages_sent += n_pages
            stats.issues_processed += 1
            if is_empty:
                stats.issues_empty += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 LLM extraction: people co-mentions and prices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single year
  python extract_pass1.py --year-start 1776 --year-end 1776

  # Process decade
  python extract_pass1.py --year-start 1880 --year-end 1890

  # Dry run to preview workload
  python extract_pass1.py --year-start 1880 --year-end 1890 --dry-run

  # Use GitHub Copilot as the LLM backend
  python extract_pass1.py --year-start 1776 --year-end 1776 --backend copilot

  # Auto mode: try LiteLLM proxy first, fall back to GitHub Copilot
  python extract_pass1.py --year-start 1880 --year-end 1890 --backend auto

  # Use a different model
  CHRONAM_MODEL=claude-haiku-4-5-20251001 python extract_pass1.py --year-start 1776 --year-end 1776

Environment:
  LITELLM_PROXY_BASE       LiteLLM proxy base URL
  LITELLM_PROXY_API_KEY    API key
  CHRONAM_MODEL            Model name (default: gpt-5-mini)
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument("--year-start", type=int, default=1770)
    parser.add_argument("--year-end", type=int, default=1963)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and count issues but do not call LLM",
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help=(
            "Delete all _pass1_done.json sentinels in the requested year range, "
            "forcing re-evaluation of all pages. Useful after new pages have been "
            "downloaded or compressed that were missed by a previous run."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel LLM worker threads (default: 1 = sequential)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--backend",
        choices=["litellm", "copilot", "auto"],
        default="litellm",
        help=(
            "LLM backend to use: "
            "litellm = HTTP proxy (default), "
            "copilot = GitHub Copilot direct access, "
            "auto = try litellm first, fall back to copilot on failure"
        ),
    )

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    set_llm_backend(args.backend)

    raw_dir = args.data_dir / "raw"
    pass0_dir = args.data_dir / "pass0"
    pass1_dir = args.data_dir / "pass1"

    # pass0_dir is used for per-issue fallback: if a compressed version of an
    # issue exists there it is preferred; otherwise raw OCR is used automatically.
    pass0_dir_arg = pass0_dir if pass0_dir.exists() else None
    text_source_desc = "pass0 (compressed) with per-issue raw fallback" if pass0_dir_arg else "raw"

    if not raw_dir.exists():
        log.error(f"Raw data directory not found: {raw_dir}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Chronicling America — Phase 1 LLM Extraction")
    log.info(f"  Data dir:    {args.data_dir.resolve()}")
    log.info(f"  Text source: {text_source_desc}")
    log.info(f"  Years:       {args.year_start}–{args.year_end}")
    log.info(f"  Model:       {args.model}")
    log.info(f"  Endpoint:    {LITELLM_ENDPOINT}")
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  Backend:     {args.backend}")
    log.info(f"  Dry run:     {args.dry_run}")
    log.info("=" * 60)

    if not API_KEY and not args.dry_run and args.backend in ("litellm", "auto"):
        log.error(
            "No API key found. Set LITELLM_PROXY_API_KEY or LITELLM_API_KEY env var."
        )
        sys.exit(1)

    # Discover all issues
    log.info("Scanning data directory...")
    issues = discover_issues(raw_dir, args.year_start, args.year_end, pass0_dir=pass0_dir_arg)
    log.info(f"Found {len(issues)} issue directories to evaluate.")

    # --recheck: remove sentinels so all pages are re-evaluated
    if args.recheck:
        removed = 0
        for issue in issues:
            sentinel = issue["dir"] / "_pass1_done.json"
            if sentinel.exists():
                sentinel.unlink()
                removed += 1
        log.info(f"--recheck: removed {removed} sentinel(s) — all pages will be re-evaluated.")

    # Attach pass1_dir to each issue so collect_pending_pages can build out_path
    for issue in issues:
        issue["pass1_dir"] = pass1_dir

    client = LLMClient(model=args.model)
    stats = Stats()

    # ------------------------------------------------------------------
    # Phase 1: bin-pack pages into batches
    # ------------------------------------------------------------------
    all_batches: list[list[dict]] = []
    pending_pages: list[dict] = []
    pending_tokens: int = 0

    for idx, issue in enumerate(issues, start=1):
        if _shutdown:
            break

        log.debug(
            f"[{idx}/{len(issues)}] Collecting {issue['lccn']}/{issue['date']}"
            f" (src:{issue.get('text_source','?')})"
        )

        new_pages = collect_pending_pages(issue, stats)

        if not new_pages:
            stats.issues_skipped += 1
            log.debug(f"  Issue {issue['lccn']}/{issue['date']}: nothing pending, skipped.")
            continue

        for page in new_pages:
            page_tokens = page["tokens"]

            if pending_pages and pending_tokens + page_tokens > MAX_TOKENS_PER_CALL:
                all_batches.append(pending_pages)
                pending_pages = []
                pending_tokens = 0

            pending_pages.append(page)
            pending_tokens += page_tokens

    if not _shutdown and pending_pages:
        all_batches.append(pending_pages)

    log.info(f"Built {len(all_batches)} batch(es) to process.")

    # ------------------------------------------------------------------
    # Phase 2: dispatch batches — sequentially or in parallel
    # ------------------------------------------------------------------
    if args.workers == 1 or args.dry_run:
        # Sequential path (original behaviour; dry-run is always sequential)
        for batch in all_batches:
            if _shutdown:
                break
            flush_batch(batch, client, stats, dry_run=args.dry_run)
    else:
        # Parallel path: sliding window of up to `workers` in-flight futures.
        # Per-sentinel locks prevent concurrent threads clobbering the same
        # sentinel/JSONL; stats_lock serialises counter updates.
        sentinel_locks: dict[str, threading.Lock] = {}
        for batch in all_batches:
            for page in batch:
                key = str(page["sentinel_path"])
                if key not in sentinel_locks:
                    sentinel_locks[key] = threading.Lock()

        stats_lock = threading.Lock()
        total = len(all_batches)
        completed = 0

        log.info(f"Submitting up to {args.workers} batches at a time ({total} total)...")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            batch_iter = iter(enumerate(all_batches))
            in_flight: dict = {}  # future → batch_index

            def _submit_next() -> bool:
                if _shutdown:
                    return False
                try:
                    i, batch = next(batch_iter)
                except StopIteration:
                    return False
                fut = executor.submit(
                    flush_batch,
                    batch,
                    client,
                    stats,
                    False,  # dry_run
                    sentinel_locks,
                    stats_lock,
                )
                in_flight[fut] = i
                return True

            # Fill the initial window
            for _ in range(args.workers):
                if not _submit_next():
                    break

            while in_flight:
                done_futures = [fut for fut in list(in_flight) if fut.done()]
                if not done_futures:
                    time.sleep(0.05)
                    continue

                for fut in done_futures[:1]:  # process one at a time to keep window full
                    batch_idx = in_flight.pop(fut)
                    completed += 1
                    try:
                        fut.result()
                    except Exception as exc:
                        log.warning(f"Batch {batch_idx} raised an exception: {exc}")

                    _submit_next()

                    if completed % max(1, min(10, total // 10)) == 0 or completed == total:
                        with stats_lock:
                            log.info(
                                f"[{completed}/{total}] "
                                f"processed={stats.issues_processed} "
                                f"failed={stats.issues_failed} "
                                f"records={stats.records_extracted} "
                                f"llm_calls={stats.llm_calls}"
                            )

                if _shutdown:
                    log.info("Shutdown: cancelling remaining queued futures...")
                    for fut in list(in_flight):
                        fut.cancel()
                    break

    # Final summary
    log.info("=" * 60)
    log.info("Phase 1 complete.")
    log.info(f"  Issues processed:    {stats.issues_processed}")
    log.info(f"  Issues empty:        {stats.issues_empty}  (processed, no records found — placeholder created)")
    log.info(f"  Issues skipped:      {stats.issues_skipped}")
    log.info(f"  Issues failed:       {stats.issues_failed}")
    log.info(f"  Pages sent to LLM:   {stats.pages_sent}")
    log.info(f"  Pages skipped short: {stats.pages_skipped_short}")
    log.info(f"  Records extracted:   {stats.records_extracted}")
    log.info(f"  Total LLM calls:     {stats.llm_calls}")
    log.info("=" * 60)

    if _shutdown:
        log.info("Stopped early. Re-run to resume from where we left off.")
        sys.exit(0)


if __name__ == "__main__":
    main()
