#!/usr/bin/env python3
"""
Chronicling America — Phase 0: OCR Compression (compress_pass0.py)
===================================================================

Reads downloaded OCR text (data/raw/…/seq-NNN.txt), fixes OCR noise and
telegraphically compresses each newspaper article to 2–3 dense sentences
that preserve ALL facts (names, brands, places, amounts, dates, numeric
values, prices, organisations) while stripping rhetorical redundancy.

The compressed text is written to data/pass0/…/seq-NNN.txt.
Downstream extract_pass1.py can then read from pass0/ instead of raw/,
working on much smaller input with identical factual content.

DESIGN PRINCIPLES:
  1. Preserve every fact: names, brands, quantities, dates, amounts, places.
  2. Compress aggressively: target output ≤ compression_ratio × input tokens.
  3. Token-aware batching: pack multiple pages per LLM call up to the model's
     context window.  The LLM returns each compressed page clearly labeled.
  4. Sentinel files: data/raw/{y}/{lccn}/{date}/_pass0_done.json records
     which page refs have been compressed.  On re-run only missing pages are
     re-submitted.
  5. Restart-safe: compressed files written atomically (tmp + rename);
     sentinel written only after successful flush.

Usage:
    python compress_pass0.py [options]

    # Compress 1776 with default 20% ratio
    python compress_pass0.py --year-start 1776 --year-end 1776

    # Compress 1880-1890, target 30% size
    python compress_pass0.py --year-start 1880 --year-end 1890 --compression-ratio 0.30

    # Dry run (shows what would be sent, no LLM calls)
    python compress_pass0.py --year-start 1776 --year-end 1776 --dry-run

    # Use GitHub Copilot as the LLM backend
    python compress_pass0.py --year-start 1776 --year-end 1776 --backend copilot

Environment variables:
    LITELLM_PROXY_BASE      LiteLLM proxy URL (default: http://ai-tools.cz.intinfra.com:4004)
    LITELLM_PROXY_API_KEY   API key (also checked as LITELLM_API_KEY)
    CHRONAM_MODEL           Model name (default: gpt-5-mini)
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    cancel_event: Optional[threading.Event] = None,
) -> Optional[str]:
    """Call GitHub Copilot chat completions API (non-streaming)."""
    token, endpoint = _get_copilot_session_token()
    if not token or not endpoint:
        return None

    if cancel_event is not None and cancel_event.is_set():
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
# Tokenizer
# ---------------------------------------------------------------------------

try:
    _TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")
except KeyError:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Context window budget
# ---------------------------------------------------------------------------

# Default usable context window.  Override with --context-window at runtime
# (e.g. 128000 for gpt-4o-mini).
DEFAULT_CONTEXT_WINDOW = 32_000

# Approximate token overhead for the system prompt (measured ~600 tokens).
_SYSTEM_PROMPT_OVERHEAD = 650


def compute_token_budget(context_window: int, compression_ratio: float) -> tuple[int, int]:
    """Return (max_input_tokens, max_output_tokens) for one LLM call.

    The context window must hold:  system_prompt + user_input + model_output.
    Expected output ≈ input × compression_ratio, so:
        input × (1 + compression_ratio) ≤ context_window − overhead
        → max_input = (context_window − overhead) / (1 + compression_ratio)
    A 5 % safety margin is applied to avoid off-by-one overflows.
    max_output is set to ceil(max_input × compression_ratio) with a 2 048 floor.
    """
    available = context_window - _SYSTEM_PROMPT_OVERHEAD
    max_input = int(available / (1.0 + compression_ratio) * 0.95)  # 5% safety margin
    max_output = max(2048, int(max_input * compression_ratio * 1.10))  # +10% headroom
    return max_input, max_output


# LLM API settings (non-budget)
TEMPERATURE = 0.2          # slight creativity helps OCR correction
LLM_MAX_RETRIES = 4
LLM_BACKOFF_FACTOR = 2.0
LLM_RETRY_STATUSES = [429, 500, 502, 503, 504]
LLM_TIMEOUT = (15, 900)    # (connect_s, read_s) — 15 min read timeout

# Application-level retry loop for upstream 504/5xx (server-side timeouts).
LLM_APP_RETRIES = 5           # up to 5 additional attempts
LLM_APP_BACKOFF_BASE = 60     # first sleep = 60 s, doubles each attempt

# Skip pages shorter than this (blank / heavily garbled pages)
MIN_PAGE_CHARS = 80

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pass0")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown = False
# Set by signal handler; checked by worker threads before issuing HTTP calls.
_cancel_event = threading.Event()


def _signal_handler(signum, frame):
    global _shutdown
    if _shutdown:
        log.warning("Force quit.")
        sys.exit(1)
    log.info("Shutdown requested — will stop after current batch...")
    _shutdown = True
    _cancel_event.set()  # unblock any threads waiting or about to send


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are an extreme-compression engine for historical OCR newspaper text (1770–1963).
Output is fed directly into another LLM — it does NOT need to be human-readable prose.
Telegraphic, note-like, abbreviated output is strongly preferred over full sentences.

TASK: For each newspaper page, emit the absolute minimum text that preserves every
extractable fact.  Aggressive truncation of everything that is not a fact is required.

WHAT TO KEEP (compress these, never drop them):
- Named people: full name + title (e.g. "Gen. Washington", "Mrs. J. Smith")
- Named places: cities, counties, states, countries, streets, buildings
- Named organisations: companies, regiments, courts, churches, governments
- All numeric values with their unit: prices, wages, counts, dates, ages,
  distances, weights, volumes, percentages, vote tallies
- Commodity names with grade/quality qualifiers (e.g. "superfine flour", "middling cotton")
- Relationship verbs that carry factual content: married, sued, sold, appointed,
  convicted, elected, died, born, arrived, departed, signed, ratified

WHAT TO DROP (zero tolerance — omit entirely):
- All rhetorical, editorial, or emotional language
- Repeated information (list each fact once only)
- Conjunctions, prepositions, articles that add no fact
- Decorative typography, column headers, masthead boilerplate
- Any sentence where every fact in it is already captured elsewhere on the page

STYLE: Comma-separated or semicolon-separated fact lists are ideal.
One line per distinct article/topic on the page.
Do NOT write full sentences unless a verb is strictly needed to preserve meaning.
Abbreviate freely: "Pres." "Gen." "Gov." "Co." "Ct." "St." "$" "£" "wt." "bu." "bbl."

TARGET: output ≤{target_pct}% of input token count.  If you are above target,
delete more filler.  Below target is fine — the floor is preserving all facts.

OUTPUT FORMAT (strictly):
For each input page output exactly one block:
=== COMPRESSED ref:<ref_key> ===
<compressed text for that page>

Use the exact ref key shown in the input header.  Output blocks in the same
order as the input.  No other text outside these blocks.
"""

_PAGE_HEADER_TMPL = "=== PAGE ref:{ref} ===\n{text}\n"


def build_system_prompt(compression_ratio: float) -> str:
    target_pct = int(round(compression_ratio * 100))
    return _SYSTEM_TEMPLATE.format(target_pct=target_pct)


def build_user_message(pages: list[dict]) -> str:
    """Pack multiple pages into a single user message."""
    parts: list[str] = []
    current_issue: str = ""
    for page in pages:
        issue_key = f"{page['lccn']}/{page['date']}"
        if issue_key != current_issue:
            current_issue = issue_key
            parts.append(
                f"--- Issue: {page.get('title', page['lccn'])} "
                f"({page['lccn']}) date:{page['date']} ---\n"
            )
        parts.append(_PAGE_HEADER_TMPL.format(ref=page["ref"], text=page["text"]))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Matches:  === COMPRESSED ref:<ref> ===\n<content until next block or EOF>
_BLOCK_RE = re.compile(
    r"=== COMPRESSED ref:([^\s=]+) ===\s*\n(.*?)(?=\n=== COMPRESSED ref:|$)",
    re.DOTALL,
)


def parse_compressed_response(raw: str) -> dict[str, str]:
    """Return {ref: compressed_text} from the LLM response."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    result: dict[str, str] = {}
    for m in _BLOCK_RE.finditer(cleaned):
        ref = m.group(1).strip()
        text = m.group(2).strip()
        result[ref] = text
    return result


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        compression_ratio: float = 0.20,
        max_output_tokens: int = 4096,
    ):
        self.model = model
        self.compression_ratio = compression_ratio
        self.max_output_tokens = max_output_tokens
        self.system_prompt = build_system_prompt(compression_ratio)
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Connection": "keep-alive",
            }
        )
        retry = Retry(
            total=LLM_MAX_RETRIES,
            backoff_factor=LLM_BACKOFF_FACTOR,
            status_forcelist=LLM_RETRY_STATUSES,
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=1,
            pool_maxsize=4,
            pool_block=False,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _post(
        self,
        payload: dict,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        """POST *payload* using SSE streaming, with application-level retry for 5xx.

        Streaming keeps the connection alive throughout generation, avoiding
        proxy idle-timeout 504s.  Returns fully assembled content string or None.
        """
        streaming_payload = {**payload, "stream": True}
        sleep_s = LLM_APP_BACKOFF_BASE

        for attempt in range(1, LLM_APP_RETRIES + 2):
            if cancel_event is not None and cancel_event.is_set():
                return None
            try:
                resp = self.session.post(
                    LITELLM_ENDPOINT,
                    data=json.dumps(streaming_payload),
                    timeout=LLM_TIMEOUT,
                    stream=True,
                )
            except requests.RequestException as e:
                if cancel_event is not None and cancel_event.is_set():
                    return None
                log.warning(f"LLM request exception (attempt {attempt}): {e}")
                if attempt > LLM_APP_RETRIES:
                    return None
                log.info(f"  Retrying in {sleep_s}s ...")
                time.sleep(sleep_s)
                sleep_s *= 2
                continue

            if cancel_event is not None and cancel_event.is_set():
                return None

            if resp.status_code == 429:
                wait = sleep_s
                log.warning(f"Rate limited (429). Waiting {wait}s...")
                if cancel_event is not None:
                    cancel_event.wait(timeout=wait)
                else:
                    time.sleep(wait)
                sleep_s = min(sleep_s * 2, 960)
                continue

            if not resp.ok:
                log.warning(f"LLM API error {resp.status_code}: {resp.text[:300]}")
                if resp.status_code not in LLM_RETRY_STATUSES or attempt > LLM_APP_RETRIES:
                    return None
                log.info(
                    f"  Upstream {resp.status_code} — retrying in {sleep_s}s "
                    f"(attempt {attempt}/{LLM_APP_RETRIES})..."
                )
                time.sleep(sleep_s)
                sleep_s *= 2
                continue

            # 2xx — consume SSE stream
            try:
                content_parts: list[str] = []
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    data_str = raw_line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"]
                        piece = delta.get("content") or ""
                        content_parts.append(piece)
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
                return "".join(content_parts).strip() or None
            except requests.RequestException as e:
                if cancel_event is not None and cancel_event.is_set():
                    return None
                log.warning(f"LLM stream read error (attempt {attempt}): {e}")
                if attempt > LLM_APP_RETRIES:
                    return None
                log.info(f"  Retrying in {sleep_s}s ...")
                time.sleep(sleep_s)
                sleep_s *= 2
                continue

        return None

    def compress(
        self,
        user_message: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[dict[str, str]]:
        """Send compression request. Returns {ref: compressed_text} or None.

        If *cancel_event* is set before or during the call, returns None
        immediately without touching the network.

        Routing:
          --backend litellm  → HTTP proxy only
          --backend copilot  → GitHub Copilot only
          --backend auto     → try HTTP proxy first, fall back to Copilot on failure
        """
        if cancel_event is not None and cancel_event.is_set():
            return None

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "temperature": TEMPERATURE,
        }

        raw: Optional[str] = None

        use_litellm = _llm_backend in ("litellm", "auto")
        use_copilot = _llm_backend in ("copilot", "auto")

        if use_litellm:
            raw = self._post(payload, cancel_event=cancel_event)

        if raw is None and use_copilot and not (
            cancel_event is not None and cancel_event.is_set()
        ):
            if _llm_backend == "auto":
                log.info("Primary LLM unavailable, switching to GitHub Copilot...")
            raw = _copilot_chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=TEMPERATURE,
                cancel_event=cancel_event,
            )

        if raw is None:
            if cancel_event is not None and cancel_event.is_set():
                return None
            return None
        return parse_compressed_response(raw)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class Stats:
    def __init__(self):
        self.pages_processed = 0
        self.pages_skipped_done = 0
        self.pages_skipped_short = 0
        self.pages_failed = 0
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------


def read_sentinel(sentinel_path: Path) -> Optional[set]:
    if not sentinel_path.exists():
        return None
    try:
        data = json.loads(sentinel_path.read_text())
        return set(data.get("pages_done", []))
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def write_sentinel(sentinel_path: Path, pages_done: set):
    tmp = sentinel_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "pages_done": sorted(pages_done),
                "updated": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    tmp.replace(sentinel_path)


# ---------------------------------------------------------------------------
# Issue discovery
# ---------------------------------------------------------------------------


def discover_issues(raw_dir: Path, year_start: int, year_end: int) -> list[dict]:
    issues = []
    for year in range(year_start, year_end + 1):
        year_dir = raw_dir / str(year)
        if not year_dir.exists():
            continue
        for lccn_dir in sorted(year_dir.iterdir()):
            if not lccn_dir.is_dir() or lccn_dir.name.startswith("_"):
                continue
            lccn = lccn_dir.name
            for date_dir in sorted(lccn_dir.iterdir()):
                if not date_dir.is_dir() or date_dir.name.startswith("_"):
                    continue
                # Collect all sequence numbers that have any text (LOC OCR or
                # local OCR).  For each seq-NNN we prefer seq-NNN.txt (non-empty),
                # then fall back to seq-NNN.ocr.txt.
                txt_files = _resolve_page_files(date_dir)
                if not txt_files:
                    continue
                issues.append(
                    {
                        "year": year,
                        "lccn": lccn,
                        "date": date_dir.name,
                        "dir": date_dir,
                        "txt_files": txt_files,
                    }
                )
    return issues


def _resolve_page_files(date_dir: Path) -> list[Path]:
    """Return one effective text file per sequence number for *date_dir*.

    Priority for each seq-NNN:
      1. seq-NNN.txt  if non-empty  → use the LOC-downloaded OCR (preferred)
      2. seq-NNN.ocr.txt if non-empty → use locally-generated Tesseract OCR (fallback)
      3. Otherwise skip (no usable text for this sequence)

    When BOTH seq-NNN.txt and seq-NNN.ocr.txt are non-empty (e.g. the LOC API
    later provided OCR for a page that was originally blank and was already
    locally OCR'd), the downloaded file is always preferred and the local OCR
    file is silently ignored — no duplication occurs.

    Returns a sorted list of Paths (each already confirmed to be non-empty).
    """
    # Collect distinct seq stems from all relevant files in the directory.
    # Both seq-NNN.txt and seq-NNN.ocr.txt map to the same stem "seq-NNN";
    # using a set ensures each sequence is considered only once.
    # Note: .ocr.txt files are checked first because they also end with .txt.
    seq_nums: set[str] = set()
    for p in date_dir.iterdir():
        name = p.name
        if not name.startswith("seq-"):
            continue
        if name.endswith(".ocr.txt"):
            seq_nums.add(name[: -len(".ocr.txt")])
        elif name.endswith(".txt"):
            seq_nums.add(name[: -len(".txt")])

    result: list[Path] = []
    for seq in sorted(seq_nums):
        loc_ocr = date_dir / f"{seq}.txt"
        local_ocr = date_dir / f"{seq}.ocr.txt"

        has_loc = loc_ocr.exists() and loc_ocr.stat().st_size > 0
        has_local = local_ocr.exists() and local_ocr.stat().st_size > 0

        if has_loc and has_local:
            # Both files are non-empty: prefer the official downloaded OCR.
            log.debug(
                f"    {seq}: both {seq}.txt (downloaded) and {seq}.ocr.txt (local)"
                " exist and are non-empty — using downloaded file"
            )
            result.append(loc_ocr)
        elif has_loc:
            result.append(loc_ocr)
        elif has_local:
            result.append(local_ocr)
        # else: both absent or both empty → no usable text, skip

    return result


def _get_title(lccn_dir: Path) -> str:
    meta = lccn_dir / "_title_meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("name", lccn_dir.name)
        except (json.JSONDecodeError, OSError):
            pass
    return lccn_dir.name


# ---------------------------------------------------------------------------
# Pending page collection
# ---------------------------------------------------------------------------


def collect_pending_pages(issue: dict, pass0_dir: Path, stats: Stats) -> list[dict]:
    """Return pages not yet compressed for this issue.

    Each page dict contains: ref, text, tokens, lccn, date, year, title,
    out_path (Path for compressed output), sentinel_path, done_refs (shared set).

    txt_files may contain either seq-NNN.txt (LOC OCR) or seq-NNN.ocr.txt
    (local Tesseract OCR); both are normalised to the seq-NNN key so the
    sentinel, ref, and output path are identical regardless of source.
    """
    lccn = issue["lccn"]
    date = issue["date"]
    year = issue["year"]

    sentinel_path = issue["dir"] / "_pass0_done.json"
    done_refs: set = read_sentinel(sentinel_path) or set()
    title = _get_title(issue["dir"].parent)

    pending: list[dict] = []
    for txt_path in issue["txt_files"]:
        # Normalise seq key: strip ".ocr" suffix so seq-001.ocr.txt → "seq-001"
        stem = txt_path.stem  # e.g. "seq-001" or "seq-001.ocr"
        if stem.endswith(".ocr"):
            seq = stem[: -len(".ocr")]
        else:
            seq = stem
        page_ref = f"{lccn}/{date}/{seq}"

        if page_ref in done_refs:
            stats.pages_skipped_done += 1
            continue

        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            log.warning(f"Cannot read {txt_path}: {e}")
            continue

        if len(text) < MIN_PAGE_CHARS:
            log.debug(f"Skipping short page {page_ref} ({len(text)} chars)")
            stats.pages_skipped_short += 1
            done_refs.add(page_ref)
            continue

        page_tokens = count_tokens(text) + 20  # +20 for header overhead

        # Output path mirrors raw structure under pass0/ (always seq-NNN.txt)
        out_path = pass0_dir / str(year) / lccn / date / f"{seq}.txt"

        # If the compressed output already exists the page was successfully
        # processed in a previous run.  Treat it as done regardless of whether
        # the sentinel was removed by --recheck.
        if out_path.exists():
            stats.pages_skipped_done += 1
            done_refs.add(page_ref)
            continue

        pending.append(
            {
                "ref": page_ref,
                "text": text,
                "seq": seq,
                "lccn": lccn,
                "date": date,
                "year": year,
                "title": title,
                "tokens": page_tokens,
                "out_path": out_path,
                "sentinel_path": sentinel_path,
                "done_refs": done_refs,  # shared mutable set
            }
        )

    # If nothing is pending but we skipped some short pages, flush sentinel now
    if not pending and done_refs:
        write_sentinel(sentinel_path, done_refs)

    return pending


# ---------------------------------------------------------------------------
# Batch flush
# ---------------------------------------------------------------------------


def flush_batch(
    batch: list[dict],
    client: LLMClient,
    stats: Stats,
    compression_ratio: float,
    dry_run: bool = False,
    sentinel_locks: Optional[dict] = None,
    stats_lock: Optional[threading.Lock] = None,
    cancel_event: Optional[threading.Event] = None,
    solo_retry_refs: Optional[set] = None,
) -> bool:
    """Compress a batch of pages via one LLM call.

    On success writes compressed text to per-page output files and updates
    sentinels.  On failure returns False; nothing is written.

    *cancel_event*: when set (Ctrl-C), the LLM call is skipped and the batch
    is left unprocessed so it will be retried on the next run.

    *solo_retry_refs*: a shared mutable set of refs that already failed in a
    batch call earlier this run.  If a ref is missing from the batch response
    AND it is already in this set, a solo retry is attempted immediately.
    Otherwise the ref is just counted as failed (leaving it out of the
    sentinel so the next run will re-attempt it in a batch first).
    """
    if not batch:
        return True

    if cancel_event is not None and cancel_event.is_set():
        return False

    user_msg = build_user_message(batch)
    token_count_in = count_tokens(user_msg)

    issue_keys = list(dict.fromkeys(f"{p['lccn']}/{p['date']}" for p in batch))
    issues_summary = (
        issue_keys[0]
        if len(issue_keys) == 1
        else f"{issue_keys[0]} … +{len(issue_keys) - 1} more"
    )
    log.info(
        f"  LLM call: {issues_summary} | "
        f"{len(batch)} page(s) from {len(issue_keys)} issue(s) | "
        f"{token_count_in} tokens in"
    )

    def _update_stats(**kwargs):
        """Update Stats counters, acquiring stats_lock when in parallel mode."""
        if stats_lock:
            with stats_lock:
                for attr, val in kwargs.items():
                    setattr(stats, attr, getattr(stats, attr) + val)
        else:
            for attr, val in kwargs.items():
                setattr(stats, attr, getattr(stats, attr) + val)

    if dry_run:
        log.info("  [DRY RUN] Would submit to LLM, skipping.")
        _update_stats(llm_calls=1, tokens_in=token_count_in, pages_processed=len(batch))
        return True

    compressed_map = client.compress(user_msg, cancel_event=cancel_event)
    _update_stats(llm_calls=1, tokens_in=token_count_in)

    if compressed_map is None:
        if cancel_event is not None and cancel_event.is_set():
            # Cancelled by Ctrl-C — not a real failure; leave sentinel untouched for retry
            return False
        log.warning(f"  Compression call failed for batch covering {issue_keys}.")
        _update_stats(pages_failed=len(batch))
        return False

    # Dispatch: write each page's compressed text and update sentinels
    from collections import defaultdict
    issue_pages: dict[str, list[dict]] = defaultdict(list)
    for page in batch:
        issue_pages[str(page["sentinel_path"])].append(page)

    all_ok = True
    batch_tokens_out = 0
    for sentinel_key, pages in issue_pages.items():
        sentinel_path: Path = pages[0]["sentinel_path"]
        # Collect refs successfully compressed in this group
        newly_done: set = set()

        for page in pages:
            ref = page["ref"]
            compressed_text = compressed_map.get(ref)

            if compressed_text is None:
                # Log a snippet of the raw response to help diagnose
                raw_snippet = (compressed_map.get("_raw_response") or "")[:400]
                if raw_snippet:
                    log.debug(f"  Raw response snippet: {raw_snippet!r}")

                # Decide whether to retry solo now or defer to the next run.
                # Solo retry is attempted only if this ref already failed in a
                # batch earlier this run (solo_retry_refs tracks those refs).
                do_solo = solo_retry_refs is not None and ref in solo_retry_refs

                if do_solo:
                    log.warning(
                        f"  LLM did not return compressed block for ref '{ref}' "
                        f"(second failure this run). Retrying solo..."
                    )
                    solo_msg = build_user_message([page])
                    solo_map = client.compress(solo_msg, cancel_event=cancel_event)
                    if solo_map:
                        compressed_text = solo_map.get(ref)
                    if compressed_text is None:
                        log.warning(
                            f"  Solo retry also failed for ref '{ref}'. "
                            "Will retry on next run."
                        )
                        _update_stats(pages_failed=1)
                        all_ok = False
                        continue
                    log.info(f"  Solo retry succeeded for ref '{ref}'.")
                else:
                    log.warning(
                        f"  LLM did not return compressed block for ref '{ref}'. "
                        "Will retry on next run."
                    )
                    if solo_retry_refs is not None:
                        solo_retry_refs.add(ref)
                    _update_stats(pages_failed=1)
                    all_ok = False
                    continue

            # Write compressed text atomically
            out_path: Path = page["out_path"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(compressed_text, encoding="utf-8")
            tmp.replace(out_path)

            page_tokens_in = page["tokens"]
            page_tokens_out = count_tokens(compressed_text)
            page_ratio = page_tokens_out / page_tokens_in if page_tokens_in > 0 else 0.0
            log.debug(
                f"    {ref}: {page_tokens_in} → {page_tokens_out} tokens "
                f"({page_ratio:.1%} of input)"
            )

            batch_tokens_out += page_tokens_out
            _update_stats(tokens_out=page_tokens_out, pages_processed=1)
            newly_done.add(ref)

        # Write sentinel for this issue under an optional per-sentinel lock
        # (parallel mode) so concurrent threads merge rather than overwrite.
        lock = (sentinel_locks or {}).get(sentinel_key)
        if lock:
            with lock:
                existing = read_sentinel(sentinel_path) or set()
                write_sentinel(sentinel_path, existing | newly_done)
        else:
            existing = read_sentinel(sentinel_path) or set()
            write_sentinel(sentinel_path, existing | newly_done)

    # Log per-batch compression result so the user can tune --compression-ratio
    batch_ratio = batch_tokens_out / token_count_in if token_count_in > 0 else 0.0
    if stats_lock:
        with stats_lock:
            cumulative_ratio = (
                stats.tokens_out / stats.tokens_in if stats.tokens_in > 0 else 0.0
            )
    else:
        cumulative_ratio = (
            stats.tokens_out / stats.tokens_in if stats.tokens_in > 0 else 0.0
        )
    log.info(
        f"  → {batch_tokens_out} tokens out | "
        f"batch ratio: {batch_ratio:.1%} | "
        f"cumulative ratio: {cumulative_ratio:.1%} "
        f"(target: {int(round(compression_ratio * 100))}%)"
    )

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: OCR compression — fix noise and compress to key facts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress 1776, default 20% target size
  python compress_pass0.py --year-start 1776 --year-end 1776

  # Compress 1880-1900, keep 30% of original
  python compress_pass0.py --year-start 1880 --year-end 1900 --compression-ratio 0.30

  # Dry run to preview workload
  python compress_pass0.py --year-start 1776 --year-end 1776 --dry-run

  # Use 4 parallel LLM workers for faster throughput
  python compress_pass0.py --year-start 1880 --year-end 1890 --workers 4

  # Large-context model (128K window) with 4 workers
  python compress_pass0.py --year-start 1880 --year-end 1890 --workers 4 --context-window 128000

  # Use a different model
  CHRONAM_MODEL=claude-haiku python compress_pass0.py --year-start 1776 --year-end 1776

  # Use GitHub Copilot backend instead of HTTP proxy
  python compress_pass0.py --year-start 1776 --year-end 1776 --backend copilot

  # Auto mode: try HTTP proxy, fall back to Copilot if unavailable
  python compress_pass0.py --year-start 1880 --year-end 1890 --backend auto

Environment:
  LITELLM_PROXY_BASE       LiteLLM proxy URL
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
        "--compression-ratio",
        type=float,
        default=0.20,
        metavar="RATIO",
        help=(
            "Target output size as a fraction of input tokens. "
            "0.20 means keep ≤20%% of input tokens (default: 0.20). "
            "Range: 0.05–0.95."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        metavar="TOKENS",
        help=(
            f"Model context window in tokens (default: {DEFAULT_CONTEXT_WINDOW}). "
            "The input budget is computed as "
            "(context_window − prompt_overhead) / (1 + compression_ratio) × 0.95. "
            "Set to 128000 for gpt-4o-mini / gpt-5-mini with full window."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel LLM requests (default: 1). "
            "Increase to speed up processing when the API supports concurrent calls. "
            "Each worker sends one batch simultaneously."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and count pages but do not call LLM",
    )
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
    parser.add_argument(
        "--recheck",
        action="store_true",
        help=(
            "Delete all _pass0_done.json sentinels in the requested year range, "
            "forcing re-evaluation of all pages. Pages whose output already exists "
            "in data/pass0/ will still be re-sent to the LLM (use to pick up "
            "newly downloaded pages that were missed)."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Configure the global LLM backend before any LLM calls
    set_llm_backend(args.backend)

    if not (0.05 <= args.compression_ratio <= 0.95):
        log.error("--compression-ratio must be between 0.05 and 0.95")
        sys.exit(1)

    if args.workers < 1:
        log.error("--workers must be >= 1")
        sys.exit(1)

    if args.context_window < 2048:
        log.error("--context-window must be >= 2048")
        sys.exit(1)

    raw_dir = args.data_dir / "raw"
    pass0_dir = args.data_dir / "pass0"

    if not raw_dir.exists():
        log.error(f"Raw data directory not found: {raw_dir}")
        sys.exit(1)

    # Compute token budgets from context window and compression ratio
    max_input_tokens, max_output_tokens = compute_token_budget(
        args.context_window, args.compression_ratio
    )

    log.info("=" * 60)
    log.info("Chronicling America — Phase 0: OCR Compression")
    log.info(f"  Data dir:          {args.data_dir.resolve()}")
    log.info(f"  Years:             {args.year_start}–{args.year_end}")
    log.info(f"  Model:             {args.model}")
    log.info(f"  Backend:           {args.backend}")
    if args.backend == "copilot":
        log.info(f"  Endpoint:          GitHub Copilot API")
    elif args.backend == "litellm":
        log.info(f"  Endpoint:          {LITELLM_ENDPOINT}")
    else:  # auto
        log.info(f"  Endpoint:          {LITELLM_ENDPOINT} (fallback: GitHub Copilot)")
    log.info(f"  Compression ratio: {args.compression_ratio:.0%}")
    log.info(f"  Context window:    {args.context_window:,} tokens")
    log.info(f"  Max input/call:    {max_input_tokens:,} tokens")
    log.info(f"  Max output/call:   {max_output_tokens:,} tokens")
    log.info(f"  Workers:           {args.workers}")
    log.info(f"  Dry run:           {args.dry_run}")
    log.info("=" * 60)

    if args.backend != "copilot" and not API_KEY and not args.dry_run:
        log.error(
            "No API key found. Set LITELLM_PROXY_API_KEY or LITELLM_API_KEY env var."
        )
        sys.exit(1)

    log.info("Scanning raw data directory...")
    issues = discover_issues(raw_dir, args.year_start, args.year_end)
    log.info(f"Found {len(issues)} issue directories to evaluate.")

    # ------------------------------------------------------------------
    # --recheck: remove _pass0_done.json sentinels so all pages are
    # re-evaluated.  This catches newly downloaded pages that the sentinel
    # was preventing from being processed.
    # ------------------------------------------------------------------
    if args.recheck:
        removed = 0
        for issue in issues:
            sentinel = issue["dir"] / "_pass0_done.json"
            if sentinel.exists():
                sentinel.unlink()
                removed += 1
        log.info(f"--recheck: removed {removed} sentinel(s) — all pages will be re-evaluated.")

    client = LLMClient(
        model=args.model,
        compression_ratio=args.compression_ratio,
        max_output_tokens=max_output_tokens,
    )
    stats = Stats()

    # ------------------------------------------------------------------
    # Phase 1: bin-pack pages into batches (same for sequential and parallel)
    # ------------------------------------------------------------------
    all_batches: list[list[dict]] = []
    pending_pages: list[dict] = []
    pending_tokens: int = 0

    for idx, issue in enumerate(issues, start=1):
        if _shutdown:
            break

        log.debug(f"[{idx}/{len(issues)}] Collecting {issue['lccn']}/{issue['date']}")

        new_pages = collect_pending_pages(issue, pass0_dir, stats)

        if not new_pages:
            log.debug(
                f"  Issue {issue['lccn']}/{issue['date']}: nothing pending, skipped."
            )
            continue

        for page in new_pages:
            page_tokens = page["tokens"]

            if pending_pages and pending_tokens + page_tokens > max_input_tokens:
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
    # Tracks refs that failed in a batch this run; used to gate solo retries.
    solo_retry_refs: set = set()

    if args.workers == 1 or args.dry_run:
        # Sequential path (original behaviour; dry-run is always sequential)
        for batch in all_batches:
            if _shutdown:
                break
            flush_batch(
                batch,
                client,
                stats,
                compression_ratio=args.compression_ratio,
                dry_run=args.dry_run,
                cancel_event=_cancel_event,
                solo_retry_refs=solo_retry_refs,
            )
    else:
        # Parallel path: submit batches *lazily* — only up to `workers` at a time.
        # This ensures Ctrl-C stops new submissions immediately instead of having
        # all batches already queued.  Per-sentinel locks prevent concurrent
        # threads clobbering sentinel updates; stats_lock serialises counters.
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
            # Use a queue-style approach: keep a sliding window of futures.
            batch_iter = iter(enumerate(all_batches))
            in_flight: dict = {}  # future → batch_index

            def _submit_next() -> bool:
                """Submit the next batch if not shut down. Returns True if submitted."""
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
                    args.compression_ratio,
                    False,  # dry_run
                    sentinel_locks,
                    stats_lock,
                    _cancel_event,
                    solo_retry_refs,
                )
                in_flight[fut] = i
                return True

            # Fill the initial window
            for _ in range(args.workers):
                if not _submit_next():
                    break

            # As each future completes, immediately submit the next batch
            while in_flight:
                # as_completed yields one at a time; we pick the next done future
                done_futures = []
                for fut in list(in_flight):
                    if fut.done():
                        done_futures.append(fut)
                        break  # process one at a time to keep the window full

                if not done_futures:
                    # No future done yet — wait a little
                    time.sleep(0.05)
                    continue

                for fut in done_futures:
                    batch_idx = in_flight.pop(fut)
                    completed += 1
                    try:
                        fut.result()
                    except Exception as exc:
                        log.warning(f"Batch {batch_idx} raised an exception: {exc}")

                    # Refill: submit one more unless shutting down
                    _submit_next()

                    if completed % max(1, min(10, total // 10)) == 0 or completed == total:
                        with stats_lock:
                            ratio_str = (
                                f"{stats.tokens_out / stats.tokens_in:.1%}"
                                if stats.tokens_in > 0
                                else "n/a"
                            )
                        log.info(
                            f"[{completed}/{total}] "
                            f"processed={stats.pages_processed} "
                            f"failed={stats.pages_failed} "
                            f"llm_calls={stats.llm_calls} "
                            f"actual_ratio={ratio_str}"
                        )

                if _shutdown:
                    log.info("Shutdown: cancelling remaining queued futures...")
                    for fut in list(in_flight):
                        fut.cancel()
                    break

    # Final summary
    actual_ratio = (
        f"{stats.tokens_out / stats.tokens_in:.1%}" if stats.tokens_in > 0 else "n/a"
    )
    log.info("=" * 60)
    log.info("Phase 0 complete.")
    log.info(f"  Pages processed:      {stats.pages_processed}")
    log.info(f"  Pages skipped (done): {stats.pages_skipped_done}")
    log.info(f"  Pages skipped (short):{stats.pages_skipped_short}")
    log.info(f"  Pages failed:         {stats.pages_failed}")
    log.info(f"  Total LLM calls:      {stats.llm_calls}")
    log.info(f"  Tokens in:            {stats.tokens_in:,}")
    log.info(f"  Tokens out:           {stats.tokens_out:,}")
    log.info(f"  Actual ratio:         {actual_ratio}")
    log.info("=" * 60)

    if _shutdown:
        log.info("Stopped early. Re-run to resume.")
        sys.exit(0)


if __name__ == "__main__":
    main()

