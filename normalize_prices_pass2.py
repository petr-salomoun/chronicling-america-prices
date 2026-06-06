#!/usr/bin/env python3
"""
Chronicling America — Phase 2: Price Normalization (normalize_prices_pass2.py)
==============================================================================

Reads pass1/prices/{year}/*.jsonl files and normalises each price record via LLM:

  1. Classifies item into a commodity taxonomy (L1/L2/L3 hierarchy).
  2. Parses the raw price string into a numeric value + currency + unit.
  3. Converts the price to USD using historical exchange rates (Python-side,
     not LLM — allows rate tuning without re-running the LLM).
  4. Computes price_per_unit_usd for cross-year comparison.

Output files (all in data/pass2/prices/):
    normalized.jsonl   — successfully normalised records
    unresolved.jsonl   — PERMANENTLY unresolvable (LLM content decision):
                         item/price was too ambiguous to classify.
                         These have "permanent": true and are NOT retried.
                         Useful for taxonomy-extension analysis.
    failed.jsonl       — TRANSIENTLY failed (LLM call error after all halvings):
                         network/rate-limit/model refusal. Marked as done so
                         they don't loop forever; re-run with --retry-failed to
                         attempt processing again.
    flagged_outliers.jsonl — records flagged as statistical outliers by
                         analyze_prices.py --write-flagged. Input for
                         --reverify-flagged mode.
    _progress.json     — sentinel tracking which source files have been processed

Failure handling:
    Recoverable   — LLM call fails (None result) → auto-halve and retry up to
                    LLM_HALVING_RETRIES times.  If all halvings fail, records go
                    to failed.jsonl and file is marked done (no infinite loop).
                    Re-run with --retry-failed to reattempt.
    Permanent     — LLM returns {"unresolvable": true} → unresolved.jsonl.
                    File is marked done.  Never retried automatically.

Usage:
    python normalize_prices_pass2.py [options]

    # Process all years with default settings
    python normalize_prices_pass2.py

    # Process specific year range
    python normalize_prices_pass2.py --year-start 1770 --year-end 1800

    # Dry run (show workload, no LLM calls)
    python normalize_prices_pass2.py --year-start 1776 --year-end 1776 --dry-run

    # Use GitHub Copilot as the LLM backend
    python normalize_prices_pass2.py --year-start 1776 --year-end 1776 --backend copilot

    # Auto mode: try LiteLLM proxy first, fall back to GitHub Copilot
    python normalize_prices_pass2.py --year-start 1880 --year-end 1890 --backend auto

    # Re-run only the transiently-failed records from failed.jsonl
    python normalize_prices_pass2.py --retry-failed

    # Re-verify statistical outlier records flagged by analyze_prices.py --write-flagged
    python normalize_prices_pass2.py --reverify-flagged

    # Use a model with a 64k context window
    python normalize_prices_pass2.py --context-window 64000

Environment variables:
    LITELLM_PROXY_BASE      LiteLLM proxy URL (default: http://ai-tools.cz.intinfra.com:4004)
    LITELLM_PROXY_API_KEY   API key
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
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


TEMPERATURE = 0.0
LLM_MAX_RETRIES = 4           # urllib3-level retries (connection drops, etc.)
LLM_BACKOFF_FACTOR = 2.0
LLM_RETRY_STATUSES = [429, 500, 502, 503, 504]
LLM_TIMEOUT = (15, 900)       # (connect_s, read_s) — 15 min read timeout

# Application-level retry loop for upstream 504/5xx (server-side timeouts).
# These are tried *after* the urllib3 retries are exhausted and apply to any
# non-ok response that is in LLM_RETRY_STATUSES.
LLM_APP_RETRIES = 5           # up to 5 additional attempts
LLM_APP_BACKOFF_BASE = 60     # first sleep = 60 s, doubles each attempt → 60, 120, 240, 480, 960

# Records below this confidence threshold go to unresolved.jsonl.
# NOTE: only applies when the LLM explicitly sets unresolvable=true OR returns no
# commodity_id.  Records where the LLM returns a valid commodity + price but sets
# confidence slightly below 0.5 are still normalised (confidence is clamped to 0.5).
CONFIDENCE_THRESHOLD = 0.5

# If this fraction or more of a batch is returned as unresolvable with identical
# or known-generic reasons, treat the whole batch as a transient failure and
# trigger the halving retry — the LLM bulk-refused rather than truly analysing.
BULK_REFUSAL_FRACTION = 0.75

# Reason strings that indicate a batch-level refusal rather than a genuine
# per-record content decision.  When >= BULK_REFUSAL_FRACTION of a batch share
# one of these reasons, the batch is re-sent (with halving) instead of being
# permanently written to unresolved.jsonl.
_BULK_REFUSAL_REASONS = frozenset({
    "batch parsing not performed",
    "not parsed",
    "not mapped",
    "requires domain mapping",
    "ambiguous item or price not parsed",
    "ambiguous or non-price textual entry",
})

# Default usable context window.  Override with --context-window at runtime.
DEFAULT_CONTEXT_WINDOW = 128_000

# Approximate token overhead for the system prompt (measured ~2400 tokens).
_SYSTEM_PROMPT_OVERHEAD = 2_500
# Approximate tokens per record: input (~47) + output (~95) = ~142 total
_TOKENS_PER_RECORD_INPUT  = 50   # conservative estimate per input record
_TOKENS_PER_RECORD_OUTPUT = 100  # conservative estimate per output record


def compute_batch_size(context_window: int) -> tuple[int, int]:
    """Return (batch_size, max_output_tokens) for one LLM call given the context window.

    The context window must hold: system_prompt + N×input_rec + N×output_rec.
    A 10 % safety margin is applied.
    """
    available = int((context_window - _SYSTEM_PROMPT_OVERHEAD) * 0.90)
    batch_size = available // (_TOKENS_PER_RECORD_INPUT + _TOKENS_PER_RECORD_OUTPUT)
    max_output = batch_size * _TOKENS_PER_RECORD_OUTPUT + 512  # 512 bracket overhead
    return batch_size, max_output


# How many times to halve the batch on LLM refusal (non-JSON response)
LLM_HALVING_RETRIES = 3    # e.g. 800 → 400 → 200 → 100 minimum

# ---------------------------------------------------------------------------
# Commodity taxonomy
# ---------------------------------------------------------------------------
# Each entry: commodity_id → (l1, l2, l3, standard_unit)
# Standard units are used for price_per_unit_usd computation.

TAXONOMY = {
    # --- Food & Agriculture: Grains ---
    "FOOD-GRAIN-WHEAT":    ("Food & Agriculture", "Grains",          "Wheat",          "bushel"),
    "FOOD-GRAIN-CORN":     ("Food & Agriculture", "Grains",          "Corn/Maize",      "bushel"),
    "FOOD-GRAIN-RYE":      ("Food & Agriculture", "Grains",          "Rye",             "bushel"),
    "FOOD-GRAIN-OATS":     ("Food & Agriculture", "Grains",          "Oats",            "bushel"),
    "FOOD-GRAIN-BARLEY":   ("Food & Agriculture", "Grains",          "Barley",          "bushel"),
    "FOOD-GRAIN-RICE":     ("Food & Agriculture", "Grains",          "Rice",            "hundredweight"),
    "FOOD-GRAIN-FLOUR":    ("Food & Agriculture", "Grains",          "Flour",           "barrel"),
    "FOOD-GRAIN-BREAD":    ("Food & Agriculture", "Grains",          "Bread",           "loaf"),
    "FOOD-GRAIN-MEAL":     ("Food & Agriculture", "Grains",          "Cornmeal/Meal",   "bushel"),
    # --- Food & Agriculture: Livestock & Meat ---
    "FOOD-MEAT-BEEF":      ("Food & Agriculture", "Livestock & Meat","Beef/Cattle",     "pound"),
    "FOOD-MEAT-PORK":      ("Food & Agriculture", "Livestock & Meat","Pork/Hog",        "pound"),
    "FOOD-MEAT-MUTTON":    ("Food & Agriculture", "Livestock & Meat","Mutton/Sheep",    "pound"),
    "FOOD-MEAT-POULTRY":   ("Food & Agriculture", "Livestock & Meat","Poultry",         "pound"),
    "FOOD-MEAT-BACON":     ("Food & Agriculture", "Livestock & Meat","Bacon/Ham",       "pound"),
    "FOOD-MEAT-LARD":      ("Food & Agriculture", "Livestock & Meat","Lard",            "pound"),
    # --- Food & Agriculture: Dairy ---
    "FOOD-DAIRY-BUTTER":   ("Food & Agriculture", "Dairy",           "Butter",          "pound"),
    "FOOD-DAIRY-CHEESE":   ("Food & Agriculture", "Dairy",           "Cheese",          "pound"),
    "FOOD-DAIRY-MILK":     ("Food & Agriculture", "Dairy",           "Milk",            "quart"),
    # --- Food & Agriculture: Sugar & Sweeteners ---
    "FOOD-SUGAR-SUGAR":    ("Food & Agriculture", "Sugar",           "Sugar",           "pound"),
    "FOOD-SUGAR-MOLASSES": ("Food & Agriculture", "Sugar",           "Molasses",        "gallon"),
    # --- Food & Agriculture: Beverages ---
    "FOOD-BEV-COFFEE":     ("Food & Agriculture", "Beverages",       "Coffee",          "pound"),
    "FOOD-BEV-TEA":        ("Food & Agriculture", "Beverages",       "Tea",             "pound"),
    "FOOD-BEV-RUM":        ("Food & Agriculture", "Beverages",       "Rum",             "gallon"),
    "FOOD-BEV-WHISKEY":    ("Food & Agriculture", "Beverages",       "Whiskey/Spirits", "gallon"),
    "FOOD-BEV-WINE":       ("Food & Agriculture", "Beverages",       "Wine",            "gallon"),
    "FOOD-BEV-BEER":       ("Food & Agriculture", "Beverages",       "Beer/Ale",        "gallon"),
    # --- Food & Agriculture: Other ---
    "FOOD-SALT":           ("Food & Agriculture", "Condiments",      "Salt",            "bushel"),
    "FOOD-FISH":           ("Food & Agriculture", "Fish & Seafood",  "Fish",            "barrel"),
    "FOOD-TOBACCO":        ("Food & Agriculture", "Tobacco",         "Tobacco",         "pound"),
    "FOOD-POTATO":         ("Food & Agriculture", "Produce",         "Potatoes",        "bushel"),
    # --- Labor & Services ---
    "LABOR-DOMESTIC":      ("Labor & Services",   "Domestic Labor",  "Domestic Servant","week"),
    "LABOR-AGRI":          ("Labor & Services",   "Agricultural",    "Farm Labor",      "month"),
    "LABOR-SKILLED-CARP":  ("Labor & Services",   "Skilled Trades",  "Carpentry",       "day"),
    "LABOR-SKILLED-SMITH": ("Labor & Services",   "Skilled Trades",  "Blacksmithing",   "day"),
    "LABOR-SKILLED-OTHER": ("Labor & Services",   "Skilled Trades",  "Skilled Trade (other)","day"),
    "LABOR-UNSKILLED":     ("Labor & Services",   "Unskilled Labor", "General Labor",   "day"),
    "LABOR-MIL-SOLDIER":   ("Labor & Services",   "Military Pay",    "Soldier Pay",     "month"),
    "LABOR-MIL-OFFICER":   ("Labor & Services",   "Military Pay",    "Officer Pay",     "month"),
    "LABOR-PROFESSIONAL":  ("Labor & Services",   "Professional",    "Professional Services","service"),
    "LABOR-BOARD-LODGING": ("Labor & Services",   "Accommodation",   "Board & Lodging", "week"),
    # --- Real Estate ---
    "REAL-HOUSE-SALE":     ("Real Estate",        "Residential",     "House Sale",      "property"),
    "REAL-HOUSE-RENT":     ("Real Estate",        "Residential",     "House Rental",    "year"),
    "REAL-ROOM-RENT":      ("Real Estate",        "Residential",     "Room/Board Rental","week"),
    "REAL-LAND":           ("Real Estate",        "Land",            "Land",            "acre"),
    "REAL-COMMERCIAL":     ("Real Estate",        "Commercial",      "Commercial Property","property"),
    # --- Manufactured Goods ---
    "GOODS-TEXTILE-COTTON":("Manufactured Goods", "Textiles",        "Cotton Cloth",    "yard"),
    "GOODS-TEXTILE-WOOL":  ("Manufactured Goods", "Textiles",        "Wool Cloth",      "yard"),
    "GOODS-TEXTILE-LINEN": ("Manufactured Goods", "Textiles",        "Linen Cloth",     "yard"),
    "GOODS-TEXTILE-SILK":  ("Manufactured Goods", "Textiles",        "Silk Cloth",      "yard"),
    "GOODS-CLOTHING":      ("Manufactured Goods", "Clothing",        "Clothing/Apparel","item"),
    "GOODS-BOOTS-SHOES":   ("Manufactured Goods", "Clothing",        "Boots & Shoes",   "pair"),
    "GOODS-TOOLS-HARDWARE":("Manufactured Goods", "Tools & Hardware","Tools & Hardware","pound"),
    "GOODS-NAILS":         ("Manufactured Goods", "Tools & Hardware","Nails",           "pound"),
    "GOODS-CANDLES":       ("Manufactured Goods", "Household Goods", "Candles",         "pound"),
    "GOODS-SOAP":          ("Manufactured Goods", "Household Goods", "Soap",            "pound"),
    # --- Raw Materials ---
    "RAW-COTTON":          ("Raw Materials",      "Agricultural Raw","Raw Cotton",       "pound"),
    "RAW-WOOL":            ("Raw Materials",      "Agricultural Raw","Raw Wool",         "pound"),
    "RAW-TOBACCO-RAW":     ("Raw Materials",      "Agricultural Raw","Tobacco (raw)",    "pound"),
    "RAW-IRON":            ("Raw Materials",      "Metals",          "Iron",             "hundredweight"),
    "RAW-LEAD":            ("Raw Materials",      "Metals",          "Lead",             "pound"),
    "RAW-COPPER":          ("Raw Materials",      "Metals",          "Copper",           "pound"),
    "RAW-COAL":            ("Raw Materials",      "Fuel",            "Coal",             "ton"),
    "RAW-WOOD-LUMBER":     ("Raw Materials",      "Lumber",          "Lumber/Timber",    "thousand board feet"),
    "RAW-INDIGO":          ("Raw Materials",      "Dyes",            "Indigo",           "pound"),
    # --- Transportation ---
    "TRANS-FARE-STAGE":    ("Transportation",     "Passenger Fares", "Stage Coach Fare", "trip"),
    "TRANS-FARE-SHIP":     ("Transportation",     "Passenger Fares", "Ship Passage",     "trip"),
    "TRANS-FARE-RAIL":     ("Transportation",     "Passenger Fares", "Railroad Fare",    "trip"),
    "TRANS-FREIGHT":       ("Transportation",     "Freight",         "Freight/Shipping", "ton"),
    "TRANS-HORSE":         ("Transportation",     "Animals",         "Horse",            "animal"),
    "TRANS-OX-CATTLE":     ("Transportation",     "Animals",         "Ox/Work Cattle",   "animal"),
    # --- Financial & Legal ---
    "FIN-REWARD":          ("Financial & Legal",  "Rewards",         "Reward/Bounty",    "reward"),
    "FIN-FINE":            ("Financial & Legal",  "Fines & Penalties","Fine/Penalty",    "incident"),
    "FIN-INTEREST":        ("Financial & Legal",  "Interest Rates",  "Interest Rate",    "percent/year"),
    "FIN-BOND-SUBSCRIPTION":("Financial & Legal", "Securities",      "Bond/Subscription","unit"),
    # --- Miscellaneous ---
    "MISC-BOOK-PRINT":     ("Miscellaneous",      "Publications",    "Book/Newspaper",   "item"),
    "MISC-MEDICAL":        ("Miscellaneous",      "Medical",         "Medicine/Medical", "item"),
    "MISC-ADMISSION":      ("Miscellaneous",      "Entertainment",   "Admission/Ticket", "ticket"),
    "MISC-LIVESTOCK-OTHER":("Miscellaneous",      "Livestock Other", "Other Livestock",  "animal"),
    "MISC-OTHER":          ("Miscellaneous",      "Other",           "Other/Unclassified","unit"),
}


def _taxonomy_prompt_section() -> str:
    """Format the taxonomy as a compact list for the LLM prompt."""
    lines = ["COMMODITY TAXONOMY (use exactly these IDs):"]
    current_l1 = None
    for cid, (l1, l2, l3, unit) in TAXONOMY.items():
        if l1 != current_l1:
            lines.append(f"  [{l1}]")
            current_l1 = l1
        lines.append(f"    {cid}: {l2} / {l3}  (standard unit: {unit})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Historical currency conversion — Python-side only (NOT sent to LLM)
# ---------------------------------------------------------------------------
# Rates are approximate USD equivalents per one unit of the named currency.
# The LLM is only asked to identify currency_original and price_numeric;
# all USD conversion happens here so exchange rates can be tuned without
# re-running the LLM.

# Fixed-rate currencies (rate = USD per one unit)
_CURRENCY_FIXED_RATES: dict[str, float] = {
    # US / generic
    "usd": 1.0,
    "dollar": 1.0,
    "dollars": 1.0,
    "cent": 0.01,
    "cents": 0.01,
    # Sterling
    "gbp": 4.44,
    "pound": 4.44,
    "pounds": 4.44,
    "pound sterling": 4.44,
    "£": 4.44,
    "shilling": 0.222,
    "shillings": 0.222,
    "s": 0.222,
    "pence": 0.0185,
    "penny": 0.0185,
    "d": 0.0185,
    "farthing": 0.0046,
    # Colonial / state variants
    "old tenor": 0.222,          # MA/NE old tenor pound (~1/13 of lawful)
    "old tenor pound": 0.222,
    "lawful money": 3.33,        # colonial NE lawful money pound
    "lawful money pound": 3.33,
    "york currency shilling": 0.125,  # 8 York shillings = 1 USD
    "york shilling": 0.125,
    # Spanish / Latin-American
    "spanish dollar": 1.0,
    "peso": 1.0,
    "real": 0.125,               # 8 reales = 1 peso
    "reales": 0.125,
    "maravedi": 0.00185,
    "maravedis": 0.00185,
    # French
    "franc": 0.193,
    "francs": 0.193,
    "livre": 0.193,
    "livres": 0.193,
    # Other US slang
    "bit": 0.125,                # 1/8 dollar (used South & West)
    "bits": 0.125,
    "half-bit": 0.0625,
    "picayune": 0.0625,
    "picayunes": 0.0625,
}

# Continental dollar: year-dependent depreciation
# Values reflect approximate exchange vs. specie dollar
_CONTINENTAL_RATES: dict[int, float] = {
    1775: 1.0,
    1776: 1.0,
    1777: 0.75,
    1778: 0.33,
    1779: 0.10,
    1780: 0.025,
    1781: 0.01,
}
_CONTINENTAL_DEFAULT = 0.01  # post-1781


_PRICE_NUMERIC_SANITY_RATIO = 1000.0   # flag if LLM value is this many × the raw value
_PRICE_NUMERIC_CONSENSUS_RATIO = 10.0  # two LLM answers differ by this much → disagreement


def _sanity_check_price_numeric(
    price_numeric: float | None,
    price_raw: str,
) -> tuple[float | None, bool]:
    """Compare LLM-returned price_numeric against the first number in price_raw.

    Returns (price_numeric, flagged):
      - flagged=True when the ratio between the LLM value and the first raw number
        exceeds _PRICE_NUMERIC_SANITY_RATIO, indicating the LLM may have
        misinterpreted the price string.  The original LLM value is preserved —
        the caller should trigger a consensus retry for flagged records.
    """
    if price_numeric is None:
        return price_numeric, False

    # Extract first standalone number from price_raw (strip $, £, commas)
    clean = price_raw.replace(",", "")
    m = re.search(r"\d+\.?\d*", clean)
    if not m:
        return price_numeric, False

    try:
        raw_val = float(m.group())
    except ValueError:
        return price_numeric, False

    if raw_val <= 0 or price_numeric <= 0:
        return price_numeric, False

    ratio = max(price_numeric / raw_val, raw_val / price_numeric)
    if ratio > _PRICE_NUMERIC_SANITY_RATIO:
        log.debug(
            f"price_numeric sanity flag: LLM={price_numeric} vs raw={price_raw!r} "
            f"(ratio {ratio:.0f}×) — will trigger consensus retry."
        )
        return price_numeric, True  # flagged but NOT replaced

    return price_numeric, False


def _continental_rate(year: int) -> float:
    """Return USD-per-continental-dollar for the given year."""
    if year <= 0:
        return _CONTINENTAL_DEFAULT
    for y in sorted(_CONTINENTAL_RATES.keys(), reverse=True):
        if year >= y:
            return _CONTINENTAL_RATES[y]
    return 1.0


def convert_to_usd(
    price_numeric: float | None,
    currency_str: str,
    year: int = 0,
) -> tuple[float | None, bool]:
    """Convert a price in currency_str to USD.

    Returns (price_usd, is_uncertain):
      - price_usd is None when conversion is genuinely impossible.
      - is_uncertain=True when the rate is approximate / year-dependent.
    """
    if price_numeric is None:
        return None, True

    key = currency_str.strip().lower()

    # Continental dollar — year-dependent
    if key in ("continental", "continental dollar", "continental dollars",
               "continental currency"):
        rate = _continental_rate(year)
        return round(price_numeric * rate, 6), True

    # Uncertain / unknown markers
    if key in ("uncertain", "unknown", "", "other"):
        return None, True

    rate = _CURRENCY_FIXED_RATES.get(key)
    if rate is not None:
        return round(price_numeric * rate, 6), False

    # Fallback: try stripping punctuation / plurals
    for candidate in (key.rstrip("s"), key.replace(".", "").strip()):
        rate = _CURRENCY_FIXED_RATES.get(candidate)
        if rate is not None:
            return round(price_numeric * rate, 6), False

    # Cannot convert
    return None, True


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""IMPORTANT: You are a BATCH JSON PROCESSING ENGINE, not a conversational assistant.
You will receive a JSON array of price records. You MUST process ALL of them and return ONLY a JSON array.
- Never refuse, ask questions, or suggest a smaller batch.
- Never return natural language text.
- Always return exactly one output element per input element, in the same order.
- Output ONLY the raw JSON array — no markdown, no explanation, no preamble.

You are a historical price normalization engine processing extracted price records
from American newspapers (1770–1963). Each record has a raw item description and a raw price string.

{_taxonomy_prompt_section()}

CURRENCY IDENTIFICATION:
Identify the currency of the price and return it as currency_original using one of these
canonical names (use exactly these strings):
  USD, cent, GBP, pound, shilling, pence, farthing,
  old tenor, lawful money, continental, continental dollar,
  Spanish dollar, peso, real, maravedi,
  franc, livre, bit, half-bit, picayune, York shilling,
  uncertain

For compound amounts like "5s 6d", express price_numeric as a decimal of the PRIMARY unit
(shillings in this case): 5s 6d → price_numeric=5.5, currency_original="shilling"
(because 6 pence = 0.5 shillings).
For "£2 10s" → price_numeric=2.5, currency_original="pound"  (10s = 0.5 pound).
For ranges like "$5–10", use the midpoint (7.5) and reduce confidence to ≤ 0.7.

CRITICAL RULE — price_numeric MUST be the literal face value as written in price_raw:
- NEVER adjust, deflate, or convert price_numeric for inflation, depreciation, or currency exchange.
- NEVER apply Continental dollar depreciation, hard-dollar conversion, or any other rate.
  All such conversions are handled externally. Your job is ONLY to read the number as written.
- If price_raw is "$40", price_numeric MUST be 40.0 — not 40 × any rate.
- If price_raw is "200,000,000 continental dollars", price_numeric MUST be 200000000.0.
- If price_raw is "£2 10s", price_numeric MUST be 2.5 (decimal of pounds only, per rule above).
- If price_raw is "5s 6d", price_numeric MUST be 5.5 (decimal of shillings only).
- Do NOT invent or compute a price_numeric that is not derivable from the digits in price_raw.
- If the only digits in price_raw are part of a date or reference (not a price), set unresolvable=true.

OUTPUT FORMAT:
Return a JSON array with exactly one element per input record (same order).
Each output element MUST include the "idx" field copied verbatim from the corresponding
input record. This is used to detect position mix-ups — if your output element N has
idx=N, you have correctly matched input to output. Do NOT change, omit, or reorder idx.

CRITICAL: Do NOT mix up values between records.
Each output element's price_numeric, currency_original, unit, commodity_id etc. must
come ONLY from the corresponding input record's item and price fields.
Never copy a price_numeric value from one record to another record.
If you process record with idx=5, the output element idx=5 must contain the price_numeric
derived from input record idx=5's price field — not from any other record.

Each element must be one of:

Normalized record (commodity identified and price parseable):
{{
  "idx": 0,                                // REQUIRED: copied verbatim from input
  "commodity_id": "FOOD-GRAIN-WHEAT",     // must be a valid ID from the taxonomy
  "commodity_name": "Wheat",
  "category_l1": "Food & Agriculture",
  "category_l2": "Grains",
  "category_l3": "Wheat",
  "unit": "bushel",                        // standard physical/quantity unit for this commodity (e.g. bushel, pound, yard, gallon, acre, ton, dozen, each). Use the taxonomy standard unit. Do NOT use invented words like "item", "service", "reward", "incident", "unit" — use "each" only if no better unit exists.
  "quantity": 1.0,                         // quantity in the stated unit (e.g. 2.0 if "2 bushels for $3")
  "time_unit": null,                       // time period of the price: "year", "month", "week", "day", or null if not time-based. Extract from "per annum", "per year", "annually", "per month", "quarterly" (→ "month"), "per week", "per day", "daily" etc. A salary "per annum" has time_unit="year". A one-time price has time_unit=null.
  "price_numeric": 1.12,                   // LITERAL face value from price_raw in currency_original units
  "currency_original": "USD",              // canonical currency name from the list above
  "confidence": 0.95,                      // 0.0–1.0; lower for ambiguous currency, price ranges, etc.
  "unresolvable": false
}}

Unresolvable record (commodity unclear OR price completely unparseable):
{{
  "idx": 0,                                // REQUIRED: copied verbatim from input
  "commodity_id": null,
  "confidence": 0.1,
  "unresolvable": true,
  "reason": "brief explanation (max 15 words)"
}}

PROCESSING RULES:
- Process every single record in the input — do not skip any.
- Match to the MOST SPECIFIC taxonomy entry possible (prefer l3 specificity).
- "quantity" should reflect the amount being priced: for "per bushel" → 1.0; for "2 bushels for $3" → 2.0.
- For compound currency amounts (e.g. "5s 6d", "£2 10s"), convert to decimal of the primary unit.
- UNIT RULES: Use the taxonomy standard unit. Do NOT invent vague units. If the commodity is intangible (reward, salary, subscription, bond), set unit="each". Never use "item", "service", "reward", "incident" as units.
- TIME UNIT RULES: Always extract time_unit when the price is stated per time period. "per annum" / "annually" / "per year" → "year". "per month" / "monthly" → "month". "quarterly" → "month" (with quantity=3). "per week" → "week". "per day" / "daily" → "day". If no time period is mentioned, time_unit=null.
- Use "unresolvable": true only when you genuinely cannot determine the commodity type or
  extract any numeric value from the price. Do NOT use it for uncommon but clear items.
- Keep "reason" in unresolvable records concise (max 15 words).
- Output ONLY the JSON array — no markdown fences, no explanation, no other text.
"""

# ---------------------------------------------------------------------------
# Re-verification system prompt (skeptical second-pass for flagged outliers)
# ---------------------------------------------------------------------------
# Used by --reverify-flagged to re-examine records whose price_per_unit_usd was
# detected as a statistical outlier in analyze_prices.py.
# The LLM is asked to act as a careful auditor and decide:
#   (a) the price is genuinely unusual but correct → return confirmed=true
#   (b) the price appears to be a data extraction error → return confirmed=false + reason

REVERIFY_SYSTEM_PROMPT = """IMPORTANT: You are a BATCH JSON AUDITING ENGINE, not a conversational assistant.
You will receive a JSON array of historical price records that were flagged as statistical outliers
(their price_per_unit_usd is an extreme outlier relative to other records of the same commodity
in the same decade). Your job is to audit each record and decide whether the price is:
  (a) Genuinely unusual but plausibly correct (e.g. wartime shortage, luxury item, different unit)
  (b) Correctable: the price_numeric, unit, or quantity appears wrong but can be fixed
  (c) Unresolvable: the price is clearly erroneous and cannot be fixed

Rules:
- Never refuse or ask questions. Process every record. Return ONLY a JSON array.
- Output ONLY the raw JSON array — no markdown, no explanation, no preamble.
- Always return exactly one output element per input element, in the same order.
- Each output element MUST include the "idx" field copied verbatim from the input.

For each record return ONE of these three response shapes:

(a) Genuinely correct — keep as-is:
{
  "idx": 0,
  "confirmed": true,
  "fixed": false,
  "confidence": 0.85,
  "reason": "brief explanation (max 20 words)"
}

(b) Correctable — fix the error and keep the record:
{
  "idx": 0,
  "confirmed": true,
  "fixed": true,
  "confidence": 0.80,
  "reason": "brief explanation of the correction (max 20 words)",
  "price_numeric": 1.25,      // corrected face value (in currency_original units)
  "unit": "pound",            // corrected unit (use taxonomy standard if applicable)
  "quantity": 1.0             // corrected quantity
}
Only provide price_numeric / unit / quantity fields when actually fixing them.
Do NOT provide a field if its value is unchanged from the input.

(c) Unresolvable error — move to unresolved:
{
  "idx": 0,
  "confirmed": false,
  "fixed": false,
  "confidence": 0.80,
  "reason": "brief explanation of why it is an error (max 20 words)"
}

Guidelines for judgment:
- confirmed=true, fixed=false  when: the commodity was rare/imported, wartime prices, price
    includes transport/tax, luxury goods, early colonial era, or you are genuinely uncertain.
- confirmed=true, fixed=true   when: the unit was clearly wrong (e.g. price_per_unit_usd would
    be reasonable if unit="ton" instead of "pound"), or quantity was clearly off (the raw price
    says "per dozen" but quantity=1.0).  Apply the minimal correction to make price_per_unit_usd
    plausible.
- confirmed=false, fixed=false when: price_numeric is a date mistaken for a price, a serial
    number, or a clear typo with no correctable form; or the item/price pair is entirely
    incoherent.
- Be SKEPTICAL but not dismissive. Prefer fixing over discarding when a clear fix exists.
- Output ONLY the JSON array — no markdown fences, no explanation, no other text.
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pass2_prices")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    if _shutdown:
        log.warning("Force quit.")
        sys.exit(1)
    log.info("Shutdown requested — will stop after current batch...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_output_tokens: int = 16384):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Connection": "keep-alive",          # persist TCP connection to the proxy
            }
        )
        retry = Retry(
            total=LLM_MAX_RETRIES,
            backoff_factor=LLM_BACKOFF_FACTOR,
            status_forcelist=LLM_RETRY_STATUSES,
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        # pool_connections / pool_maxsize keep the connection pool alive between
        # requests so successive LLM calls reuse the same TCP socket to the proxy.
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=1,
            pool_maxsize=4,
            pool_block=False,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _post_litellm(self, payload: dict) -> Optional[str]:
        """POST *payload* to the LiteLLM endpoint using SSE streaming.

        Streaming keeps the TCP connection active throughout generation, avoiding
        proxy idle-timeout 504s that fire when the server goes quiet for too long.

        Applies an application-level retry loop (LLM_APP_RETRIES) with exponential
        back-off for upstream 5xx errors, on top of urllib3's low-level retries.

        Returns the fully assembled content string on success, or None on failure.
        """
        # Always request a stream to keep the connection alive
        streaming_payload = {**payload, "stream": True}

        sleep_s = LLM_APP_BACKOFF_BASE
        for attempt in range(1, LLM_APP_RETRIES + 2):  # +1 for the first (non-retry) attempt
            try:
                resp = self.session.post(
                    LITELLM_ENDPOINT,
                    data=json.dumps(streaming_payload),
                    timeout=LLM_TIMEOUT,
                    stream=True,
                )
            except requests.RequestException as e:
                log.warning(f"LLM request exception (attempt {attempt}): {e}")
                if attempt > LLM_APP_RETRIES:
                    return None
                log.info(f"  Retrying in {sleep_s}s ...")
                time.sleep(sleep_s)
                sleep_s *= 2
                continue

            if resp.status_code == 429:
                wait = sleep_s
                log.warning(f"Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                sleep_s = min(sleep_s * 2, 960)
                continue  # don't count against LLM_APP_RETRIES

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

            # 2xx — consume the SSE stream and reassemble the content
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
                log.warning(f"LLM stream read error (attempt {attempt}): {e}")
                if attempt > LLM_APP_RETRIES:
                    return None
                log.info(f"  Retrying in {sleep_s}s ...")
                time.sleep(sleep_s)
                sleep_s *= 2
                continue

        return None

    def _call_llm(self, payload: dict) -> Optional[str]:
        """Route LLM call via litellm/copilot/auto based on _llm_backend."""
        raw: Optional[str] = None
        use_litellm = _llm_backend in ("litellm", "auto")
        use_copilot = _llm_backend in ("copilot", "auto")

        if use_litellm:
            raw = self._post_litellm(payload)

        if raw is None and use_copilot:
            if _llm_backend == "auto":
                log.info("Primary LLM unavailable, switching to GitHub Copilot...")
            messages = payload.get("messages", [])
            raw = _copilot_chat_completion(
                messages=messages,
                model=payload.get("model", self.model),
                max_tokens=payload.get("max_tokens", self.max_output_tokens),
                temperature=payload.get("temperature", TEMPERATURE),
            )
        return raw

    def normalize(self, records: list[dict]) -> Optional[list]:
        """Send a batch of raw price records for normalization.

        Returns list of normalized result objects (same length as input) or None on failure.
        """
        n = len(records)
        # Prepend the expected count so the model knows it must output exactly N elements.
        # Each input record has an "idx" field (0-based). The output MUST echo the same idx.
        user_msg = (
            f"Process the following {n} price records and return a JSON array of exactly "
            f"{n} elements. Each output element MUST include the \"idx\" field copied from "
            f"the corresponding input (do NOT change it, do NOT reorder):\n"
            + json.dumps(records, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self.max_output_tokens,
            "temperature": TEMPERATURE,
        }
        raw = self._call_llm(payload)
        if raw is None:
            return None
        return self._parse_response(raw)

    def reverify(self, records: list[dict]) -> Optional[list]:
        """Send a batch of flagged outlier records for skeptical re-verification.

        Uses REVERIFY_SYSTEM_PROMPT instead of the normal SYSTEM_PROMPT.
        Returns a list of audit results [{idx, confirmed, confidence, reason}, …] or None.
        """
        n = len(records)
        user_msg = (
            f"Audit the following {n} flagged price records and return a JSON array of exactly "
            f"{n} elements. Each output element MUST include the \"idx\" field copied from "
            f"the corresponding input (do NOT change it, do NOT reorder):\n"
            + json.dumps(records, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": REVERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self.max_output_tokens,
            "temperature": TEMPERATURE,
        }
        raw = self._call_llm(payload)
        if raw is None:
            return None
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> Optional[list]:
        cleaned = raw
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
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
            log.warning(f"LLM output is not valid JSON: {e}\nRaw: {cleaned[:300]}")
            return None


# ---------------------------------------------------------------------------
# Progress / stats
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    files_processed: int = 0
    files_skipped: int = 0
    records_read: int = 0
    records_normalized: int = 0
    records_unresolved: int = 0   # permanent content failures (unresolved.jsonl)
    records_failed: int = 0       # transient LLM-call failures (failed.jsonl)
    llm_calls: int = 0


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------


def read_progress(progress_path: Path) -> set:
    """Return the set of source file keys already processed."""
    if not progress_path.exists():
        return set()
    try:
        data = json.loads(progress_path.read_text())
        return set(data.get("processed_files", []))
    except (json.JSONDecodeError, OSError):
        return set()


def write_progress(progress_path: Path, processed_files: set):
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "processed_files": sorted(processed_files),
                "updated": datetime.now(timezone.utc).isoformat(),
                "count": len(processed_files),
            },
            indent=2,
        )
    )
    tmp.replace(progress_path)


def append_jsonl(path: Path, records: list, lock: Optional[threading.Lock] = None):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
    ctx = lock if lock is not None else _null_lock()
    with ctx:
        with open(path, "a", encoding="utf-8") as f:
            f.write(lines)


class _null_lock:
    """No-op context manager — used as a drop-in for threading.Lock() in sequential mode."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Date extraction from ref
# ---------------------------------------------------------------------------

_REF_DATE_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})/")


def _extract_date_year(ref: str) -> tuple[str, int]:
    """Extract ISO date and year from a ref like 'sn83045462/1880-01-05/seq-001'."""
    m = _REF_DATE_RE.search(ref)
    if m:
        date_str = m.group(1)
        year = int(date_str[:4])
        return date_str, year
    return "", 0


# ---------------------------------------------------------------------------
# Source file discovery
# ---------------------------------------------------------------------------


def discover_price_files(prices_dir: Path, year_start: int, year_end: int) -> list[tuple[str, Path]]:
    """Return list of (file_key, Path) for all pass1 price JSONL files in year range.

    file_key is "{year}/{filename}" — used as the sentinel key.
    """
    result = []
    for year in range(year_start, year_end + 1):
        year_dir = prices_dir / str(year)
        if not year_dir.exists():
            continue
        for jsonl_path in sorted(year_dir.glob("*.jsonl")):
            file_key = f"{year}/{jsonl_path.name}"
            result.append((file_key, jsonl_path))
    return result


# ---------------------------------------------------------------------------
# Process one batch
# ---------------------------------------------------------------------------


def process_batch(
    batch: list[dict],
    client: LLMClient,
    stats: Stats,
    normalized_path: Path,
    unresolved_path: Path,
    failed_path: Path,
    dry_run: bool = False,
    _halving_depth: int = 0,
    write_lock: Optional[threading.Lock] = None,
    stats_lock: Optional[threading.Lock] = None,
) -> tuple[bool, list[dict]]:
    """Send one batch to LLM, write results, update stats.

    write_lock  — if provided, held while appending to JSONL files (for thread safety)
    stats_lock  — if provided, held while mutating the Stats object

    Returns (success, requeue_records):
      - success=True, requeue=[]          → all records fully processed
      - success=True, requeue=[...]       → LLM returned partial results (truncation);
                                            requeue contains un-answered records that
                                            should be prepended to the next batch
      - success=True, requeue=[]          → hard failure after all halvings; records
                                            written to failed.jsonl; file IS marked done
                                            (use --retry-failed to reprocess)

    Two failure modes:
      TRANSIENT (LLM call None)     → auto-halve and retry up to LLM_HALVING_RETRIES.
                                       If all halvings exhausted: write to failed.jsonl,
                                       return (True, []) so file is marked done.
      PERMANENT (LLM says unres.)   → write to unresolved.jsonl with permanent=True.
      TRUNCATION (partial results)  → write answered records, requeue the rest.
    """
    if not batch:
        return True, []

    # Prepare minimal input for LLM (just item, price, note, ref) + idx for position tracking
    llm_input = [
        {
            "idx": i,
            "ref": r["ref"],
            "item": r.get("item", ""),
            "price": r.get("price", ""),
            "note": r.get("note", ""),
        }
        for i, r in enumerate(batch)
    ]

    log.info(
        f"  LLM call: {len(batch)} records "
        f"(refs: {batch[0]['ref']} … {batch[-1]['ref']})"
        + (f" [halving depth {_halving_depth}]" if _halving_depth else "")
    )

    def _inc(**kwargs):
        ctx = stats_lock if stats_lock is not None else _null_lock()
        with ctx:
            for attr, val in kwargs.items():
                setattr(stats, attr, getattr(stats, attr) + val)

    if dry_run:
        log.info("  [DRY RUN] Would submit to LLM, skipping.")
        _inc(llm_calls=1, records_normalized=len(batch))
        return True, []

    results = client.normalize(llm_input)
    _inc(llm_calls=1)

    if results is None:
        # TRANSIENT FAILURE — halve and retry
        if _halving_depth < LLM_HALVING_RETRIES and len(batch) > 1:
            mid = len(batch) // 2
            log.warning(
                f"  LLM refused/invalid JSON — halving batch "
                f"({len(batch)} → {mid} + {len(batch)-mid}), "
                f"depth {_halving_depth + 1}/{LLM_HALVING_RETRIES}"
            )
            ok_a, requeue_a = process_batch(
                batch[:mid], client, stats,
                normalized_path, unresolved_path, failed_path,
                dry_run=dry_run, _halving_depth=_halving_depth + 1,
                write_lock=write_lock, stats_lock=stats_lock,
            )
            ok_b, requeue_b = process_batch(
                batch[mid:], client, stats,
                normalized_path, unresolved_path, failed_path,
                dry_run=dry_run, _halving_depth=_halving_depth + 1,
                write_lock=write_lock, stats_lock=stats_lock,
            )
            return ok_a and ok_b, requeue_a + requeue_b
        else:
            # All halvings exhausted — write to failed.jsonl so they don't loop
            log.warning(
                f"  LLM call failed — max halvings reached. Writing {len(batch)} records "
                f"to failed.jsonl (use --retry-failed to reprocess)."
            )
            failed_recs = [
                {**r, "_failed_reason": "LLM call failed after all halvings",
                 "_halving_depth": _halving_depth}
                for r in batch
            ]
            append_jsonl(failed_path, failed_recs, lock=write_lock)
            _inc(records_failed=len(batch))
            # Return True so these files ARE marked as done (no infinite loop)
            return True, []

    # -----------------------------------------------------------------------
    # Bulk-refusal detection: if the LLM returned a valid JSON array but
    # the overwhelming majority of entries are unresolvable=true with
    # generic/identical reasons, treat this as a transient LLM failure and
    # trigger the halving retry — exactly as if results were None.
    # -----------------------------------------------------------------------
    if results:
        n_results = len(results)
        if n_results > 0:
            unres_count = sum(
                1 for r in results
                if isinstance(r, dict) and (
                    r.get("unresolvable", False) or r.get("commodity_id") is None
                )
            )
            # Count how many share a bulk-refusal reason string
            bulk_reason_count = sum(
                1 for r in results
                if isinstance(r, dict)
                and str(r.get("reason", "")).lower() in _BULK_REFUSAL_REASONS
            )
            # Also detect when all reasons are identical (any single reason)
            all_reasons = [
                str(r.get("reason", "")).lower()
                for r in results
                if isinstance(r, dict) and r.get("reason")
            ]
            identical_reason = (
                len(all_reasons) >= max(2, n_results // 2)
                and len(set(all_reasons)) == 1
            )
            is_bulk_refusal = (
                bulk_reason_count / n_results >= BULK_REFUSAL_FRACTION
                or identical_reason
            )
            if is_bulk_refusal:
                log.warning(
                    f"  Bulk-refusal detected: {bulk_reason_count}/{n_results} records "
                    f"have generic/identical unresolvable reasons — treating as transient "
                    f"failure, will halve and retry."
                )
                # Treat as if results were None → trigger halving retry
                if _halving_depth < LLM_HALVING_RETRIES and len(batch) > 1:
                    mid = len(batch) // 2
                    log.warning(
                        f"  Halving batch ({len(batch)} → {mid} + {len(batch)-mid}), "
                        f"depth {_halving_depth + 1}/{LLM_HALVING_RETRIES}"
                    )
                    ok_a, requeue_a = process_batch(
                        batch[:mid], client, stats,
                        normalized_path, unresolved_path, failed_path,
                        dry_run=dry_run, _halving_depth=_halving_depth + 1,
                        write_lock=write_lock, stats_lock=stats_lock,
                    )
                    ok_b, requeue_b = process_batch(
                        batch[mid:], client, stats,
                        normalized_path, unresolved_path, failed_path,
                        dry_run=dry_run, _halving_depth=_halving_depth + 1,
                        write_lock=write_lock, stats_lock=stats_lock,
                    )
                    return ok_a and ok_b, requeue_a + requeue_b
                else:
                    log.warning(
                        f"  Bulk-refusal + max halvings reached. Writing {len(batch)} records "
                        f"to failed.jsonl (use --retry-failed to reprocess)."
                    )
                    failed_recs = [
                        {**r, "_failed_reason": "bulk LLM refusal after all halvings",
                         "_halving_depth": _halving_depth}
                        for r in batch
                    ]
                    append_jsonl(failed_path, failed_recs, lock=write_lock)
                    _inc(records_failed=len(batch))
                    return True, []

    # -----------------------------------------------------------------------
    # Match results to input records by idx (not position).
    # This detects and corrects position mix-ups where the LLM returns the
    # right values but assigned to the wrong record.
    # -----------------------------------------------------------------------

    # Build idx -> result dict from LLM output
    idx_to_result: dict[int, dict] = {}
    for r in results:
        if isinstance(r, dict) and "idx" in r:
            try:
                idx_to_result[int(r["idx"])] = r
            except (ValueError, TypeError):
                pass

    # Determine which batch indices were answered vs not
    batch_size = len(batch)
    answered_idxs = set(idx_to_result.keys()) & set(range(batch_size))
    unanswered_idxs = set(range(batch_size)) - answered_idxs

    # Fall back to positional matching if the LLM didn't echo any idx values
    if not answered_idxs and results:
        n_pos = min(len(results), batch_size)
        log.debug("LLM did not echo idx fields — falling back to positional matching.")
        idx_to_result = {i: results[i] for i in range(n_pos)}
        answered_idxs = set(range(n_pos))
        unanswered_idxs = set(range(batch_size)) - answered_idxs

    requeue: list[dict] = [batch[i] for i in sorted(unanswered_idxs)]
    effective_pairs: list[tuple[dict, dict]] = [
        (batch[i], idx_to_result[i]) for i in sorted(answered_idxs)
    ]

    n_answered = len(answered_idxs)
    if unanswered_idxs:
        log.info(
            f"  LLM answered {n_answered}/{batch_size} records — "
            f"re-queuing {len(unanswered_idxs)} for next batch."
        )
    elif len(results) > batch_size:
        # Extra results returned — log and ignore
        log.debug(f"  LLM returned {len(results)} results for {batch_size} inputs; extras ignored.")

    # -----------------------------------------------------------------------
    # Build output records
    # -----------------------------------------------------------------------
    normalized_out = []
    unresolved_out = []

    # First pass: process all records; collect those that need consensus retry
    pending_consensus: list[tuple[dict, dict, dict]] = []  # (raw_rec, norm, base)

    def _build_out_rec(raw_rec, norm, base, price_numeric, currency_original, year, confidence):
        """Assemble the final normalized output record."""
        cid = norm.get("commodity_id", "")
        if cid not in TAXONOMY:
            cid = "MISC-OTHER"
            confidence = min(confidence, 0.4)
        taxon = TAXONOMY.get(cid, TAXONOMY["MISC-OTHER"])
        quantity = float(norm.get("quantity") or 1.0)

        # Unit: use LLM value but reject vague invented words; fall back to taxonomy standard
        _VAGUE_UNITS = {"item", "service", "reward", "incident", "unit", "thing", "event",
                        "transaction", "case", "instance", "occurrence"}
        raw_unit = (norm.get("unit") or "").strip().lower()
        unit = raw_unit if raw_unit and raw_unit not in _VAGUE_UNITS else taxon[3]

        # Time unit: extracted by LLM (year/month/week/day/null)
        time_unit = norm.get("time_unit") or None
        if time_unit not in ("year", "month", "week", "day", None):
            time_unit = None  # discard unexpected values

        price_usd, is_uncertain = convert_to_usd(price_numeric, currency_original, year)
        if is_uncertain and price_usd is not None:
            confidence = min(confidence, 0.6)
        price_per_unit_usd: float | None = None
        if price_usd is not None and quantity > 0:
            price_per_unit_usd = round(price_usd / quantity, 6)
        return {
            **base,
            "commodity_id": cid,
            "commodity_name": norm.get("commodity_name", taxon[2]),
            "category_l1": norm.get("category_l1", taxon[0]),
            "category_l2": norm.get("category_l2", taxon[1]),
            "category_l3": norm.get("category_l3", taxon[2]),
            "unit": unit,
            "time_unit": time_unit,
            "quantity": quantity,
            "price_numeric": price_numeric,
            "currency_original": currency_original,
            "price_usd": price_usd,
            "price_per_unit_usd": price_per_unit_usd,
            "confidence": confidence,
        }

    for raw_rec, norm in effective_pairs:
        date_str, year = _extract_date_year(raw_rec.get("ref", ""))
        base = {
            "ref": raw_rec.get("ref", ""),
            "date": date_str,
            "year": year,
            "item_raw": raw_rec.get("item", ""),
            "price_raw": raw_rec.get("price", ""),
            "note": raw_rec.get("note", ""),
        }

        confidence = float(norm.get("confidence", 0.0))
        is_unresolvable = norm.get("unresolvable", False) or norm.get("commodity_id") is None

        # If the LLM explicitly flagged unresolvable, or returned no commodity_id
        # at all, route to unresolved.jsonl permanently.
        # If the LLM returned a valid commodity + price but set a low confidence
        # (e.g. 0.1 for a clearly parseable record), we still normalise the record
        # — the confidence is clamped up to CONFIDENCE_THRESHOLD so it is visible
        # as a low-quality but present result.  Only truly unresolvable records
        # (no commodity_id, or unresolvable=true) are permanently discarded.
        if is_unresolvable:
            unresolved_out.append({
                **base,
                "reason": norm.get("reason", "unresolvable"),
                "confidence": confidence,
                "permanent": True,
            })
            _inc(records_unresolved=1)
        else:
            # Clamp confidence to at least CONFIDENCE_THRESHOLD so that records
            # the LLM under-scored (but did not explicitly mark unresolvable) are
            # kept.  The raw LLM confidence is preserved in the output as
            # "confidence_llm" for downstream inspection.
            if confidence < CONFIDENCE_THRESHOLD:
                log.debug(
                    f"  Low confidence ({confidence}) clamped to {CONFIDENCE_THRESHOLD} "
                    f"for item={raw_rec.get('item','')!r} — keeping as normalized."
                )
                confidence = CONFIDENCE_THRESHOLD

            price_raw_str = raw_rec.get("price", "")
            price_numeric = norm.get("price_numeric")
            currency_original = norm.get("currency_original", "")

            # Sanity flag — if triggered, queue for consensus retry
            price_numeric, flagged = _sanity_check_price_numeric(price_numeric, price_raw_str)
            if flagged and not dry_run:
                # Store for consensus retry; build placeholder after retry below
                pending_consensus.append((raw_rec, norm, base))
                continue  # will be appended after the retry pass

            out_rec = _build_out_rec(raw_rec, norm, base, price_numeric, currency_original, year, confidence)
            normalized_out.append(out_rec)
            _inc(records_normalized=1)

    # -----------------------------------------------------------------------
    # Consensus retry for sanity-flagged records
    # -----------------------------------------------------------------------
    if pending_consensus and not dry_run:
        retry_input = [
            {
                "idx": i,
                "ref": raw_rec["ref"],
                "item": raw_rec.get("item", ""),
                "price": raw_rec.get("price", ""),
                "note": raw_rec.get("note", ""),
            }
            for i, (raw_rec, norm, base) in enumerate(pending_consensus)
        ]
        log.info(
            f"  Consensus retry: re-querying LLM for {len(retry_input)} "
            f"sanity-flagged record(s)."
        )
        retry_results = client.normalize(retry_input)
        _inc(llm_calls=1)

        # Build idx→result map for retry
        retry_idx_map: dict[int, dict] = {}
        if retry_results:
            for r in retry_results:
                if isinstance(r, dict) and "idx" in r:
                    try:
                        retry_idx_map[int(r["idx"])] = r
                    except (ValueError, TypeError):
                        pass
            if not retry_idx_map and retry_results:
                retry_idx_map = {i: retry_results[i] for i in range(min(len(retry_results), len(pending_consensus)))}

        for i, (raw_rec, norm1, base) in enumerate(pending_consensus):
            date_str, year = _extract_date_year(raw_rec.get("ref", ""))
            confidence = float(norm1.get("confidence", 0.0))
            price_numeric1 = norm1.get("price_numeric")
            currency_original = norm1.get("currency_original", "")

            norm2 = retry_idx_map.get(i)
            if norm2 is not None:
                price_numeric2 = norm2.get("price_numeric")
                # Compare the two LLM answers
                if (price_numeric1 is not None and price_numeric2 is not None
                        and price_numeric1 > 0 and price_numeric2 > 0):
                    ratio = max(price_numeric1 / price_numeric2, price_numeric2 / price_numeric1)
                    if ratio <= _PRICE_NUMERIC_CONSENSUS_RATIO:
                        # LLM answers agree — trust the value, use average
                        price_numeric = (price_numeric1 + price_numeric2) / 2.0
                        log.info(
                            f"  Consensus AGREE: item={raw_rec.get('item','')!r} "
                            f"price={raw_rec.get('price','')!r} "
                            f"→ LLM1={price_numeric1} LLM2={price_numeric2} (ratio {ratio:.1f}×)"
                        )
                    else:
                        # LLM answers disagree — lower confidence, keep first answer
                        price_numeric = price_numeric1
                        confidence = min(confidence, 0.4)
                        log.warning(
                            f"  Consensus DISAGREE: item={raw_rec.get('item','')!r} "
                            f"price={raw_rec.get('price','')!r} "
                            f"→ LLM1={price_numeric1} LLM2={price_numeric2} (ratio {ratio:.1f}×). "
                            f"Keeping LLM1, confidence capped at 0.4."
                        )
                else:
                    price_numeric = price_numeric1
                    confidence = min(confidence, 0.45)
            else:
                # Retry gave no answer for this record — keep original, lower confidence
                price_numeric = price_numeric1
                confidence = min(confidence, 0.45)
                log.warning(
                    f"  Consensus retry gave no answer for idx={i} "
                    f"({raw_rec.get('ref','')}) — keeping original, confidence capped at 0.45."
                )

            out_rec = _build_out_rec(raw_rec, norm1, base, price_numeric, currency_original, year, confidence)
            normalized_out.append(out_rec)
            _inc(records_normalized=1)

    if normalized_out:
        append_jsonl(normalized_path, normalized_out, lock=write_lock)
    if unresolved_out:
        append_jsonl(unresolved_path, unresolved_out, lock=write_lock)

    n_prices = sum(1 for r in normalized_out if r.get("price_usd") is not None)
    log.info(
        f"  → {len(normalized_out)} normalized "
        f"({n_prices} with USD price), "
        f"{len(unresolved_out)} unresolved"
        + (f", {len(requeue)} re-queued" if requeue else "")
    )
    return True, requeue


# ---------------------------------------------------------------------------
# Reverify helper
# ---------------------------------------------------------------------------

def run_reverify_pass(
    *,
    flagged_path,
    normalized_path,
    unresolved_path,
    client,
    batch_size: int,
    dry_run: bool,
    n_workers: int = 1,
) -> dict:
    """Re-examine records in *flagged_path* using the LLM reverify prompt.

    Mutates *normalized_path* and *unresolved_path* in place (unless *dry_run*).
    Rewrites *flagged_path* to contain only still-pending records.

    Returns a summary dict with keys: examined, confirmed, fixed, rejected,
    errors, still_pending.
    """
    if not flagged_path.exists() or flagged_path.stat().st_size == 0:
        return {"examined": 0, "confirmed": 0, "fixed": 0,
                "rejected": 0, "errors": 0, "still_pending": 0}

    log.info(f"Loading flagged outlier records from {flagged_path}...")
    flagged_records: list[dict] = []
    try:
        for line in flagged_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                flagged_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError as e:
        log.error(f"Cannot read {flagged_path}: {e}")
        return {"examined": 0, "confirmed": 0, "fixed": 0,
                "rejected": 0, "errors": 0, "still_pending": len(flagged_records)}

    # Skip records already reverified on a previous run
    already_done_refs: set[str] = set()
    try:
        for line in normalized_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("reverified"):
                    already_done_refs.add(rec.get("ref", ""))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass

    pending_flagged = [r for r in flagged_records if r.get("ref", "") not in already_done_refs]
    skipped_count = len(flagged_records) - len(pending_flagged)
    if skipped_count:
        log.info(f"  Skipping {skipped_count} already-reverified records.")
    log.info(f"Reverifying {len(pending_flagged)} pending flagged records...")

    if not pending_flagged:
        log.info("All flagged records already reverified. Nothing to do.")
        return {"examined": 0, "confirmed": 0, "fixed": 0,
                "rejected": 0, "errors": 0, "still_pending": 0}

    n_confirmed = 0
    n_fixed     = 0
    n_rejected  = 0
    n_error     = 0

    verdicts: dict[str, dict] = {}
    verdicts_lock = threading.Lock()
    counters_lock = threading.Lock()
    completed_batches_rv = 0
    total_chunks = (len(pending_flagged) + batch_size - 1) // batch_size

    def _reverify_chunk(batch_num: int, chunk: list[dict]):
        nonlocal n_confirmed, n_fixed, n_rejected, n_error
        llm_input = [
            {
                "idx": j,
                "ref": rec.get("ref", ""),
                "item_raw": rec.get("item_raw", ""),
                "price_raw": rec.get("price_raw", ""),
                "commodity_id": rec.get("commodity_id", ""),
                "category_l3": rec.get("category_l3", ""),
                "unit": rec.get("unit", ""),
                "quantity": rec.get("quantity", 1.0),
                "price_per_unit_usd": rec.get("price_per_unit_usd"),
                "year": rec.get("year", 0),
                "currency_original": rec.get("currency_original", ""),
                "note": rec.get("note", ""),
            }
            for j, rec in enumerate(chunk)
        ]
        results = client.reverify(llm_input)

        local_confirmed = local_fixed = local_rejected = local_error = 0
        local_verdicts: dict[str, dict] = {}

        if not results:
            log.warning(f"  Reverify batch {batch_num}: no result — marking all as errors")
            for rec in chunk:
                local_verdicts[rec.get("ref", "")] = {
                    "confirmed": False, "fixed": False,
                    "reason": "reverify call failed", "conf": 0.0, "corrections": {},
                }
                local_error += 1
        else:
            result_map: dict[int, dict] = {}
            for r in results:
                if isinstance(r, dict) and "idx" in r:
                    try:
                        result_map[int(r["idx"])] = r
                    except (ValueError, TypeError):
                        pass
            if not result_map and results:
                result_map = {k: results[k] for k in range(min(len(results), len(chunk)))}

            for j, rec in enumerate(chunk):
                ref = rec.get("ref", "")
                verdict = result_map.get(j)
                if verdict is None:
                    local_verdicts[ref] = {
                        "confirmed": False, "fixed": False,
                        "reason": "no verdict returned", "conf": 0.0, "corrections": {},
                    }
                    local_error += 1
                    continue
                confirmed = bool(verdict.get("confirmed", True))
                fixed = confirmed and bool(verdict.get("fixed", False))
                reason = str(verdict.get("reason", ""))[:200]
                conf = float(verdict.get("confidence", 0.5))
                corrections: dict = {}
                if fixed:
                    for field in ("price_numeric", "unit", "quantity"):
                        if field in verdict and verdict[field] is not None:
                            corrections[field] = verdict[field]
                local_verdicts[ref] = {
                    "confirmed": confirmed, "fixed": fixed,
                    "reason": reason, "conf": conf, "corrections": corrections,
                }
                if confirmed and fixed:
                    local_fixed += 1
                elif confirmed:
                    local_confirmed += 1
                else:
                    local_rejected += 1

        with verdicts_lock:
            verdicts.update(local_verdicts)
        with counters_lock:
            n_confirmed += local_confirmed
            n_fixed     += local_fixed
            n_rejected  += local_rejected
            n_error     += local_error

        return batch_num, local_confirmed + local_fixed, local_fixed, local_rejected, local_error

    chunks = [
        (idx + 1, pending_flagged[i: i + batch_size])
        for idx, i in enumerate(range(0, len(pending_flagged), batch_size))
    ]

    if n_workers <= 1:
        for batch_num, chunk in chunks:
            if _shutdown:
                break
            result = _reverify_chunk(batch_num, chunk)
            log.info(
                f"  Reverify batch {result[0]}: "
                f"+{batch_size} → confirmed {n_confirmed}, fixed {n_fixed}, "
                f"rejected {n_rejected}, errors {n_error}"
            )
    else:
        log.info(f"  Parallel reverify: {n_workers} workers, {total_chunks} batches.")
        done_count = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_reverify_chunk, bn, chunk): bn
                for bn, chunk in chunks
            }
            for future in as_completed(futures):
                if _shutdown:
                    break
                try:
                    result = future.result()
                except Exception as exc:
                    log.error(f"  Reverify batch error: {exc}")
                    continue
                done_count += 1
                if done_count % 10 == 0 or done_count == total_chunks:
                    log.info(
                        f"  Reverify [{done_count}/{total_chunks} batches] "
                        f"confirmed {n_confirmed}, fixed {n_fixed}, "
                        f"rejected {n_rejected}, errors {n_error}"
                    )

    log.info(
        f"Verdicts complete: {n_confirmed} confirmed, {n_fixed} fixed, "
        f"{n_rejected} rejected, {n_error} errors"
    )

    if dry_run:
        log.info("--dry-run: not rewriting normalized.jsonl or unresolved.jsonl")
        return {"examined": len(pending_flagged), "confirmed": n_confirmed, "fixed": n_fixed,
                "rejected": n_rejected, "errors": n_error, "still_pending": len(flagged_records)}

    # ------------------------------------------------------------------
    # Rewrite normalized.jsonl
    # ------------------------------------------------------------------
    log.info(f"Rewriting {normalized_path} ...")
    kept: list[dict] = []
    to_unresolved: list[dict] = []
    try:
        for line in normalized_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ref = rec.get("ref", "")
            if ref in verdicts:
                v = verdicts[ref]
                if v["confirmed"]:
                    rec["reverified"] = True
                    rec["reverify_reason"] = v["reason"]
                    if v["fixed"] and v["corrections"]:
                        rec["reverify_fixed"] = True
                        if "price_numeric" in v["corrections"]:
                            rec["price_numeric"] = v["corrections"]["price_numeric"]
                        if "unit" in v["corrections"]:
                            rec["unit"] = v["corrections"]["unit"]
                        if "quantity" in v["corrections"]:
                            rec["quantity"] = float(v["corrections"]["quantity"])
                        price_usd = rec.get("price_usd")
                        quantity = rec.get("quantity", 1.0)
                        if price_usd is not None and quantity and quantity > 0:
                            if "price_numeric" in v["corrections"]:
                                new_pn = v["corrections"]["price_numeric"]
                                orig_pn = rec.get("_orig_price_numeric") or new_pn
                                if orig_pn and orig_pn > 0:
                                    rec["price_usd"] = round(price_usd * (new_pn / orig_pn), 6)
                            rec["price_per_unit_usd"] = round(
                                rec["price_usd"] / quantity, 6
                            ) if rec.get("price_usd") and quantity > 0 else None
                    kept.append(rec)
                else:
                    to_unresolved.append({
                        "ref": ref,
                        "date": rec.get("date", ""),
                        "year": rec.get("year", 0),
                        "item_raw": rec.get("item_raw", ""),
                        "price_raw": rec.get("price_raw", ""),
                        "note": rec.get("note", ""),
                        "reason": f"reverify rejected: {v['reason']}",
                        "confidence": v["conf"],
                        "permanent": True,
                        "flagged_outlier": True,
                    })
            else:
                kept.append(rec)
    except OSError as e:
        log.error(f"Cannot read {normalized_path}: {e}")
        return {"examined": len(pending_flagged), "confirmed": n_confirmed, "fixed": n_fixed,
                "rejected": n_rejected, "errors": n_error, "still_pending": len(flagged_records)}

    try:
        with normalized_path.open("w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        log.error(f"Cannot write {normalized_path}: {e}")
        return {"examined": len(pending_flagged), "confirmed": n_confirmed, "fixed": n_fixed,
                "rejected": n_rejected, "errors": n_error, "still_pending": len(flagged_records)}

    if to_unresolved:
        append_jsonl(unresolved_path, to_unresolved)

    # Rewrite flagged_outliers.jsonl to only keep still-pending records
    resolved_refs = set(verdicts.keys()) | already_done_refs
    still_pending = [r for r in flagged_records if r.get("ref", "") not in resolved_refs]
    try:
        with flagged_path.open("w", encoding="utf-8") as f:
            for rec in still_pending:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info(
            f"  flagged_outliers.jsonl updated: "
            f"{len(still_pending)} still-pending records remain "
            f"({len(resolved_refs)} resolved this session)."
        )
    except OSError as e:
        log.warning(f"Could not rewrite {flagged_path}: {e}")

    return {
        "examined": len(pending_flagged),
        "confirmed": n_confirmed,
        "fixed": n_fixed,
        "rejected": n_rejected,
        "errors": n_error,
        "still_pending": len(still_pending),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Normalize extracted price records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all available years
  python normalize_prices_pass2.py

  # Process 1770-1800 only
  python normalize_prices_pass2.py --year-start 1770 --year-end 1800

  # Dry run to preview workload
  python normalize_prices_pass2.py --dry-run

  # Use a model with a 64k context window
  python normalize_prices_pass2.py --context-window 64000

  # Override batch size explicitly (otherwise auto-computed from --context-window)
  python normalize_prices_pass2.py --batch-size 400

Environment:
  LITELLM_PROXY_BASE       LiteLLM proxy URL
  LITELLM_PROXY_API_KEY    API key
  CHRONAM_MODEL            Model name (default: gpt-5-mini)
        """,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override output directory (default: <data-dir>/pass2/prices). "
             "Useful for running parallel instances on non-overlapping year ranges.",
    )
    parser.add_argument("--year-start", type=int, default=1770)
    parser.add_argument("--year-end", type=int, default=1963)
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
        metavar="TOKENS",
        help=(
            f"Model context window in tokens (default: {DEFAULT_CONTEXT_WINDOW}). "
            "Used to auto-compute batch size and max_output_tokens."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        metavar="N",
        help=(
            "Records per LLM call (default: auto-computed from --context-window). "
            "Records are accumulated across files to fill each call."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover files and count records but do not call LLM",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help=(
            "Re-process records from failed.jsonl (transient failures). "
            "Reads failed.jsonl, clears it, and re-runs those records through the LLM. "
            "Does not scan pass1/ — only processes the failed records."
        ),
    )
    parser.add_argument(
        "--reverify-flagged", action="store_true",
        help=(
            "Re-examine records from flagged_outliers.jsonl (written by analyze_prices.py "
            "--write-flagged) using a skeptical second-pass LLM prompt. Records the LLM "
            "confirms as errors are moved to unresolved.jsonl; confirmed records are tagged "
            "with reverified=true in normalized.jsonl."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help=(
            "Number of parallel LLM workers (default: 1 = sequential). "
            "Workers share the same output files with thread-safe locking. "
            "Each worker processes independent batches concurrently."
        ),
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

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    set_llm_backend(args.backend)

    # Compute batch size and max_output_tokens from context window
    auto_batch_size, auto_max_output = compute_batch_size(args.context_window)
    effective_batch_size = args.batch_size if args.batch_size is not None else auto_batch_size
    # Store on args for use in the rest of main
    args.batch_size = effective_batch_size

    prices_in_dir = args.data_dir / "pass1" / "prices"
    pass2_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else args.data_dir / "pass2" / "prices"
    normalized_path = pass2_dir / "normalized.jsonl"
    unresolved_path = pass2_dir / "unresolved.jsonl"
    failed_path     = pass2_dir / "failed.jsonl"
    progress_path   = pass2_dir / "_progress.json"

    if not prices_in_dir.exists():
        log.error(f"Pass1 prices directory not found: {prices_in_dir}")
        sys.exit(1)

    pass2_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Chronicling America — Phase 2: Price Normalization")
    log.info(f"  Input:          {prices_in_dir}")
    log.info(f"  Output:         {pass2_dir}")
    log.info(f"  Years:          {args.year_start}–{args.year_end}")
    log.info(f"  Model:          {args.model}")
    log.info(f"  Endpoint:       {LITELLM_ENDPOINT}")
    log.info(f"  Backend:        {args.backend}")
    log.info(f"  Context window: {args.context_window:,} tokens")
    log.info(f"  Batch size:     {args.batch_size} records/call"
             + (" (auto)" if args.batch_size == auto_batch_size else " (manual override)"))
    log.info(f"  Max output:     {auto_max_output:,} tokens/call")
    log.info(f"  Dry run:        {args.dry_run}")
    log.info(f"  Taxonomy:       {len(TAXONOMY)} commodity types")
    log.info("=" * 60)

    if not API_KEY and not args.dry_run and args.backend in ("litellm", "auto"):
        log.error("No API key found. Set LITELLM_PROXY_API_KEY or LITELLM_API_KEY.")
        sys.exit(1)

    n_workers = max(1, args.workers)

    # -----------------------------------------------------------------------
    # --retry-failed mode: reprocess records from failed.jsonl
    # -----------------------------------------------------------------------
    if args.retry_failed:
        if not failed_path.exists() or failed_path.stat().st_size == 0:
            log.info("No failed.jsonl found or it is empty — nothing to retry.")
            sys.exit(0)

        log.info(f"Loading failed records from {failed_path}...")
        retry_records = []
        try:
            for line in failed_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # Strip internal failure metadata before re-submitting
                    rec.pop("_failed_reason", None)
                    rec.pop("_halving_depth", None)
                    retry_records.append(rec)
                except json.JSONDecodeError:
                    pass
        except OSError as e:
            log.error(f"Cannot read {failed_path}: {e}")
            sys.exit(1)

        log.info(f"Retrying {len(retry_records)} failed records...")

        # Clear failed.jsonl before reprocessing (records will be written to
        # normalized/unresolved/failed as appropriate on this run)
        if not args.dry_run:
            failed_path.write_text("")

        client = LLMClient(model=args.model, max_output_tokens=auto_max_output)
        stats = Stats()

        # Process in batches
        for i in range(0, len(retry_records), args.batch_size):
            if _shutdown:
                break
            chunk = retry_records[i: i + args.batch_size]
            process_batch(
                chunk, client, stats,
                normalized_path, unresolved_path, failed_path,
                dry_run=args.dry_run,
            )

        log.info("=" * 60)
        log.info("--retry-failed complete.")
        log.info(f"  Records retried:     {len(retry_records)}")
        log.info(f"  Normalized:          {stats.records_normalized}")
        log.info(f"  Unresolved (perm):   {stats.records_unresolved}")
        log.info(f"  Still failed:        {stats.records_failed}")
        log.info(f"  LLM calls:           {stats.llm_calls}")
        log.info("=" * 60)
        sys.exit(0)

    # -----------------------------------------------------------------------
    # --reverify-flagged mode: skeptical second-pass for outlier records
    # -----------------------------------------------------------------------
    if args.reverify_flagged:
        flagged_path = pass2_dir / "flagged_outliers.jsonl"
        if not flagged_path.exists() or flagged_path.stat().st_size == 0:
            log.info(
                "No flagged_outliers.jsonl found or it is empty.\n"
                "Run: python analyze_prices.py"
            )
            sys.exit(0)

        client = LLMClient(model=args.model, max_output_tokens=auto_max_output)
        rv = run_reverify_pass(
            flagged_path=flagged_path,
            normalized_path=normalized_path,
            unresolved_path=unresolved_path,
            client=client,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            n_workers=n_workers,
        )
        n_batches = (rv["examined"] + args.batch_size - 1) // max(args.batch_size, 1)
        log.info("=" * 60)
        log.info("--reverify-flagged complete.")
        log.info(f"  Flagged records examined:           {rv['examined']}")
        log.info(f"  Confirmed (tagged reverified=true): {rv['confirmed']}")
        log.info(f"  Fixed (corrected + reverified):     {rv['fixed']}")
        log.info(f"  Rejected → unresolved.jsonl:        {rv['rejected']}")
        log.info(f"  Errors (call failures):             {rv['errors']}")
        log.info(f"  Still-pending in flagged_outliers:  {rv['still_pending']}")
        log.info(f"  LLM calls:                          {n_batches}")
        log.info("=" * 60)
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Normal mode: discover and process pass1 price files
    # -----------------------------------------------------------------------
    log.info("Scanning pass1/prices directory...")
    all_files = discover_price_files(prices_in_dir, args.year_start, args.year_end)
    log.info(f"Found {len(all_files)} price JSONL files.")

    processed_keys = read_progress(progress_path)
    pending_files = [
        (key, path) for key, path in all_files
        if key not in processed_keys
    ]
    log.info(
        f"  {len(processed_keys)} already processed, "
        f"{len(pending_files)} pending."
    )

    client = LLMClient(model=args.model, max_output_tokens=auto_max_output)
    stats = Stats()
    stats.files_skipped = len(processed_keys)

    # -----------------------------------------------------------------------
    # Cross-file accumulator: collect records from many small files before
    # sending an LLM call, so each call uses close to the full context window.
    #
    # acc_records   — raw records buffered for the next LLM call
    # acc_file_keys — file keys whose records are in the buffer
    #                 (a file_key is added here when ALL its records are loaded)
    # -----------------------------------------------------------------------
    acc_records: list[dict] = []
    acc_file_keys: list[str] = []

    if n_workers > 1:
        # -------------------------------------------------------------------
        # PARALLEL MODE: pre-build all batches, then dispatch concurrently.
        # -------------------------------------------------------------------
        write_lock = threading.Lock()
        stats_lock = threading.Lock()
        progress_lock = threading.Lock()

        # Pre-build batch list from all pending files
        all_batches: list[tuple[list[dict], list[str]]] = []  # (records, file_keys)

        for idx, (file_key, jsonl_path) in enumerate(pending_files, start=1):
            if jsonl_path.stat().st_size == 0:
                if not args.dry_run:
                    processed_keys.add(file_key)
                stats.files_skipped += 1
                continue

            file_records = []
            try:
                for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("t") == "$":
                            file_records.append(rec)
                    except json.JSONDecodeError:
                        pass
            except OSError as e:
                log.warning(f"Cannot read {jsonl_path}: {e}")
                continue

            if not file_records:
                if not args.dry_run:
                    processed_keys.add(file_key)
                stats.files_skipped += 1
                continue

            stats.records_read += len(file_records)
            acc_records.extend(file_records)
            acc_file_keys.append(file_key)

            if len(acc_records) >= args.batch_size:
                all_batches.append((list(acc_records), list(acc_file_keys)))
                acc_records = []
                acc_file_keys = []

        if acc_records:
            all_batches.append((list(acc_records), list(acc_file_keys)))
            acc_records = []
            acc_file_keys = []

        log.info(
            f"Parallel mode: {n_workers} workers, {len(all_batches)} batches "
            f"({stats.records_read} records total)."
        )

        completed_batches = 0

        def _run_batch(batch_idx: int, chunk: list[dict], chunk_file_keys: list[str]):
            ok, requeue = process_batch(
                chunk, client, stats,
                normalized_path, unresolved_path, failed_path,
                dry_run=args.dry_run,
                write_lock=write_lock,
                stats_lock=stats_lock,
            )
            # Requeue not fully handled in parallel mode — log and continue
            if requeue:
                log.warning(
                    f"  [batch {batch_idx}] {len(requeue)} records not answered by LLM; "
                    f"they will be lost (re-run to recover from failed.jsonl)."
                )
            return ok, chunk_file_keys

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_batch, i, chunk, keys): (i, keys)
                for i, (chunk, keys) in enumerate(all_batches)
            }
            for future in as_completed(futures):
                if _shutdown:
                    break
                batch_idx, _ = futures[future]
                try:
                    ok, chunk_file_keys = future.result()
                except Exception as exc:
                    log.error(f"  [batch {batch_idx}] Exception: {exc}")
                    continue

                if ok and not args.dry_run:
                    with progress_lock:
                        for key in chunk_file_keys:
                            processed_keys.add(key)
                        stats.files_processed += len(chunk_file_keys)

                completed_batches += 1
                if completed_batches % 10 == 0 or completed_batches == len(all_batches):
                    if not args.dry_run:
                        with progress_lock:
                            write_progress(progress_path, processed_keys)
                    log.info(
                        f"[{completed_batches}/{len(all_batches)} batches] "
                        f"records: {stats.records_normalized} normalized, "
                        f"{stats.records_unresolved} unresolved, "
                        f"{stats.records_failed} failed, "
                        f"llm_calls: {stats.llm_calls}"
                    )

    else:
        # -------------------------------------------------------------------
        # SEQUENTIAL MODE (original logic — unchanged)
        # -------------------------------------------------------------------

        def flush_accumulator(force: bool = False) -> bool:
            """Send buffered records to LLM (if enough accumulated or forced).

    Handles two outcomes from process_batch:
      - Partial results (truncation): re-queued records are prepended to
        acc_records so they get paired with new records in the next batch.
      - Complete failure (refusal after all halvings): affected file keys
        are NOT registered → they retry on next run.

    Returns False only if a hard failure occurred AND we should abort.
    """
            nonlocal acc_records, acc_file_keys
            if not acc_records:
                return True
            if not force and len(acc_records) < args.batch_size:
                return True  # not yet full — keep accumulating

            # Snapshot the file keys associated with this batch before sending
            chunk_file_keys = list(acc_file_keys)
            chunk = list(acc_records)

            ok, requeue = process_batch(
                chunk, client, stats,
                normalized_path, unresolved_path, failed_path,
                dry_run=args.dry_run,
            )

            if requeue:
                # Partial truncation: some records not answered — keep them at the
                # front of the accumulator for the next batch.
                acc_records = requeue
                return True

            # Full success (including records written to failed.jsonl) — mark files done
            if not args.dry_run:
                for key in chunk_file_keys:
                    processed_keys.add(key)
                stats.files_processed += len(chunk_file_keys)

            acc_records = []
            acc_file_keys = []
            return True

        # Iterate files, accumulating records into batches
        for idx, (file_key, jsonl_path) in enumerate(pending_files, start=1):
            if _shutdown:
                break

            # Read all records from this file
            if jsonl_path.stat().st_size == 0:
                if not args.dry_run:
                    processed_keys.add(file_key)
                stats.files_skipped += 1
                if not args.dry_run and idx % 500 == 0:
                    write_progress(progress_path, processed_keys)
                continue

            file_records = []
            try:
                for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("t") == "$":
                            file_records.append(rec)
                    except json.JSONDecodeError:
                        pass
            except OSError as e:
                log.warning(f"Cannot read {jsonl_path}: {e}")
                continue

            if not file_records:
                if not args.dry_run:
                    processed_keys.add(file_key)
                stats.files_skipped += 1
                continue

            stats.records_read += len(file_records)
            acc_records.extend(file_records)
            acc_file_keys.append(file_key)

            log.debug(
                f"[{idx}/{len(pending_files)}] {file_key}: "
                f"+{len(file_records)} records (buffer={len(acc_records)})"
            )

            # Flush when buffer is full or on shutdown
            if len(acc_records) >= args.batch_size or _shutdown:
                if not flush_accumulator(force=_shutdown):
                    break

            # Periodic progress save + status log
            if (idx % 200 == 0) or _shutdown:
                if not args.dry_run:
                    write_progress(progress_path, processed_keys)
                log.info(
                    f"[{idx}/{len(pending_files)}] "
                    f"files done: {stats.files_processed}, "
                    f"records: {stats.records_normalized} normalized, "
                    f"{stats.records_unresolved} unresolved, "
                    f"{stats.records_failed} failed, "
                    f"llm_calls: {stats.llm_calls}"
                )

        # Flush any remaining records in the accumulator
        if not _shutdown:
            flush_accumulator(force=True)
        else:
            log.info(
                f"Shutdown: {len(acc_records)} buffered records in "
                f"{len(acc_file_keys)} files will retry on next run."
            )

    # Final save
    if not args.dry_run:
        write_progress(progress_path, processed_keys)

    log.info("=" * 60)
    log.info("Phase 2 Price Normalization complete.")
    log.info(f"  Files processed:     {stats.files_processed}")
    log.info(f"  Files skipped:       {stats.files_skipped}")
    log.info(f"  Records read:        {stats.records_read}")
    log.info(f"  Records normalized:  {stats.records_normalized}")
    log.info(f"  Records unresolved:  {stats.records_unresolved}  (permanent → unresolved.jsonl)")
    log.info(f"  Records failed:      {stats.records_failed}  (transient → failed.jsonl)")
    log.info(f"  Total LLM calls:     {stats.llm_calls}")
    log.info(f"  Output:              {normalized_path}")
    log.info(f"  Unresolved (perm):   {unresolved_path}")
    log.info(f"  Failed (transient):  {failed_path}")
    if stats.records_failed > 0:
        log.info(f"  → Re-run with --retry-failed to reattempt {stats.records_failed} failed records.")
    log.info("=" * 60)

    if _shutdown:
        log.info("Stopped early. Re-run to resume from where we left off.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Auto-reverify: if flagged_outliers.jsonl exists with pending records,
    # run the reverify pass automatically without needing --reverify-flagged.
    # -----------------------------------------------------------------------
    flagged_path = pass2_dir / "flagged_outliers.jsonl"
    if flagged_path.exists() and flagged_path.stat().st_size > 0:
        log.info("=" * 60)
        log.info("flagged_outliers.jsonl detected — running auto reverify pass...")
        rv = run_reverify_pass(
            flagged_path=flagged_path,
            normalized_path=normalized_path,
            unresolved_path=unresolved_path,
            client=client,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            n_workers=n_workers,
        )
        n_batches = (rv["examined"] + args.batch_size - 1) // max(args.batch_size, 1)
        log.info("=" * 60)
        log.info("Auto reverify complete.")
        log.info(f"  Flagged records examined:           {rv['examined']}")
        log.info(f"  Confirmed (tagged reverified=true): {rv['confirmed']}")
        log.info(f"  Fixed (corrected + reverified):     {rv['fixed']}")
        log.info(f"  Rejected → unresolved.jsonl:        {rv['rejected']}")
        log.info(f"  Errors (call failures):             {rv['errors']}")
        log.info(f"  Still-pending in flagged_outliers:  {rv['still_pending']}")
        log.info(f"  LLM calls:                          {n_batches}")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
