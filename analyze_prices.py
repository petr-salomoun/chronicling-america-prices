#!/usr/bin/env python3
"""
Chronicling America — Price Analysis & Visualization (analyze_prices.py)
=========================================================================

Reads the normalized price records from data/pass2/prices/normalized.jsonl
and produces:

  Console output:
    - Summary statistics (total records, coverage, top commodities)
    - Per-category breakdown table
    - Long-term price trend summary

  Plots (saved to price_analysis_output/):
    fig01_coverage_per_year.png       — Records per year (data density)
    fig02_category_distribution.png   — L1 category distribution (donut)
    fig03_top_commodities.png         — Top 25 commodities by record count
    fig04_currency_breakdown.png      — Currency type breakdown by era
    fig05_staples_over_time.png       — Staple goods price trends (wheat, corn, pork, butter)
    fig06_labor_over_time.png         — Labor wage trends over time
    fig07_commodity_grid.png          — Grid of top-N commodities with price trends
    fig08_decade_volatility.png       — Price volatility (IQR) by decade for key commodities
    fig09_confidence_dist.png         — Distribution of normalization confidence scores
    fig10_unresolved_analysis.png     — Top unresolved item categories (from unresolved.jsonl)
    fig11_availability_heatmap.png    — Commodity sub-category × year data heatmap (pandas)
    fig12_outlier_map.png             — Statistical price outliers per year (IQR×3 per commodity+decade)

Usage:
    python analyze_prices.py [options]

    # Run full analysis with default settings
    python analyze_prices.py

    # Limit to specific year range
    python analyze_prices.py --year-start 1770 --year-end 1850

    # Specify minimum records per commodity for trend plots
    python analyze_prices.py --min-records 10

    # Outliers are flagged and written to flagged_outliers.jsonl by default.
    # To disable this behaviour:
    python analyze_prices.py --no-write-flagged

Requirements:
    pip install pandas matplotlib
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import to_rgba
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _print_table(rows, headers, col_widths):
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "  ".join("-" * w for w in col_widths)
    print(header_row)
    print(sep)
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_normalized(path: Path, year_start: int, year_end: int) -> list[dict]:
    """Load normalized price records from JSONL, filtered by year range."""
    records = []
    if not path.exists():
        return records
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                year = rec.get("year", 0)
                if year_start <= year <= year_end:
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    except OSError as e:
        print(f"ERROR: Cannot read {path}: {e}", file=sys.stderr)
    return records


def load_unresolved(path: Path) -> list[dict]:
    """Load unresolved price records from JSONL."""
    records = []
    if not path.exists():
        return records
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return records


# ---------------------------------------------------------------------------
# Console reports
# ---------------------------------------------------------------------------

def print_summary(records: list[dict], unresolved: list[dict]):
    total = len(records) + len(unresolved)
    if total == 0:
        print("No records found. Run normalize_prices_pass2.py first.")
        return

    years = [r["year"] for r in records if r.get("year")]
    years_with_usd = [r["year"] for r in records if r.get("price_per_unit_usd") is not None and r.get("year")]
    n_with_usd = len(years_with_usd)

    print("=" * 72)
    print("PHASE 2 — Price Normalization Summary")
    print("=" * 72)
    print(f"  Total records processed:   {_fmt_count(total)}")
    print(f"  Normalized records:        {_fmt_count(len(records))}")
    print(f"    → with USD price:        {_fmt_count(n_with_usd)}  ({100*n_with_usd/max(1,len(records)):.0f}%)")
    print(f"  Unresolved records:        {_fmt_count(len(unresolved))}")
    if years:
        print(f"  Year range:                {min(years)}–{max(years)}")
        year_cnt = Counter(years)
        peak_year = max(year_cnt, key=year_cnt.get)
        print(f"  Peak year:                 {peak_year} ({year_cnt[peak_year]:,} records)")
    n_time_based = sum(1 for r in records if r.get("time_unit"))
    if n_time_based:
        from collections import Counter as _C
        tu_counts = _C(r["time_unit"] for r in records if r.get("time_unit"))
        tu_str = "  ".join(f"{tu}:{cnt}" for tu, cnt in tu_counts.most_common())
        print(f"  Time-based prices:         {_fmt_count(n_time_based)} ({tu_str})")
    print()


def print_category_report(records: list[dict]):
    if not records:
        return

    # Group by L1
    l1_counts: dict[str, int] = defaultdict(int)
    l2_counts: dict[str, int] = defaultdict(int)
    l3_counts: dict[str, int] = defaultdict(int)

    for r in records:
        l1 = r.get("category_l1", "Unknown")
        l2 = r.get("category_l2", "Unknown")
        l3 = r.get("category_l3", "Unknown")
        l1_counts[l1] += 1
        l2_counts[f"{l1} / {l2}"] += 1
        l3_counts[f"{l3} ({r.get('commodity_id','?')})"] += 1

    print("=" * 72)
    print("Top-level categories (L1):")
    print("=" * 72)
    total = len(records)
    rows = [
        (l1, count, f"{100*count/total:.1f}%")
        for l1, count in sorted(l1_counts.items(), key=lambda x: -x[1])
    ]
    _print_table(rows, ["Category", "Records", "Share"], [30, 10, 8])

    print("=" * 72)
    print("Top 30 specific commodities (L3):")
    print("=" * 72)
    rows = [
        (name, count, f"{100*count/total:.1f}%")
        for name, count in sorted(l3_counts.items(), key=lambda x: -x[1])[:30]
    ]
    _print_table(rows, ["Commodity", "Records", "Share"], [45, 10, 8])


def _dominant_unit_records(recs: list[dict]) -> tuple[list[dict], str]:
    """Return only records with the most frequent unit_raw value, plus that unit string."""
    from collections import Counter
    unit_counts = Counter(r.get("unit_raw") or "" for r in recs)
    dominant = unit_counts.most_common(1)[0][0]
    filtered = [r for r in recs if (r.get("unit_raw") or "") == dominant]
    return filtered, dominant


def _remove_outliers(prices: list[float]) -> list[float]:
    """Remove outliers using IQR×1.5 (Tukey). Falls back to MAD for small n."""
    import math
    if len(prices) < 2:
        return prices
    s = sorted(prices)
    n = len(s)
    if n >= 4:
        q1 = s[n // 4]
        q3 = s[(3 * n) // 4]
        iqr = q3 - q1
        if iqr == 0:
            return prices
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return [p for p in prices if lo <= p <= hi]
    else:
        # MAD for n=2-3
        med = s[n // 2]
        mad = sorted(abs(p - med) for p in s)[n // 2]
        if mad == 0:
            return prices
        return [p for p in prices if abs(p - med) / mad <= 3.0]


def _remove_outliers_log(prices: list[float], iqr_factor: float = 1.5) -> list[float]:
    """Remove outliers using IQR×iqr_factor in log space.

    Applies a two-pass approach:
    1. Crude pre-filter: remove values > 2.5 log-decades from the median
       (physically impossible for any commodity at any point in history)
    2. IQR×iqr_factor in log space on the pre-filtered data
    """
    import math
    pos = [p for p in prices if p > 0]
    if len(pos) < 2:
        return pos

    # Pre-filter: discard values more than 2.5 log-decades from median
    logs_sorted = sorted(math.log10(p) for p in pos)
    n = len(logs_sorted)
    log_med = logs_sorted[n // 2]
    pos = [p for p in pos if abs(math.log10(p) - log_med) <= 2.5]
    if len(pos) < 2:
        return pos

    logs = sorted(math.log10(p) for p in pos)
    n = len(logs)
    q1 = logs[n // 4]
    q3 = logs[(3 * n) // 4]
    iqr = q3 - q1
    if iqr == 0:
        return pos
    lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    return [p for p in pos if lo <= math.log10(p) <= hi]


def _cross_year_fence(prices: list[float], factor: float = 1.5) -> tuple[float, float]:
    """Return (lo, hi) price bounds as median × 10^±factor in linear space.

    Pre-filters values more than 4 log-decades from the raw median before computing
    the fence (prevents a few extreme outliers from dominating the bounds).
    """
    import math
    pos = [p for p in prices if p > 0]
    if not pos:
        return 0.0, float("inf")
    raw_logs = sorted(math.log10(p) for p in pos)
    raw_med = raw_logs[len(raw_logs) // 2]
    # Pre-filter: discard physically impossible values (>2.5 decades from median)
    pos = [p for p in pos if abs(math.log10(p) - raw_med) <= 2.5]
    if not pos:
        return 0.0, float("inf")
    log_med = sorted(math.log10(p) for p in pos)[len(pos) // 2]
    return 10 ** (log_med - factor), 10 ** (log_med + factor)


# ---------------------------------------------------------------------------
# Canonical unit maps — per commodity, the normalised unit strings we trust.
# Records with other units are excluded from trend analysis.
# Keys are commodity_id prefixes (matched with startswith).
# ---------------------------------------------------------------------------

_UNIT_ALIASES: dict[str, str] = {
    # bulk volume / dry measure
    "bu": "bushel", "bu.": "bushel", "bus": "bushel", "buses": "bushel",
    "bushels": "bushel",
    "peck": "peck", "pecks": "peck",
    "gallon": "gallon", "gallons": "gallon", "gal": "gallon",
    "barrel": "barrel", "barrels": "barrel", "bbl": "barrel",
    "hogshead": "hogshead",
    # weight
    "pound": "pound", "pounds": "pound", "lb": "pound", "lbs": "pound",
    "oz": "ounce", "ounce": "ounce", "ounces": "ounce",
    "hundredweight": "hundredweight", "cwt": "hundredweight",
    "ton": "ton", "tons": "ton",
    "cental": "hundredweight",   # cental = 100 lbs = cwt
    "100 lb": "hundredweight", "100 lbs": "hundredweight",
    "60 lbs": "bushel",          # 60 lbs = standard wheat bushel
    # count
    "dozen": "dozen", "doz": "dozen",
    "each": "each", "unit": "each", "item": "each",
    "head": "head",
    # area
    "acre": "acre", "acres": "acre",
    # time
    "day": "day", "daily": "day",
    "week": "week", "weekly": "week",
    "month": "month", "monthly": "month",
    "year": "year", "annual": "year", "annually": "year",
    "hour": "hour",
    # misc
    "cord": "cord",
    "box": "box", "case": "case", "sack": "sack",
    "package": "package",
}

def _normalize_unit(unit: str) -> str:
    """Lowercase and alias a unit string to its canonical form."""
    u = (unit or "").lower().strip()
    return _UNIT_ALIASES.get(u, u)


# Canonical units per commodity prefix.
# Records whose normalised unit is NOT in this set are excluded from trend analysis.
# None = no restriction (use dominant-unit fallback).
_CANONICAL_UNITS: dict[str, set[str] | None] = {
    "FOOD-GRAIN-WHEAT":    {"bushel", "hundredweight", "pound"},
    "FOOD-GRAIN-CORN":     {"bushel", "hundredweight", "pound"},
    "FOOD-GRAIN-FLOUR":    {"barrel", "hundredweight", "pound", "sack"},
    "FOOD-GRAIN-OATS":     {"bushel", "pound", "hundredweight"},
    "FOOD-GRAIN-RYE":      {"bushel", "pound"},
    "FOOD-GRAIN-BARLEY":   {"bushel", "pound"},
    "FOOD-GRAIN-RICE":     {"pound", "hundredweight", "bushel"},
    "FOOD-MEAT-BEEF":      {"pound", "hundredweight", "barrel"},
    "FOOD-MEAT-PORK":      {"pound", "hundredweight", "barrel"},
    "FOOD-MEAT-POULTRY":   {"pound", "dozen", "each"},
    "FOOD-MEAT-MUTTON":    {"pound", "hundredweight"},
    "FOOD-DAIRY-BUTTER":   {"pound", "dozen"},
    "FOOD-DAIRY-CHEESE":   {"pound"},
    "FOOD-DAIRY-MILK":     {"gallon", "quart", "pint"},
    "FOOD-DAIRY-EGGS":     {"dozen", "each"},
    "FOOD-POTATO":         {"bushel", "pound", "hundredweight"},
    "FOOD-BEV-COFFEE":     {"pound"},
    "FOOD-BEV-TEA":        {"pound", "ounce"},
    "FOOD-BEV-SUGAR":      {"pound", "hundredweight"},
    "FOOD-BEV-MOLASSES":   {"gallon", "barrel"},
    "FOOD-BEV-WHISKEY":    {"gallon", "barrel"},
    "FOOD-BEV-WINE":       {"gallon", "bottle"},
    "LABOR-UNSKILLED":     {"day", "week", "month", "hour", "year"},
    "LABOR-SKILLED":       {"day", "week", "month", "hour", "year"},
    "LABOR-AGRI":          {"day", "week", "month", "year"},
    "LABOR-DOMESTIC":      {"week", "month", "year"},
    "LABOR-FARM":          {"day", "week", "month", "year"},
    "LABOR-PROFESSIONAL":  {"year", "month", "week", "day"},
    "TRANS-HORSE":         {"each", "head"},
    "TRANS-FARE-RAIL":     {"each"},
    "REAL-LAND":           {"acre"},
    "REAL-HOUSE-SALE":     {"each"},
    "RAW-COAL":            {"ton", "hundredweight", "bushel"},
    "RAW-WOOD":            {"cord", "thousand", "board"},
    "RAW-IRON":            {"ton", "hundredweight", "pound"},
    "RAW-COTTON":          {"pound", "bale"},
    "RAW-WOOL":            {"pound"},
    "RAW-LEATHER":         {"pound", "side", "each"},
    "GOODS-CLOTHING":      {"each", "yard", "dozen"},
    "GOODS-BOOTS-SHOES":   {"each", "pair"},
}

# ---------------------------------------------------------------------------
# Physical price bounds: (min_USD, max_USD) per unit for analysis/plotting.
# Records outside these bounds are excluded from trend analysis — they are
# physically implausible even accounting for inflation/deflation over 200 years.
# Keys are commodity_id prefixes (same matching logic as _CANONICAL_UNITS).
# ---------------------------------------------------------------------------
_PRICE_BOUNDS: dict[str, tuple[float, float]] = {
    # Grains per bushel (~32-60 lbs)
    "FOOD-GRAIN-WHEAT":  (0.05, 20.0),
    "FOOD-GRAIN-CORN":   (0.02, 15.0),
    "FOOD-GRAIN-OATS":   (0.02, 10.0),
    "FOOD-GRAIN-RYE":    (0.02, 10.0),
    "FOOD-GRAIN-BARLEY": (0.02, 10.0),
    "FOOD-GRAIN-RICE":   (0.005, 5.0),   # per pound
    # Flour per barrel (~196 lbs) or hundredweight
    "FOOD-GRAIN-FLOUR":  (0.50, 50.0),
    # Meats per pound
    "FOOD-MEAT-BEEF":    (0.01, 5.0),
    "FOOD-MEAT-PORK":    (0.01, 5.0),
    "FOOD-MEAT-POULTRY": (0.01, 5.0),
    "FOOD-MEAT-MUTTON":  (0.01, 5.0),
    # Dairy per pound
    "FOOD-DAIRY-BUTTER": (0.01, 3.0),
    "FOOD-DAIRY-CHEESE": (0.01, 3.0),
    "FOOD-DAIRY-MILK":   (0.01, 5.0),   # per gallon
    "FOOD-DAIRY-EGGS":   (0.01, 5.0),   # per dozen
    # Staples
    "FOOD-POTATO":       (0.01, 10.0),
    "FOOD-BEV-COFFEE":   (0.01, 10.0),
    "FOOD-BEV-TEA":      (0.01, 20.0),
    "FOOD-BEV-SUGAR":    (0.001, 5.0),
    "FOOD-BEV-MOLASSES": (0.01, 5.0),
    # Labor per day/week/month/year
    "LABOR":             (0.01, 100000.0),
    # Land per acre
    "REAL-LAND":         (0.10, 100000.0),
    # Horses
    "TRANS-HORSE":       (1.0, 5000.0),
}


def _apply_price_bounds(recs: list[dict], cid: str) -> list[dict]:
    """Filter records to those within physical price bounds for this commodity."""
    # Find matching bounds
    bounds = None
    for prefix in sorted(_PRICE_BOUNDS.keys(), key=len, reverse=True):
        if cid.startswith(prefix) or cid == prefix:
            bounds = _PRICE_BOUNDS[prefix]
            break
    if bounds is None:
        return recs
    lo, hi = bounds
    return [r for r in recs if lo <= (r.get("price_per_unit_usd") or 0) <= hi]


def _canonical_unit_records(recs: list[dict], cid: str) -> tuple[list[dict], str]:
    """Filter records to only those with canonical units for the commodity.

    Returns (filtered_records, dominant_canonical_unit).
    Falls back to _dominant_unit_records if no canonical map entry exists for cid.
    """
    # Find matching canonical set (longest matching prefix wins)
    canon: set[str] | None = None
    for prefix in sorted(_CANONICAL_UNITS.keys(), key=len, reverse=True):
        if cid.startswith(prefix) or cid == prefix:
            canon = _CANONICAL_UNITS[prefix]
            break

    if canon is None:
        # No canonical map → fall back to dominant-unit heuristic
        return _dominant_unit_records(recs)

    # Normalise units and filter
    accepted = []
    for r in recs:
        norm_unit = _normalize_unit(r.get("unit") or r.get("unit_raw") or "")
        if norm_unit in canon:
            accepted.append({**r, "_norm_unit": norm_unit})

    if not accepted:
        # Nothing passed the whitelist — fall back to dominant-unit
        return _dominant_unit_records(recs)

    # Among accepted, pick dominant normalised unit
    unit_counts = Counter(r["_norm_unit"] for r in accepted)
    dominant = unit_counts.most_common(1)[0][0]
    dominant_recs = [r for r in accepted if r["_norm_unit"] == dominant]
    # Apply physical price bounds last
    dominant_recs = _apply_price_bounds(dominant_recs, cid)
    if not dominant_recs:
        # Bounds eliminated everything — return original dominant without bounds
        dominant_recs = [r for r in accepted if r["_norm_unit"] == dominant]
    return dominant_recs, dominant


def print_trend_summary(records: list[dict], min_year_n: int = 3):
    """Print year-level median price for key commodities.

    Only years with at least min_year_n samples (after outlier removal) are shown.
    Groups per commodity are filtered to the dominant unit to avoid mixing
    barrel-priced and pound-priced records.
    """
    import math as _math
    key_commodities = [
        "FOOD-GRAIN-WHEAT",
        "FOOD-GRAIN-CORN",
        "FOOD-GRAIN-FLOUR",
        "FOOD-MEAT-PORK",
        "FOOD-DAIRY-BUTTER",
        "LABOR-DOMESTIC",
        "LABOR-AGRI",
    ]

    usable = [
        r for r in records
        if r.get("price_per_unit_usd") is not None
        and r.get("commodity_id") in key_commodities
        and r.get("year", 0) > 0
    ]

    if not usable:
        print("No records with USD prices found for trend summary.\n")
        return

    print("=" * 72)
    print("Year median price (USD/unit) for key commodities:")
    print(f"  (only years with n≥{min_year_n} samples after outlier removal shown)")
    print("=" * 72)

    for cid in key_commodities:
        cid_recs = [r for r in usable if r["commodity_id"] == cid]
        if not cid_recs:
            continue
        cid_recs, dominant_unit = _canonical_unit_records(cid_recs, cid)
        unit = dominant_unit or cid_recs[0].get("unit", "unit")
        commodity_name = cid_recs[0].get("category_l3", cid)

        data: dict[int, list] = defaultdict(list)
        for r in cid_recs:
            data[r["year"]].append(r["price_per_unit_usd"])

        all_flat = sorted(v for vals in data.values() for v in vals)
        cross_cleaned = _remove_outliers_log(all_flat)
        log_lo, log_hi = _cross_year_fence(cross_cleaned)
        cross_lo = max(min(cross_cleaned) if cross_cleaned else 0.0, log_lo)
        cross_hi = min(max(cross_cleaned) if cross_cleaned else float("inf"), log_hi)

        # Apply cross-year bounds — clip to within fence range
        # (do NOT skip wide-span series — they may be legitimately 200-year data)
        pos_clean = [v for v in cross_cleaned if cross_lo <= v <= cross_hi and v > 0]

        year_strs = []
        for year in sorted(data):
            clipped = [v for v in data[year] if cross_lo <= v <= cross_hi]
            prices = _remove_outliers(clipped) if clipped else []
            n = len(prices)
            if n < min_year_n:
                continue
            median = prices[n // 2]
            year_strs.append(f"{year}: ${median:.3f}(n={n})")

        if not year_strs:
            print(f"  {commodity_name} (/{unit}): no years with n≥{min_year_n}")
            print()
            continue

        print(f"  {commodity_name} (/{unit})")
        for chunk_start in range(0, len(year_strs), 6):
            print("    " + "  |  ".join(year_strs[chunk_start:chunk_start + 6]))
        print()


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------


def flag_outliers(
    records: list[dict],
    iqr_factor: float = 1.5,
    min_bucket_size: int = 5,
) -> set[str]:
    """Flag records whose price_per_unit_usd is a statistical outlier within their
    commodity_id + decade bucket.

    A record is flagged when its price_per_unit_usd lies outside
    [Q1 − iqr_factor×IQR, Q3 + iqr_factor×IQR] for the (commodity_id, decade) group.
    Buckets with fewer than min_bucket_size records are skipped (too little data to judge).

    Returns a set of 'ref' strings for all flagged records.
    """
    bucket: dict[tuple, list] = defaultdict(list)
    for r in records:
        cid = r.get("commodity_id")
        year = r.get("year", 0)
        price = r.get("price_per_unit_usd")
        ref = r.get("ref", "")
        if not cid or year <= 0 or price is None or price <= 0:
            continue
        decade = (year // 10) * 10
        bucket[(cid, decade)].append((ref, price))

    flagged_refs: set[str] = set()
    for (_cid, _decade), items in bucket.items():
        if len(items) < min_bucket_size:
            continue
        prices = sorted(v for _, v in items)
        n = len(prices)
        q1 = prices[n // 4]
        q3 = prices[3 * n // 4]
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo = q1 - iqr_factor * iqr
        hi = q3 + iqr_factor * iqr
        for ref, price in items:
            if price < lo or price > hi:
                flagged_refs.add(ref)

    return flagged_refs


def write_flagged_outliers(
    records: list[dict],
    flagged_refs: set[str],
    output_path: Path,
) -> int:
    """Write records whose 'ref' is in flagged_refs to output_path as JSONL.

    Each written record has 'flagged_outlier': true added.  Returns the count written.
    """
    ref_to_rec = {r.get("ref", ""): r for r in records if r.get("ref")}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for ref in sorted(flagged_refs):
            rec = ref_to_rec.get(ref)
            if rec is not None:
                out = {**rec, "flagged_outlier": True}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
    return written


# ---------------------------------------------------------------------------
# Colour palette helpers
# ---------------------------------------------------------------------------

L1_COLORS = {
    "Food & Agriculture":  "#4CAF50",
    "Labor & Services":    "#2196F3",
    "Real Estate":         "#FF9800",
    "Manufactured Goods":  "#9C27B0",
    "Raw Materials":       "#795548",
    "Transportation":      "#00BCD4",
    "Financial & Legal":   "#F44336",
    "Miscellaneous":       "#9E9E9E",
}


def _l1_color(l1: str) -> str:
    return L1_COLORS.get(l1, "#607D8B")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(
    records: list[dict],
    unresolved: list[dict],
    output_dir: Path,
    min_records: int = 5,
    min_year_n: int = 3,
):
    import math as _math
    if not _HAS_MATPLOTLIB:
        print("matplotlib not available — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # -----------------------------------------------------------------------
    # Precompute common aggregations
    # -----------------------------------------------------------------------
    all_years = sorted({r["year"] for r in records if r.get("year", 0) > 0})

    # -----------------------------------------------------------------------
    # Fig 01: Records per year
    # -----------------------------------------------------------------------
    year_counts = Counter(r["year"] for r in records if r.get("year", 0) > 0)
    if year_counts:
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.bar(list(year_counts.keys()), list(year_counts.values()), width=0.8, color="#2196F3")
        ax.set_title("Data coverage: records per year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Records")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        fig.tight_layout()
        fig.savefig(output_dir / "fig01_coverage_per_year.png", dpi=120)
        plt.close(fig)
        print("  Saved fig01_coverage_per_year.png")

    # -----------------------------------------------------------------------
    # Fig 02: L1 category donut
    # -----------------------------------------------------------------------
    l1_counts = Counter(r.get("category_l1", "Unknown") for r in records)
    if l1_counts:
        labels = [k for k, _ in l1_counts.most_common()]
        sizes = [l1_counts[k] for k in labels]
        colors = [_l1_color(k) for k in labels]
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.8,
            wedgeprops={"width": 0.5},
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title("Records by top-level category")
        fig.tight_layout()
        fig.savefig(output_dir / "fig02_category_distribution.png", dpi=120)
        plt.close(fig)
        print("  Saved fig02_category_distribution.png")

    # -----------------------------------------------------------------------
    # Fig 03: Top 25 commodities
    # -----------------------------------------------------------------------
    l3_counts = Counter(
        f"{r.get('category_l3','?')} ({r.get('commodity_id','?')})"
        for r in records
    )
    top25 = l3_counts.most_common(25)
    if top25:
        names, counts = zip(*top25)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(names)), counts, color="#4CAF50")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Records")
        ax.set_title("Top 25 commodities by record count")
        fig.tight_layout()
        fig.savefig(output_dir / "fig03_top_commodities.png", dpi=120)
        plt.close(fig)
        print("  Saved fig03_top_commodities.png")

    # -----------------------------------------------------------------------
    # Fig 04: Currency breakdown by era
    # -----------------------------------------------------------------------
    era_map = {
        "colonial (1770–1791)": (1770, 1791),
        "early republic (1792–1860)": (1792, 1860),
        "civil war & after (1861–1882)": (1861, 1882),
    }
    currency_buckets = {"USD/cents": [], "shilling/pence": [], "other colonial": [], "other": []}

    def _bucket_currency(cur: str) -> str:
        c = (cur or "").lower()
        if c in ("usd", "dollar", "cent", "cents", "$"):
            return "USD/cents"
        if c in ("shilling", "pence", "penny", "pound sterling", "s", "d"):
            return "shilling/pence"
        if c in ("old tenor", "lawful money", "continental", "livre", "real", "reale", "maravedi"):
            return "other colonial"
        return "other"

    era_currency: dict[str, Counter] = {era: Counter() for era in era_map}
    for r in records:
        year = r.get("year", 0)
        cur = r.get("currency_original", "")
        for era, (lo, hi) in era_map.items():
            if lo <= year <= hi:
                era_currency[era][_bucket_currency(cur)] += 1
                break

    eras = list(era_map.keys())
    buckets = list(currency_buckets.keys())
    bucket_colors = {
        "USD/cents": "#4CAF50", "shilling/pence": "#2196F3",
        "other colonial": "#FF9800", "other": "#9E9E9E",
    }
    if any(era_currency[e] for e in eras):
        fig, ax = plt.subplots(figsize=(10, 5))
        bottoms = [0] * len(eras)
        for bucket in buckets:
            heights = [era_currency[e].get(bucket, 0) for e in eras]
            if sum(heights) == 0:
                continue
            ax.bar(eras, heights, bottom=bottoms,
                   label=bucket, color=bucket_colors.get(bucket, "#607D8B"))
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax.set_title("Currency type breakdown by era")
        ax.set_xlabel("Era")
        ax.set_ylabel("Records")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / "fig04_currency_breakdown.png", dpi=120)
        plt.close(fig)
        print("  Saved fig04_currency_breakdown.png")

    # -----------------------------------------------------------------------
    # Figs 05–06: Long-term price trends for staples and labor
    # -----------------------------------------------------------------------
    def _trend_plot(commodity_ids: list[str], title: str, filename: str,
                    time_unit_filter: str | None = "any"):
        """Multi-line plot of annual median price per unit (USD) over time.

        Filters each commodity to its dominant unit, applies cross-year log-IQR
        outlier fence, and requires min_year_n samples per plotted year.
        """
        def _time_match(r):
            if time_unit_filter == "any":
                return True
            return r.get("time_unit") == time_unit_filter

        subset = [
            r for r in records
            if r.get("commodity_id") in commodity_ids
            and r.get("price_per_unit_usd") is not None
            and r.get("year", 0) > 0
            and _time_match(r)
        ]
        if not subset:
            print(f"  No USD price data for {filename}, skipping.")
            return

        data: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        name_map: dict[str, str] = {}
        unit_map: dict[str, str] = {}

        for cid in commodity_ids:
            cid_recs = [r for r in subset if r["commodity_id"] == cid]
            if not cid_recs:
                continue
            cid_recs, dominant_unit = _canonical_unit_records(cid_recs, cid)
            for r in cid_recs:
                data[cid][r["year"]].append(r["price_per_unit_usd"])
            if cid not in name_map:
                sample = cid_recs[0]
                name_map[cid] = sample.get("category_l3", cid)
                unit_map[cid] = dominant_unit or sample.get("unit", "unit")

        plot_data = {
            cid: years for cid, years in data.items()
            if sum(len(v) for v in years.values()) >= min_records
        }

        if not plot_data:
            print(f"  Insufficient data (< {min_records} records) for {filename}, skipping.")
            return

        cmap = plt.cm.get_cmap("tab10", len(plot_data))
        fig, ax = plt.subplots(figsize=(14, 6))

        for i, (cid, years) in enumerate(sorted(plot_data.items())):
            all_vals_flat = sorted(v for vals in years.values() for v in vals)
            cross_cleaned = _remove_outliers_log(all_vals_flat)
            log_lo, log_hi = _cross_year_fence(cross_cleaned)
            cross_lo = max(min(cross_cleaned) if cross_cleaned else 0.0, log_lo)
            cross_hi = min(max(cross_cleaned) if cross_cleaned else float("inf"), log_hi)

            # Skip only clearly corrupt data (> 2.5 log-decades even after cleaning)
            pos_clean = [v for v in cross_cleaned if cross_lo <= v <= cross_hi and v > 0]
            if len(pos_clean) >= 2:
                log_span = _math.log10(max(pos_clean)) - _math.log10(min(pos_clean))
                if log_span > 3.0:
                    continue

            sorted_years = []
            medians = []
            q1s = []
            q3s = []
            for year in sorted(years):
                clipped = [v for v in years[year] if cross_lo <= v <= cross_hi]
                if not clipped:
                    continue
                vals = _remove_outliers(clipped)
                if len(vals) < min_year_n:
                    continue
                n = len(vals)
                sorted_years.append(year)
                medians.append(vals[n // 2])
                q1s.append(vals[n // 4])
                q3s.append(vals[3 * n // 4])

            if not sorted_years:
                continue

            label = f"{name_map[cid]} (/{unit_map.get(cid, 'unit')})"
            color = cmap(i)
            ax.plot(sorted_years, medians, marker="o", markersize=3,
                    linewidth=1.2, label=label, color=color)
            ax.fill_between(sorted_years, q1s, q3s, alpha=0.12, color=color)

        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Median price (USD / standard unit)")
        ax.legend(loc="upper left", fontsize=8, ncol=1)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=120)
        plt.close(fig)
        print(f"  Saved {filename}")

    _trend_plot(
        ["FOOD-GRAIN-WHEAT", "FOOD-GRAIN-CORN", "FOOD-GRAIN-FLOUR",
         "FOOD-MEAT-PORK", "FOOD-DAIRY-BUTTER", "FOOD-SUGAR-SUGAR"],
        "Staple food prices over time (annual median, USD/dominant unit, n≥3/yr)",
        "fig05_staples_over_time.png",
    )

    _trend_plot(
        ["LABOR-DOMESTIC", "LABOR-AGRI", "LABOR-UNSKILLED",
         "LABOR-SKILLED-CARP", "LABOR-MIL-SOLDIER"],
        "Labor wages over time (annual median, USD/dominant unit, n≥3/yr)",
        "fig06_labor_over_time.png",
        time_unit_filter="any",
    )

    # -----------------------------------------------------------------------
    # Fig 07: Grid of top-N commodities with USD price data
    # -----------------------------------------------------------------------
    cid_usd_counts = Counter(
        r["commodity_id"] for r in records
        if r.get("price_per_unit_usd") is not None and r.get("year", 0) > 0
    )
    top_cids = [cid for cid, cnt in cid_usd_counts.most_common(16) if cnt >= min_records]

    if top_cids:
        n_plots = len(top_cids)
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(16, 4 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.35)

        for i, cid in enumerate(top_cids):
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])

            subset = [
                r for r in records
                if r.get("commodity_id") == cid
                and r.get("price_per_unit_usd") is not None
                and r.get("year", 0) > 0
            ]

            subset, dominant_unit = _canonical_unit_records(subset, cid)
            by_year_clean: dict[int, list] = defaultdict(list)
            for r in subset:
                by_year_clean[r["year"]].append(r["price_per_unit_usd"])

            all_flat = sorted(v for vals in by_year_clean.values() for v in vals)
            cross_cleaned = _remove_outliers_log(all_flat)
            log_lo, log_hi = _cross_year_fence(cross_cleaned)
            cross_lo = max(min(cross_cleaned) if cross_cleaned else 0.0, log_lo)
            cross_hi = min(max(cross_cleaned) if cross_cleaned else float("inf"), log_hi)

            sorted_years = []
            medians = []
            for y in sorted(by_year_clean):
                clipped = [v for v in by_year_clean[y] if cross_lo <= v <= cross_hi]
                if not clipped:
                    continue
                vals = _remove_outliers(clipped)
                if len(vals) < 3:
                    continue
                sorted_years.append(y)
                medians.append(vals[len(vals) // 2])

            name = subset[0].get("category_l3", cid) if subset else cid
            unit = dominant_unit or (subset[0].get("unit", "unit") if subset else "unit")
            l1 = subset[0].get("category_l1", "") if subset else ""
            color = _l1_color(l1)

            if not sorted_years:
                ax.set_title(f"{name}\n(no data after filtering)", fontsize=8)
                ax.tick_params(axis="both", labelsize=7)
                continue

            ax.fill_between(sorted_years, medians, alpha=0.3, color=color)
            ax.plot(sorted_years, medians, color=color, linewidth=1.5,
                    marker="o", markersize=2)
            ax.set_title(f"{name}\n(/{unit})", fontsize=8)
            ax.set_ylabel("USD", fontsize=7)
            ax.tick_params(axis="both", labelsize=7)
            total_n = sum(len(v) for v in by_year_clean.values())
            ax.set_xlabel(f"n={total_n}", fontsize=7)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(50))

        fig.suptitle("Price trends for top commodities (annual medians, USD/unit)", fontsize=11)
        fig.savefig(output_dir / "fig07_commodity_grid.png", dpi=120)
        plt.close(fig)
        print("  Saved fig07_commodity_grid.png")

    # -----------------------------------------------------------------------
    # Fig 08: Price volatility (IQR / median) by year for key commodities
    # -----------------------------------------------------------------------
    vol_commodities = [
        "FOOD-GRAIN-WHEAT", "FOOD-GRAIN-CORN", "FOOD-MEAT-PORK",
        "FOOD-DAIRY-BUTTER", "FOOD-GRAIN-FLOUR",
    ]
    vol_subset = [
        r for r in records
        if r.get("commodity_id") in vol_commodities
        and r.get("price_per_unit_usd") is not None
        and r.get("year", 0) > 0
    ]

    if vol_subset:
        vol_data: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        vol_names: dict[str, str] = {}
        for cid in vol_commodities:
            cid_recs = [r for r in vol_subset if r["commodity_id"] == cid]
            if not cid_recs:
                continue
            cid_recs, _ = _canonical_unit_records(cid_recs, cid)
            for r in cid_recs:
                vol_data[cid][r["year"]].append(r["price_per_unit_usd"])
                vol_names[cid] = r.get("category_l3", cid)

        cmap = plt.cm.get_cmap("Set2", len(vol_commodities))
        fig, ax = plt.subplots(figsize=(14, 5))

        for i, cid in enumerate(vol_commodities):
            if cid not in vol_data:
                continue
            all_flat = sorted(v for vals in vol_data[cid].values() for v in vals)
            cross_cleaned = _remove_outliers_log(all_flat)
            log_lo, log_hi = _cross_year_fence(cross_cleaned)
            cross_lo = max(min(cross_cleaned) if cross_cleaned else 0.0, log_lo)
            cross_hi = min(max(cross_cleaned) if cross_cleaned else float("inf"), log_hi)

            plot_years_valid = []
            cv_list = []
            for year in sorted(vol_data[cid]):
                clipped = [v for v in vol_data[cid][year] if cross_lo <= v <= cross_hi]
                vals = _remove_outliers(clipped) if clipped else []
                n = len(vals)
                if n < 3:
                    continue
                median = vals[n // 2]
                q1 = vals[n // 4]
                q3 = vals[3 * n // 4]
                iqr = q3 - q1
                cv = iqr / median if median > 0 else None
                if cv is not None:
                    plot_years_valid.append(year)
                    cv_list.append(cv)

            if plot_years_valid:
                ax.plot(plot_years_valid, cv_list, marker="o", markersize=3,
                        label=vol_names.get(cid, cid), color=cmap(i), linewidth=1.2)

        ax.set_title("Price volatility (IQR/median) by year for staple commodities")
        ax.set_xlabel("Year")
        ax.set_ylabel("IQR / median (relative volatility)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        fig.tight_layout()
        fig.savefig(output_dir / "fig08_decade_volatility.png", dpi=120)
        plt.close(fig)
        print("  Saved fig08_decade_volatility.png")

    # -----------------------------------------------------------------------
    # Fig 09: Confidence score distribution
    # -----------------------------------------------------------------------
    confidences = [r["confidence"] for r in records if "confidence" in r]
    if confidences:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(confidences, bins=20, range=(0, 1), color="#2196F3", edgecolor="white")
        ax.set_title("Distribution of normalization confidence scores")
        ax.set_xlabel("Confidence score")
        ax.set_ylabel("Records")
        ax.axvline(0.5, color="red", linestyle="--", label="Threshold (0.5)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "fig09_confidence_dist.png", dpi=120)
        plt.close(fig)
        print("  Saved fig09_confidence_dist.png")

    # -----------------------------------------------------------------------
    # Fig 10: Top unresolved item analysis
    # -----------------------------------------------------------------------
    if unresolved:
        def _item_cluster(item_raw: str) -> str:
            words = item_raw.lower().split()[:3]
            return " ".join(words)

        item_clusters = Counter(_item_cluster(r.get("item_raw", "?")) for r in unresolved)
        top_unresolved = item_clusters.most_common(20)

        if top_unresolved:
            names, counts = zip(*top_unresolved)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(range(len(names)), counts, color="#FF9800")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Count")
            ax.set_title(f"Top unresolved item types (total {_fmt_count(len(unresolved))} unresolved)")
            fig.tight_layout()
            fig.savefig(output_dir / "fig10_unresolved_analysis.png", dpi=120)
            plt.close(fig)
            print("  Saved fig10_unresolved_analysis.png")

    print()


def make_outlier_plot(
    records: list[dict],
    flagged_refs: set[str],
    output_dir: Path,
):
    """Fig 12 — outlier records per year (bar chart)."""
    if not _HAS_MATPLOTLIB or not flagged_refs:
        return
    year_counts: Counter = Counter()
    for r in records:
        if r.get("ref", "") in flagged_refs:
            year = r.get("year", 0)
            if year > 0:
                year_counts[year] += 1
    if not year_counts:
        return
    years = sorted(year_counts)
    counts = [year_counts[y] for y in years]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(years, counts, width=0.8, color="#F44336")
    ax.set_title(
        f"Statistical price outliers per year (total {len(flagged_refs):,} records, "
        "IQR×1.5 within commodity+decade bucket)"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Outlier records")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    fig.tight_layout()
    fig.savefig(output_dir / "fig12_outlier_map.png", dpi=120)
    plt.close(fig)
    print("  Saved fig12_outlier_map.png")


# ---------------------------------------------------------------------------
# Pandas-enhanced analysis (optional)
# ---------------------------------------------------------------------------

def pandas_analysis(records: list[dict], output_dir: Path):
    """Additional analysis using pandas if available."""
    if not _HAS_PANDAS or not _HAS_MATPLOTLIB:
        return

    usable = [r for r in records if r.get("price_per_unit_usd") is not None and r.get("year", 0) > 0]
    if not usable:
        return

    df = pd.DataFrame(usable)

    pivot = df[df["price_per_unit_usd"].notna()].groupby(
        ["year", "category_l2"]
    ).size().unstack(fill_value=0)

    if not pivot.empty and pivot.shape[0] > 2 and pivot.shape[1] > 2:
        fig, ax = plt.subplots(figsize=(14, max(6, pivot.shape[1] * 0.35)))
        im = ax.imshow(pivot.T, aspect="auto", cmap="YlGn", interpolation="nearest")
        ax.set_xticks(range(0, len(pivot.index), max(1, len(pivot.index) // 30)))
        ax.set_xticklabels(
            [pivot.index[i] for i in range(0, len(pivot.index), max(1, len(pivot.index) // 30))],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels(pivot.columns, fontsize=8)
        plt.colorbar(im, ax=ax, label="Records")
        ax.set_title("Data availability heatmap: commodity sub-category × year")
        ax.set_xlabel("Year")
        fig.tight_layout()
        fig.savefig(output_dir / "fig11_availability_heatmap.png", dpi=120)
        plt.close(fig)
        print("  Saved fig11_availability_heatmap.png")

    key_l2_cats = df.groupby("category_l2").size().nlargest(5).index.tolist()
    if key_l2_cats:
        print("=" * 72)
        print("Year × sub-category record counts (top 5 sub-categories):")
        print("=" * 72)
        for cat in key_l2_cats:
            sub = df[df["category_l2"] == cat]
            by_year = sub.groupby("year").size()
            years_str = "  ".join(f"{y}:{n}" for y, n in sorted(by_year.items()))
            print(f"  {cat:<30} {years_str}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize normalized historical price records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_prices.py
  python analyze_prices.py --year-start 1770 --year-end 1850
  python analyze_prices.py --min-records 20 --no-plots
        """,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./price_analysis_output"),
        help="Output directory for plots (default: ./price_analysis_output)",
    )
    parser.add_argument("--year-start", type=int, default=1770)
    parser.add_argument("--year-end", type=int, default=1963)
    parser.add_argument(
        "--min-records", type=int, default=5,
        help="Minimum total records per commodity for trend plots (default: 5)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Print statistics only; do not generate plot files",
    )
    parser.add_argument(
        "--no-write-flagged", dest="write_flagged", action="store_false",
        help=(
            "Disable writing statistical outlier records to "
            "data/pass2/prices/flagged_outliers.jsonl."
        ),
    )
    parser.set_defaults(write_flagged=True)
    parser.add_argument(
        "--outlier-iqr-factor", type=float, default=1.5, metavar="K",
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    parser.add_argument(
        "--outlier-min-bucket", type=int, default=5, metavar="N",
        help="Minimum records per commodity+decade bucket to attempt outlier flagging (default: 5)",
    )
    parser.add_argument(
        "--min-year-n", type=int, default=3, metavar="N",
        help="Minimum samples per year (after outlier removal) to include in trend plots/summary (default: 3)",
    )
    args = parser.parse_args()

    pass2_dir = args.data_dir / "pass2" / "prices"
    normalized_path = pass2_dir / "normalized.jsonl"
    unresolved_path = pass2_dir / "unresolved.jsonl"

    if not normalized_path.exists():
        print(
            f"ERROR: {normalized_path} not found.\n"
            "Run normalize_prices_pass2.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("Loading price data...")
    records = load_normalized(normalized_path, args.year_start, args.year_end)
    unresolved = load_unresolved(unresolved_path)
    print(f"  Loaded {len(records):,} normalized records, {len(unresolved):,} unresolved.")
    print()

    if not records:
        print("No records to analyze.")
        sys.exit(0)

    print_summary(records, unresolved)
    print_category_report(records)
    print_trend_summary(records, min_year_n=args.min_year_n)

    flagged_jsonl_path = pass2_dir / "flagged_outliers.jsonl"
    pending_outlier_refs: set[str] = set()
    if flagged_jsonl_path.exists():
        try:
            for line in flagged_jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    ref = r.get("ref", "")
                    if ref:
                        pending_outlier_refs.add(ref)
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass

    if pending_outlier_refs:
        print(
            f"Known pending outliers (from flagged_outliers.jsonl): "
            f"{len(pending_outlier_refs):,} records excluded from trend/volatility stats."
        )
        print()

    clean_records = [r for r in records if r.get("ref", "") not in pending_outlier_refs]

    flagged_refs = flag_outliers(
        clean_records,
        iqr_factor=args.outlier_iqr_factor,
        min_bucket_size=args.outlier_min_bucket,
    )
    print(f"Statistical outliers flagged: {len(flagged_refs):,} records "
          f"(IQR×{args.outlier_iqr_factor}, min bucket {args.outlier_min_bucket})")
    if flagged_refs:
        print(f"  ({100 * len(flagged_refs) / max(1, len(clean_records)):.1f}% of clean normalized records)")
    print()

    if args.write_flagged and flagged_refs:
        flagged_path = pass2_dir / "flagged_outliers.jsonl"
        all_flagged = flagged_refs | pending_outlier_refs
        n_written = write_flagged_outliers(records, all_flagged, flagged_path)
        print(f"Wrote {n_written:,} flagged outlier records → {flagged_path}")
        print("  Re-verification will run automatically on next: python normalize_prices_pass2.py")
        print()

    if _HAS_PANDAS:
        pandas_analysis(clean_records, args.output_dir)

    if not args.no_plots:
        if _HAS_MATPLOTLIB:
            print(f"Generating plots → {args.output_dir}/")
            make_plots(clean_records, unresolved, args.output_dir,
                       min_records=args.min_records, min_year_n=args.min_year_n)
            make_outlier_plot(records, flagged_refs | pending_outlier_refs, args.output_dir)
            print(f"All plots saved to {args.output_dir.resolve()}")
        else:
            print("matplotlib not installed — skipping plots.")
            print("Install with: pip install matplotlib")
    else:
        print("(Plots skipped — use --no-plots=False to generate them)")

    print("Done.")


if __name__ == "__main__":
    main()
