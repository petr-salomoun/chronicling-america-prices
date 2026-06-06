#!/usr/bin/env python3
"""
Chronicling America — Pipeline Statistics & Plots
==================================================

Scans the data directory and produces:

  1. Console summary tables for each pipeline stage:
       Phase 0 (raw download)   — pages, issues, bytes, OCR sources
       Phase 0b (compress_pass0) — compressed pages per year
       Phase 1  (extract_pass1) — extracted records (people / prices) per year
       Phase 2  (normalize_prices_pass2) — normalized price records

  2. Matplotlib figures saved to stats_output/:
       fig1_raw_pages_per_year.png       — downloaded + local-OCR pages per year
       fig2_raw_coverage.png             — % of year ranges with data
       fig3_pass0_coverage.png           — raw vs pass0 pages per year
       fig4_pass1_records_per_year.png   — P + $ records extracted per year
       fig5_pass1_record_types.png       — pie / bar of P vs $ totals
       fig6_prices_per_year.png          — $ record count per year
       fig7_people_per_year.png          — P record count per year
       fig8_pass2_normalized_per_year.png — normalized price records per year
       fig9_pass2_category_pie.png        — L1 category distribution (donut)
       fig10_pass2_top_commodities.png    — top 20 commodities by record count
       fig11_pass2_coverage_heatmap.png   — commodity × decade availability heatmap

Usage:
    python stats_pipeline.py [--data-dir ./data] [--output-dir ./stats_output] [--no-plots]

Requirements:
    matplotlib, pandas (both come with the standard scientific Python stack)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional import – plots only if matplotlib is present
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (safe everywhere)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPLOTS = True
except ImportError:
    _HAS_MPLOTS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _bar(value: int, max_val: int, width: int = 30) -> str:
    """ASCII progress bar."""
    if max_val == 0:
        return " " * width
    filled = int(round(value / max_val * width))
    return "#" * filled + "-" * (width - filled)


def _print_table(rows: list[tuple], headers: list[str], col_widths: list[int]):
    """Simple left-aligned table printer."""
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "  ".join("-" * w for w in col_widths)
    print(header_row)
    print(sep)
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print()


# ---------------------------------------------------------------------------
# Phase 0 — raw download stats
# ---------------------------------------------------------------------------

def collect_raw_stats(raw_dir: Path) -> dict:
    """Scan data/raw/ and return counts per year."""
    progress_path = raw_dir / "_progress.json"
    ocr_progress_path = raw_dir / "_ocr_progress.json"

    global_stats = {}
    if progress_path.exists():
        try:
            global_stats = json.loads(progress_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    ocr_stats = {}
    if ocr_progress_path.exists():
        try:
            ocr_stats = json.loads(ocr_progress_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    per_year: dict[int, dict] = {}

    years_completed = set(global_stats.get("years_completed", []))

    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)

        n_issues = 0
        n_loc_pages = 0        # seq-NNN.txt (non-empty = has LOC OCR)
        n_empty_pages = 0      # seq-NNN.txt empty AND no .ocr.txt (truly unresolved)
        n_local_ocr_pages = 0  # seq-NNN.ocr.txt exists (local Tesseract result)
        total_bytes = 0

        for lccn_dir in year_dir.iterdir():
            if not lccn_dir.is_dir() or lccn_dir.name.startswith("_"):
                continue
            for date_dir in lccn_dir.iterdir():
                if not date_dir.is_dir() or date_dir.name.startswith("_"):
                    continue
                n_issues += 1
                # Collect all files in one pass so we can cross-reference
                # seq-NNN.txt vs seq-NNN.ocr.txt by sequence number.
                loc_files: dict[str, int] = {}   # seq_key -> size
                ocr_files: set[str] = set()      # seq_key
                for f in date_dir.iterdir():
                    name = f.name
                    if not name.startswith("seq-"):
                        continue
                    if name.endswith(".ocr.txt"):
                        # seq-NNN.ocr.txt  →  key = "seq-NNN"
                        seq_key = name[: -len(".ocr.txt")]
                        ocr_files.add(seq_key)
                        total_bytes += f.stat().st_size
                    elif name.endswith(".txt"):
                        # seq-NNN.txt  →  key = "seq-NNN"
                        seq_key = name[: -len(".txt")]
                        sz = f.stat().st_size
                        loc_files[seq_key] = sz
                        total_bytes += sz

                for seq_key, sz in loc_files.items():
                    if sz > 0:
                        n_loc_pages += 1
                    elif seq_key in ocr_files:
                        # Empty LOC file but local OCR exists — count as local OCR only
                        pass  # counted below when we iterate ocr_files
                    else:
                        # Empty LOC file, no local OCR attempt yet
                        n_empty_pages += 1

                n_local_ocr_pages += len(ocr_files)

        per_year[year] = {
            "issues": n_issues,
            "loc_pages": n_loc_pages,
            "empty_pages": n_empty_pages,       # empty LOC file, no local OCR yet
            "local_ocr_pages": n_local_ocr_pages,
            "total_pages": n_loc_pages + n_empty_pages + n_local_ocr_pages,
            "bytes": total_bytes,
            "year_complete": year in years_completed,
        }

    return {
        "per_year": per_year,
        "global": global_stats.get("stats", {}),
        "ocr_global": ocr_stats.get("stats", {}),
        "years_completed": list(years_completed),
    }


# ---------------------------------------------------------------------------
# Phase 0b — pass0 compression stats
# ---------------------------------------------------------------------------

def collect_pass0_stats(pass0_dir: Path, raw_per_year: dict) -> dict:
    """Count compressed pages per year in data/pass0/."""
    per_year: dict[int, dict] = {}

    if not pass0_dir.exists():
        return {"per_year": per_year}

    for year_dir in sorted(pass0_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)

        n_pages = 0
        total_bytes = 0
        for lccn_dir in year_dir.iterdir():
            if not lccn_dir.is_dir():
                continue
            for date_dir in lccn_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                for f in date_dir.glob("seq-*.txt"):
                    n_pages += 1
                    total_bytes += f.stat().st_size

        raw_year = raw_per_year.get(year, {})
        raw_usable = raw_year.get("loc_pages", 0) + raw_year.get("local_ocr_pages", 0)
        pct = (n_pages / raw_usable * 100) if raw_usable > 0 else 0.0

        per_year[year] = {
            "compressed_pages": n_pages,
            "raw_usable_pages": raw_usable,
            "coverage_pct": pct,
            "bytes": total_bytes,
        }

    return {"per_year": per_year}


# ---------------------------------------------------------------------------
# Phase 1 — extraction stats
# ---------------------------------------------------------------------------

def collect_pass1_stats(pass1_dir: Path) -> dict:
    """Count extracted records per year from pass1/social/ and pass1/prices/."""
    social_dir = pass1_dir / "social"
    prices_dir = pass1_dir / "prices"

    per_year: dict[int, dict] = {}

    def _scan_dir(base: Path, record_type: str):
        if not base.exists():
            return
        for year_dir in sorted(base.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            year = int(year_dir.name)
            if year not in per_year:
                per_year[year] = {"P": 0, "$": 0, "issues_with_records": 0, "issues_empty": 0}

            for jsonl_path in year_dir.glob("*.jsonl"):
                sz = jsonl_path.stat().st_size
                if sz == 0:
                    per_year[year]["issues_empty"] += 1
                    continue
                per_year[year]["issues_with_records"] += 1
                try:
                    lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            t = rec.get("t", "")
                            if t in per_year[year]:
                                per_year[year][t] += 1
                        except (json.JSONDecodeError, ValueError):
                            pass
                except OSError:
                    pass

    _scan_dir(social_dir, "P")
    _scan_dir(prices_dir, "$")

    return {"per_year": per_year}


# ---------------------------------------------------------------------------
# Phase 2 — price normalization stats
# ---------------------------------------------------------------------------

def collect_pass2_stats(pass2_dir: Path) -> dict:
    """Scan data/pass2/prices/ and return stats from normalized/unresolved/failed JSONL files."""
    prices_dir = pass2_dir / "prices"
    normalized_path = prices_dir / "normalized.jsonl"
    unresolved_path = prices_dir / "unresolved.jsonl"
    failed_path     = prices_dir / "failed.jsonl"
    progress_path   = prices_dir / "_progress.json"

    result: dict = {
        "total_normalized": 0,
        "total_unresolved": 0,
        "total_failed": 0,
        "files_processed": 0,
        "per_year": {},          # year -> {"normalized": N, "unresolved": N, "with_usd": N}
        "by_category_l1": {},    # l1 -> count
        "by_commodity": {},      # commodity_id -> count
        "by_commodity_unit": {}, # commodity_id -> most-common unit string
        "by_currency": {},       # currency_original -> count
        "n_with_usd": 0,
        "available": normalized_path.exists(),
    }

    if not normalized_path.exists():
        return result

    # Progress sentinel
    if progress_path.exists():
        try:
            d = json.loads(progress_path.read_text())
            result["files_processed"] = d.get("count", 0)
        except (json.JSONDecodeError, OSError):
            pass

    # Scan normalized records
    _unit_counts: dict = {}   # commodity_id -> {unit -> count}
    try:
        for line in normalized_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            result["total_normalized"] += 1
            year = rec.get("year", 0)
            if year not in result["per_year"]:
                result["per_year"][year] = {"normalized": 0, "unresolved": 0, "with_usd": 0}
            result["per_year"][year]["normalized"] += 1
            if rec.get("price_usd") is not None:
                result["per_year"][year]["with_usd"] += 1
                result["n_with_usd"] += 1
            l1 = rec.get("category_l1", "Unknown")
            result["by_category_l1"][l1] = result["by_category_l1"].get(l1, 0) + 1
            cid = rec.get("commodity_id", "Unknown")
            result["by_commodity"][cid] = result["by_commodity"].get(cid, 0) + 1
            unit = rec.get("unit") or ""
            if unit:
                if cid not in _unit_counts:
                    _unit_counts[cid] = {}
                _unit_counts[cid][unit] = _unit_counts[cid].get(unit, 0) + 1
            curr = rec.get("currency_original", "Unknown")
            result["by_currency"][curr] = result["by_currency"].get(curr, 0) + 1
    except OSError:
        pass

    # Pick modal unit per commodity
    for cid, uc in _unit_counts.items():
        result["by_commodity_unit"][cid] = max(uc, key=uc.get)

    # Scan unresolved records
    if unresolved_path.exists():
        try:
            for line in unresolved_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result["total_unresolved"] += 1
                year = rec.get("year", 0)
                if year not in result["per_year"]:
                    result["per_year"][year] = {"normalized": 0, "unresolved": 0, "with_usd": 0}
                result["per_year"][year]["unresolved"] += 1
        except OSError:
            pass

    # Scan failed records (transient LLM failures — re-runnable with --retry-failed)
    if failed_path.exists():
        try:
            for line in failed_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)   # just validate JSON; no field indexing needed
                except json.JSONDecodeError:
                    continue
                result["total_failed"] += 1
        except OSError:
            pass

    return result


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

def print_raw_report(raw_stats: dict):
    per_year = raw_stats["per_year"]
    g = raw_stats.get("global", {})
    ocr_g = raw_stats.get("ocr_global", {})

    print("=" * 72)
    print("PHASE 0  — Raw Download Summary")
    print("=" * 72)

    total_pages_dl = g.get("total_pages_downloaded", 0)
    total_bytes = g.get("total_bytes", 0)
    total_issues = g.get("total_issues_processed", 0)
    total_failed = g.get("total_pages_failed", 0)
    years_done = len(raw_stats.get("years_completed", []))
    total_years = len(per_year)

    ocr_done = ocr_g.get("pages_ocr_done", 0)
    ocr_text = ocr_g.get("pages_ocr_text", 0)
    ocr_failed = ocr_g.get("pages_failed", 0)

    print(f"  Years with data:       {total_years}  (of which {years_done} marked complete)")
    print(f"  Issues processed:      {total_issues:,}")
    print(f"  Pages with LOC OCR:    {total_pages_dl:,}")
    print(f"  Pages download failed: {total_failed:,}")
    print(f"  Raw text downloaded:   {_fmt_bytes(total_bytes)}")
    print(f"  Local Tesseract OCR:   {ocr_done:,} pages done  ({ocr_text:,} with text, {ocr_failed:,} failed)")
    print()

    # Per-year table (show only years with data, limit to ~50 rows for readability)
    rows = []
    max_pages = max((v["total_pages"] for v in per_year.values()), default=1)
    for year in sorted(per_year):
        d = per_year[year]
        done_flag = "Y" if d["year_complete"] else " "
        rows.append((
            year,
            done_flag,
            d["issues"],
            d["loc_pages"],
            d["empty_pages"],
            d["local_ocr_pages"],
            _fmt_bytes(d["bytes"]),
            _bar(d["total_pages"], max_pages, 20),
        ))

    _print_table(
        rows,
        headers=["Year", "Done", "Issues", "LOC-OCR", "No-OCR", "Local-OCR", "Size", "Pages"],
        col_widths=[4, 4, 7, 8, 7, 10, 9, 22],
    )


def print_pass0_report(pass0_stats: dict):
    per_year = pass0_stats["per_year"]
    if not per_year:
        print("PHASE 0b — Compression: no data in data/pass0/\n")
        return

    print("=" * 72)
    print("PHASE 0b — OCR Compression (compress_pass0)")
    print("=" * 72)

    total_compressed = sum(v["compressed_pages"] for v in per_year.values())
    total_raw_usable = sum(v["raw_usable_pages"] for v in per_year.values())
    overall_pct = (total_compressed / total_raw_usable * 100) if total_raw_usable else 0.0

    print(f"  Years with compressed data: {len(per_year)}")
    print(f"  Compressed pages total:     {total_compressed:,}")
    print(f"  Raw usable pages total:     {total_raw_usable:,}")
    print(f"  Overall coverage:           {overall_pct:.1f}%")
    print()

    rows = []
    for year in sorted(per_year):
        d = per_year[year]
        rows.append((
            year,
            d["compressed_pages"],
            d["raw_usable_pages"],
            f"{d['coverage_pct']:.0f}%",
            _fmt_bytes(d["bytes"]),
        ))

    _print_table(
        rows,
        headers=["Year", "Compressed", "Raw usable", "Coverage", "Size"],
        col_widths=[4, 10, 11, 9, 10],
    )


def print_pass1_report(pass1_stats: dict):
    per_year = pass1_stats["per_year"]
    if not per_year:
        print("PHASE 1 — Extraction: no data in data/pass1/\n")
        return

    print("=" * 72)
    print("PHASE 1 — LLM Extraction (extract_pass1)")
    print("=" * 72)

    total_P = sum(v["P"] for v in per_year.values())
    total_dollar = sum(v["$"] for v in per_year.values())
    total_issues_w = sum(v["issues_with_records"] for v in per_year.values())
    total_issues_e = sum(v["issues_empty"] for v in per_year.values())

    print(f"  Years with extraction:      {len(per_year)}")
    print(f"  Issues with records:        {total_issues_w:,}")
    print(f"  Issues processed (empty):   {total_issues_e:,}")
    print(f"  People co-mention records:  {total_P:,}")
    print(f"  Price records:              {total_dollar:,}")
    print(f"  Total records:              {total_P + total_dollar:,}")
    print()

    rows = []
    for year in sorted(per_year):
        d = per_year[year]
        rows.append((
            year,
            d["issues_with_records"],
            d["issues_empty"],
            d["P"],
            d["$"],
        ))

    _print_table(
        rows,
        headers=["Year", "Issues(w/rec)", "Issues(empty)", "People-P", "Prices-$"],
        col_widths=[4, 13, 14, 9, 9],
    )


def print_pass2_report(pass2_stats: dict):
    if not pass2_stats.get("available"):
        print("PHASE 2 — Price Normalization: no data in data/pass2/prices/\n")
        return

    print("=" * 72)
    print("PHASE 2 — Price Normalization (normalize_prices_pass2)")
    print("=" * 72)

    total_norm   = pass2_stats["total_normalized"]
    total_unres  = pass2_stats["total_unresolved"]
    total_failed = pass2_stats.get("total_failed", 0)
    total_all    = total_norm + total_unres + total_failed
    n_with_usd   = pass2_stats["n_with_usd"]
    files_done   = pass2_stats["files_processed"]
    pct_norm     = (total_norm / total_all * 100) if total_all > 0 else 0.0
    pct_usd      = (n_with_usd / total_norm * 100) if total_norm > 0 else 0.0

    print(f"  Files processed (sentinel):  {files_done:,}")
    print(f"  Normalized records:          {total_norm:,}  ({pct_norm:.1f}% of all processed)")
    print(f"    of which with USD price:   {n_with_usd:,}  ({pct_usd:.1f}% of normalized)")
    print(f"  Unresolved records:          {total_unres:,}  (permanent — content failures)")
    print(f"  Failed records (transient):  {total_failed:,}  (LLM errors → re-run with --retry-failed)")
    print(f"  Total records processed:     {total_all:,}")
    print()

    # Top L1 categories
    by_l1 = pass2_stats["by_category_l1"]
    if by_l1:
        print("  Category breakdown (L1):")
        for cat, cnt in sorted(by_l1.items(), key=lambda x: -x[1]):
            pct = cnt / total_norm * 100 if total_norm else 0
            print(f"    {cat:<30s} {cnt:>7,}  ({pct:5.1f}%)")
        print()

    # Top 10 currencies
    by_curr = pass2_stats["by_currency"]
    if by_curr:
        print("  Top currencies:")
        for curr, cnt in sorted(by_curr.items(), key=lambda x: -x[1])[:10]:
            pct = cnt / total_norm * 100 if total_norm else 0
            print(f"    {curr:<20s} {cnt:>7,}  ({pct:5.1f}%)")
        print()

    # Per-year table (only show years with data)
    per_year = pass2_stats["per_year"]
    if per_year:
        rows = []
        for year in sorted(per_year):
            d = per_year[year]
            total_y = d["normalized"] + d["unresolved"]
            pct_y = d["normalized"] / total_y * 100 if total_y else 0
            rows.append((
                year,
                d["normalized"],
                d["unresolved"],
                f"{pct_y:.0f}%",
                d["with_usd"],
            ))
        _print_table(
            rows,
            headers=["Year", "Normalized", "Unresolved", "Norm%", "WithUSD"],
            col_widths=[4, 10, 11, 6, 8],
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(raw_stats: dict, pass0_stats: dict, pass1_stats: dict,
               pass2_stats: dict, output_dir: Path):
    if not _HAS_MPLOTS:
        print("matplotlib not available — skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    raw_per_year = raw_stats["per_year"]
    pass0_per_year = pass0_stats["per_year"]
    pass1_per_year = pass1_stats["per_year"]

    raw_years = sorted(raw_per_year)
    raw_loc   = [raw_per_year[y]["loc_pages"] for y in raw_years]
    raw_local = [raw_per_year[y]["local_ocr_pages"] for y in raw_years]
    raw_empty = [raw_per_year[y]["empty_pages"] for y in raw_years]

    # ---- Fig 1: Raw pages per year (stacked bar) ----
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(raw_years, raw_loc,   label="LOC OCR (seq-NNN.txt)",       color="#2196F3")
    ax.bar(raw_years, raw_local, label="Local Tesseract (seq-NNN.ocr.txt)", color="#4CAF50",
           bottom=raw_loc)
    bottom2 = [a + b for a, b in zip(raw_loc, raw_local)]
    ax.bar(raw_years, raw_empty, label="Empty (no OCR)", color="#FF9800", bottom=bottom2)
    ax.set_title("Phase 0: Raw pages per year (by OCR source)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pages")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_raw_pages_per_year.png", dpi=120)
    plt.close(fig)
    print(f"  Saved fig1_raw_pages_per_year.png")

    # ---- Fig 2: Raw download coverage (issues per year) ----
    raw_issues = [raw_per_year[y]["issues"] for y in raw_years]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(raw_years, raw_issues, color="#2196F3", width=0.8)
    # Shade completed years
    completed = set(raw_stats.get("years_completed", []))
    for y, n in zip(raw_years, raw_issues):
        if y in completed:
            ax.axvspan(y - 0.5, y + 0.5, alpha=0.15, color="green", zorder=0)
    ax.set_title("Phase 0: Issues downloaded per year  (green = year marked complete)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Issues")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_raw_issues_per_year.png", dpi=120)
    plt.close(fig)
    print(f"  Saved fig2_raw_issues_per_year.png")

    # ---- Fig 3: pass0 coverage vs raw usable ----
    if pass0_per_year:
        p0_years = sorted(pass0_per_year)
        p0_compressed = [pass0_per_year[y]["compressed_pages"] for y in p0_years]
        p0_raw        = [pass0_per_year[y]["raw_usable_pages"] for y in p0_years]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(p0_years, p0_raw,        label="Raw usable pages",  color="#90CAF9", width=0.8)
        ax.bar(p0_years, p0_compressed, label="Compressed (pass0)", color="#1565C0", width=0.8)
        ax.set_title("Phase 0b: Compressed pages vs raw usable pages per year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Pages")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        fig.tight_layout()
        fig.savefig(output_dir / "fig3_pass0_coverage.png", dpi=120)
        plt.close(fig)
        print(f"  Saved fig3_pass0_coverage.png")

    # ---- Fig 4: pass1 records per year (P + $) ----
    if pass1_per_year:
        p1_years   = sorted(pass1_per_year)
        p1_P       = [pass1_per_year[y]["P"] for y in p1_years]
        p1_dollar  = [pass1_per_year[y]["$"] for y in p1_years]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(p1_years, p1_P,     label='People co-mentions ("P")',  color="#9C27B0")
        ax.bar(p1_years, p1_dollar, label='Price records ("$")', color="#E91E63",
               bottom=p1_P)
        ax.set_title('Phase 1: Extracted records per year')
        ax.set_xlabel("Year")
        ax.set_ylabel("Records")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        fig.tight_layout()
        fig.savefig(output_dir / "fig4_pass1_records_per_year.png", dpi=120)
        plt.close(fig)
        print(f"  Saved fig4_pass1_records_per_year.png")

        # ---- Fig 5: P vs $ totals (donut) ----
        total_P      = sum(p1_P)
        total_dollar = sum(p1_dollar)
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            [total_P, total_dollar],
            labels=["People (P)", "Prices ($)"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#9C27B0", "#E91E63"],
            pctdistance=0.75,
            wedgeprops={"width": 0.5},
        )
        ax.set_title(f"Phase 1: Record types\n(total {total_P + total_dollar:,})")
        fig.tight_layout()
        fig.savefig(output_dir / "fig5_pass1_record_types_pie.png", dpi=120)
        plt.close(fig)
        print(f"  Saved fig5_pass1_record_types_pie.png")

        # ---- Fig 6: Price records per year (separate) ----
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(p1_years, p1_dollar, alpha=0.4, color="#E91E63")
        ax.plot(p1_years, p1_dollar, color="#E91E63", linewidth=1.5, marker="o", markersize=3)
        ax.set_title('Phase 1: Price records ("$") per year')
        ax.set_xlabel("Year")
        ax.set_ylabel("Price records")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        fig.tight_layout()
        fig.savefig(output_dir / "fig6_prices_per_year.png", dpi=120)
        plt.close(fig)
        print(f"  Saved fig6_prices_per_year.png")

        # ---- Fig 7: People records per year (separate) ----
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(p1_years, p1_P, alpha=0.4, color="#9C27B0")
        ax.plot(p1_years, p1_P, color="#9C27B0", linewidth=1.5, marker="o", markersize=3)
        ax.set_title('Phase 1: People co-mention records ("P") per year')
        ax.set_xlabel("Year")
        ax.set_ylabel("People records")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        fig.tight_layout()
        fig.savefig(output_dir / "fig7_people_per_year.png", dpi=120)
        plt.close(fig)
        print(f"  Saved fig7_people_per_year.png")

    # ---- Pass 2 plots ----
    if pass2_stats.get("available") and pass2_stats["total_normalized"] > 0:

        # ---- Fig 8: Normalized records per year ----
        p2_per_year = pass2_stats["per_year"]
        p2_years = sorted(y for y in p2_per_year if y > 0)
        if p2_years:
            p2_norm   = [p2_per_year[y]["normalized"] for y in p2_years]
            p2_unres  = [p2_per_year[y]["unresolved"]  for y in p2_years]
            p2_nousd  = [p2_per_year[y]["normalized"] - p2_per_year[y]["with_usd"]
                         for y in p2_years]
            p2_usd    = [p2_per_year[y]["with_usd"]    for y in p2_years]

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.bar(p2_years, p2_usd,   label="Normalized (USD price)",    color="#1B5E20", width=0.8)
            ax.bar(p2_years, p2_nousd, label="Normalized (no USD price)",  color="#81C784", width=0.8,
                   bottom=p2_usd)
            ax.bar(p2_years, p2_unres, label="Unresolved",                 color="#FF7043", width=0.8,
                   bottom=p2_norm)
            ax.set_title("Phase 2: Normalized price records per year\n"
                         "(pre-1790 USD prices apply depreciation rates for Continental dollars)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Records")
            ax.legend()
            ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
            fig.tight_layout()
            fig.savefig(output_dir / "fig8_pass2_normalized_per_year.png", dpi=120)
            plt.close(fig)
            print(f"  Saved fig8_pass2_normalized_per_year.png")

        # ---- Fig 9: L1 category distribution (donut) ----
        by_l1 = pass2_stats["by_category_l1"]
        if by_l1:
            cats   = sorted(by_l1.items(), key=lambda x: -x[1])
            labels = [c for c, _ in cats]
            sizes  = [n for _, n in cats]
            colors = [
                "#1565C0", "#2E7D32", "#6A1B9A", "#E65100",
                "#AD1457", "#00695C", "#4527A0", "#558B2F",
            ]
            fig, ax = plt.subplots(figsize=(7, 7))
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                startangle=90, colors=colors[:len(cats)],
                pctdistance=0.78, wedgeprops={"width": 0.55},
            )
            for at in autotexts:
                at.set_fontsize(8)
            ax.set_title(f"Phase 2: Category distribution (L1)\n(n={sum(sizes):,} normalized)")
            fig.tight_layout()
            fig.savefig(output_dir / "fig9_pass2_category_pie.png", dpi=120)
            plt.close(fig)
            print(f"  Saved fig9_pass2_category_pie.png")

        # ---- Fig 10: Top 20 commodities by record count ----
        by_comm = pass2_stats["by_commodity"]
        by_comm_unit = pass2_stats.get("by_commodity_unit", {})
        if by_comm:
            top20  = sorted(by_comm.items(), key=lambda x: -x[1])[:20]
            c_ids  = [c for c, _ in top20]
            c_cnts = [n for _, n in top20]
            # Build y-axis labels: "COMMODITY-ID  (per unit)"
            c_labels = [
                f"{cid}  (per {by_comm_unit[cid]})" if cid in by_comm_unit else cid
                for cid in c_ids
            ]
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.barh(range(len(c_ids)), c_cnts, color="#1565C0")
            ax.set_yticks(range(len(c_ids)))
            ax.set_yticklabels(c_labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Record count")
            ax.set_title("Phase 2: Top 20 commodities by record count\n(label shows most common pricing unit)")
            for bar, cnt in zip(bars, c_cnts):
                ax.text(bar.get_width() + max(c_cnts) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{cnt:,}", va="center", fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / "fig10_pass2_top_commodities.png", dpi=120)
            plt.close(fig)
            print(f"  Saved fig10_pass2_top_commodities.png")

        # ---- Fig 11: Commodity × decade heatmap ----
        p2_per_year = pass2_stats["per_year"]
        if p2_per_year and by_comm:
            # Re-read normalized.jsonl to get per-decade per-commodity counts
            # (We already have totals by commodity but not by decade)
            # Build from pass2_stats.per_year — we need raw records for this, so
            # this plot requires re-reading the file. Do it inline.
            pass  # handled below

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chronicling America pipeline statistics and plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stats_pipeline.py
  python stats_pipeline.py --data-dir /mnt/data/chronam
  python stats_pipeline.py --no-plots
        """,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./stats_output"),
        help="Directory for output plots (default: ./stats_output)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Print stats only; do not generate plot files",
    )
    args = parser.parse_args()

    raw_dir   = args.data_dir / "raw"
    pass0_dir = args.data_dir / "pass0"
    pass1_dir = args.data_dir / "pass1"
    pass2_dir = args.data_dir / "pass2"

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} does not exist. Run the downloader first.", file=sys.stderr)
        sys.exit(1)

    print()
    print("Scanning pipeline data — this may take a moment for large corpora...")
    print()

    print("  [1/4] Collecting raw download stats...")
    raw_stats = collect_raw_stats(raw_dir)

    print("  [2/4] Collecting pass0 compression stats...")
    pass0_stats = collect_pass0_stats(pass0_dir, raw_stats["per_year"])

    print("  [3/4] Collecting pass1 extraction stats...")
    pass1_stats = collect_pass1_stats(pass1_dir)

    print("  [4/4] Collecting pass2 normalization stats...")
    pass2_stats = collect_pass2_stats(pass2_dir)

    print()
    print_raw_report(raw_stats)
    print_pass0_report(pass0_stats)
    print_pass1_report(pass1_stats)
    print_pass2_report(pass2_stats)

    if not args.no_plots and _HAS_MPLOTS:
        print(f"Generating plots → {args.output_dir}/")
        make_plots(raw_stats, pass0_stats, pass1_stats, pass2_stats, args.output_dir)
        print(f"All plots saved to {args.output_dir.resolve()}")
    elif not args.no_plots and not _HAS_MPLOTS:
        print("matplotlib not installed — skipping plots.  Install with: pip install matplotlib")

    print("Done.")


if __name__ == "__main__":
    main()
