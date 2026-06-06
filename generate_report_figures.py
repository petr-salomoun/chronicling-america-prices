#!/usr/bin/env python3
"""
Generate report figures for the full 1770-1963 Chronicling America price dataset.
Run AFTER merge_and_finalize.py has been executed.

Produces 9 focused figures in report_figures/:
  fig1  - Record volume by decade (full span)
  fig2  - Category distribution (donut)
  fig3  - Civil War + WWI + Great Depression flour/wheat inflation
  fig4  - Labor wages over time (farm, domestic, professional)
  fig5  - Food price trends: wheat, corn, butter (1800-1960)
  fig6  - Price volatility by category
  fig7  - Currency diversity over time
  fig8  - 20th-century commodity price index (vs 1880 baseline)
  fig9  - Wheat price dispersion (coefficient of variation by year)
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA = Path("data/pass2/prices/normalized.jsonl")
OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)

FULL_YEAR_RANGE = (1770, 1963)

print("Loading data...")
records = []
with DATA.open() as f:
    for line in f:
        r = json.loads(line)
        if FULL_YEAR_RANGE[0] <= r.get("year", 0) <= FULL_YEAR_RANGE[1]:
            records.append(r)
print(f"  {len(records):,} records in range {FULL_YEAR_RANGE[0]}-{FULL_YEAR_RANGE[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def dominant_unit(recs):
    from collections import Counter
    c = Counter(r.get("unit", "") for r in recs)
    return c.most_common(1)[0][0] if c else ""


def clean_median(vals):
    """IQR-cleaned median."""
    s = sorted(v for v in vals if v and v > 0)
    if not s:
        return None
    n = len(s)
    if n < 2:
        return s[0]
    q1, q3 = s[n // 4], s[3 * n // 4]
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        s = [v for v in s if lo <= v <= hi]
    return s[len(s) // 2] if s else None


def log_cross_fence(vals, factor=1.5):
    """Return (lo, hi) bounds from log-median ± factor decades."""
    pos = [v for v in vals if v and v > 0]
    if not pos:
        return 0, float("inf")
    log_med = sorted(math.log10(v) for v in pos)[len(pos) // 2]
    return 10 ** (log_med - factor), 10 ** (log_med + factor)


def yearly_medians(cid, min_n=3, unit_override=None):
    """Return (years, medians, unit) for a commodity, with cross-year fence."""
    recs = [r for r in records if r.get("commodity_id") == cid
            and r.get("price_per_unit_usd") and r["price_per_unit_usd"] > 0]
    if not recs:
        return [], [], ""
    unit = unit_override or dominant_unit(recs)
    recs = [r for r in recs if r.get("unit", "") == unit]
    by_year = defaultdict(list)
    for r in recs:
        by_year[r["year"]].append(r["price_per_unit_usd"])
    all_vals = [v for vals in by_year.values() for v in vals]
    lo, hi = log_cross_fence(all_vals)
    years, meds = [], []
    for y in sorted(by_year):
        clipped = [v for v in by_year[y] if lo <= v <= hi]
        m = clean_median(clipped)
        if m and len(clipped) >= min_n:
            years.append(y)
            meds.append(m)
    return years, meds, unit


WAR_SPANS = [
    (1861, 1865, "Civil War", "#e74c3c"),
    (1898, 1898, "Spanish-Am.", "#e67e22"),
    (1917, 1918, "WWI", "#c0392b"),
    (1929, 1933, "Depression", "#8e44ad"),
    (1939, 1945, "WWII", "#2c3e50"),
    (1950, 1953, "Korea", "#7f8c8d"),
]


def add_war_bands(ax, alpha=0.15):
    for y1, y2, label, color in WAR_SPANS:
        ax.axvspan(y1, y2 + 0.5, alpha=alpha, color=color)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Record volume by decade
# ─────────────────────────────────────────────────────────────────────────────
def fig1_volume():
    decade_counts = defaultdict(int)
    for r in records:
        decade_counts[(r["year"] // 10) * 10] += 1
    decades = sorted(decade_counts)
    counts = [decade_counts[d] for d in decades]
    labels = [str(d) + "s" for d in decades]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#3498db" if d < 1884 else "#2ecc71" for d in decades]
    ax.bar(decades, counts, width=8, color=colors, edgecolor="white", linewidth=0.4)
    add_war_bands(ax)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Price records")
    ax.set_title("Chronicling America Price Records by Decade, 1770–1963", fontsize=13, fontweight="bold")
    ax.set_xticks(decades)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    blue = mpatches.Patch(color="#3498db", label="1770–1883 (original run)")
    green = mpatches.Patch(color="#2ecc71", label="1884–1963 (recovered)")
    ax.legend(handles=[blue, green], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_volume_by_decade.png", dpi=150)
    plt.close()
    print(f"  fig1: {sum(counts):,} total records, {len(decades)} decades")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Category distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig2_categories():
    from collections import Counter
    cats = Counter(r.get("category_l1", "Unknown") for r in records)
    labels = sorted(cats, key=cats.get, reverse=True)
    sizes = [cats[l] for l in labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        pctdistance=0.82, startangle=90, textprops={"fontsize": 9},
    )
    ax.add_artist(plt.Circle((0, 0), 0.52, fc="white"))
    ax.set_title("Price Records by Economic Category\n(full 1770–1963 dataset)", fontsize=13, fontweight="bold")
    ax.text(0, 0, f"{len(records):,}\nrecords", ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_category_distribution.png", dpi=150)
    plt.close()
    print(f"  fig2: {len(labels)} categories")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Flour price — all wars visible
# ─────────────────────────────────────────────────────────────────────────────
def fig3_flour_wars():
    years, meds, unit = yearly_medians("FOOD-GRAIN-FLOUR", min_n=2)
    if not years:
        print("  fig3: no flour data"); return

    fig, ax = plt.subplots(figsize=(14, 5))
    add_war_bands(ax)
    ax.plot(years, meds, "o-", color="#2c3e50", markersize=4, linewidth=1.5, zorder=5)
    # Annotate key events
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Median price (USD / {unit})")
    ax.set_title("Flour Prices in American Newspapers, 1800–1963", fontsize=13, fontweight="bold")
    ax.set_xlim(1800, 1965)

    # Legend for war bands
    patches = [mpatches.Patch(color=c, alpha=0.4, label=f"{lbl} ({y1}–{y2})")
               for y1, y2, lbl, c in WAR_SPANS if y1 >= 1800]
    ax.legend(handles=patches, fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_flour_all_wars.png", dpi=150)
    plt.close()
    print(f"  fig3: {len(years)} flour year-points")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Food staples long-run (wheat, butter, sugar)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_food_staples():
    commodities = [
        ("FOOD-GRAIN-WHEAT",   "Wheat",  "#e67e22"),
        ("FOOD-DAIRY-BUTTER",  "Butter", "#f1c40f"),
        ("FOOD-SUGAR-SUGAR",   "Sugar",  "#e74c3c"),
        ("FOOD-GRAIN-CORN",    "Corn",   "#27ae60"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    add_war_bands(ax, alpha=0.1)
    any_plotted = False
    for cid, label, color in commodities:
        yrs, meds, unit = yearly_medians(cid, min_n=2)
        if len(yrs) >= 3:
            ax.plot(yrs, meds, "o-", color=color, markersize=3, linewidth=1.3, label=f"{label} (/{unit})")
            any_plotted = True
    if not any_plotted:
        print("  fig4: no food data"); plt.close(); return
    ax.set_xlabel("Year")
    ax.set_ylabel("Median price (USD/unit)")
    ax.set_title("Food Staple Prices, 1770–1963", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
    fig.tight_layout()
    fig.savefig(OUT / "fig4_food_staples.png", dpi=150)
    plt.close()
    print("  fig4: food staples plotted")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Labor wages over time
# ─────────────────────────────────────────────────────────────────────────────
def fig5_labor():
    labor_cats = [
        ("LABOR-UNSKILLED",   "General Labor", "#e74c3c"),
        ("LABOR-AGRI",        "Farm Labor",    "#27ae60"),
        ("LABOR-DOMESTIC",    "Domestic Svc",  "#3498db"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    add_war_bands(ax, alpha=0.1)
    for cid, label, color in labor_cats:
        yrs, meds, unit = yearly_medians(cid, min_n=2)
        if len(yrs) >= 2:
            ax.plot(yrs, meds, "o-", color=color, markersize=4, linewidth=1.3,
                    label=f"{label} (/{unit})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median wage (USD/unit)")
    ax.set_title("Labor Wages in American Newspapers, 1770–1963", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_labor_wages.png", dpi=150)
    plt.close()
    print("  fig5: labor wages plotted")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Price volatility by category
# ─────────────────────────────────────────────────────────────────────────────
def fig6_volatility():
    cat_year_prices = defaultdict(lambda: defaultdict(list))
    for r in records:
        p = r.get("price_per_unit_usd")
        y = r.get("year", 0)
        cat = r.get("category_l1", "")
        if p and p > 0 and cat:
            cat_year_prices[cat][y].append(p)

    results = []
    for cat, yd in cat_year_prices.items():
        meds = []
        for y, vals in sorted(yd.items()):
            m = clean_median(vals)
            if m:
                meds.append(m)
        if len(meds) >= 5:
            logs = [math.log10(m) for m in meds]
            cv = np.std(logs) / abs(np.mean(logs)) if np.mean(logs) != 0 else 0
            results.append((cat, cv, len(meds)))

    results.sort(key=lambda x: -x[1])
    cats = [r[0] for r in results]
    cvs = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cats)))
    ax.barh(cats[::-1], cvs[::-1], color=colors[::-1], edgecolor="#2c3e50", linewidth=0.4)
    ax.set_xlabel("Log-scale Coefficient of Variation (higher = more volatile)")
    ax.set_title("Price Volatility by Economic Category, 1770–1963", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_volatility.png", dpi=150)
    plt.close()
    print(f"  fig6: {len(results)} categories")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Horse prices as transport proxy + railroad fare
# ─────────────────────────────────────────────────────────────────────────────
def fig7_transport():
    transport_items = [
        ("TRANS-HORSE",      "Horse price",    "#8e44ad"),
        ("TRANS-FARE-RAIL",  "Rail fare",      "#3498db"),
        ("TRANS-FARE-SHIP",  "Ship fare",      "#1abc9c"),
        ("TRANS-FARE-STAGE", "Stage fare",     "#e67e22"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    add_war_bands(ax, alpha=0.1)
    for cid, label, color in transport_items:
        yrs, meds, unit = yearly_medians(cid, min_n=2)
        if len(yrs) >= 2:
            ax.plot(yrs, meds, "o-", color=color, markersize=4, linewidth=1.3,
                    label=f"{label} (/{unit})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median price (USD)")
    ax.set_title("Transportation Prices, 1770–1963", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
    fig.tight_layout()
    fig.savefig(OUT / "fig7_transport.png", dpi=150)
    plt.close()
    print("  fig7: transport plotted")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Multi-commodity price index (1880 baseline = 100)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_price_index():
    index_commodities = [
        "FOOD-GRAIN-WHEAT", "FOOD-GRAIN-FLOUR", "FOOD-GRAIN-CORN",
        "FOOD-DAIRY-BUTTER", "FOOD-SUGAR-SUGAR",
    ]
    baseline_years = range(1875, 1886)

    all_year_data = defaultdict(list)
    for cid in index_commodities:
        yrs, meds, _ = yearly_medians(cid, min_n=2)
        if not yrs:
            continue
        # Compute baseline
        baseline_vals = [m for y, m in zip(yrs, meds) if y in baseline_years]
        if not baseline_vals:
            continue
        baseline = sum(baseline_vals) / len(baseline_vals)
        for y, m in zip(yrs, meds):
            all_year_data[y].append(m / baseline * 100)

    years = sorted(y for y in all_year_data if len(all_year_data[y]) >= 2)
    index_vals = [np.median(all_year_data[y]) for y in years]

    if not years:
        print("  fig8: no data"); return

    fig, ax = plt.subplots(figsize=(14, 5))
    add_war_bands(ax)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="1880 baseline")
    ax.fill_between(years, index_vals, 100,
                    where=[v > 100 for v in index_vals], alpha=0.3, color="#e74c3c", label="Above baseline")
    ax.fill_between(years, index_vals, 100,
                    where=[v <= 100 for v in index_vals], alpha=0.3, color="#27ae60", label="Below baseline")
    ax.plot(years, index_vals, color="#2c3e50", linewidth=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price index (1880 = 100)")
    ax.set_title("Food Price Index, 1770–1963\n(Wheat, Flour, Corn, Butter, Sugar — 1880 baseline)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_price_index.png", dpi=150)
    plt.close()
    print(f"  fig8: {len(years)} year-points in index")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Wheat price dispersion (coefficient of variation)
# ─────────────────────────────────────────────────────────────────────────────
def fig9_dispersion():
    wheat_recs = [
        r for r in records
        if r.get("commodity_id") == "FOOD-GRAIN-WHEAT"
        and r.get("price_per_unit_usd")
        and r["price_per_unit_usd"] > 0
    ]
    if not wheat_recs:
        print("  fig9: no wheat data"); return

    unit = dominant_unit(wheat_recs)
    wheat_recs = [r for r in wheat_recs if r.get("unit", "") == unit]

    by_year = defaultdict(list)
    for r in wheat_recs:
        by_year[r["year"]].append(r["price_per_unit_usd"])

    years, cvs = [], []
    for y in sorted(by_year):
        vals = [v for v in by_year[y] if v > 0]
        if len(vals) < 2:
            continue
        mean_v = float(np.mean(vals))
        if mean_v <= 0:
            continue
        cv = float(np.std(vals) / mean_v)
        years.append(y)
        cvs.append(cv)

    if not years:
        print("  fig9: insufficient yearly wheat data"); return

    fig, ax = plt.subplots(figsize=(14, 5))
    add_war_bands(ax, alpha=0.1)
    ax.plot(years, cvs, "o-", color="#d35400", markersize=3.5, linewidth=1.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient of variation (std / mean)")
    ax.set_title("Wheat Price Dispersion by Year, 1770–1963", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_wheat_dispersion.png", dpi=150)
    plt.close()
    print(f"  fig9: {len(years)} year-points (unit: {unit or 'mixed/unknown'})")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Wheat price dispersion (coefficient of variation)
# ─────────────────────────────────────────────────────────────────────────────
def fig9_dispersion():
    wheat_recs = [
        r for r in records
        if r.get("commodity_id") == "FOOD-GRAIN-WHEAT"
        and r.get("price_per_unit_usd")
        and r["price_per_unit_usd"] > 0
    ]
    if not wheat_recs:
        print("  fig9: no wheat data"); return

    unit = dominant_unit(wheat_recs)
    wheat_recs = [r for r in wheat_recs if r.get("unit", "") == unit]

    by_year = defaultdict(list)
    for r in wheat_recs:
        by_year[r["year"]].append(r["price_per_unit_usd"])

    years, cvs = [], []
    for y in sorted(by_year):
        vals = [v for v in by_year[y] if v > 0]
        if len(vals) < 2:
            continue
        mean_v = float(np.mean(vals))
        if mean_v <= 0:
            continue
        cv = float(np.std(vals) / mean_v)
        years.append(y)
        cvs.append(cv)

    if not years:
        print("  fig9: insufficient yearly wheat data"); return

    fig, ax = plt.subplots(figsize=(14, 5))
    add_war_bands(ax, alpha=0.1)
    ax.plot(years, cvs, "o-", color="#d35400", markersize=3.5, linewidth=1.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient of variation (std / mean)")
    ax.set_title("Wheat Price Dispersion by Year, 1770–1963", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_wheat_dispersion.png", dpi=150)
    plt.close()
    print(f"  fig9: {len(years)} year-points (unit: {unit or 'mixed/unknown'})")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating report figures...")
    fig1_volume()
    fig2_categories()
    fig3_flour_wars()
    fig4_food_staples()
    fig5_labor()
    fig6_volatility()
    fig7_transport()
    fig8_price_index()
    fig9_dispersion()
    print(f"\nAll figures saved to {OUT}/")
