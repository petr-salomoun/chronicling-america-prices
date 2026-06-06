#!/usr/bin/env python3
"""Monitor parallel normalization workers."""
import json, subprocess
from pathlib import Path

RANGES = [('1884_1900', 4947), ('1901_1915', 6294), ('1916_1930', 3820), ('1931_1945', 1517), ('1946_1963', 1118)]

total_pending = sum(n for _, n in RANGES)
total_norm = 0
total_unres = 0

print(f"{'Range':<12} {'Files':>6} {'Norm':>7} {'Unres':>6} {'%':>5}")
print("-" * 45)
for label, total_files in RANGES:
    d = Path(f"data/pass2/prices_{label}")
    try:
        n_norm = sum(1 for _ in open(d / 'normalized.jsonl'))
        n_unres = sum(1 for _ in open(d / 'unresolved.jsonl')) if (d / 'unresolved.jsonl').exists() else 0
        prog = json.load(open(d / '_progress.json'))['count'] if (d / '_progress.json').exists() else 0
        pct = 100 * prog / total_files
        total_norm += n_norm
        total_unres += n_unres
        print(f"{label:<12} {prog:>6}/{total_files} {n_norm:>7} {n_unres:>6} {pct:>4.1f}%")
    except Exception as e:
        print(f"{label:<12} (not started)")

alive = subprocess.run(['pgrep', '-c', '-f', 'normalize_prices_pass2'], capture_output=True, text=True)
n_alive = alive.stdout.strip()

print("-" * 45)
print(f"{'TOTAL':<12} {'':>6}  {total_norm:>7} {total_unres:>6}")
print(f"Workers alive: {n_alive}")
print(f"\nMain normalized.jsonl: {sum(1 for _ in open('data/pass2/prices/normalized.jsonl'))} lines")
