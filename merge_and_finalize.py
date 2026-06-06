#!/usr/bin/env python3
"""
Merge parallel normalization outputs into the main data/pass2/prices/ directory.
Run this after all parallel workers have finished.

Usage:
    python3 merge_and_finalize.py [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

RANGES = ['1884_1900', '1901_1915', '1916_1930', '1931_1945', '1946_1963']
BASE = Path("data/pass2/prices")


def count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in open(path, encoding='utf-8', errors='replace'))
    except FileNotFoundError:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=== Merge parallel normalization outputs ===\n")

    # 1. Check worker status
    import subprocess
    alive = int(subprocess.run(['pgrep', '-c', '-f', 'normalize_prices_pass2'],
                               capture_output=True, text=True).stdout.strip() or '0')
    if alive > 0:
        print(f"WARNING: {alive} normalize_prices_pass2 worker(s) still running!")
        print("Wait for them to finish before merging, or use --force to merge anyway.")
        if '--force' not in sys.argv:
            print("Aborting. Add --force to override.")
            sys.exit(1)

    # 2. Show what we'd merge
    for label in RANGES:
        d = Path(f"data/pass2/prices_{label}")
        if not d.exists():
            print(f"  {label}: no output dir")
            continue
        n_norm = count_lines(d / 'normalized.jsonl')
        n_unres = count_lines(d / 'unresolved.jsonl')
        n_fail = count_lines(d / 'failed.jsonl')
        prog = json.load(open(d / '_progress.json'))['count'] if (d / '_progress.json').exists() else 0
        print(f"  {label}: {prog} files, {n_norm} normalized, {n_unres} unresolved, {n_fail} failed")

    print(f"\nMain normalized.jsonl currently: {count_lines(BASE/'normalized.jsonl')} lines")
    print(f"Main unresolved.jsonl currently: {count_lines(BASE/'unresolved.jsonl')} lines")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # 3. Merge
    print("\nMerging...")
    for label in RANGES:
        d = Path(f"data/pass2/prices_{label}")
        if not d.exists():
            continue

        for fname, main_path in [
            ('normalized.jsonl', BASE / 'normalized.jsonl'),
            ('unresolved.jsonl', BASE / 'unresolved.jsonl'),
            ('failed.jsonl', BASE / 'failed.jsonl'),
        ]:
            src = d / fname
            if src.exists() and src.stat().st_size > 0:
                content = src.read_text(encoding='utf-8', errors='replace')
                n = content.count('\n')
                with open(main_path, 'a', encoding='utf-8') as f:
                    f.write(content)
                print(f"  [{label}] +{n} lines → {fname}")

        # Merge progress
        src_prog = d / '_progress.json'
        main_prog = BASE / '_progress.json'
        if src_prog.exists() and main_prog.exists():
            src_data = json.load(open(src_prog))
            main_data = json.load(open(main_prog))
            combined = sorted(set(main_data['processed_files']) | set(src_data['processed_files']))
            main_data['processed_files'] = combined
            main_data['count'] = len(combined)
            with open(main_prog, 'w') as f:
                json.dump(main_data, f, indent=2)
            print(f"  [{label}] progress merged: {len(combined)} total files")

    print(f"\nFinal normalized.jsonl: {count_lines(BASE/'normalized.jsonl')} lines")
    print(f"Final unresolved.jsonl: {count_lines(BASE/'unresolved.jsonl')} lines")
    print("\nDone. Now run: python3 analyze_prices.py --no-plots")
    print("Then:         python3 generate_report_figures.py")


if __name__ == '__main__':
    main()
