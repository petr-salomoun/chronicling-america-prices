#!/usr/bin/env python3
"""Run normalize_prices_pass2.py in parallel over non-overlapping year ranges.

Usage:
    python run_parallel_normalize.py [--workers N] [--batch-size N] [--backend BACKEND]

Each worker writes to its own output dir (data/pass2/prices_YYYY-YYYY/),
then the merge step combines everything into data/pass2/prices/.

Defaults:
    --workers 5   (one per year-range partition)
    --batch-size 150
    --backend copilot
"""

import argparse
import json
import subprocess
import sys
import time
import threading
from pathlib import Path

YEAR_RANGES = [
    (1884, 1900),
    (1901, 1915),
    (1916, 1930),
    (1931, 1945),
    (1946, 1963),
]

BASE_OUT = Path("data/pass2/prices")


def run_worker(y1: int, y2: int, batch_size: int, backend: str, log_path: Path):
    out_dir = Path(f"data/pass2/prices_{y1}_{y2}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "normalize_prices_pass2.py",
        "--year-start", str(y1),
        "--year-end", str(y2),
        "--out-dir", str(out_dir),
        "--batch-size", str(batch_size),
        "--backend", backend,
    ]
    print(f"[{y1}-{y2}] Starting: out={out_dir}")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf)
    return proc, y1, y2, out_dir


def merge_results():
    """Merge per-range output dirs into the main data/pass2/prices/ directory."""
    print("\n=== Merging results ===")
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    normalized_main = BASE_OUT / "normalized.jsonl"
    unresolved_main = BASE_OUT / "unresolved.jsonl"
    failed_main = BASE_OUT / "failed.jsonl"

    for y1, y2 in YEAR_RANGES:
        out_dir = Path(f"data/pass2/prices_{y1}_{y2}")
        if not out_dir.exists():
            print(f"  [{y1}-{y2}] No output dir found, skipping")
            continue

        for fname, main_path in [
            ("normalized.jsonl", normalized_main),
            ("unresolved.jsonl", unresolved_main),
            ("failed.jsonl", failed_main),
        ]:
            src = out_dir / fname
            if src.exists() and src.stat().st_size > 0:
                lines = src.read_text(encoding="utf-8", errors="replace")
                with open(main_path, "a", encoding="utf-8") as f:
                    f.write(lines)
                n = lines.count("\n")
                print(f"  [{y1}-{y2}] Merged {fname}: {n} lines → {main_path}")

        # Merge progress
        src_prog = out_dir / "_progress.json"
        main_prog = BASE_OUT / "_progress.json"
        if src_prog.exists() and main_prog.exists():
            with open(src_prog) as f:
                src_data = json.load(f)
            with open(main_prog) as f:
                main_data = json.load(f)
            combined = sorted(set(main_data["processed_files"]) | set(src_data["processed_files"]))
            main_data["processed_files"] = combined
            main_data["count"] = len(combined)
            with open(main_prog, "w") as f:
                json.dump(main_data, f, indent=2)
            print(f"  [{y1}-{y2}] Merged _progress.json: {len(combined)} total entries")

    print("Merge complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    parser.add_argument("--batch-size", type=int, default=150, help="Records per LLM call (default: 150)")
    parser.add_argument("--backend", default="copilot", choices=["litellm", "copilot", "auto"])
    parser.add_argument("--merge-only", action="store_true", help="Skip processing, just merge existing output dirs")
    args = parser.parse_args()

    if args.merge_only:
        merge_results()
        return

    ranges_to_run = YEAR_RANGES[:args.workers]
    procs = []
    for y1, y2 in ranges_to_run:
        log_path = Path(f"/tmp/norm_{y1}_{y2}.log")
        proc, *info = run_worker(y1, y2, args.batch_size, args.backend, log_path)
        procs.append((proc, y1, y2, log_path))
        time.sleep(1)  # stagger starts slightly

    print(f"\nRunning {len(procs)} workers. Monitoring...")
    try:
        while True:
            alive = [(p, y1, y2, lp) for p, y1, y2, lp in procs if p.poll() is None]
            done = [(p, y1, y2, lp) for p, y1, y2, lp in procs if p.poll() is not None]
            print(f"\n[{time.strftime('%H:%M:%S')}] {len(alive)} running, {len(done)} done")
            for _, y1, y2, lp in alive:
                # Show last log line
                try:
                    lines = Path(lp).read_text(errors="replace").strip().splitlines()
                    last = lines[-1] if lines else "(no output)"
                    print(f"  [{y1}-{y2}] {last[-120:]}")
                except Exception:
                    pass
            if not alive:
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nInterrupted — sending SIGTERM to workers...")
        for proc, y1, y2, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        print("Workers terminated. Run again to resume (progress is saved).")
        return

    print("\nAll workers finished. Merging results...")
    merge_results()


if __name__ == "__main__":
    main()
