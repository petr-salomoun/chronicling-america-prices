#!/bin/bash
# Start 5 parallel normalization workers for different year ranges
# Each writes to its own output directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$(which python3)"
BATCH=150
BACKEND="copilot"

ranges=("1884 1900" "1901 1915" "1916 1930" "1931 1945" "1946 1963")

for range in "${ranges[@]}"; do
    y1=$(echo $range | cut -d' ' -f1)
    y2=$(echo $range | cut -d' ' -f2)
    outdir="data/pass2/prices_${y1}_${y2}"
    logfile="/tmp/norm_${y1}_${y2}.log"
    mkdir -p "$outdir"
    echo "Starting worker $y1-$y2 -> $outdir (log: $logfile)"
    nohup "$PYTHON" normalize_prices_pass2.py \
        --year-start "$y1" --year-end "$y2" \
        --out-dir "$outdir" \
        --batch-size "$BATCH" \
        --backend "$BACKEND" \
        > "$logfile" 2>&1 &
    echo "  PID: $!"
    sleep 2
done

echo ""
echo "All workers started. Monitor with:"
echo "  tail -f /tmp/norm_*.log"
echo "Check progress with:"
echo "  python3 -c \"import json; [print(y, len(list(open('data/pass2/prices_'+y+'/_progress.json').read().split('\"processed_files\"')[1].split(']')[0].split(',')))) for y in ['1884_1900','1901_1915','1916_1930','1931_1945','1946_1963'] if __import__('os').path.exists('data/pass2/prices_'+y+'/_progress.json')]\""
