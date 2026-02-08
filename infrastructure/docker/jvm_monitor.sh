#!/bin/bash
# JVM Monitor for Forge Daemon
# Periodically logs heap usage, GC stats, and thread count.
# Output goes to stdout (captured by Docker logs) and optionally to a file.
#
# Usage:
#   ./jvm_monitor.sh [interval_seconds] [output_file]
#   ./jvm_monitor.sh 30                          # Log every 30s to stdout
#   ./jvm_monitor.sh 60 /forge/logs/jvm_stats.log  # Log every 60s to file + stdout

INTERVAL="${1:-30}"
OUTPUT_FILE="${2:-}"

log_msg() {
    local timestamp
    timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    local msg="[JVM-MONITOR $timestamp] $1"
    echo "$msg"
    if [ -n "$OUTPUT_FILE" ]; then
        echo "$msg" >> "$OUTPUT_FILE"
    fi
}

log_msg "Starting JVM monitor (interval: ${INTERVAL}s)"

# Wait for the JVM to start
sleep 10

while true; do
    # Find the Forge JVM process
    JAVA_PID=$(pgrep -f "forge.jar" | head -1)

    if [ -z "$JAVA_PID" ]; then
        log_msg "WARN: Forge JVM not found, waiting..."
        sleep "$INTERVAL"
        continue
    fi

    # --- Heap usage via jcmd ---
    HEAP_INFO=$(jcmd "$JAVA_PID" GC.heap_info 2>/dev/null | tail -n +2)
    if [ -n "$HEAP_INFO" ]; then
        # Extract used/committed from G1 regions
        HEAP_USED=$(echo "$HEAP_INFO" | grep -oP 'used \K[0-9]+[KMG]' | head -1)
        HEAP_COMMITTED=$(echo "$HEAP_INFO" | grep -oP 'committed \K[0-9]+[KMG]' | head -1)
        log_msg "HEAP used=$HEAP_USED committed=$HEAP_COMMITTED"
    fi

    # --- GC stats via jstat ---
    # Columns: S0C S1C S0U S1U EC EU OC OU MC MU CCSC CCSU YGC YGCT FGC FGCT GCT
    GC_LINE=$(jstat -gc "$JAVA_PID" 2>/dev/null | tail -1)
    if [ -n "$GC_LINE" ]; then
        # Parse jstat output (space-separated floating point values)
        read -r S0C S1C S0U S1U EC EU OC OU MC MU CCSC CCSU YGC YGCT FGC FGCT GCT <<< "$GC_LINE"
        # Convert KB to MB for readability
        OU_MB=$(echo "$OU" | awk '{printf "%.0f", $1/1024}')
        OC_MB=$(echo "$OC" | awk '{printf "%.0f", $1/1024}')
        EU_MB=$(echo "$EU" | awk '{printf "%.0f", $1/1024}')
        EC_MB=$(echo "$EC" | awk '{printf "%.0f", $1/1024}')
        MU_MB=$(echo "$MU" | awk '{printf "%.0f", $1/1024}')
        log_msg "GC young_gc=$YGC young_time=${YGCT}s full_gc=$FGC full_time=${FGCT}s total_gc_time=${GCT}s"
        log_msg "MEM eden=${EU_MB}/${EC_MB}MB old=${OU_MB}/${OC_MB}MB metaspace=${MU_MB}MB"
    fi

    # --- Thread count ---
    THREAD_COUNT=$(jcmd "$JAVA_PID" Thread.print 2>/dev/null | grep -c "^\"" || echo "?")
    log_msg "THREADS count=$THREAD_COUNT"

    sleep "$INTERVAL"
done
