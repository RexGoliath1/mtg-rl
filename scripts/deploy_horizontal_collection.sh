#!/bin/bash
set -e

# ============================================================================
# Horizontal Data Collection — Launch N parallel instances
# ============================================================================
# Wraps deploy_data_collection_docker.sh to launch multiple spot instances
# in parallel, each writing to a unique S3 prefix.
#
# Cost estimate: 20 instances × $0.078/hr (c5.2xlarge spot) = $1.56/hr
# Expected output: 20K games (~8.2M decisions) in ~35 minutes
#
# Prerequisites:
#   - AWS CLI configured, spot capacity available
#   - GHCR images pushed (CI on main branch)
#
# Usage:
#   ./scripts/deploy_horizontal_collection.sh --instances 20 --games-per 1000
#   ./scripts/deploy_horizontal_collection.sh --instances 5 --games-per 500 --workers 4
# ============================================================================

# Configuration
NUM_INSTANCES="${NUM_INSTANCES:-20}"
GAMES_PER_INSTANCE="${GAMES_PER_INSTANCE:-1000}"
WORKERS_PER_INSTANCE="${WORKERS_PER_INSTANCE:-8}"
TIMEOUT="${TIMEOUT:-60}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.2xlarge}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --instances)
            NUM_INSTANCES="$2"
            shift 2
            ;;
        --games-per)
            GAMES_PER_INSTANCE="$2"
            shift 2
            ;;
        --workers)
            WORKERS_PER_INSTANCE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--instances N] [--games-per N] [--workers N] [--timeout N] [--instance-type TYPE] [--dry-run]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLEET_ID="fleet_$(date +%Y%m%d_%H%M%S)"
TOTAL_GAMES=$((NUM_INSTANCES * GAMES_PER_INSTANCE))

echo "============================================================"
echo "HORIZONTAL DATA COLLECTION"
echo "============================================================"
echo "Fleet ID:            $FLEET_ID"
echo "Instances:           $NUM_INSTANCES"
echo "Games per instance:  $GAMES_PER_INSTANCE"
echo "Workers per instance: $WORKERS_PER_INSTANCE"
echo "Total games:         $TOTAL_GAMES"
echo "Instance type:       $INSTANCE_TYPE"
echo ""
echo "Estimated cost: $NUM_INSTANCES × \$0.078/hr = \$$(echo "scale=2; $NUM_INSTANCES * 0.078" | bc)/hr"
echo "Estimated time: ~35 min (parallel)"
echo "============================================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[DRY RUN] Would launch $NUM_INSTANCES instances. Exiting."
    exit 0
fi

echo ""
read -p "Proceed? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Track instance IDs for monitoring
INSTANCE_IDS=()
PIDS=()

echo ""
echo "Launching $NUM_INSTANCES instances..."

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    INSTANCE_PREFIX="${FLEET_ID}/instance_${i}"
    RUN_ID="${FLEET_ID}_instance_${i}"

    echo "  [$i/$NUM_INSTANCES] Launching with s3-prefix=$INSTANCE_PREFIX ..."

    # Launch in background, capture PID
    "$SCRIPT_DIR/deploy_data_collection_docker.sh" \
        --games "$GAMES_PER_INSTANCE" \
        --workers "$WORKERS_PER_INSTANCE" \
        --timeout "$TIMEOUT" \
        --instance-type "$INSTANCE_TYPE" \
        --run-id "$RUN_ID" \
        --s3-prefix "$INSTANCE_PREFIX" \
        > "/tmp/fleet_${i}.log" 2>&1 &

    PIDS+=($!)

    # Small delay to avoid API rate limits
    sleep 2
done

echo ""
echo "All $NUM_INSTANCES instances launched. Fleet ID: $FLEET_ID"
echo ""
echo "Monitor with:"
echo "  ./scripts/monitor_collection.py --fleet-id $FLEET_ID"
echo ""
echo "Merge results after completion:"
echo "  python3 scripts/merge_fleet_hdf5.py --fleet-id $FLEET_ID"
echo ""
echo "Instance launch logs: /tmp/fleet_*.log"

# Wait for all launch scripts to complete (they return after instance starts)
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "All launch commands completed. Instances are running autonomously."
echo "They will auto-terminate after data collection + S3 upload."
