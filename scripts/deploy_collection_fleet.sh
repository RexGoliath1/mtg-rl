#!/bin/bash
set -e

# ============================================================================
# Deploy Fleet of Data Collection Instances for Horizontal Scaling
# ============================================================================
# Launches N parallel EC2 spot instances, each running its own Forge daemon
# + collector. Games are split evenly across instances for near-linear
# throughput scaling.
#
# Each instance uploads to its own S3 subdirectory:
#   s3://BUCKET/imitation_data/fleet_TIMESTAMP/instance_0/
#   s3://BUCKET/imitation_data/fleet_TIMESTAMP/instance_1/
#   ...
#
# After all instances complete, use merge_collection_data.sh to combine
# the HDF5 files into a single training dataset.
#
# Prerequisites:
# - AWS CLI configured
# - GHCR images exist (CI pushes on main branch)
# - S3 bucket exists (from terraform)
# - deploy_data_collection_docker.sh in same directory
#
# Usage:
#   ./scripts/deploy_collection_fleet.sh --games 5000 --instances 5
#   ./scripts/deploy_collection_fleet.sh --games 10000 --instances 10 --workers 8
# ============================================================================

S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.2xlarge}"
NUM_GAMES="${NUM_GAMES:-3000}"
NUM_INSTANCES="${NUM_INSTANCES:-3}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-60}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --games)
            NUM_GAMES="$2"
            shift 2
            ;;
        --instances)
            NUM_INSTANCES="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--games N] [--instances N] [--workers N] [--timeout N] [--instance-type TYPE]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_SCRIPT="${SCRIPT_DIR}/deploy_data_collection_docker.sh"

if [ ! -x "$DEPLOY_SCRIPT" ]; then
    echo "ERROR: Deploy script not found or not executable: $DEPLOY_SCRIPT"
    echo "Run: chmod +x $DEPLOY_SCRIPT"
    exit 1
fi

# Calculate games per instance (distribute evenly, remainder goes to last instance)
GAMES_PER_INSTANCE=$((NUM_GAMES / NUM_INSTANCES))
REMAINDER=$((NUM_GAMES % NUM_INSTANCES))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FLEET_ID="fleet_${TIMESTAMP}"

echo "============================================================"
echo "FLEET DATA COLLECTION DEPLOYMENT"
echo "============================================================"
echo "Fleet ID:        $FLEET_ID"
echo "Total Games:     $NUM_GAMES"
echo "Instances:       $NUM_INSTANCES"
echo "Games/Instance:  $GAMES_PER_INSTANCE (+ $REMAINDER remainder to last)"
echo "Workers/Instance: $WORKERS"
echo "Instance Type:   $INSTANCE_TYPE"
echo "Game Timeout:    ${TIMEOUT}s"
echo "Region:          $REGION"
echo ""
echo "S3 Output:       s3://${S3_BUCKET}/imitation_data/${FLEET_ID}/"
echo "============================================================"
echo ""

# Track launched instances for summary
declare -a INSTANCE_IDS
declare -a INSTANCE_IPS
declare -a INSTANCE_GAMES

# Launch each instance
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    # Last instance gets the remainder games
    if [ $i -eq $((NUM_INSTANCES - 1)) ]; then
        INST_GAMES=$((GAMES_PER_INSTANCE + REMAINDER))
    else
        INST_GAMES=$GAMES_PER_INSTANCE
    fi

    INST_RUN_ID="${FLEET_ID}/instance_${i}"
    INST_S3_PREFIX="${FLEET_ID}/instance_${i}"

    echo "------------------------------------------------------------"
    echo "Launching instance $i/$((NUM_INSTANCES - 1)) ($INST_GAMES games)"
    echo "  Run ID:    $INST_RUN_ID"
    echo "  S3 Prefix: $INST_S3_PREFIX"
    echo "------------------------------------------------------------"

    # Capture the output to extract instance ID and IP
    OUTPUT=$("$DEPLOY_SCRIPT" \
        --games "$INST_GAMES" \
        --workers "$WORKERS" \
        --timeout "$TIMEOUT" \
        --instance-type "$INSTANCE_TYPE" \
        --run-id "$INST_RUN_ID" \
        --s3-prefix "$INST_S3_PREFIX" \
        2>&1)

    echo "$OUTPUT"

    # Extract instance ID and IP from output
    INST_ID=$(echo "$OUTPUT" | grep "Instance ID:" | tail -1 | awk '{print $NF}')
    INST_IP=$(echo "$OUTPUT" | grep "Public IP:" | tail -1 | awk '{print $NF}')

    INSTANCE_IDS+=("$INST_ID")
    INSTANCE_IPS+=("$INST_IP")
    INSTANCE_GAMES+=("$INST_GAMES")

    echo ""
    echo "Instance $i launched: $INST_ID ($INST_IP) - $INST_GAMES games"
    echo ""

    # Brief pause between launches to avoid API throttling
    if [ $i -lt $((NUM_INSTANCES - 1)) ]; then
        echo "Waiting 5s before next launch..."
        sleep 5
    fi
done

# Print fleet summary
echo ""
echo "============================================================"
echo "FLEET DEPLOYMENT COMPLETE"
echo "============================================================"
echo "Fleet ID: $FLEET_ID"
echo "Total:    $NUM_GAMES games across $NUM_INSTANCES instances"
echo ""
echo "Instance Summary:"
echo "  %-4s %-22s %-16s %-8s" "IDX" "INSTANCE_ID" "IP" "GAMES"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    printf "  %-4s %-22s %-16s %-8s\n" "$i" "${INSTANCE_IDS[$i]}" "${INSTANCE_IPS[$i]}" "${INSTANCE_GAMES[$i]}"
done

echo ""
echo "S3 Output:"
echo "  s3://${S3_BUCKET}/imitation_data/${FLEET_ID}/"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    echo "    instance_${i}/  (${INSTANCE_GAMES[$i]} games)"
done

echo ""
echo "============================================================"
echo "MONITORING COMMANDS"
echo "============================================================"
echo ""
echo "# Check which instances are complete:"
echo "for i in $(seq 0 $((NUM_INSTANCES - 1))); do"
echo "  echo -n \"instance_\$i: \""
echo "  aws s3 ls s3://${S3_BUCKET}/imitation_data/${FLEET_ID}/instance_\${i}/collection_complete.json 2>/dev/null && echo 'DONE' || echo 'RUNNING'"
echo "done"
echo ""
echo "# Quick fleet status (all instances):"
INST_IDS_STR=$(IFS=' '; echo "${INSTANCE_IDS[*]}")
echo "aws ec2 describe-instances --region $REGION --instance-ids $INST_IDS_STR --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output table"
echo ""
echo "# View live log for instance N:"
echo "aws s3 cp s3://${S3_BUCKET}/imitation_data/${FLEET_ID}/instance_N/live_log.txt - | tail -50"
echo ""
echo "# After all complete, merge data:"
echo "./scripts/merge_collection_data.sh --fleet-prefix ${FLEET_ID}"
echo ""
echo "All instances auto-terminate after collection."
echo "============================================================"

# Save fleet manifest to S3 for merge script
MANIFEST="{\"fleet_id\":\"${FLEET_ID}\",\"timestamp\":\"${TIMESTAMP}\",\"num_instances\":${NUM_INSTANCES},\"total_games\":${NUM_GAMES},\"instance_ids\":[$(printf '"%s",' "${INSTANCE_IDS[@]}" | sed 's/,$//')],\"games_per_instance\":[$(printf '%s,' "${INSTANCE_GAMES[@]}" | sed 's/,$//')]}"
echo "$MANIFEST" | aws s3 cp - "s3://${S3_BUCKET}/imitation_data/${FLEET_ID}/fleet_manifest.json"
echo ""
echo "Fleet manifest uploaded to s3://${S3_BUCKET}/imitation_data/${FLEET_ID}/fleet_manifest.json"
