#!/bin/bash
# =============================================================================
# Check Training Status and Download Results
# =============================================================================
# Usage:
#   ./scripts/check_training.sh                    # Check status
#   ./scripts/check_training.sh --download         # Download all artifacts
#   ./scripts/check_training.sh --live             # Stream live log
#   ./scripts/check_training.sh --report           # Generate report from results
# =============================================================================

set -e

# Configuration
BUCKET_PREFIX="mtg-rl-checkpoints"
OUTPUT_DIR="./training_output"

# Find the bucket
BUCKET=$(aws s3 ls | grep "$BUCKET_PREFIX" | awk '{print $3}' | head -1)

if [ -z "$BUCKET" ]; then
    echo "Error: No S3 bucket found matching $BUCKET_PREFIX"
    echo "Make sure training infrastructure is deployed."
    exit 1
fi

echo "Found bucket: $BUCKET"
echo ""

# Parse arguments
ACTION="${1:-status}"

case "$ACTION" in
    --status|status)
        echo "=== Training Status ==="

        # Check for completion marker
        if aws s3 ls "s3://$BUCKET/training_complete.json" &>/dev/null; then
            echo "Status: COMPLETE"
            aws s3 cp "s3://$BUCKET/training_complete.json" - 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Result: {d['status']}\"); print(f\"  Time: {d['timestamp']}\")"
        else
            echo "Status: IN PROGRESS (or not started)"
        fi

        echo ""
        echo "=== Checkpoints ==="
        aws s3 ls "s3://$BUCKET/checkpoints/" 2>/dev/null | tail -5 || echo "  No checkpoints yet"

        echo ""
        echo "=== Logs ==="
        aws s3 ls "s3://$BUCKET/logs/" 2>/dev/null | tail -5 || echo "  No logs yet"

        echo ""
        echo "=== Live Log (last 20 lines) ==="
        aws s3 cp "s3://$BUCKET/logs/training_live.log" - 2>/dev/null | tail -20 || echo "  No live log available"
        ;;

    --live|live)
        echo "=== Streaming Live Log (Ctrl+C to stop) ==="
        while true; do
            clear
            echo "=== Live Training Log ($(date)) ==="
            echo ""
            aws s3 cp "s3://$BUCKET/logs/training_live.log" - 2>/dev/null | tail -40 || echo "Waiting for log..."
            sleep 10
        done
        ;;

    --download|download)
        echo "=== Downloading Training Artifacts ==="
        mkdir -p "$OUTPUT_DIR"

        echo "Downloading checkpoints..."
        aws s3 sync "s3://$BUCKET/checkpoints/" "$OUTPUT_DIR/checkpoints/"

        echo "Downloading logs..."
        aws s3 sync "s3://$BUCKET/logs/" "$OUTPUT_DIR/logs/"

        echo "Downloading TensorBoard logs..."
        aws s3 sync "s3://$BUCKET/tensorboard-logs/" "$OUTPUT_DIR/tensorboard-logs/"

        echo ""
        echo "All artifacts downloaded to: $OUTPUT_DIR/"
        echo ""
        echo "To view TensorBoard:"
        echo "  tensorboard --logdir $OUTPUT_DIR/tensorboard-logs/"
        ;;

    --report|report)
        echo "=== Generating Training Report ==="

        # First download
        mkdir -p "$OUTPUT_DIR"
        aws s3 cp "s3://$BUCKET/checkpoints/final_results.json" "$OUTPUT_DIR/final_results.json" 2>/dev/null || {
            echo "Error: No final_results.json found. Training may not be complete."
            exit 1
        }

        # Find latest training log
        LATEST_LOG=$(aws s3 ls "s3://$BUCKET/logs/" 2>/dev/null | grep "training_final" | sort | tail -1 | awk '{print $4}')
        if [ -n "$LATEST_LOG" ]; then
            aws s3 cp "s3://$BUCKET/logs/$LATEST_LOG" "$OUTPUT_DIR/training.log"
        fi

        # Generate report
        python3 generate_training_report.py \
            --results "$OUTPUT_DIR/final_results.json" \
            --log "$OUTPUT_DIR/training.log" \
            --output-dir "$OUTPUT_DIR/reports"

        echo ""
        echo "Report generated in: $OUTPUT_DIR/reports/"
        ;;

    --costs|costs)
        echo "=== Training Costs ==="
        echo ""

        # Get running instances
        INSTANCES=$(aws ec2 describe-instances \
            --filters "Name=tag:Project,Values=mtg-rl" "Name=instance-state-name,Values=running" \
            --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime]' \
            --output text 2>/dev/null)

        if [ -n "$INSTANCES" ]; then
            echo "Running Instances:"
            echo "$INSTANCES"
            echo ""
            echo "Estimated hourly cost:"
            echo "  g4dn.xlarge spot: ~\$0.16/hr"
            echo "  g4dn.xlarge on-demand: ~\$0.53/hr"
        else
            echo "No running training instances."
        fi

        echo ""
        echo "S3 Storage:"
        aws s3 ls "s3://$BUCKET" --recursive --summarize 2>/dev/null | tail -2
        ;;

    --terminate|terminate)
        echo "=== Terminating Training Instances ==="

        INSTANCES=$(aws ec2 describe-instances \
            --filters "Name=tag:Project,Values=mtg-rl" "Name=instance-state-name,Values=running" \
            --query 'Reservations[*].Instances[*].InstanceId' \
            --output text 2>/dev/null)

        if [ -n "$INSTANCES" ]; then
            echo "Terminating: $INSTANCES"
            aws ec2 terminate-instances --instance-ids $INSTANCES
            echo "Instances terminated."
        else
            echo "No running instances to terminate."
        fi
        ;;

    *)
        echo "Usage: $0 [--status|--live|--download|--report|--costs|--terminate]"
        echo ""
        echo "Commands:"
        echo "  --status     Check training status (default)"
        echo "  --live       Stream live training log"
        echo "  --download   Download all training artifacts"
        echo "  --report     Generate post-training report"
        echo "  --costs      Check training costs"
        echo "  --terminate  Terminate running training instances"
        ;;
esac
