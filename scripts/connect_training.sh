#!/bin/bash
# =============================================================================
# Connect to MTG-RL Training Instance
# =============================================================================
#
# Provides multiple connection methods:
#   1. SSM Session Manager (keyless, recommended)
#   2. SSH with key pair
#   3. TensorBoard port forwarding
#
# Usage:
#   ./scripts/connect_training.sh          # Auto-detect and connect
#   ./scripts/connect_training.sh ssm      # Use SSM Session Manager
#   ./scripts/connect_training.sh ssh      # Use SSH with key
#   ./scripts/connect_training.sh tensorboard  # Forward TensorBoard
#   ./scripts/connect_training.sh status   # Show instance status
#
# Prerequisites:
#   - AWS CLI configured
#   - For SSM: AWS Session Manager Plugin installed
#     brew install --cask session-manager-plugin
#
# =============================================================================

set -e

REGION="${AWS_REGION:-us-west-2}"
PROJECT="mtg-rl"
KEY_FILE="$HOME/.ssh/mtg-rl-training.pem"

# Get instance ID and IP
get_instance_info() {
    # Try multiple methods to find the training instance:
    # 1. By Project tag (on-demand instances)
    # 2. By IAM instance profile (spot instances don't inherit tags)
    # 3. By instance type (g4dn.*)

    # Method 1: Tag-based lookup
    INSTANCE_INFO=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=$PROJECT" \
                  "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \
        --output text \
        --region "$REGION" 2>/dev/null || echo "None None")

    INSTANCE_ID=$(echo "$INSTANCE_INFO" | awk '{print $1}')
    INSTANCE_IP=$(echo "$INSTANCE_INFO" | awk '{print $2}')

    # Method 2: IAM profile lookup (for spot instances)
    if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
        INSTANCE_INFO=$(aws ec2 describe-instances \
            --filters "Name=iam-instance-profile.arn,Values=*${PROJECT}-training-profile*" \
                      "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \
            --output text \
            --region "$REGION" 2>/dev/null || echo "None None")

        INSTANCE_ID=$(echo "$INSTANCE_INFO" | awk '{print $1}')
        INSTANCE_IP=$(echo "$INSTANCE_INFO" | awk '{print $2}')
    fi

    # Method 3: GPU instance type lookup (fallback)
    if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
        INSTANCE_INFO=$(aws ec2 describe-instances \
            --filters "Name=instance-type,Values=g4dn.*" \
                      "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \
            --output text \
            --region "$REGION" 2>/dev/null || echo "None None")

        INSTANCE_ID=$(echo "$INSTANCE_INFO" | awk '{print $1}')
        INSTANCE_IP=$(echo "$INSTANCE_INFO" | awk '{print $2}')
    fi

    if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
        echo "No running training instance found in $REGION"
        echo ""
        echo "To launch a training instance:"
        echo "  cd infrastructure && terraform apply -var=\"enable_training_instance=true\""
        exit 1
    fi
}

# Show status
show_status() {
    echo "============================================================"
    echo "MTG-RL Training Instance Status"
    echo "============================================================"
    get_instance_info
    echo "Region:      $REGION"
    echo "Instance ID: $INSTANCE_ID"
    echo "Public IP:   $INSTANCE_IP"
    echo "TensorBoard: http://$INSTANCE_IP:6006"
    echo ""

    # Try to get training progress
    S3_BUCKET=$(aws s3 ls | grep mtg-rl-checkpoints | awk '{print $3}' | head -1)
    if [ -n "$S3_BUCKET" ]; then
        echo "S3 Bucket: $S3_BUCKET"
        echo ""

        # Show live training log (last 15 lines)
        echo "=== Training Progress (live log) ==="
        aws s3 cp "s3://$S3_BUCKET/logs/training_live.log" - --region "$REGION" 2>/dev/null | tail -15 || echo "  Waiting for training to start..."
        echo ""

        # Check if training is complete
        COMPLETE=$(aws s3 cp "s3://$S3_BUCKET/training_complete.json" - --region "$REGION" 2>/dev/null)
        if [ -n "$COMPLETE" ]; then
            echo "=== Training Complete ==="
            echo "$COMPLETE"
        fi
    fi
}

# Connect via SSM
connect_ssm() {
    get_instance_info
    echo "Connecting via SSM Session Manager..."
    echo "Instance: $INSTANCE_ID"
    echo ""
    aws ssm start-session --target "$INSTANCE_ID" --region "$REGION"
}

# Connect via SSH
connect_ssh() {
    get_instance_info
    if [ ! -f "$KEY_FILE" ]; then
        echo "SSH key not found: $KEY_FILE"
        echo ""
        echo "Options:"
        echo "  1. Create key: ./scripts/setup_ssh.sh"
        echo "  2. Use SSM instead: ./scripts/connect_training.sh ssm"
        exit 1
    fi
    echo "Connecting via SSH..."
    echo "Instance: $INSTANCE_IP"
    echo ""
    ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ubuntu@"$INSTANCE_IP"
}

# Forward TensorBoard
forward_tensorboard() {
    get_instance_info
    echo "============================================================"
    echo "TensorBoard Port Forwarding"
    echo "============================================================"

    # Prefer SSM for port forwarding (keyless)
    if command -v session-manager-plugin &> /dev/null; then
        echo "Using SSM for port forwarding (keyless)..."
        echo ""
        echo "TensorBoard will be available at: http://localhost:6006"
        echo "Press Ctrl+C to stop"
        echo ""
        aws ssm start-session \
            --target "$INSTANCE_ID" \
            --document-name AWS-StartPortForwardingSession \
            --parameters '{"portNumber":["6006"],"localPortNumber":["6006"]}' \
            --region "$REGION"
    elif [ -f "$KEY_FILE" ]; then
        echo "Using SSH for port forwarding..."
        echo ""
        echo "TensorBoard will be available at: http://localhost:6006"
        echo "Press Ctrl+C to stop"
        echo ""
        ssh -i "$KEY_FILE" -N -L 6006:localhost:6006 ubuntu@"$INSTANCE_IP"
    else
        echo "Neither SSM nor SSH available."
        echo ""
        echo "Install SSM plugin: brew install --cask session-manager-plugin"
        echo "Or create SSH key: ./scripts/setup_ssh.sh"
        echo ""
        echo "Direct TensorBoard URL (if publicly accessible):"
        echo "  http://$INSTANCE_IP:6006"
        exit 1
    fi
}

# Main
case "${1:-auto}" in
    ssm)
        connect_ssm
        ;;
    ssh)
        connect_ssh
        ;;
    tensorboard|tb)
        forward_tensorboard
        ;;
    status)
        show_status
        ;;
    auto|"")
        get_instance_info
        echo "============================================================"
        echo "MTG-RL Training Instance"
        echo "============================================================"
        echo "Instance: $INSTANCE_ID ($INSTANCE_IP)"
        echo ""
        echo "Commands:"
        echo "  ./scripts/connect_training.sh ssm         # Shell access (keyless)"
        echo "  ./scripts/connect_training.sh ssh         # Shell access (SSH key)"
        echo "  ./scripts/connect_training.sh tensorboard # TensorBoard"
        echo "  ./scripts/connect_training.sh status      # Check status"
        echo ""
        echo "Quick check training log:"
        echo "  aws ssm start-session --target $INSTANCE_ID --region $REGION"
        echo "  Then: tail -f /home/ubuntu/mtg-rl/training.log"
        ;;
    *)
        echo "Usage: $0 [ssm|ssh|tensorboard|status]"
        exit 1
        ;;
esac
