#!/bin/bash
# =============================================================================
# SSH Setup for MTG-RL Training Instances
# =============================================================================
#
# This script creates an SSH key pair in AWS and configures your local machine
# to connect to training instances.
#
# Usage:
#   ./scripts/setup_ssh.sh
#
# After running, add this to infrastructure/terraform.tfvars:
#   ssh_key_name = "mtg-rl-training"
#
# =============================================================================

set -e

KEY_NAME="mtg-rl-training"
REGION="${AWS_REGION:-us-west-2}"
SSH_DIR="$HOME/.ssh"
KEY_FILE="$SSH_DIR/$KEY_NAME.pem"

echo "============================================================"
echo "MTG-RL SSH Key Setup"
echo "============================================================"
echo "Region: $REGION"
echo "Key name: $KEY_NAME"
echo ""

# Check if key already exists locally
if [ -f "$KEY_FILE" ]; then
    echo "Key file already exists: $KEY_FILE"
    echo "To regenerate, delete it first: rm $KEY_FILE"
    echo ""
    echo "Your terraform.tfvars should have:"
    echo '  ssh_key_name = "mtg-rl-training"'
    exit 0
fi

# Check if key exists in AWS
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" 2>/dev/null; then
    echo "Key pair '$KEY_NAME' exists in AWS but not locally."
    echo "Options:"
    echo "  1. Delete the AWS key: aws ec2 delete-key-pair --key-name $KEY_NAME --region $REGION"
    echo "  2. Use a different key name"
    exit 1
fi

echo "Creating new key pair in AWS..."
mkdir -p "$SSH_DIR"

# Create key pair and save locally
aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --query 'KeyMaterial' \
    --output text \
    --region "$REGION" > "$KEY_FILE"

# Set correct permissions
chmod 400 "$KEY_FILE"

echo ""
echo "============================================================"
echo "SUCCESS: SSH key created"
echo "============================================================"
echo ""
echo "Key file: $KEY_FILE"
echo ""
echo "Add this to infrastructure/terraform.tfvars:"
echo '  ssh_key_name = "mtg-rl-training"'
echo ""
echo "To SSH into a training instance:"
echo "  ssh -i $KEY_FILE ubuntu@<INSTANCE_IP>"
echo ""
echo "To forward TensorBoard:"
echo "  ssh -i $KEY_FILE -L 6006:localhost:6006 ubuntu@<INSTANCE_IP>"
echo "  Then open: http://localhost:6006"
echo ""
echo "============================================================"
