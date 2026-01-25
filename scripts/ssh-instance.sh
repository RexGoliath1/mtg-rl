#!/bin/bash
# SSH into MTG RL training/collection instance using EC2 Instance Connect
# No PEM files needed - uses your AWS CLI credentials
#
# Usage:
#   ./scripts/ssh-instance.sh              # SSH into running instance
#   ./scripts/ssh-instance.sh logs         # Tail collection logs
#   ./scripts/ssh-instance.sh status       # Check Forge daemon status

set -e

REGION="${AWS_REGION:-us-west-2}"

# Find running instance
INSTANCE_INFO=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=instance-state-name,Values=running" \
              "Name=tag:Project,Values=mtg-rl" \
    --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress,Tags[?Key==`Name`].Value|[0]]' \
    --output text 2>/dev/null)

if [ -z "$INSTANCE_INFO" ] || [ "$INSTANCE_INFO" = "None" ]; then
    echo "No running MTG RL instance found."
    echo ""
    echo "To start an instance:"
    echo "  cd infrastructure && terraform apply -var='enable_training_instance=true'"
    exit 1
fi

INSTANCE_ID=$(echo "$INSTANCE_INFO" | awk '{print $1}')
PUBLIC_IP=$(echo "$INSTANCE_INFO" | awk '{print $2}')
INSTANCE_NAME=$(echo "$INSTANCE_INFO" | awk '{print $3}')

echo "Found instance: $INSTANCE_NAME"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo ""

# Generate temporary SSH key if needed
SSH_KEY_DIR="$HOME/.ssh/ec2-instance-connect"
mkdir -p "$SSH_KEY_DIR"
SSH_KEY="$SSH_KEY_DIR/temp_key"

if [ ! -f "$SSH_KEY" ]; then
    echo "Generating temporary SSH key..."
    ssh-keygen -t rsa -b 2048 -f "$SSH_KEY" -N "" -q
fi

# Push public key to instance
echo "Pushing SSH key to instance via EC2 Instance Connect..."
aws ec2-instance-connect send-ssh-public-key \
    --region "$REGION" \
    --instance-id "$INSTANCE_ID" \
    --instance-os-user ubuntu \
    --ssh-public-key "file://${SSH_KEY}.pub" \
    --output text > /dev/null

# Handle command argument
case "${1:-ssh}" in
    logs)
        echo "Tailing collection logs..."
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            "ubuntu@$PUBLIC_IP" "tail -f /home/ubuntu/collection.log 2>/dev/null || tail -f /var/log/imitation-setup.log"
        ;;
    status)
        echo "Checking status..."
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            "ubuntu@$PUBLIC_IP" "echo '=== Setup Log ===' && tail -20 /var/log/imitation-setup.log 2>/dev/null; echo ''; echo '=== Docker ===' && docker ps 2>/dev/null; echo ''; echo '=== Collection ===' && tail -10 /home/ubuntu/collection.log 2>/dev/null || echo 'Not started yet'"
        ;;
    ssh|*)
        echo "Connecting via SSH..."
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            "ubuntu@$PUBLIC_IP"
        ;;
esac
