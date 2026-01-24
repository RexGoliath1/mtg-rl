#!/bin/bash
# =============================================================================
# Deploy Training to AWS
# =============================================================================
#
# This script deploys the draft model training to AWS using spot instances.
#
# Prerequisites:
#   - AWS CLI configured
#   - Terraform installed
#   - Docker (optional, for container builds)
#
# Usage:
#   ./scripts/deploy_training.sh [options]
#
# Options:
#   --instance-type    EC2 instance type (default: g4dn.xlarge)
#   --spot-price       Max spot price (default: 0.20)
#   --s3-bucket        S3 bucket for checkpoints (created if not exists)
#   --dry-run          Show what would be deployed without deploying
#
# Example:
#   ./scripts/deploy_training.sh --instance-type g4dn.xlarge --s3-bucket my-bucket
# =============================================================================

set -e

# Defaults
INSTANCE_TYPE="g4dn.xlarge"
SPOT_PRICE="0.20"
S3_BUCKET=""
DRY_RUN=false
REGION="${AWS_REGION:-us-west-2}"
PROJECT_NAME="mtg-rl"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --spot-price)
            SPOT_PRICE="$2"
            shift 2
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "MTG RL Training Deployment"
echo "============================================================"
echo "Region: $REGION"
echo "Instance Type: $INSTANCE_TYPE"
echo "Max Spot Price: \$$SPOT_PRICE"
echo "S3 Bucket: ${S3_BUCKET:-'(will be created)'}"
echo "Dry Run: $DRY_RUN"
echo "============================================================"

# Check AWS credentials
echo ""
echo "Checking AWS credentials..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: $ACCOUNT_ID"

# Get or create S3 bucket name
if [ -z "$S3_BUCKET" ]; then
    S3_BUCKET="${PROJECT_NAME}-checkpoints-${ACCOUNT_ID}"
fi

# Check if bucket exists, create if not
echo ""
echo "Checking S3 bucket: $S3_BUCKET"
if aws s3 ls "s3://$S3_BUCKET" 2>/dev/null; then
    echo "  Bucket exists"
else
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would create bucket: $S3_BUCKET"
    else
        echo "  Creating bucket..."
        if [ "$REGION" = "us-east-1" ]; then
            aws s3 mb "s3://$S3_BUCKET"
        else
            aws s3 mb "s3://$S3_BUCKET" --region "$REGION"
        fi
        echo "  Bucket created"
    fi
fi

# Upload training data to S3
echo ""
echo "Syncing training data to S3..."
if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Would sync data/17lands/ to s3://$S3_BUCKET/data/17lands/"
else
    aws s3 sync data/17lands/ "s3://$S3_BUCKET/data/17lands/" --exclude "*.json"
    echo "  Data synced"
fi

# Upload training code
echo ""
echo "Uploading training code..."
if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Would upload training scripts"
else
    # Create a tarball of necessary files
    tar -czf /tmp/mtg-training-code.tar.gz \
        train_draft.py \
        train_draft_cloud.py \
        data_loader_17lands.py \
        requirements.txt \
        2>/dev/null || true

    aws s3 cp /tmp/mtg-training-code.tar.gz "s3://$S3_BUCKET/code/training-code.tar.gz"
    rm /tmp/mtg-training-code.tar.gz
    echo "  Code uploaded"
fi

# Deploy infrastructure with Terraform
echo ""
echo "Deploying infrastructure with Terraform..."
cd infrastructure

if [ "$DRY_RUN" = true ]; then
    terraform plan \
        -var="enable_training_instance=true" \
        -var="training_instance_type=$INSTANCE_TYPE" \
        -var="spot_max_price=$SPOT_PRICE" \
        -var="use_spot_instances=true"
else
    terraform apply -auto-approve \
        -var="enable_training_instance=true" \
        -var="training_instance_type=$INSTANCE_TYPE" \
        -var="spot_max_price=$SPOT_PRICE" \
        -var="use_spot_instances=true"
fi

cd ..

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN COMPLETE"
    echo "Run without --dry-run to actually deploy"
else
    echo "DEPLOYMENT COMPLETE"
    echo ""
    echo "S3 Bucket: s3://$S3_BUCKET"
    echo ""
    echo "To monitor training:"
    echo "  # View logs"
    echo "  aws s3 ls s3://$S3_BUCKET/checkpoints/"
    echo ""
    echo "  # Download latest checkpoint"
    echo "  aws s3 cp s3://$S3_BUCKET/checkpoints/latest.pt checkpoints/"
    echo ""
    echo "  # SSH to instance (get IP from EC2 console)"
    echo "  ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>"
    echo ""
    echo "  # View TensorBoard (after SSH)"
    echo "  ssh -L 6006:localhost:6006 ubuntu@<instance-ip>"
    echo "  # Then open http://localhost:6006"
fi
echo "============================================================"
