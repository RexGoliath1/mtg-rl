#!/bin/bash
# MTG RL Deployment Script

set -e

REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/mtg-rl-inference"

echo "=== MTG RL Deployment ==="
echo "Region: $REGION"
echo "Account: $ACCOUNT_ID"
echo "ECR Repo: $ECR_REPO"
echo ""

# Build Docker image
echo "Building Docker image..."
docker build -t mtg-rl-inference .

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO

# Tag and push
echo "Pushing to ECR..."
docker tag mtg-rl-inference:latest $ECR_REPO:latest
docker push $ECR_REPO:latest

# Update ECS service
echo "Updating ECS service..."
aws ecs update-service \
  --cluster mtg-rl-cluster \
  --service mtg-rl-inference \
  --force-new-deployment \
  --region $REGION

echo ""
echo "=== Deployment Complete ==="
echo "Monitor with: aws ecs describe-services --cluster mtg-rl-cluster --services mtg-rl-inference"
