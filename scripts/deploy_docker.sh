#!/bin/bash
set -e

# ============================================================================
# Docker-based Forge Deployment
# ============================================================================
# Builds Docker images, pushes to ECR, and runs on EC2.
# Much faster than tar-based deployment.
#
# Usage:
#   ./scripts/deploy_docker.sh [--build] [--push] [--deploy] [--all]
# ============================================================================

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text 2>/dev/null)
ECR_REPO="mtg-rl"
S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
NUM_GAMES="${NUM_GAMES:-100}"

# Image tags
DAEMON_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:daemon-latest"
TRAINING_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:training-latest"

# Parse arguments
BUILD=false
PUSH=false
DEPLOY=false
LOCAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --deploy)
            DEPLOY=true
            shift
            ;;
        --all)
            BUILD=true
            PUSH=true
            DEPLOY=true
            shift
            ;;
        --local)
            LOCAL=true
            shift
            ;;
        --games)
            NUM_GAMES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--build] [--push] [--deploy] [--all] [--local] [--games N]"
            exit 1
            ;;
    esac
done

# If no action specified, show help
if ! $BUILD && ! $PUSH && ! $DEPLOY && ! $LOCAL; then
    echo "Docker-based Forge Deployment"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build    Build Docker images locally"
    echo "  --push     Push images to ECR"
    echo "  --deploy   Deploy to EC2"
    echo "  --all      Build + Push + Deploy"
    echo "  --local    Run locally with docker-compose"
    echo "  --games N  Number of games to profile (default: 100)"
    echo ""
    echo "Examples:"
    echo "  $0 --local                    # Test locally"
    echo "  $0 --build                    # Build images"
    echo "  $0 --all --games 500          # Full deployment with 500 games"
    exit 0
fi

cd "$(dirname "$0")/.."

# ============================================================================
# Build Docker images
# ============================================================================
if $BUILD; then
    echo "============================================================"
    echo "Building Docker images..."
    echo "============================================================"

    # Check if forge-repo exists
    if [ ! -d "forge-repo" ]; then
        echo "ERROR: forge-repo not found"
        echo "Clone with: git clone git@github.com:RexGoliath1/forge.git forge-repo"
        exit 1
    fi

    echo "Building daemon image..."
    docker build -t mtg-daemon:latest -f infrastructure/docker/Dockerfile.daemon .

    echo "Building training image..."
    docker build -t mtg-training:latest -f infrastructure/docker/Dockerfile.training .

    echo "Images built successfully!"
    docker images | grep mtg-
fi

# ============================================================================
# Run locally with docker-compose
# ============================================================================
if $LOCAL; then
    echo "============================================================"
    echo "Running locally with docker-compose..."
    echo "============================================================"

    # Start daemon
    echo "Starting Forge daemon..."
    docker-compose -f infrastructure/docker-compose.yml up -d daemon

    # Wait for daemon to be healthy
    echo "Waiting for daemon to be ready..."
    for i in {1..60}; do
        if docker-compose -f infrastructure/docker-compose.yml exec -T daemon sh -c "echo 'STATUS' | nc -w 5 localhost 17171" 2>/dev/null | grep -q "OK"; then
            echo "Daemon is ready!"
            break
        fi
        echo "Waiting... ($i/60)"
        sleep 5
    done

    # Run profiling
    echo "Running profiling with ${NUM_GAMES} games..."
    docker-compose -f infrastructure/docker-compose.yml run --rm training python /app/scripts/profile_forge_games.py \
        --host daemon --port 17171 --num-games ${NUM_GAMES} --verbose

    # Show logs
    docker-compose -f infrastructure/docker-compose.yml logs daemon

    # Cleanup
    echo "Stopping containers..."
    docker-compose -f infrastructure/docker-compose.yml down
fi

# ============================================================================
# Push to ECR
# ============================================================================
if $PUSH; then
    echo "============================================================"
    echo "Pushing images to ECR..."
    echo "============================================================"

    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

    # Create repository if it doesn't exist
    aws ecr describe-repositories --repository-names ${ECR_REPO} --region ${AWS_REGION} 2>/dev/null || \
        aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}

    # Tag and push
    docker tag mtg-daemon:latest ${DAEMON_IMAGE}
    docker push ${DAEMON_IMAGE}

    docker tag mtg-training:latest ${TRAINING_IMAGE}
    docker push ${TRAINING_IMAGE}

    echo "Images pushed to ECR!"
fi

# ============================================================================
# Deploy to EC2
# ============================================================================
if $DEPLOY; then
    echo "============================================================"
    echo "Deploying to EC2..."
    echo "============================================================"

    # Get latest Deep Learning AMI
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text \
        --region ${AWS_REGION})

    echo "Using AMI: ${AMI_ID}"
    echo "Instance type: ${INSTANCE_TYPE}"

    # User data script
    USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/user-data.log) 2>&1

echo "Starting deployment..."
cd /home/ubuntu

# Configure AWS CLI
aws configure set region AWS_REGION_PLACEHOLDER

# Login to ECR
aws ecr get-login-password --region AWS_REGION_PLACEHOLDER | \
    docker login --username AWS --password-stdin AWS_ACCOUNT_PLACEHOLDER.dkr.ecr.AWS_REGION_PLACEHOLDER.amazonaws.com

# Pull images
echo "Pulling images..."
docker pull DAEMON_IMAGE_PLACEHOLDER
docker pull TRAINING_IMAGE_PLACEHOLDER

# Create docker-compose.yml
cat > docker-compose.yml <<'COMPOSE'
version: '3.8'
services:
  daemon:
    image: DAEMON_IMAGE_PLACEHOLDER
    container_name: mtg-daemon
    ports:
      - "17171:17171"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 3G

  training:
    image: TRAINING_IMAGE_PLACEHOLDER
    container_name: mtg-training
    depends_on:
      - daemon
    environment:
      - DAEMON_HOST=daemon
      - DAEMON_PORT=17171
      - S3_BUCKET=S3_BUCKET_PLACEHOLDER
    command: >
      sh -c "sleep 120 && python /app/scripts/profile_forge_games.py
             --host daemon --port 17171 --num-games NUM_GAMES_PLACEHOLDER --verbose
             --output /tmp/results.json &&
             aws s3 cp /tmp/results.json s3://S3_BUCKET_PLACEHOLDER/forge-test/profiling_results.json"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
COMPOSE

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for completion
echo "Waiting for profiling to complete..."
for i in {1..60}; do
    if aws s3 ls s3://S3_BUCKET_PLACEHOLDER/forge-test/profiling_results.json 2>/dev/null; then
        echo "Results uploaded!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 60
done

echo "Done! Shutting down..."
shutdown -h now
USERDATA
)

    # Replace placeholders
    USER_DATA=$(echo "$USER_DATA" | \
        sed "s|AWS_REGION_PLACEHOLDER|${AWS_REGION}|g" | \
        sed "s|AWS_ACCOUNT_PLACEHOLDER|${AWS_ACCOUNT_ID}|g" | \
        sed "s|DAEMON_IMAGE_PLACEHOLDER|${DAEMON_IMAGE}|g" | \
        sed "s|TRAINING_IMAGE_PLACEHOLDER|${TRAINING_IMAGE}|g" | \
        sed "s|S3_BUCKET_PLACEHOLDER|${S3_BUCKET}|g" | \
        sed "s|NUM_GAMES_PLACEHOLDER|${NUM_GAMES}|g")

    # Encode user data
    USER_DATA_B64=$(echo "$USER_DATA" | base64)

    # Get IAM instance profile
    INSTANCE_PROFILE=$(aws iam list-instance-profiles --query 'InstanceProfiles[?contains(InstanceProfileName, `mtg`) || contains(InstanceProfileName, `training`)].InstanceProfileName' --output text | head -1)

    if [ -z "$INSTANCE_PROFILE" ]; then
        echo "WARNING: No IAM instance profile found. Creating one..."
        # Use existing role if available
        INSTANCE_PROFILE="mtg-training-profile"
    fi

    # Get default VPC and subnet
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text --region ${AWS_REGION})
    SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" --query 'Subnets[0].SubnetId' --output text --region ${AWS_REGION})

    # Create security group if needed
    SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=mtg-forge-test" --query 'SecurityGroups[0].GroupId' --output text --region ${AWS_REGION} 2>/dev/null)

    if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
        echo "Creating security group..."
        SG_ID=$(aws ec2 create-security-group \
            --group-name mtg-forge-test \
            --description "Security group for Forge testing" \
            --vpc-id ${VPC_ID} \
            --query 'GroupId' \
            --output text \
            --region ${AWS_REGION})

        # Allow SSH (optional)
        aws ec2 authorize-security-group-ingress \
            --group-id ${SG_ID} \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0 \
            --region ${AWS_REGION} 2>/dev/null || true
    fi

    # Launch instance
    echo "Launching EC2 instance..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id ${AMI_ID} \
        --instance-type ${INSTANCE_TYPE} \
        --iam-instance-profile Name=${INSTANCE_PROFILE} \
        --security-group-ids ${SG_ID} \
        --subnet-id ${SUBNET_ID} \
        --user-data "${USER_DATA_B64}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mtg-forge-test},{Key=Project,Value=mtg-rl}]" \
        --instance-initiated-shutdown-behavior terminate \
        --query 'Instances[0].InstanceId' \
        --output text \
        --region ${AWS_REGION})

    echo "============================================================"
    echo "Instance launched: ${INSTANCE_ID}"
    echo "============================================================"
    echo ""
    echo "Monitor with:"
    echo "  aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --query 'Reservations[0].Instances[0].State.Name'"
    echo ""
    echo "Check results:"
    echo "  aws s3 ls s3://${S3_BUCKET}/forge-test/"
    echo "  aws s3 cp s3://${S3_BUCKET}/forge-test/profiling_results.json -"
    echo ""
    echo "Console output:"
    echo "  aws ec2 get-console-output --instance-id ${INSTANCE_ID} --query 'Output' --output text | tail -50"
fi

echo ""
echo "Done!"
