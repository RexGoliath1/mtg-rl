#!/bin/bash
set -e

# ============================================================================
# Deploy AI Training Data Collection to AWS Spot Instance (Docker-based)
# ============================================================================
# Launches a spot instance that:
# 1. Installs Docker and pulls daemon + collection images from GHCR
# 2. Runs docker compose to start Forge daemon + data collection
# 3. Uploads HDF5 training data + logs to S3
# 4. Auto-terminates when complete
#
# Prerequisites:
# - AWS CLI configured
# - GHCR images exist (CI pushes on main branch)
# - S3 bucket exists (from terraform)
#
# Usage:
#   ./scripts/deploy_data_collection_docker.sh --games 1000
#   ./scripts/deploy_data_collection_docker.sh --games 5000 --workers 8
#   ./scripts/deploy_data_collection_docker.sh --games 1000 --run-id fleet_123/instance_0 --s3-prefix fleet_123/instance_0
# ============================================================================

# Configuration
S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.2xlarge}"
NUM_GAMES="${NUM_GAMES:-1000}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-60}"
CUSTOM_RUN_ID=""
CUSTOM_S3_PREFIX=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --games)
            NUM_GAMES="$2"
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
        --run-id)
            CUSTOM_RUN_ID="$2"
            shift 2
            ;;
        --s3-prefix)
            CUSTOM_S3_PREFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--games N] [--workers N] [--timeout N] [--instance-type TYPE] [--run-id ID] [--s3-prefix PREFIX]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "AI TRAINING DATA COLLECTION - DOCKER DEPLOYMENT"
echo "============================================================"
echo "Instance Type: $INSTANCE_TYPE"
echo "Games: $NUM_GAMES"
echo "Workers: $WORKERS"
echo "Game Timeout: ${TIMEOUT}s"
echo "Region: $REGION"
echo ""

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS credentials not configured"
    exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS credentials OK (account: $ACCOUNT_ID)"

# Image registry: GHCR (public, no auth needed to pull)
IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/rexgoliath1}"
echo "Image Registry: $IMAGE_REGISTRY"

# Check S3 bucket exists
if ! aws s3 ls "s3://${S3_BUCKET}" &>/dev/null; then
    echo "ERROR: S3 bucket ${S3_BUCKET} does not exist"
    echo "Run: cd infrastructure && terraform apply"
    exit 1
fi
echo "S3 bucket OK"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$CUSTOM_RUN_ID" ]; then
    RUN_ID="$CUSTOM_RUN_ID"
else
    RUN_ID="collection_${TIMESTAMP}"
fi

# S3 prefix defaults to RUN_ID (fleet script overrides for per-instance subdirs)
S3_PREFIX="${CUSTOM_S3_PREFIX:-$RUN_ID}"

# Find AMI: prefer Deep Learning Base AMI with Single CUDA (35GB, has Docker pre-installed)
# This is lighter than the full GPU DLAMI (75GB) and works on CPU instances (c5)
# Falls back to plain Ubuntu 22.04 if DLAMI not available
echo ""
echo "Finding AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text) || true

if [ "$AMI_ID" != "None" ] && [ -n "$AMI_ID" ]; then
    AMI_NAME=$(aws ec2 describe-images --region "$REGION" --image-ids "$AMI_ID" --query 'Images[0].Name' --output text)
    echo "Using DLAMI: $AMI_ID"
    echo "  Name: $AMI_NAME"
    echo "  Docker pre-installed — faster bootstrap"
else
    echo "DLAMI not found, falling back to Ubuntu 22.04..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)

    if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
        echo "ERROR: Could not find any suitable AMI"
        exit 1
    fi
    echo "Using AMI: $AMI_ID (Ubuntu 22.04)"
fi

# Get default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[0].SubnetId' --output text)

# Get or create security group
SG_NAME="forge-collection-sg"
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Security group for Forge data collection" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
fi

# Ensure IAM instance profile exists (Terraform-managed, but check anyway)
INSTANCE_PROFILE="mtg-rl-training-profile"
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" &>/dev/null; then
    # Fall back to the manually-created profile
    INSTANCE_PROFILE="mtg-rl-training"
    if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" &>/dev/null; then
        echo "ERROR: IAM instance profile not found. Run terraform apply first."
        exit 1
    fi
fi
echo "IAM profile: $INSTANCE_PROFILE"

# Create user data script
USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
set -ex

exec > >(tee /var/log/data-collection.log) 2>&1
echo "=========================================="
echo "DATA COLLECTION SETUP (Docker)"
echo "=========================================="
date

# --- Instance metadata ---
INSTANCE_TYPE=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
AZ=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null || echo "unknown")
echo "Instance type: $INSTANCE_TYPE"
echo "Instance ID: $INSTANCE_ID"
echo "Availability zone: $AZ"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/Mem:/ {print $2}')"

# --- Early heartbeat: upload log before anything else can fail ---
# This ensures we always get diagnostic output in S3, even if Docker
# install or image pull fails. Uses pre-installed AWS CLI v1 (ubuntu AMI).
(aws s3 cp /var/log/data-collection.log \
    s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/live_log.txt 2>/dev/null || true) &

# --- Continuous log uploader (every 60s) ---
# Replaces the 5-min uploader that started too late to catch bootstrap failures.
(
    while true; do
        sleep 60
        aws s3 cp /var/log/data-collection.log \
            s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/live_log.txt 2>/dev/null || true
    done
) &
LOG_UPLOADER_PID=$!
echo "Continuous log uploader started (PID: $LOG_UPLOADER_PID, interval: 60s)"

# --- Install Docker and AWS CLI ---
# DLAMI has Docker + AWS CLI pre-installed; plain Ubuntu does not
echo ""
echo "[1/4] Installing Docker..."
if command -v docker &> /dev/null; then
    echo "  Docker already installed (DLAMI) — skipping"
    systemctl start docker 2>/dev/null || true
else
    apt-get update -qq
    apt-get install -y -qq docker.io docker-compose-v2 unzip > /dev/null
    systemctl start docker
    systemctl enable docker
fi

# Ensure docker-compose v2 plugin is available
if ! docker compose version &> /dev/null; then
    echo "  Installing docker-compose plugin..."
    apt-get update -qq
    apt-get install -y -qq docker-compose-v2 > /dev/null
fi

echo ""
echo "[2/4] Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install --update
    rm -rf aws awscliv2.zip
else
    echo "  AWS CLI already installed — skipping"
fi

# --- Pull Docker images ---
echo ""
echo "[3/4] Pulling Docker images from IMAGE_REGISTRY_PLACEHOLDER..."
# GHCR images are public — no login needed
docker pull IMAGE_REGISTRY_PLACEHOLDER/mtg-rl-daemon:latest
docker pull IMAGE_REGISTRY_PLACEHOLDER/mtg-rl-collection:latest

echo "Images pulled successfully"
docker images

# --- Start incremental HDF5 uploader (protects against spot interruption) ---
(
    LAST_UPLOADED=""
    while true; do
        sleep 120
        # Find the newest HDF5 file (checkpoint or final)
        NEWEST_H5=$(ls -t /home/ubuntu/training_data/*.h5 2>/dev/null | head -1)
        if [ -n "$NEWEST_H5" ] && [ "$NEWEST_H5" != "$LAST_UPLOADED" ]; then
            echo "[S3_SYNC] Incremental upload: $(basename $NEWEST_H5)"
            aws s3 cp "$NEWEST_H5" \
                "s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/$(basename $NEWEST_H5)" 2>/dev/null || true
            LAST_UPLOADED="$NEWEST_H5"
        fi
    done
) &
S3_SYNC_PID=$!
echo "Incremental S3 uploader started (PID: $S3_SYNC_PID, interval: 2min)"

# --- Run data collection via Docker Compose ---
echo ""
echo "[4/4] Starting data collection..."
echo "=========================================="
echo "STARTING DATA COLLECTION"
echo "=========================================="
echo "Games: NUM_GAMES_PLACEHOLDER"
echo "Workers: WORKERS_PLACEHOLDER"
echo "Timeout: TIMEOUT_PLACEHOLDER"
date

# Create a minimal docker-compose file inline
# (avoids needing to clone the repo just for the compose file)
mkdir -p /home/ubuntu/collection /home/ubuntu/daemon-logs
cat > /home/ubuntu/collection/docker-compose.yml << 'COMPOSE_EOF'
services:
  daemon:
    image: IMAGE_REGISTRY_PLACEHOLDER/mtg-rl-daemon:latest
    container_name: mtg-daemon
    ports:
      - "17171:17171"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "sh", "-c", "echo 'STATUS' | nc -w 5 localhost 17171 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 180s
    volumes:
      - /home/ubuntu/daemon-logs:/forge/logs
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

  collection:
    image: IMAGE_REGISTRY_PLACEHOLDER/mtg-rl-collection:latest
    container_name: mtg-collection
    depends_on:
      daemon:
        condition: service_healthy
    volumes:
      - /home/ubuntu/training_data:/app/training_data
    environment:
      - PYTHONUNBUFFERED=1
    command: >
      python -u scripts/collect_ai_training_data.py
      --games NUM_GAMES_PLACEHOLDER
      --workers WORKERS_PLACEHOLDER
      --host daemon
      --port 17171
      --output /app/training_data
      --save-interval 500
      --timeout TIMEOUT_PLACEHOLDER
COMPOSE_EOF

cd /home/ubuntu/collection

# --- Start daemon first, run warmup, then start collection ---
# Start daemon in background
docker compose up -d daemon
echo "Waiting for daemon health check..."

# Wait for daemon to be healthy (up to 5 minutes)
for i in $(seq 1 60); do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' mtg-daemon 2>/dev/null || echo "starting")
    if [ "$STATUS" = "healthy" ]; then
        echo "Daemon is healthy after ${i}x5s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Daemon failed to become healthy after 5 minutes"
        docker compose logs daemon
        docker compose down
        shutdown -h now
    fi
    sleep 5
done

# --- JVM Warmup Phase ---
# Run 3 throwaway games to trigger JIT compilation of hot code paths.
# This prevents the first real collection games from being slow.
echo ""
echo "=========================================="
echo "JVM WARMUP PHASE (3 throwaway games)"
echo "=========================================="
DECKS=$(docker exec mtg-daemon ls /forge/userdata/decks/constructed/ | grep '\.dck$' | head -2)
DECK1=$(echo "$DECKS" | head -1)
DECK2=$(echo "$DECKS" | tail -1)

if [ -n "$DECK1" ] && [ -n "$DECK2" ]; then
    for warmup_i in 1 2 3; do
        echo "  Warmup game $warmup_i/3..."
        echo "NEWGAME ${DECK1} ${DECK2} -i -c 30" | nc -w 60 localhost 17171 > /dev/null 2>&1 || true
        sleep 2
    done
    echo "Warmup complete. JIT should have compiled hot paths."
else
    echo "WARN: Could not find decks for warmup, skipping."
fi

# --- Start JVM Monitor ---
# Run monitoring in background inside the daemon container.
# Logs every 60s to /forge/logs/jvm_stats.log (volume-mounted to host).
docker exec -d mtg-daemon /forge/jvm_monitor.sh 60 /forge/logs/jvm_stats.log
echo "JVM monitor started (60s interval)"

# --- Start collection ---
echo ""
echo "Starting collection container..."
docker compose up --abort-on-container-exit --exit-code-from collection 2>&1 | tee /var/log/docker-compose.log
COMPOSE_EXIT=$?

echo "=========================================="
echo "COLLECTION COMPLETE (exit code: $COMPOSE_EXIT)"
echo "=========================================="
date

# Show container logs for debugging
docker compose logs daemon > /var/log/forge-daemon.log 2>&1 || true

# Stop background uploaders
kill $LOG_UPLOADER_PID 2>/dev/null || true
kill $S3_SYNC_PID 2>/dev/null || true

# --- Upload results to S3 ---
echo "Uploading results to S3..."
aws s3 sync /home/ubuntu/training_data/ \
    s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/ \
    --exclude "*.tex"

# Upload logs
aws s3 cp /var/log/data-collection.log \
    s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/collection_log.txt
aws s3 cp /var/log/forge-daemon.log \
    s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/forge_daemon.log 2>/dev/null || true
aws s3 cp /var/log/docker-compose.log \
    s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/docker_compose.log 2>/dev/null || true

# Upload JVM performance logs (GC log + monitor stats)
aws s3 sync /home/ubuntu/daemon-logs/ \
    s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/jvm_logs/ 2>/dev/null || true

echo "Results uploaded to s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/"

# Signal completion
echo '{"status":"complete","timestamp":"TIMESTAMP_PLACEHOLDER","games":NUM_GAMES_PLACEHOLDER,"method":"docker","run_id":"RUN_ID_PLACEHOLDER"}' | \
    aws s3 cp - s3://BUCKET_PLACEHOLDER/imitation_data/S3_PREFIX_PLACEHOLDER/collection_complete.json

# Stop all containers
docker compose down 2>/dev/null || true

echo "Shutting down..."
shutdown -h now
USERDATA
)

# Replace placeholders
USER_DATA="${USER_DATA//BUCKET_PLACEHOLDER/$S3_BUCKET}"
USER_DATA="${USER_DATA//IMAGE_REGISTRY_PLACEHOLDER/$IMAGE_REGISTRY}"
USER_DATA="${USER_DATA//S3_PREFIX_PLACEHOLDER/$S3_PREFIX}"
USER_DATA="${USER_DATA//RUN_ID_PLACEHOLDER/$RUN_ID}"
USER_DATA="${USER_DATA//TIMESTAMP_PLACEHOLDER/$TIMESTAMP}"
USER_DATA="${USER_DATA//NUM_GAMES_PLACEHOLDER/$NUM_GAMES}"
USER_DATA="${USER_DATA//WORKERS_PLACEHOLDER/$WORKERS}"
USER_DATA="${USER_DATA//TIMEOUT_PLACEHOLDER/$TIMEOUT}"

# Encode user data
USER_DATA_B64=$(echo "$USER_DATA" | base64)

# Launch spot instance
# Volume: 50GB gp3 — DLAMI base is ~35GB, need headroom for Docker images + data
echo ""
echo "Launching spot instance..."
LAUNCH_SPEC=$(cat << EOF
{
    "ImageId": "$AMI_ID",
    "InstanceType": "$INSTANCE_TYPE",
    "SecurityGroupIds": ["$SG_ID"],
    "SubnetId": "$SUBNET_ID",
    "IamInstanceProfile": {"Name": "$INSTANCE_PROFILE"},
    "UserData": "$USER_DATA_B64",
    "BlockDeviceMappings": [{
        "DeviceName": "/dev/sda1",
        "Ebs": {"VolumeSize": 50, "VolumeType": "gp3"}
    }]
}
EOF
)

# Try spot instance first
SPOT_RESPONSE=$(aws ec2 request-spot-instances \
    --region "$REGION" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "$LAUNCH_SPEC" \
    --query 'SpotInstanceRequests[0]' \
    --output json 2>/dev/null || echo "{}")

SPOT_REQUEST_ID=$(echo "$SPOT_RESPONSE" | jq -r '.SpotInstanceRequestId // empty')
INSTANCE_ID=""

if [ -n "$SPOT_REQUEST_ID" ]; then
    echo "Spot request: $SPOT_REQUEST_ID"
    echo "Waiting for spot fulfillment..."

    for i in {1..30}; do
        INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
            --region "$REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID" \
            --query 'SpotInstanceRequests[0].InstanceId' \
            --output text 2>/dev/null || echo "None")

        if [ "$INSTANCE_ID" != "None" ] && [ -n "$INSTANCE_ID" ]; then
            break
        fi
        echo "  Waiting... ($i/30)"
        sleep 10
    done

    if [ "$INSTANCE_ID" == "None" ] || [ -z "$INSTANCE_ID" ]; then
        echo "Spot not fulfilled, cancelling..."
        aws ec2 cancel-spot-instance-requests --region "$REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID" 2>/dev/null || true
        INSTANCE_ID=""
    fi
fi

# Fall back to on-demand
if [ -z "$INSTANCE_ID" ]; then
    echo "Using on-demand instance..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --security-group-ids "$SG_ID" \
        --subnet-id "$SUBNET_ID" \
        --iam-instance-profile "Name=$INSTANCE_PROFILE" \
        --user-data "$USER_DATA" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":50,\"VolumeType\":\"gp3\"}}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
fi

# Tag instance
aws ec2 create-tags --region "$REGION" \
    --resources "$INSTANCE_ID" \
    --tags "Key=Name,Value=data-collection-docker-$TIMESTAMP" "Key=Project,Value=mtg-rl" "Key=Task,Value=data-collection"

# Wait for instance to start
echo "Waiting for instance..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "============================================================"
echo "DATA COLLECTION INSTANCE LAUNCHED (Docker)"
echo "============================================================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo "Games:       $NUM_GAMES"
echo "Workers:     $WORKERS"
echo "Run ID:      $RUN_ID"
echo "S3 Prefix:   $S3_PREFIX"
echo ""
echo "Monitor with:"
echo "  # Check if complete:"
echo "  aws s3 ls s3://${S3_BUCKET}/imitation_data/${S3_PREFIX}/collection_complete.json"
echo ""
echo "  # View live log (updates every 5 min):"
echo "  aws s3 cp s3://${S3_BUCKET}/imitation_data/${S3_PREFIX}/live_log.txt - | tail -50"
echo ""
echo "  # Check instance state:"
echo "  aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text"
echo ""
echo "  # Download results when complete:"
echo "  aws s3 sync s3://${S3_BUCKET}/imitation_data/${S3_PREFIX}/ training_data/${TIMESTAMP}/"
echo ""
echo "Instance auto-terminates after collection."
echo "============================================================"
