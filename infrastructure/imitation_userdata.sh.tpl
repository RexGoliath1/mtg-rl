#!/bin/bash
# =============================================================================
# MTG RL Imitation Learning - Docker-based Data Collection on AWS
# =============================================================================
# Pulls daemon + collection images from ECR and runs via docker compose.
# No JDK, Maven, Python deps, or Xvfb needed on the instance.
# =============================================================================
set -ex
exec > >(tee /var/log/imitation-setup.log) 2>&1

echo "============================================================"
echo "MTG RL Imitation Learning Data Collection (Docker)"
echo "Started at: $(date)"
echo "============================================================"

# Configuration from Terraform
S3_BUCKET="${s3_bucket}"
ECR_REPO="${ecr_repo}"
NUM_GAMES="${num_games}"
WORKERS="${workers}"
REGION="us-east-1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="collection_$${TIMESTAMP}"

echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  ECR Repo: $ECR_REPO"
echo "  Games: $NUM_GAMES"
echo "  Workers: $WORKERS"
echo "  Run ID: $RUN_ID"

# --- Instance metadata ---
INSTANCE_TYPE=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
echo "Instance type: $INSTANCE_TYPE"
echo "Instance ID: $INSTANCE_ID"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/Mem:/ {print $2}')"

# --- Install Docker ---
echo ""
echo "[1/4] Installing Docker..."
apt-get update -y -qq
apt-get install -y -qq docker.io docker-compose-v2 unzip > /dev/null
systemctl start docker
systemctl enable docker

# Ensure SSM agent is running
snap install amazon-ssm-agent --classic 2>/dev/null || true
systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true

# --- Install AWS CLI ---
echo ""
echo "[2/4] Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install --update
    rm -rf awscliv2.zip aws/
fi

# --- Pull Docker images from ECR ---
echo ""
echo "[3/4] Pulling Docker images from ECR..."
# ECR_REPO is the full registry URL (e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/mtg-rl-training)
# We need the registry base URL for login
ECR_REGISTRY=$(echo "$ECR_REPO" | cut -d'/' -f1)
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

# Pull daemon and collection images
docker pull $ECR_REGISTRY/mtg-rl-daemon:latest
docker pull $ECR_REGISTRY/mtg-rl-collection:latest
echo "Images pulled successfully"
docker images

# --- Start background log uploader ---
(
    while true; do
        sleep 300
        aws s3 cp /var/log/imitation-setup.log \
            "s3://$S3_BUCKET/imitation_data/$RUN_ID/live_log.txt" 2>/dev/null || true
    done
) &
LOG_UPLOADER_PID=$!

# --- Run data collection via Docker Compose ---
echo ""
echo "[4/4] Starting data collection..."
mkdir -p /home/ubuntu/collection /home/ubuntu/training_data

cat > /home/ubuntu/collection/docker-compose.yml << COMPOSE_EOF
services:
  daemon:
    image: $ECR_REGISTRY/mtg-rl-daemon:latest
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
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  collection:
    image: $ECR_REGISTRY/mtg-rl-collection:latest
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
      --games $NUM_GAMES
      --workers $WORKERS
      --host daemon
      --port 17171
      --output /app/training_data
      --save-interval 500
      --timeout 60
COMPOSE_EOF

cd /home/ubuntu/collection
echo "=========================================="
echo "STARTING DATA COLLECTION"
echo "=========================================="
echo "Games: $NUM_GAMES"
echo "Workers: $WORKERS"
date

docker compose up --abort-on-container-exit --exit-code-from collection 2>&1 | tee /var/log/docker-compose.log
COMPOSE_EXIT=$?

echo "=========================================="
echo "COLLECTION COMPLETE (exit code: $COMPOSE_EXIT)"
echo "=========================================="
date

# Save daemon logs
docker compose logs daemon > /var/log/forge-daemon.log 2>&1 || true

# Stop log uploader
kill $LOG_UPLOADER_PID 2>/dev/null || true

# --- Upload results to S3 ---
echo "Uploading results to S3..."
aws s3 sync /home/ubuntu/training_data/ \
    "s3://$S3_BUCKET/imitation_data/$RUN_ID/" \
    --exclude "*.tex"

# Upload logs
aws s3 cp /var/log/imitation-setup.log \
    "s3://$S3_BUCKET/imitation_data/$RUN_ID/collection_log.txt"
aws s3 cp /var/log/forge-daemon.log \
    "s3://$S3_BUCKET/imitation_data/$RUN_ID/forge_daemon.log" 2>/dev/null || true
aws s3 cp /var/log/docker-compose.log \
    "s3://$S3_BUCKET/imitation_data/$RUN_ID/docker_compose.log" 2>/dev/null || true

echo "Results uploaded to s3://$S3_BUCKET/imitation_data/$RUN_ID/"

# Signal completion
echo "{\"status\":\"complete\",\"timestamp\":\"$TIMESTAMP\",\"games\":$NUM_GAMES,\"method\":\"docker\"}" | \
    aws s3 cp - "s3://$S3_BUCKET/imitation_data/$RUN_ID/collection_complete.json"

# Stop containers
docker compose down 2>/dev/null || true

echo "Collection complete at $(date)"

# Auto-shutdown
if [ "${auto_shutdown}" = "true" ]; then
    echo "Auto-shutdown enabled. Shutting down..."
    shutdown -h now
fi
