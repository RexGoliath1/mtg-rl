#!/bin/bash
# =============================================================================
# MTG RL Imitation Learning Model Training (Docker-based)
# =============================================================================
# Pulls mtg-rl-training image from GHCR, downloads HDF5 data from S3,
# runs train_imitation.py, uploads checkpoints + logs to S3.
# =============================================================================
set -ex

exec > >(tee /var/log/imitation-train.log) 2>&1

echo "============================================================"
echo "MTG RL IMITATION TRAINING (Docker)"
echo "Started at: $(date)"
echo "============================================================"

# --- Configuration from Terraform ---
S3_BUCKET="${s3_bucket}"
IMAGE_REGISTRY="${image_registry}"
EPOCHS="${epochs}"
BATCH_SIZE="${batch_size}"
HIDDEN_DIM="${hidden_dim}"
LEARNING_RATE="${learning_rate}"
AUTO_SHUTDOWN="${auto_shutdown}"

echo "Configuration:"
echo "  S3 Bucket:     $S3_BUCKET"
echo "  Image Registry: $IMAGE_REGISTRY"
echo "  Epochs:        $EPOCHS"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Hidden Dim:    $HIDDEN_DIM"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Auto Shutdown: $AUTO_SHUTDOWN"

# --- Instance metadata ---
INSTANCE_TYPE=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
AZ=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null || echo "unknown")
echo "Instance type: $INSTANCE_TYPE"
echo "Instance ID:   $INSTANCE_ID"
echo "AZ:            $AZ"
echo "CPU cores:     $(nproc)"
echo "Memory:        $(free -h | awk '/Mem:/ {print $2}')"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
else
    echo "GPU: none detected"
fi

# --- [1/6] Install Docker and AWS CLI ---
echo ""
echo "[1/6] Installing Docker..."
apt-get update -qq
apt-get install -y -qq docker.io unzip > /dev/null
systemctl start docker
systemctl enable docker

# Install NVIDIA container toolkit if GPU present
if command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA container toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit > /dev/null
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "NVIDIA container toolkit installed"
fi

echo ""
echo "[2/6] Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install --update
    rm -rf aws awscliv2.zip
fi

# --- [3/6] Pull training image from GHCR ---
echo ""
echo "[3/6] Pulling training image from GHCR..."
# GHCR images are public — no login needed
docker pull "$IMAGE_REGISTRY/mtg-rl-training:latest"
echo "Image pulled successfully"
docker images

# --- [4/6] Download HDF5 training data from S3 ---
echo ""
echo "[4/6] Downloading training data from S3..."
mkdir -p /home/ubuntu/training_data
aws s3 sync "s3://$S3_BUCKET/imitation_data/" /home/ubuntu/training_data/ \
    --exclude "*" --include "*.h5"

echo "Downloaded files:"
ls -lh /home/ubuntu/training_data/
H5_COUNT=$(find /home/ubuntu/training_data -name "*.h5" | wc -l)
echo "Total HDF5 files: $H5_COUNT"

if [ "$H5_COUNT" -eq 0 ]; then
    echo "ERROR: No HDF5 files found in s3://$S3_BUCKET/imitation_data/"
    echo '{"status":"failed","reason":"no_data","timestamp":"'$(date -Iseconds)'"}' | \
        aws s3 cp - "s3://$S3_BUCKET/training_runs/imitation_train_$(date +%Y%m%d_%H%M%S)/training_complete.json"
    shutdown -h now
    exit 1
fi

# --- [5/6] Create output directories ---
echo ""
echo "[5/6] Preparing output directories..."
mkdir -p /home/ubuntu/checkpoints /home/ubuntu/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="imitation_train_$TIMESTAMP"

# --- Background log uploader (every 5 min) ---
(
    while true; do
        sleep 300
        aws s3 cp /var/log/imitation-train.log \
            "s3://$S3_BUCKET/training_runs/$RUN_ID/live_log.txt" 2>/dev/null || true
    done
) &
LOG_UPLOADER_PID=$!
echo "Background log uploader started (PID: $LOG_UPLOADER_PID)"

# --- [6/6] Run training in Docker container ---
echo ""
echo "[6/6] Starting training..."
echo "============================================================"
echo "TRAINING: $EPOCHS epochs, batch=$BATCH_SIZE, hidden=$HIDDEN_DIM, lr=$LEARNING_RATE"
echo "============================================================"

# Build Docker run flags
DOCKER_FLAGS="-v /home/ubuntu/training_data:/data -v /home/ubuntu/checkpoints:/checkpoints"

# Add GPU support if available
if command -v nvidia-smi &> /dev/null; then
    DOCKER_FLAGS="$DOCKER_FLAGS --gpus all"
    echo "GPU mode enabled"
else
    echo "CPU mode (no GPU detected)"
fi

# Fetch W&B API key from Secrets Manager (optional, non-fatal)
WANDB_KEY=""
WANDB_KEY=$(aws secretsmanager get-secret-value \
    --secret-id mtg-rl/wandb-api-key \
    --region "$REGION" \
    --query SecretString \
    --output text 2>/dev/null || echo "")

WANDB_FLAGS=""
if [ -n "$WANDB_KEY" ]; then
    WANDB_FLAGS="-e WANDB_API_KEY=$WANDB_KEY -e WANDB_PROJECT=mtg-rl-imitation"
    echo "W&B tracking enabled"
else
    WANDB_FLAGS="-e WANDB_MODE=disabled"
    echo "W&B tracking disabled (no secret found)"
fi

# Run training
docker run --rm \
    $DOCKER_FLAGS \
    $WANDB_FLAGS \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE_REGISTRY/mtg-rl-training:latest" \
    python /app/scripts/train_imitation.py \
        --data-dir /data \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --hidden-dim "$HIDDEN_DIM" \
        --lr "$LEARNING_RATE" \
        --output /checkpoints/imitation_policy.pt

TRAINING_EXIT=$?

echo "============================================================"
if [ $TRAINING_EXIT -eq 0 ]; then
    echo "TRAINING COMPLETED SUCCESSFULLY"
else
    echo "TRAINING FAILED (exit code: $TRAINING_EXIT)"
fi
echo "Finished at: $(date)"
echo "============================================================"

# --- Upload results to S3 ---
echo "Uploading results to S3..."
aws s3 sync /home/ubuntu/checkpoints/ "s3://$S3_BUCKET/training_runs/$RUN_ID/checkpoints/"
aws s3 cp /var/log/imitation-train.log "s3://$S3_BUCKET/training_runs/$RUN_ID/training_log.txt"

# Also copy best model to a stable path for easy access
if [ -f /home/ubuntu/checkpoints/imitation_policy.pt ]; then
    aws s3 cp /home/ubuntu/checkpoints/imitation_policy.pt \
        "s3://$S3_BUCKET/models/imitation_policy_latest.pt"
    echo "Best model uploaded to s3://$S3_BUCKET/models/imitation_policy_latest.pt"
fi
if [ -f /home/ubuntu/checkpoints/imitation_policy.json ]; then
    aws s3 cp /home/ubuntu/checkpoints/imitation_policy.json \
        "s3://$S3_BUCKET/training_runs/$RUN_ID/training_summary.json"
fi

# Signal completion
echo "{\"status\":\"$([ $TRAINING_EXIT -eq 0 ] && echo success || echo failed)\",\"timestamp\":\"$(date -Iseconds)\",\"epochs\":$EPOCHS,\"exit_code\":$TRAINING_EXIT,\"method\":\"docker\",\"run_id\":\"$RUN_ID\"}" | \
    aws s3 cp - "s3://$S3_BUCKET/training_runs/$RUN_ID/training_complete.json"

echo "Results uploaded to s3://$S3_BUCKET/training_runs/$RUN_ID/"

# Stop background log uploader
kill $LOG_UPLOADER_PID 2>/dev/null || true

# Auto-shutdown
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "Auto-shutdown enabled. Shutting down in 60 seconds..."
    sleep 60
    shutdown -h now
fi
