#!/bin/bash
# =============================================================================
# EC2 Training Instance User Data Script
# =============================================================================
# This script runs on EC2 instance startup to:
# 1. Install dependencies
# 2. Download training code and data from S3
# 3. Start training with monitoring
# =============================================================================

set -e
exec > >(tee /var/log/training-setup.log) 2>&1

echo "============================================================"
echo "MTG RL Training Instance Setup"
echo "Started at: $(date)"
echo "============================================================"

# Variables (replaced by Terraform)
S3_BUCKET="${s3_bucket}"
SETS="${sets}"
EPOCHS="${epochs}"
BATCH_SIZE="${batch_size}"
MAX_SAMPLES="${max_samples}"

# Install system dependencies
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -y
apt-get install -y python3-pip python3-venv awscli htop nvtop

# Setup working directory
echo ""
echo "[2/6] Setting up working directory..."
mkdir -p /home/ubuntu/mtg-rl
cd /home/ubuntu/mtg-rl

# Download code from S3
echo ""
echo "[3/6] Downloading training code..."
aws s3 cp "s3://$S3_BUCKET/code/training-code.tar.gz" .
tar -xzf training-code.tar.gz
rm training-code.tar.gz

# Download training data
echo ""
echo "[4/6] Downloading training data..."
mkdir -p data/17lands
aws s3 sync "s3://$S3_BUCKET/data/17lands/" data/17lands/

# Setup Python environment
echo ""
echo "[5/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard boto3 requests numpy pandas

# Create training wrapper script
echo ""
echo "[6/6] Creating training wrapper..."
cat > run_training.sh << 'TRAINING_SCRIPT'
#!/bin/bash
cd /home/ubuntu/mtg-rl
source venv/bin/activate

# Start TensorBoard in background
tensorboard --logdir logs/ --port 6006 --bind_all &
TENSORBOARD_PID=$!

# Run training
echo "Starting training at $(date)"
PYTHONUNBUFFERED=1 python3 train_draft_cloud.py \
    --sets ${SETS} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES} \
    --s3-bucket ${S3_BUCKET} \
    --early-stopping-patience 10 \
    --checkpoint-every 2 \
    2>&1 | tee training.log

# Training complete
echo "Training finished at $(date)"

# Upload final logs
aws s3 cp training.log "s3://${S3_BUCKET}/logs/training_$(date +%Y%m%d_%H%M%S).log"
aws s3 sync logs/ "s3://${S3_BUCKET}/tensorboard-logs/"

# Cleanup
kill $TENSORBOARD_PID 2>/dev/null || true

echo "All done!"
TRAINING_SCRIPT

chmod +x run_training.sh
chown -R ubuntu:ubuntu /home/ubuntu/mtg-rl

# Start training as ubuntu user
echo ""
echo "============================================================"
echo "Setup complete. Starting training..."
echo "============================================================"

# Run training in background with nohup
su - ubuntu -c "cd /home/ubuntu/mtg-rl && nohup ./run_training.sh > training_output.log 2>&1 &"

echo "Training started in background."
echo "Monitor with: tail -f /home/ubuntu/mtg-rl/training_output.log"
echo "Setup completed at: $(date)"
