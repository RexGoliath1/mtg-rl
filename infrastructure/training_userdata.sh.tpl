#!/bin/bash
# =============================================================================
# MTG RL Training Instance Setup (Terraform Template)
# =============================================================================
set -e
exec > >(tee /var/log/training-setup.log) 2>&1

echo "============================================================"
echo "MTG RL Training Instance Setup"
echo "Started at: $(date)"
echo "============================================================"

# Training configuration from Terraform
S3_BUCKET="${s3_bucket}"
SETS="${sets}"
EPOCHS="${epochs}"
BATCH_SIZE="${batch_size}"
MAX_SAMPLES="${max_samples}"

echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Sets: $SETS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Samples: $$${MAX_SAMPLES:-all}"

# Install system dependencies
echo ""
echo "[1/7] Installing system dependencies..."
apt-get update -y
apt-get install -y python3-pip python3-venv htop nvtop unzip

# Install AWS CLI v2
echo ""
echo "[2/7] Installing AWS CLI..."
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install
rm -rf awscliv2.zip aws/

# Setup working directory
echo ""
echo "[3/7] Setting up working directory..."
mkdir -p /home/ubuntu/mtg-rl
cd /home/ubuntu/mtg-rl

# Download code from S3
echo ""
echo "[4/7] Downloading training code..."
aws s3 cp "s3://$S3_BUCKET/code/training-code.tar.gz" . || {
    echo "Code not in S3, cloning from GitHub..."
    cd /home/ubuntu
    git clone https://github.com/RexGoliath1/mtg-rl.git
    cd mtg-rl
}

if [ -f training-code.tar.gz ]; then
    tar -xzf training-code.tar.gz
    rm training-code.tar.gz
fi

# Download training data
echo ""
echo "[5/7] Downloading training data..."
mkdir -p data/17lands
aws s3 sync "s3://$S3_BUCKET/data/17lands/" data/17lands/ || echo "No data in S3, will need to upload"

# Setup Python environment
echo ""
echo "[6/7] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard boto3 requests numpy pandas

# Create training wrapper script with monitoring
echo ""
echo "[7/7] Creating training scripts..."

cat > run_training.sh << 'TRAINSCRIPT'
#!/bin/bash
set -e
cd /home/ubuntu/mtg-rl
source venv/bin/activate

S3_BUCKET="${s3_bucket}"
SETS="${sets}"
EPOCHS="${epochs}"
BATCH_SIZE="${batch_size}"
MAX_SAMPLES="${max_samples}"

# Start TensorBoard in background
echo "Starting TensorBoard on port 6006..."
tensorboard --logdir logs/ --port 6006 --bind_all &
TENSORBOARD_PID=$!

# Function to handle shutdown
cleanup() {
    echo ""
    echo "[SHUTDOWN] Received signal, cleaning up..."
    kill $TENSORBOARD_PID 2>/dev/null || true

    # Upload final state
    echo "Uploading final logs to S3..."
    aws s3 cp training.log "s3://$S3_BUCKET/logs/training_$(date +%Y%m%d_%H%M%S).log" || true
    aws s3 sync logs/ "s3://$S3_BUCKET/tensorboard-logs/" || true

    echo "Cleanup complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Build max_samples argument
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

# Run training
echo "============================================================"
echo "Starting training at $(date)"
echo "Sets: $SETS"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Samples: $$${MAX_SAMPLES:-all}"
echo "============================================================"

PYTHONUNBUFFERED=1 python3 train_draft_cloud.py \
    --sets $SETS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    $MAX_SAMPLES_ARG \
    --s3-bucket $S3_BUCKET \
    --early-stopping-patience 10 \
    --checkpoint-every 2 \
    2>&1 | tee training.log

# Training complete
echo ""
echo "============================================================"
echo "Training finished at $(date)"
echo "============================================================"

# Upload final logs
aws s3 cp training.log "s3://$S3_BUCKET/logs/training_final_$(date +%Y%m%d_%H%M%S).log"
aws s3 sync logs/ "s3://$S3_BUCKET/tensorboard-logs/"

# Keep TensorBoard running for inspection
echo "Training complete. TensorBoard still running on port 6006."
echo "Instance will remain active for inspection."
echo "Run 'sudo shutdown -h now' when done."

# Wait for TensorBoard
wait $TENSORBOARD_PID
TRAINSCRIPT

chmod +x run_training.sh

# Create monitoring script
cat > monitor.sh << 'MONITORSCRIPT'
#!/bin/bash
# Quick monitoring commands
echo "=== GPU Status ==="
nvidia-smi

echo ""
echo "=== Training Progress ==="
tail -20 /home/ubuntu/mtg-rl/training.log 2>/dev/null || echo "Training not started yet"

echo ""
echo "=== Latest Checkpoints ==="
ls -lt /home/ubuntu/mtg-rl/checkpoints/ 2>/dev/null | head -5 || echo "No checkpoints yet"
MONITORSCRIPT

chmod +x monitor.sh

# Set ownership
chown -R ubuntu:ubuntu /home/ubuntu/mtg-rl

# Start training
echo ""
echo "============================================================"
echo "Setup complete. Starting training..."
echo "============================================================"

su - ubuntu -c "cd /home/ubuntu/mtg-rl && nohup ./run_training.sh > training_output.log 2>&1 &"

echo ""
echo "Training started in background."
echo "Monitor with: tail -f /home/ubuntu/mtg-rl/training_output.log"
echo "Or run: /home/ubuntu/mtg-rl/monitor.sh"
echo ""
echo "TensorBoard: http://<instance-ip>:6006"
echo ""
echo "Setup completed at: $(date)"
