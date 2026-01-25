#!/bin/bash
# =============================================================================
# MTG RL Training Instance Setup (Terraform Template)
# =============================================================================
# Note: Not using set -e to allow graceful handling of pre-installed packages
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
ENCODER_TYPE="${encoder_type}"

echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Sets: $SETS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Samples: $${MAX_SAMPLES:-all}"
echo "  Encoder Type: $${ENCODER_TYPE:-hybrid}"

# Install system dependencies
echo ""
echo "[1/7] Installing system dependencies..."
apt-get update -y || true
apt-get install -y python3-pip python3-venv htop nvtop unzip || true

# Check AWS CLI (Deep Learning AMI has it pre-installed)
echo ""
echo "[2/7] Checking AWS CLI..."
if command -v aws &> /dev/null; then
    echo "AWS CLI already installed: $(aws --version)"
else
    echo "Installing AWS CLI..."
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install || ./aws/install --update
    rm -rf awscliv2.zip aws/
fi

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
pip install tensorboard boto3 requests numpy pandas sentence-transformers wandb

# Create training wrapper script with monitoring
echo ""
echo "[7/7] Creating training scripts..."

cat > run_training.sh << 'TRAINSCRIPT'
#!/bin/bash
# Note: Not using set -e to allow capturing errors
cd /home/ubuntu/mtg-rl
source venv/bin/activate

S3_BUCKET="${s3_bucket}"
SETS="${sets}"
EPOCHS="${epochs}"
BATCH_SIZE="${batch_size}"
MAX_SAMPLES="${max_samples}"
ENCODER_TYPE="${encoder_type}"
AUTO_SHUTDOWN="${auto_shutdown}"

echo "============================================================"
echo "MTG-RL Training Run (v2 Hybrid Encoder)"
echo "============================================================"
echo "Started at: $(date)"
echo "Encoder Type: $ENCODER_TYPE"
echo "Auto-shutdown: $AUTO_SHUTDOWN"
echo "============================================================"

# Start TensorBoard in background
echo "Starting TensorBoard on port 6006..."
mkdir -p logs checkpoints
tensorboard --logdir logs/ --port 6006 --bind_all &
TENSORBOARD_PID=$!

# Upload log to S3 every 60 seconds in background
upload_logs() {
    while true; do
        sleep 60
        aws s3 cp training.log "s3://$S3_BUCKET/logs/training_live.log" 2>/dev/null || true
    done
}
upload_logs &
LOG_UPLOADER_PID=$!

# Function to handle shutdown
cleanup() {
    echo ""
    echo "[CLEANUP] Saving all artifacts to S3..."
    kill $TENSORBOARD_PID 2>/dev/null || true
    kill $LOG_UPLOADER_PID 2>/dev/null || true

    # Upload all artifacts
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "Uploading training log..."
    aws s3 cp training.log "s3://$S3_BUCKET/logs/training_$TIMESTAMP.log" || true

    echo "Uploading TensorBoard logs..."
    aws s3 sync logs/ "s3://$S3_BUCKET/tensorboard-logs/" || true

    echo "Uploading any remaining checkpoints..."
    aws s3 sync checkpoints/ "s3://$S3_BUCKET/checkpoints/" || true

    echo "Cleanup complete"
}

trap cleanup SIGTERM SIGINT EXIT

# Build max_samples argument
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "0" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

# Run training
echo ""
echo "============================================================"
echo "Starting training at $(date)"
echo "Sets: $SETS"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Samples: $${MAX_SAMPLES:-all}"
echo "Encoder Type: $${ENCODER_TYPE:-hybrid}"
echo "============================================================"

PYTHONUNBUFFERED=1 python3 train_draft_cloud.py \
    --sets $SETS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    $MAX_SAMPLES_ARG \
    --s3-bucket $S3_BUCKET \
    --encoder-type $${ENCODER_TYPE:-hybrid} \
    --early-stopping-patience 10 \
    --checkpoint-every 2 \
    2>&1 | tee training.log
TRAINING_EXIT_CODE=$?

# Training complete (or failed)
echo ""
echo "============================================================"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "TRAINING COMPLETED SUCCESSFULLY"
else
    echo "TRAINING FAILED (exit code: $TRAINING_EXIT_CODE)"
fi
echo "Finished at: $(date)"
echo "============================================================"

# Final artifact upload
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo ""
echo "Uploading final artifacts to S3..."
aws s3 cp training.log "s3://$S3_BUCKET/logs/training_final_$TIMESTAMP.log"
aws s3 sync logs/ "s3://$S3_BUCKET/tensorboard-logs/"
aws s3 sync checkpoints/ "s3://$S3_BUCKET/checkpoints/"

# Generate completion marker
echo "{\"status\": \"$( [ $TRAINING_EXIT_CODE -eq 0 ] && echo 'success' || echo 'failed')\", \"timestamp\": \"$(date -Iseconds)\", \"exit_code\": $TRAINING_EXIT_CODE}" > training_complete.json
aws s3 cp training_complete.json "s3://$S3_BUCKET/training_complete.json"

echo ""
echo "============================================================"
echo "All artifacts saved to: s3://$S3_BUCKET/"
echo "============================================================"

# Stop background processes
kill $LOG_UPLOADER_PID 2>/dev/null || true
kill $TENSORBOARD_PID 2>/dev/null || true

# Auto-shutdown if enabled
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo ""
    echo "[AUTO-SHUTDOWN] Instance will shut down in 60 seconds..."
    echo "To prevent shutdown: sudo shutdown -c"
    sleep 60
    sudo shutdown -h now
else
    echo ""
    echo "Instance will remain active."
    echo "TensorBoard: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):6006"
    echo "To shut down: sudo shutdown -h now"
    echo ""
    # Keep container running
    tail -f /dev/null
fi
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
