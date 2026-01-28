#!/bin/bash
# =============================================================================
# MTG RL Imitation Learning Model Training
# =============================================================================
exec > >(tee /var/log/imitation-train-setup.log) 2>&1

echo "============================================================"
echo "MTG RL Imitation Learning Model Training"
echo "Started at: $(date)"
echo "============================================================"

# Configuration from Terraform
S3_BUCKET="${s3_bucket}"
EPOCHS="${epochs}"
BATCH_SIZE="${batch_size}"
HIDDEN_DIM="${hidden_dim}"

echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Hidden Dim: $HIDDEN_DIM"

# Install system dependencies
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -y
apt-get install -y python3-pip python3-venv htop unzip ec2-instance-connect

# Ensure SSM agent is running
snap install amazon-ssm-agent --classic 2>/dev/null || true
systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true

# Check AWS CLI
echo ""
echo "[2/6] Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install
    rm -rf awscliv2.zip aws/
fi

# Clone code
echo ""
echo "[3/6] Cloning repository..."
cd /home/ubuntu

GITHUB_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id mtg-rl/github-token \
    --region us-west-2 \
    --query SecretString \
    --output text)

git clone https://$GITHUB_TOKEN@github.com/RexGoliath1/mtg-rl.git
cd mtg-rl
chown -R ubuntu:ubuntu /home/ubuntu/mtg-rl

# Download collected imitation data from S3
echo ""
echo "[4/6] Downloading imitation learning data..."
mkdir -p training_data/imitation_aws
aws s3 sync "s3://$S3_BUCKET/imitation_data/" training_data/imitation_aws/
echo "Downloaded files:"
ls -lh training_data/imitation_aws/

# Setup Python environment
echo ""
echo "[5/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install h5py numpy tensorboard boto3

# Create training script
echo ""
echo "[6/6] Starting model training..."

cat > /home/ubuntu/run_imitation_train.sh << 'SCRIPT'
#!/bin/bash
cd /home/ubuntu/mtg-rl
source venv/bin/activate

S3_BUCKET="${s3_bucket}"
EPOCHS="${epochs}"
BATCH_SIZE="${batch_size}"
HIDDEN_DIM="${hidden_dim}"
AUTO_SHUTDOWN="${auto_shutdown}"

echo "============================================================"
echo "PASS/PLAY POLICY TRAINING"
echo "============================================================"
echo "Started at: $(date)"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Note: Training binary classifier (pass vs play) on existing data"
echo "============================================================"

mkdir -p checkpoints logs

# Start TensorBoard
tensorboard --logdir logs/ --port 6006 --bind_all &
TENSORBOARD_PID=$!

# Upload log every 60 seconds
upload_logs() {
    while true; do
        sleep 60
        aws s3 cp imitation_training.log "s3://$S3_BUCKET/logs/imitation_training_live.log" 2>/dev/null || true
    done
}
upload_logs &
LOG_UPLOADER_PID=$!

# Run training
# Note: Using pass/play policy since existing data only has pass vs play labels
# (action indices were not properly encoded in data collection v1)
echo ""
echo "Starting pass/play policy training..."
PYTHONUNBUFFERED=1 python3 scripts/train_pass_policy.py \
    --data-dir training_data/imitation_aws \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --output checkpoints/pass_policy.pt \
    --balanced \
    2>&1 | tee imitation_training.log

TRAINING_EXIT_CODE=$?

# Upload results
echo ""
echo "Uploading results to S3..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
aws s3 cp imitation_training.log "s3://$S3_BUCKET/logs/pass_policy_training_$TIMESTAMP.log"
aws s3 cp checkpoints/pass_policy.pt "s3://$S3_BUCKET/checkpoints/pass_policy.pt" || true
aws s3 cp checkpoints/pass_policy.json "s3://$S3_BUCKET/checkpoints/pass_policy_summary.json" || true

echo ""
echo "============================================================"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "TRAINING COMPLETED SUCCESSFULLY"
else
    echo "TRAINING FAILED (exit code: $TRAINING_EXIT_CODE)"
fi
echo "Finished at: $(date)"
echo "============================================================"

# Generate completion marker
echo "{\"status\": \"$( [ $TRAINING_EXIT_CODE -eq 0 ] && echo 'success' || echo 'failed')\", \"timestamp\": \"$(date -Iseconds)\", \"exit_code\": $TRAINING_EXIT_CODE, \"model\": \"pass_policy\"}" > pass_policy_training_complete.json
aws s3 cp pass_policy_training_complete.json "s3://$S3_BUCKET/pass_policy_training_complete.json"

# Cleanup
kill $TENSORBOARD_PID 2>/dev/null || true
kill $LOG_UPLOADER_PID 2>/dev/null || true

# Auto-shutdown
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "Auto-shutdown enabled. Shutting down in 60 seconds..."
    sleep 60
    sudo shutdown -h now
fi
SCRIPT

chmod +x /home/ubuntu/run_imitation_train.sh
chown ubuntu:ubuntu /home/ubuntu/run_imitation_train.sh

# Run training as ubuntu user
sudo -u ubuntu bash /home/ubuntu/run_imitation_train.sh > /home/ubuntu/imitation_output.log 2>&1 &

echo "Training started in background."
echo "Monitor with: tail -f /home/ubuntu/imitation_output.log"
echo "Setup complete at $(date)"
