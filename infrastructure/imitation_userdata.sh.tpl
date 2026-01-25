#!/bin/bash
# =============================================================================
# MTG RL Imitation Learning - Data Collection on AWS
# =============================================================================
exec > >(tee /var/log/imitation-setup.log) 2>&1

echo "============================================================"
echo "MTG RL Imitation Learning Data Collection"
echo "Started at: $(date)"
echo "============================================================"

# Configuration from Terraform
S3_BUCKET="${s3_bucket}"
ECR_REPO="${ecr_repo}"
NUM_GAMES="${num_games}"
WORKERS="${workers}"

echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  ECR Repo: $ECR_REPO"
echo "  Games: $NUM_GAMES"
echo "  Workers: $WORKERS"

# Install Docker and EC2 Instance Connect (for keyless SSH via AWS CLI)
echo ""
echo "[1/5] Installing Docker and EC2 Instance Connect..."
apt-get update -y
apt-get install -y docker.io python3-pip python3-venv unzip htop ec2-instance-connect
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Ensure SSM agent is running (for AWS Systems Manager access)
snap install amazon-ssm-agent --classic 2>/dev/null || true
systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true
systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true

# Install AWS CLI
echo ""
echo "[2/5] Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install
    rm -rf awscliv2.zip aws/
fi

# Login to ECR and pull Forge daemon image (amd64 version for EC2)
echo ""
echo "[3/5] Pulling Forge daemon from ECR..."
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_REPO
docker pull $ECR_REPO:forge-daemon-amd64

# Clone code and setup
echo ""
echo "[4/5] Setting up collection environment..."
cd /home/ubuntu
git clone https://github.com/RexGoliath1/mtg-rl.git
cd mtg-rl

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install h5py numpy requests

# Create collection script
echo ""
echo "[5/5] Starting data collection..."

cat > /home/ubuntu/run_collection.sh << 'SCRIPT'
#!/bin/bash
cd /home/ubuntu/mtg-rl
source venv/bin/activate

S3_BUCKET="${s3_bucket}"
NUM_GAMES="${num_games}"
WORKERS="${workers}"

# Start Forge daemon (amd64 version for EC2)
echo "Starting Forge daemon..."
docker run -d --name forge-daemon -p 17171:17171 ${ecr_repo}:forge-daemon-amd64

# Wait for daemon to be ready
echo "Waiting for daemon to initialize..."
sleep 90

# Verify daemon is healthy
for i in {1..10}; do
    if echo "STATUS" | nc -w 5 localhost 17171 | grep -q "FORGE DAEMON"; then
        echo "Daemon is ready!"
        break
    fi
    echo "Waiting for daemon... attempt $i"
    sleep 30
done

# Run collection
echo "Starting collection of $NUM_GAMES games..."
python -u scripts/collect_ai_training_data.py \
    --games $NUM_GAMES \
    --output training_data/imitation_aws \
    --workers $WORKERS \
    --save-interval 500 \
    --timeout 60

# Upload results to S3
echo "Uploading results to S3..."
aws s3 sync training_data/imitation_aws/ s3://$S3_BUCKET/imitation_data/

echo "Collection complete at $(date)"

# Shutdown instance to save costs
if [ "${auto_shutdown}" = "true" ]; then
    echo "Auto-shutdown enabled. Shutting down in 5 minutes..."
    sudo shutdown -h +5
fi
SCRIPT

chmod +x /home/ubuntu/run_collection.sh
chown ubuntu:ubuntu /home/ubuntu/run_collection.sh

# Run as ubuntu user
sudo -u ubuntu bash /home/ubuntu/run_collection.sh > /home/ubuntu/collection.log 2>&1 &

echo "Collection started in background. Monitor with: tail -f /home/ubuntu/collection.log"
echo "Setup complete at $(date)"
