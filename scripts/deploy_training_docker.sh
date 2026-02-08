#!/bin/bash
set -e

# ============================================================================
# Deploy Imitation Learning Training to AWS Spot Instance (Docker-based)
# ============================================================================
# Launches a GPU spot instance that:
# 1. Installs Docker + NVIDIA container toolkit
# 2. Pulls mtg-rl-training image from ECR
# 3. Downloads HDF5 training data from S3
# 4. Runs train_imitation.py in Docker container
# 5. Uploads checkpoints + logs to S3
# 6. Auto-terminates when complete
#
# Prerequisites:
# - AWS CLI configured
# - mtg-rl-training:latest pushed to ECR
# - HDF5 data in S3 (from data collection run)
#
# Usage:
#   ./scripts/deploy_training_docker.sh
#   ./scripts/deploy_training_docker.sh --epochs 100 --lr 0.0005
#   ./scripts/deploy_training_docker.sh --data-path collection_20260207_123456
# ============================================================================

# Configuration
S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
DATA_PATH=""  # S3 sub-path under imitation_data/ (empty = all)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --lr|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--epochs N] [--batch-size N] [--hidden-dim N] [--lr RATE] [--data-path S3_PREFIX] [--instance-type TYPE]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "IMITATION LEARNING TRAINING - DOCKER DEPLOYMENT"
echo "============================================================"
echo "Instance Type:  $INSTANCE_TYPE"
echo "Epochs:         $EPOCHS"
echo "Batch Size:     $BATCH_SIZE"
echo "Hidden Dim:     $HIDDEN_DIM"
echo "Learning Rate:  $LEARNING_RATE"
echo "Data Path:      ${DATA_PATH:-all imitation_data/}"
echo "Region:         $REGION"
echo ""

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS credentials not configured"
    exit 1
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS credentials OK (account: $ACCOUNT_ID)"

# Derive ECR registry URL
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
ECR_REPO="${ECR_REGISTRY}/mtg-rl-training"
echo "ECR Registry: $ECR_REGISTRY"

# Check S3 bucket exists
if ! aws s3 ls "s3://${S3_BUCKET}" &>/dev/null; then
    echo "ERROR: S3 bucket ${S3_BUCKET} does not exist"
    echo "Run: cd infrastructure && terraform apply"
    exit 1
fi
echo "S3 bucket OK"

# Check for HDF5 data in S3
S3_DATA_PREFIX="imitation_data/"
if [ -n "$DATA_PATH" ]; then
    S3_DATA_PREFIX="imitation_data/${DATA_PATH}/"
fi
H5_COUNT=$(aws s3 ls "s3://${S3_BUCKET}/${S3_DATA_PREFIX}" --recursive | grep -c '\.h5$' || echo "0")
echo "HDF5 files in S3: $H5_COUNT"
if [ "$H5_COUNT" -eq 0 ]; then
    echo "ERROR: No HDF5 files found in s3://${S3_BUCKET}/${S3_DATA_PREFIX}"
    echo "Run data collection first: ./scripts/deploy_data_collection_docker.sh --games 1000"
    exit 1
fi

# Verify ECR training image exists
echo ""
echo "Checking ECR image..."
if ! aws ecr describe-images --region "$REGION" --repository-name "mtg-rl-training" --image-ids imageTag=latest &>/dev/null 2>&1; then
    echo "ERROR: Image mtg-rl-training:latest not found in ECR"
    echo "Push the training image first:"
    echo "  aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY"
    echo "  docker build -t $ECR_REGISTRY/mtg-rl-training:latest -f infrastructure/docker/Dockerfile.training ."
    echo "  docker push $ECR_REGISTRY/mtg-rl-training:latest"
    exit 1
fi
echo "  mtg-rl-training:latest OK"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="imitation_train_${TIMESTAMP}"

# Find Ubuntu 22.04 Deep Learning AMI (has NVIDIA drivers pre-installed)
echo ""
echo "Finding Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
    echo "WARN: Deep Learning AMI not found, falling back to standard Ubuntu 22.04"
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi
echo "Using AMI: $AMI_ID"

# Get default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[0].SubnetId' --output text)

# Get or create security group
SG_NAME="mtg-rl-training-sg"
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Security group for MTG RL training" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
fi

# Ensure IAM instance profile exists
INSTANCE_PROFILE="mtg-rl-training-profile"
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" &>/dev/null; then
    INSTANCE_PROFILE="mtg-rl-training"
    if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" &>/dev/null; then
        echo "ERROR: IAM instance profile not found. Run terraform apply first."
        exit 1
    fi
fi
echo "IAM profile: $INSTANCE_PROFILE"

# Build the S3 sync command based on whether a specific data path was given
if [ -n "$DATA_PATH" ]; then
    S3_SYNC_CMD="aws s3 sync \"s3://S3_BUCKET_PLACEHOLDER/imitation_data/DATA_PATH_PLACEHOLDER/\" /home/ubuntu/training_data/ --exclude \"*\" --include \"*.h5\""
else
    S3_SYNC_CMD="aws s3 sync \"s3://S3_BUCKET_PLACEHOLDER/imitation_data/\" /home/ubuntu/training_data/ --exclude \"*\" --include \"*.h5\""
fi

# Create user data script
USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
set -ex

exec > >(tee /var/log/imitation-train.log) 2>&1

echo "============================================================"
echo "MTG RL IMITATION TRAINING (Docker)"
echo "Started at: $(date)"
echo "============================================================"

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

# --- [1/6] Install Docker ---
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

# --- [2/6] Install AWS CLI ---
echo ""
echo "[2/6] Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install --update
    rm -rf aws awscliv2.zip
fi

# --- [3/6] Login to ECR and pull training image ---
echo ""
echo "[3/6] Pulling training image from ECR..."
aws ecr get-login-password --region ECR_REGION_PLACEHOLDER | \
    docker login --username AWS --password-stdin ECR_REGISTRY_PLACEHOLDER

docker pull ECR_REPO_PLACEHOLDER:latest
echo "Image pulled successfully"
docker images

# --- [4/6] Download HDF5 training data from S3 ---
echo ""
echo "[4/6] Downloading training data from S3..."
mkdir -p /home/ubuntu/training_data
S3_SYNC_CMD_PLACEHOLDER

echo "Downloaded files:"
ls -lh /home/ubuntu/training_data/
H5_COUNT=$(find /home/ubuntu/training_data -name "*.h5" | wc -l)
echo "Total HDF5 files: $H5_COUNT"

if [ "$H5_COUNT" -eq 0 ]; then
    echo "ERROR: No HDF5 files found"
    echo '{"status":"failed","reason":"no_data","timestamp":"'$(date -Iseconds)'"}' | \
        aws s3 cp - "s3://S3_BUCKET_PLACEHOLDER/training_runs/RUN_ID_PLACEHOLDER/training_complete.json"
    shutdown -h now
    exit 1
fi

# --- [5/6] Prepare output directories ---
echo ""
echo "[5/6] Preparing output directories..."
mkdir -p /home/ubuntu/checkpoints /home/ubuntu/logs

# Background log uploader (every 5 min)
(
    while true; do
        sleep 300
        aws s3 cp /var/log/imitation-train.log \
            "s3://S3_BUCKET_PLACEHOLDER/training_runs/RUN_ID_PLACEHOLDER/live_log.txt" 2>/dev/null || true
    done
) &
LOG_UPLOADER_PID=$!

# --- [6/6] Run training in Docker container ---
echo ""
echo "[6/6] Starting training..."
echo "============================================================"
echo "TRAINING: EPOCHS_PLACEHOLDER epochs, batch=BATCH_SIZE_PLACEHOLDER, hidden=HIDDEN_DIM_PLACEHOLDER, lr=LR_PLACEHOLDER"
echo "============================================================"

# Build Docker run flags
DOCKER_FLAGS="-v /home/ubuntu/training_data:/data -v /home/ubuntu/checkpoints:/checkpoints -v /home/ubuntu/logs:/app/logs"

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
    --region ECR_REGION_PLACEHOLDER \
    --query SecretString \
    --output text 2>/dev/null || echo "")

WANDB_FLAGS=""
if [ -n "$WANDB_KEY" ]; then
    WANDB_FLAGS="-e WANDB_API_KEY=$WANDB_KEY -e WANDB_PROJECT=forgerl -e WANDB_ENTITY=sgoncia-self"
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
    ECR_REPO_PLACEHOLDER:latest \
    python /app/scripts/train_imitation.py \
        --data-dir /data \
        --epochs EPOCHS_PLACEHOLDER \
        --batch-size BATCH_SIZE_PLACEHOLDER \
        --hidden-dim HIDDEN_DIM_PLACEHOLDER \
        --lr LR_PLACEHOLDER \
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
aws s3 sync /home/ubuntu/checkpoints/ "s3://S3_BUCKET_PLACEHOLDER/training_runs/RUN_ID_PLACEHOLDER/checkpoints/"
aws s3 sync /home/ubuntu/logs/ "s3://S3_BUCKET_PLACEHOLDER/training_runs/RUN_ID_PLACEHOLDER/tensorboard/" 2>/dev/null || true
aws s3 cp /var/log/imitation-train.log "s3://S3_BUCKET_PLACEHOLDER/training_runs/RUN_ID_PLACEHOLDER/training_log.txt"

# Copy best model to stable path
if [ -f /home/ubuntu/checkpoints/imitation_policy.pt ]; then
    aws s3 cp /home/ubuntu/checkpoints/imitation_policy.pt \
        "s3://S3_BUCKET_PLACEHOLDER/models/imitation_policy_latest.pt"
    echo "Best model uploaded to s3://S3_BUCKET_PLACEHOLDER/models/imitation_policy_latest.pt"
fi
if [ -f /home/ubuntu/checkpoints/imitation_policy.json ]; then
    aws s3 cp /home/ubuntu/checkpoints/imitation_policy.json \
        "s3://S3_BUCKET_PLACEHOLDER/training_runs/RUN_ID_PLACEHOLDER/training_summary.json"
fi

# Signal completion
echo "{\"status\":\"$([ $TRAINING_EXIT -eq 0 ] && echo success || echo failed)\",\"timestamp\":\"TIMESTAMP_PLACEHOLDER\",\"epochs\":EPOCHS_PLACEHOLDER,\"exit_code\":$TRAINING_EXIT,\"method\":\"docker\",\"run_id\":\"RUN_ID_PLACEHOLDER\"}" | \
    aws s3 cp - "s3://S3_BUCKET_PLACEHOLDER/training_runs/RUN_ID_PLACEHOLDER/training_complete.json"

echo "Results uploaded to s3://S3_BUCKET_PLACEHOLDER/training_runs/RUN_ID_PLACEHOLDER/"

# Stop background log uploader
kill $LOG_UPLOADER_PID 2>/dev/null || true

echo "Shutting down..."
shutdown -h now
USERDATA
)

# Replace placeholders
USER_DATA="${USER_DATA//S3_BUCKET_PLACEHOLDER/$S3_BUCKET}"
USER_DATA="${USER_DATA//ECR_REGISTRY_PLACEHOLDER/$ECR_REGISTRY}"
USER_DATA="${USER_DATA//ECR_REPO_PLACEHOLDER/$ECR_REPO}"
USER_DATA="${USER_DATA//ECR_REGION_PLACEHOLDER/$REGION}"
USER_DATA="${USER_DATA//RUN_ID_PLACEHOLDER/$RUN_ID}"
USER_DATA="${USER_DATA//TIMESTAMP_PLACEHOLDER/$TIMESTAMP}"
USER_DATA="${USER_DATA//EPOCHS_PLACEHOLDER/$EPOCHS}"
USER_DATA="${USER_DATA//BATCH_SIZE_PLACEHOLDER/$BATCH_SIZE}"
USER_DATA="${USER_DATA//HIDDEN_DIM_PLACEHOLDER/$HIDDEN_DIM}"
USER_DATA="${USER_DATA//LR_PLACEHOLDER/$LEARNING_RATE}"
USER_DATA="${USER_DATA//DATA_PATH_PLACEHOLDER/$DATA_PATH}"
USER_DATA="${USER_DATA//S3_SYNC_CMD_PLACEHOLDER/$S3_SYNC_CMD}"

# Encode user data
USER_DATA_B64=$(echo "$USER_DATA" | base64)

# Launch spot instance
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
    --tags "Key=Name,Value=imitation-train-docker-$TIMESTAMP" "Key=Project,Value=mtg-rl" "Key=Task,Value=imitation-training"

# Wait for instance to start
echo "Waiting for instance..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "============================================================"
echo "IMITATION TRAINING INSTANCE LAUNCHED (Docker)"
echo "============================================================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo "Run ID:      $RUN_ID"
echo ""
echo "Training config:"
echo "  Epochs:        $EPOCHS"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Hidden Dim:    $HIDDEN_DIM"
echo "  Learning Rate: $LEARNING_RATE"
echo "  HDF5 files:    $H5_COUNT"
echo ""
echo "Monitor with:"
echo "  # Check if complete:"
echo "  aws s3 ls s3://${S3_BUCKET}/training_runs/${RUN_ID}/training_complete.json"
echo ""
echo "  # View live log (updates every 5 min):"
echo "  aws s3 cp s3://${S3_BUCKET}/training_runs/${RUN_ID}/live_log.txt - | tail -50"
echo ""
echo "  # Check instance state:"
echo "  aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text"
echo ""
echo "  # Download results when complete:"
echo "  aws s3 sync s3://${S3_BUCKET}/training_runs/${RUN_ID}/ training_results/${RUN_ID}/"
echo ""
echo "  # Download best model:"
echo "  aws s3 cp s3://${S3_BUCKET}/models/imitation_policy_latest.pt checkpoints/"
echo ""
echo "Instance auto-terminates after training."
echo "============================================================"
