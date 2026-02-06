#!/bin/bash
set -e

# ============================================================================
# Deploy AI Training Data Collection to AWS Spot Instance
# ============================================================================
# Launches a spot instance that:
# 1. Starts Forge daemon
# 2. Runs collect_ai_training_data.py with deck rotation
# 3. Uploads HDF5 training data + reports to S3
# 4. Auto-terminates when complete
#
# Prerequisites:
# - AWS CLI configured
# - S3 bucket exists (from terraform)
# - forge-repo built locally (or will build on instance)
#
# Usage:
#   ./scripts/deploy_data_collection.sh --games 1000
#   ./scripts/deploy_data_collection.sh --games 5000 --workers 8
# ============================================================================

# Configuration
S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
REGION="us-west-2"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.2xlarge}"  # CPU-only (no GPU needed for collection)
NUM_GAMES="${NUM_GAMES:-1000}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-60}"

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--games N] [--workers N] [--timeout N] [--instance-type TYPE]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "AI TRAINING DATA COLLECTION - AWS DEPLOYMENT"
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
echo "AWS credentials OK"

# Check S3 bucket exists
if ! aws s3 ls "s3://${S3_BUCKET}" &>/dev/null; then
    echo "ERROR: S3 bucket ${S3_BUCKET} does not exist"
    echo "Run: cd infrastructure && terraform apply"
    exit 1
fi
echo "S3 bucket OK"

# Package the code
echo ""
echo "Packaging code..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="data_collection_${TIMESTAMP}.tar.gz"

COPYFILE_DISABLE=1 tar -czf "/tmp/${PACKAGE_NAME}" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='forge-repo' \
    --exclude='data' \
    --exclude='checkpoints' \
    --exclude='wandb' \
    --exclude='training_output' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.h5' \
    -C "$PROJECT_DIR" \
    src scripts decks requirements.txt 2>/dev/null || true

PACKAGE_SIZE=$(ls -lh "/tmp/${PACKAGE_NAME}" | awk '{print $5}')
echo "  Package: $PACKAGE_NAME ($PACKAGE_SIZE)"

# Package Forge JAR if available
FORGE_JAR_PACKAGE=""
FORGE_JAR_LOCAL=$(find "$PROJECT_DIR/forge-repo/forge-gui-desktop/target" -name "*jar-with-dependencies.jar" 2>/dev/null | head -1)
if [ -n "$FORGE_JAR_LOCAL" ]; then
    FORGE_JAR_REL="${FORGE_JAR_LOCAL#$PROJECT_DIR/forge-repo/}"
    echo "Packaging Forge JAR ($FORGE_JAR_REL)..."
    COPYFILE_DISABLE=1 tar -czf "/tmp/forge_jar_${TIMESTAMP}.tar.gz" \
        -C "$PROJECT_DIR/forge-repo" \
        "$FORGE_JAR_REL" \
        forge-gui/res
    FORGE_JAR_SIZE=$(ls -lh "/tmp/forge_jar_${TIMESTAMP}.tar.gz" | awk '{print $5}')
    FORGE_JAR_PACKAGE="/tmp/forge_jar_${TIMESTAMP}.tar.gz"
    echo "  Forge JAR packaged ($FORGE_JAR_SIZE)"
else
    echo "  Forge JAR not found locally — will build on instance"
fi

# Upload to S3
echo ""
echo "Uploading to S3..."
aws s3 cp "/tmp/${PACKAGE_NAME}" "s3://${S3_BUCKET}/test_packages/${PACKAGE_NAME}" --quiet
if [ -n "$FORGE_JAR_PACKAGE" ]; then
    aws s3 cp "$FORGE_JAR_PACKAGE" "s3://${S3_BUCKET}/test_packages/forge_jar_${TIMESTAMP}.tar.gz" --quiet
fi
echo "  Upload complete"

# Find Ubuntu 22.04 AMI (no GPU needed)
echo ""
echo "Finding AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners 099720109477 \
    --filters \
        "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
    echo "ERROR: Could not find Ubuntu 22.04 AMI"
    exit 1
fi
echo "Using AMI: $AMI_ID (Ubuntu 22.04)"

# Get default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[0].SubnetId' --output text)

# Get or create security group
SG_NAME="forge-test-sg"
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Security group for Forge testing" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
fi

# Ensure IAM instance profile exists
INSTANCE_PROFILE="mtg-rl-training"
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" &>/dev/null; then
    echo "Creating IAM instance profile..."
    aws iam create-role --role-name "$INSTANCE_PROFILE" \
        --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' || true
    aws iam attach-role-policy --role-name "$INSTANCE_PROFILE" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess" || true
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE" || true
    aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE" --role-name "$INSTANCE_PROFILE" || true
    sleep 10
fi

# Create user data script
USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
set -ex

exec > >(tee /var/log/data-collection.log) 2>&1
echo "=========================================="
echo "DATA COLLECTION SETUP"
echo "=========================================="
date

cd /home/ubuntu

# Install dependencies
apt-get update -qq
apt-get install -y -qq openjdk-17-jdk maven python3-pip python3-venv unzip > /dev/null

# Install AWS CLI v2
curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install --update
rm -rf aws awscliv2.zip

# Download code package
aws s3 cp s3://BUCKET_PLACEHOLDER/test_packages/PACKAGE_PLACEHOLDER code.tar.gz
mkdir -p mtg && cd mtg
tar -xzf ../code.tar.gz

# Install Python dependencies
python3 -m pip install -q numpy h5py requests

# Check for pre-built Forge JAR
if aws s3 ls s3://BUCKET_PLACEHOLDER/test_packages/forge_jar_TIMESTAMP_PLACEHOLDER.tar.gz &>/dev/null; then
    echo "Using pre-built Forge JAR..."
    aws s3 cp s3://BUCKET_PLACEHOLDER/test_packages/forge_jar_TIMESTAMP_PLACEHOLDER.tar.gz forge_jar.tar.gz
    mkdir -p forge-repo
    tar -xzf forge_jar.tar.gz -C forge-repo
    FORGE_JAR=$(find forge-repo -name "*jar-with-dependencies.jar" ! -name "._*" | head -1)
else
    echo "Building Forge from source..."
    git clone --depth 1 -b feature/rl-daemon-mode https://github.com/RexGoliath1/forge.git forge-repo
    cd forge-repo
    mvn package -DskipTests -pl forge-gui-desktop -am -q
    cd ..
    FORGE_JAR=$(find forge-repo -name "*jar-with-dependencies.jar" ! -name "._*" | head -1)
fi

echo "Forge JAR: $FORGE_JAR"

# Start Forge daemon
echo "Starting Forge daemon..."
export JAVA_OPTS="-Xmx6g"
java -jar "$FORGE_JAR" --daemon --port 17171 &
FORGE_PID=$!

# Wait for daemon to start
echo "Waiting for Forge daemon to initialize..."
sleep 30

if ! kill -0 $FORGE_PID 2>/dev/null; then
    echo "ERROR: Forge daemon failed to start"
    exit 1
fi
echo "Forge daemon running (PID: $FORGE_PID)"

# Run data collection
echo "=========================================="
echo "STARTING DATA COLLECTION"
echo "=========================================="
echo "Games: NUM_GAMES_PLACEHOLDER"
echo "Workers: WORKERS_PLACEHOLDER"
echo "Timeout: TIMEOUT_PLACEHOLDER"
date

python3 scripts/collect_ai_training_data.py \
    --games NUM_GAMES_PLACEHOLDER \
    --output /home/ubuntu/training_data \
    --host localhost \
    --port 17171 \
    --workers WORKERS_PLACEHOLDER \
    --timeout TIMEOUT_PLACEHOLDER \
    --save-interval 500

echo "=========================================="
echo "COLLECTION COMPLETE"
echo "=========================================="
date

# Upload results to S3
echo "Uploading results to S3..."
aws s3 sync /home/ubuntu/training_data/ \
    s3://BUCKET_PLACEHOLDER/imitation_data/collection_TIMESTAMP_PLACEHOLDER/ \
    --exclude "*.tex"

# Upload log
aws s3 cp /var/log/data-collection.log \
    s3://BUCKET_PLACEHOLDER/imitation_data/collection_TIMESTAMP_PLACEHOLDER/collection_log.txt

echo "Results uploaded to s3://BUCKET_PLACEHOLDER/imitation_data/collection_TIMESTAMP_PLACEHOLDER/"

# Signal completion
echo '{"status":"complete","timestamp":"TIMESTAMP_PLACEHOLDER","games":NUM_GAMES_PLACEHOLDER}' | \
    aws s3 cp - s3://BUCKET_PLACEHOLDER/imitation_data/collection_TIMESTAMP_PLACEHOLDER/collection_complete.json

# Stop Forge daemon
kill $FORGE_PID 2>/dev/null || true

echo "Shutting down..."
shutdown -h now
USERDATA
)

# Replace placeholders
USER_DATA="${USER_DATA//BUCKET_PLACEHOLDER/$S3_BUCKET}"
USER_DATA="${USER_DATA//PACKAGE_PLACEHOLDER/$PACKAGE_NAME}"
USER_DATA="${USER_DATA//TIMESTAMP_PLACEHOLDER/$TIMESTAMP}"
USER_DATA="${USER_DATA//NUM_GAMES_PLACEHOLDER/$NUM_GAMES}"
USER_DATA="${USER_DATA//WORKERS_PLACEHOLDER/$WORKERS}"
USER_DATA="${USER_DATA//TIMEOUT_PLACEHOLDER/$TIMEOUT}"

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
    --tags "Key=Name,Value=data-collection-$TIMESTAMP" "Key=Project,Value=mtg-rl" "Key=Task,Value=data-collection"

# Wait for instance to start
echo "Waiting for instance..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "============================================================"
echo "DATA COLLECTION INSTANCE LAUNCHED"
echo "============================================================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo "Games:       $NUM_GAMES"
echo "Workers:     $WORKERS"
echo ""
echo "Monitor with:"
echo "  # Check if complete:"
echo "  aws s3 ls s3://${S3_BUCKET}/imitation_data/collection_${TIMESTAMP}/collection_complete.json"
echo ""
echo "  # View live log:"
echo "  aws s3 cp s3://${S3_BUCKET}/imitation_data/collection_${TIMESTAMP}/collection_log.txt - | tail -50"
echo ""
echo "  # Check instance state:"
echo "  aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text"
echo ""
echo "  # Download results when complete:"
echo "  aws s3 sync s3://${S3_BUCKET}/imitation_data/collection_${TIMESTAMP}/ training_data/${TIMESTAMP}/"
echo ""
echo "Instance auto-terminates after collection."
echo "============================================================"
