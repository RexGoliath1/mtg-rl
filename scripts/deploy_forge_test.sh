#!/bin/bash
set -e

# ============================================================================
# Forge Integration Cloud Test
# ============================================================================
# Deploys to AWS and runs actual Forge games to profile real latency.
# This helps us understand the actual bottlenecks vs simulated games.
#
# Prerequisites:
# - AWS CLI configured with valid credentials
# - Terraform infrastructure deployed (for S3 bucket, IAM roles)
# - forge-repo with daemon mode implementation
#
# Usage:
#   ./scripts/deploy_forge_test.sh [--games N] [--duration MIN] [--instance-type TYPE]
# ============================================================================

# Configuration
S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
REGION="us-west-2"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
NUM_GAMES="${NUM_GAMES:-50}"
DURATION_MINUTES="${DURATION_MINUTES:-15}"
KEY_NAME="${KEY_NAME:-}"  # Optional SSH key

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --games)
            NUM_GAMES="$2"
            shift 2
            ;;
        --duration)
            DURATION_MINUTES="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --key)
            KEY_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "FORGE INTEGRATION CLOUD TEST"
echo "============================================================"
echo "Instance Type: $INSTANCE_TYPE"
echo "Games: $NUM_GAMES"
echo "Duration: $DURATION_MINUTES minutes"
echo "Region: $REGION"
echo ""

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS credentials not configured"
    exit 1
fi

# Check S3 bucket exists
if ! aws s3 ls "s3://${S3_BUCKET}" &>/dev/null; then
    echo "ERROR: S3 bucket ${S3_BUCKET} does not exist"
    echo "Run: cd infrastructure && terraform apply"
    exit 1
fi

# Package the code
echo "Packaging code..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="forge_test_${TIMESTAMP}.tar.gz"

# Create package with essential files
tar -czf "/tmp/${PACKAGE_NAME}" \
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
    -C "$(dirname "$0")/.." \
    src scripts decks requirements.txt 2>/dev/null || true

# Also package the forge daemon (built jar)
if [ -d "forge-repo/forge-gui-desktop/target" ]; then
    echo "Packaging Forge JAR..."
    tar -czf "/tmp/forge_jar_${TIMESTAMP}.tar.gz" \
        -C "$(dirname "$0")/../forge-repo" \
        forge-gui-desktop/target/forge-gui-desktop-*-SNAPSHOT-jar-with-dependencies.jar \
        forge-gui/res 2>/dev/null || true
    FORGE_JAR_PACKAGE="/tmp/forge_jar_${TIMESTAMP}.tar.gz"
else
    echo "WARNING: Forge JAR not found. Will build on instance."
    FORGE_JAR_PACKAGE=""
fi

# Upload to S3
echo "Uploading to S3..."
aws s3 cp "/tmp/${PACKAGE_NAME}" "s3://${S3_BUCKET}/test_packages/${PACKAGE_NAME}"
if [ -n "$FORGE_JAR_PACKAGE" ]; then
    aws s3 cp "$FORGE_JAR_PACKAGE" "s3://${S3_BUCKET}/test_packages/forge_jar_${TIMESTAMP}.tar.gz"
fi

# Get latest Ubuntu Deep Learning AMI
echo "Finding AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
        "Name=state,Values=available" \
        "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
    echo "ERROR: Could not find suitable AMI"
    exit 1
fi
echo "Using AMI: $AMI_ID"

# Get default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[0].SubnetId' --output text)

# Create or get security group
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

    # Allow SSH (optional)
    aws ec2 authorize-security-group-ingress --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
fi

# Create IAM instance profile if needed (for S3 access)
INSTANCE_PROFILE="mtg-rl-training"
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" &>/dev/null; then
    echo "Creating IAM instance profile..."
    aws iam create-role --role-name "$INSTANCE_PROFILE" \
        --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' || true
    aws iam attach-role-policy --role-name "$INSTANCE_PROFILE" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess" || true
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE" || true
    aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE" --role-name "$INSTANCE_PROFILE" || true
    sleep 10  # Wait for propagation
fi

# Create user data script
USER_DATA=$(cat << 'USERDATA'
#!/bin/bash
set -ex

# Log everything
exec > >(tee /var/log/forge-test.log) 2>&1

echo "Starting Forge test setup..."
cd /home/ubuntu

# Install Java (for Forge)
apt-get update
apt-get install -y openjdk-17-jdk maven

# Download code package
aws s3 cp s3://BUCKET_PLACEHOLDER/test_packages/PACKAGE_PLACEHOLDER code.tar.gz
mkdir -p mtg && cd mtg
tar -xzf ../code.tar.gz

# Check for pre-built Forge JAR
if aws s3 ls s3://BUCKET_PLACEHOLDER/test_packages/forge_jar_TIMESTAMP_PLACEHOLDER.tar.gz &>/dev/null; then
    echo "Using pre-built Forge JAR..."
    aws s3 cp s3://BUCKET_PLACEHOLDER/test_packages/forge_jar_TIMESTAMP_PLACEHOLDER.tar.gz forge_jar.tar.gz
    mkdir -p forge-repo
    tar -xzf forge_jar.tar.gz -C forge-repo
    FORGE_JAR=$(find forge-repo -name "*jar-with-dependencies.jar" | head -1)
else
    echo "Building Forge from source..."
    git clone --depth 1 -b feature/rl-daemon-mode https://github.com/RexGoliath1/forge.git forge-repo
    cd forge-repo
    mvn package -DskipTests -pl forge-gui-desktop -am
    cd ..
    FORGE_JAR=$(find forge-repo -name "*jar-with-dependencies.jar" | head -1)
fi

echo "Forge JAR: $FORGE_JAR"

# Set up Python environment
python3 -m pip install -r requirements.txt
python3 -m pip install numpy

# Create a simple test deck if needed
mkdir -p decks
cat > decks/test_deck1.dck << 'DECK'
[metadata]
Name=Test Deck 1
[main]
24 Mountain
4 Lightning Bolt
4 Monastery Swiftspear
4 Goblin Guide
4 Eidolon of the Great Revel
4 Lava Spike
4 Rift Bolt
4 Searing Blaze
4 Shard Volley
4 Light Up the Stage
DECK

cat > decks/test_deck2.dck << 'DECK'
[metadata]
Name=Test Deck 2
[main]
24 Plains
4 Savannah Lions
4 Elite Vanguard
4 Soldier of the Pantheon
4 Dryad Militant
4 Imposing Sovereign
4 Thalia, Guardian of Thraben
4 Honor of the Pure
4 Path to Exile
4 Brave the Elements
DECK

# Start Forge daemon in background
echo "Starting Forge daemon..."
FORGE_RES=$(dirname "$FORGE_JAR")/../forge-gui/res
export JAVA_OPTS="-Xmx4g"
java -jar "$FORGE_JAR" --daemon --port 17171 &
FORGE_PID=$!

# Wait for daemon to start
echo "Waiting for Forge daemon..."
sleep 30

# Check if daemon is running
if ! kill -0 $FORGE_PID 2>/dev/null; then
    echo "ERROR: Forge daemon failed to start"
    cat /var/log/forge-test.log
    exit 1
fi

# Run profiling
echo "Running Forge game profiler..."
python3 scripts/profile_forge_games.py \
    --host localhost \
    --port 17171 \
    --deck1 decks/test_deck1.dck \
    --deck2 decks/test_deck2.dck \
    --games NUM_GAMES_PLACEHOLDER \
    --timeout 120 \
    --verbose

# Upload results
echo "Uploading results..."
aws s3 cp forge_profile_results.json s3://BUCKET_PLACEHOLDER/test_results/forge_profile_TIMESTAMP_PLACEHOLDER.json

# Also upload full log
aws s3 cp /var/log/forge-test.log s3://BUCKET_PLACEHOLDER/test_results/forge_test_log_TIMESTAMP_PLACEHOLDER.txt

echo "Test complete! Results uploaded to S3."

# Stop Forge daemon
kill $FORGE_PID 2>/dev/null || true

# Shutdown instance after completion
shutdown -h now
USERDATA
)

# Replace placeholders in user data
USER_DATA="${USER_DATA//BUCKET_PLACEHOLDER/$S3_BUCKET}"
USER_DATA="${USER_DATA//PACKAGE_PLACEHOLDER/$PACKAGE_NAME}"
USER_DATA="${USER_DATA//TIMESTAMP_PLACEHOLDER/$TIMESTAMP}"
USER_DATA="${USER_DATA//NUM_GAMES_PLACEHOLDER/$NUM_GAMES}"

# Encode user data
USER_DATA_B64=$(echo "$USER_DATA" | base64)

# Launch spot instance
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
        "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}
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

if [ -n "$SPOT_REQUEST_ID" ]; then
    echo "Spot request ID: $SPOT_REQUEST_ID"
    echo "Waiting for spot instance..."

    # Wait for spot instance
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
        echo "Spot instance not fulfilled, cancelling..."
        aws ec2 cancel-spot-instance-requests --region "$REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID" 2>/dev/null || true
        INSTANCE_ID=""
    fi
fi

# Fall back to on-demand if spot failed
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
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
fi

echo ""
echo "============================================================"
echo "INSTANCE LAUNCHED"
echo "============================================================"
echo "Instance ID: $INSTANCE_ID"
echo ""

# Tag instance
aws ec2 create-tags --region "$REGION" \
    --resources "$INSTANCE_ID" \
    --tags "Key=Name,Value=forge-test-$TIMESTAMP" "Key=Project,Value=mtg-rl"

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance running at: $PUBLIC_IP"
echo ""

# Show monitoring commands
echo "Monitor the test with:"
echo "  # View logs:"
echo "  aws s3 cp s3://${S3_BUCKET}/test_results/forge_test_log_${TIMESTAMP}.txt - | tail -100"
echo ""
echo "  # Check instance status:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name'"
echo ""
echo "  # SSH (if key provided):"
if [ -n "$KEY_NAME" ]; then
    echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
else
    echo "  (no SSH key provided)"
fi
echo ""
echo "  # Download results when complete:"
echo "  aws s3 cp s3://${S3_BUCKET}/test_results/forge_profile_${TIMESTAMP}.json forge_profile_results.json"
echo ""
echo "Instance will automatically terminate after test completes."
echo "Estimated completion time: ~${DURATION_MINUTES} minutes"
echo ""

# Wait for results
echo "Waiting for results (timeout: ${DURATION_MINUTES} minutes)..."
TIMEOUT=$((DURATION_MINUTES * 60))
START_TIME=$(date +%s)

while true; do
    # Check if results exist
    if aws s3 ls "s3://${S3_BUCKET}/test_results/forge_profile_${TIMESTAMP}.json" &>/dev/null; then
        echo ""
        echo "Results available! Downloading..."
        aws s3 cp "s3://${S3_BUCKET}/test_results/forge_profile_${TIMESTAMP}.json" "forge_test_results_${TIMESTAMP}.json"

        echo ""
        echo "============================================================"
        echo "RESULTS"
        echo "============================================================"
        cat "forge_test_results_${TIMESTAMP}.json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Games: {data['num_games']}\")
print(f\"Total time: {data['total_time_s']:.1f}s\")
print(f\"Games/second: {data['games_per_second']:.2f}\")
print(f\"Decisions/second: {data['decisions_per_second']:.1f}\")
print(f\"Samples/hour: {data['samples_per_hour']:,.0f}\")
print(f\"Hours to 1M samples: {data['hours_to_1m']:.2f}\")
print(f\"Avg decision latency: {data['avg_decision_ms']:.2f}ms\")
print(f\"P95 decision latency: {data['p95_decision_ms']:.2f}ms\")
"
        break
    fi

    # Check timeout
    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo ""
        echo "Timeout waiting for results. Instance may still be running."
        echo "Check manually with the commands above."
        break
    fi

    # Check instance state
    STATE=$(aws ec2 describe-instances --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "unknown")

    if [ "$STATE" == "terminated" ] || [ "$STATE" == "stopped" ]; then
        echo ""
        echo "Instance terminated before completing. Check logs."
        aws s3 cp "s3://${S3_BUCKET}/test_results/forge_test_log_${TIMESTAMP}.txt" - 2>/dev/null | tail -50 || true
        break
    fi

    echo -n "."
    sleep 30
done

echo ""
echo "Done."
