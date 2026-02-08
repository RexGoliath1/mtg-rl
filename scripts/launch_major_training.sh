#!/bin/bash
set -euo pipefail

# ============================================================================
# Major Training Run Launcher
# ============================================================================
# Orchestrates a full training pipeline on AWS:
#   Phase 1: Data collection (10K games across multiple spot instances)
#   Phase 2: GPU training (30 epochs on g4dn.xlarge)
#   Phase 3: Evaluation + email notification with high priority
#
# Budget guard: refuses to launch if estimated cost exceeds remaining budget.
#
# Usage:
#   ./scripts/launch_major_training.sh                    # Full run
#   ./scripts/launch_major_training.sh --dry-run          # Show plan only
#   ./scripts/launch_major_training.sh --phase training   # Skip collection
#   ./scripts/launch_major_training.sh --games 5000       # Custom game count
# ============================================================================

# --- Configuration ---
S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
REGION="${REGION:-us-east-1}"
MONTHLY_BUDGET=100  # Hard cap in USD
COST_TRACKING_FILE="${HOME}/.mtg_rl_cost_tracking.json"

# Collection defaults
COLLECTION_INSTANCE_TYPE="c5.2xlarge"
COLLECTION_SPOT_PRICE="0.17"   # c5.2xlarge spot ~$0.10-0.13/hr
NUM_GAMES=10000
COLLECTION_WORKERS=8
GAME_TIMEOUT=60

# Training defaults
TRAINING_INSTANCE_TYPE="g4dn.xlarge"
TRAINING_SPOT_PRICE="0.20"     # g4dn.xlarge spot ~$0.16/hr
TRAINING_EPOCHS=30
TRAINING_BATCH_SIZE=256
TRAINING_LR="1e-3"
TRAINING_WARMUP=5
TRAINING_GRAD_ACCUM=1

# Flags
DRY_RUN=false
PHASE="all"  # all, collection, training
SKIP_COST_CHECK=false
NOTIFY_EMAIL="${FORGERL_NOTIFY_EMAIL:-}"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)     DRY_RUN=true; shift ;;
        --phase)       PHASE="$2"; shift 2 ;;
        --games)       NUM_GAMES="$2"; shift 2 ;;
        --epochs)      TRAINING_EPOCHS="$2"; shift 2 ;;
        --batch-size)  TRAINING_BATCH_SIZE="$2"; shift 2 ;;
        --skip-cost-check) SKIP_COST_CHECK=true; shift ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Show plan without launching"
            echo "  --phase PHASE      Run phase: all, collection, training"
            echo "  --games N          Number of games to collect (default: 10000)"
            echo "  --epochs N         Training epochs (default: 30)"
            echo "  --batch-size N     Training batch size (default: 256)"
            echo "  --skip-cost-check  Skip budget validation"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="major_${TIMESTAMP}"

# ============================================================================
# Cost Estimation
# ============================================================================

estimate_costs() {
    # Collection cost: c5.2xlarge spot, ~1hr for 10K games
    local collection_hours
    collection_hours=$(echo "scale=2; $NUM_GAMES / 10000 * 1.5" | bc)  # ~1.5hr for 10K
    local collection_cost
    collection_cost=$(echo "scale=2; $collection_hours * $COLLECTION_SPOT_PRICE" | bc)

    # Training cost: g4dn.xlarge spot, ~24hr for 30 epochs on 4M decisions
    local training_hours
    training_hours=$(echo "scale=2; $TRAINING_EPOCHS / 30 * 24" | bc)  # ~24hr for 30 epochs
    local training_cost
    training_cost=$(echo "scale=2; $training_hours * $TRAINING_SPOT_PRICE" | bc)

    # S3 storage cost (negligible but included): ~$0.02/GB/month, data is <100MB
    local storage_cost="0.01"

    local total_cost
    total_cost=$(echo "scale=2; $collection_cost + $training_cost + $storage_cost" | bc)

    echo "$collection_hours $collection_cost $training_hours $training_cost $storage_cost $total_cost"
}

get_current_month_spend() {
    # Try AWS Cost Explorer first (requires ce:GetCostAndUsage permission)
    local current_spend="0.00"

    if aws ce get-cost-and-usage \
        --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
        --granularity MONTHLY \
        --metrics "UnblendedCost" \
        --query 'ResultsByTime[0].Total.UnblendedCost.Amount' \
        --output text 2>/dev/null; then
        current_spend=$(aws ce get-cost-and-usage \
            --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
            --granularity MONTHLY \
            --metrics "UnblendedCost" \
            --query 'ResultsByTime[0].Total.UnblendedCost.Amount' \
            --output text 2>/dev/null || echo "0.00")
    fi

    # Fallback: check local tracking file
    if [ "$current_spend" = "0.00" ] && [ -f "$COST_TRACKING_FILE" ]; then
        local month_key
        month_key=$(date +%Y-%m)
        current_spend=$(python3 -c "
import json, sys
try:
    with open('$COST_TRACKING_FILE') as f:
        data = json.load(f)
    print(data.get('$month_key', {}).get('total_spent', '0.00'))
except Exception:
    print('0.00')
" 2>/dev/null || echo "0.00")
    fi

    echo "$current_spend"
}

update_cost_tracking() {
    local estimated_cost="$1"
    local run_id="$2"
    local month_key
    month_key=$(date +%Y-%m)

    python3 -c "
import json, os
from datetime import datetime

path = '$COST_TRACKING_FILE'
try:
    with open(path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}

month = '$month_key'
if month not in data:
    data[month] = {'total_spent': 0.0, 'runs': []}

data[month]['runs'].append({
    'run_id': '$run_id',
    'estimated_cost': float('$estimated_cost'),
    'timestamp': datetime.now().isoformat(),
    'status': 'launched'
})
data[month]['total_spent'] += float('$estimated_cost')

with open(path, 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true
}

# ============================================================================
# Validation
# ============================================================================

validate_prerequisites() {
    echo "Checking prerequisites..."

    # AWS credentials
    if ! aws sts get-caller-identity &>/dev/null; then
        echo "ERROR: AWS credentials not configured"
        echo "Run: aws configure"
        exit 1
    fi
    echo "  [OK] AWS credentials"

    # S3 bucket
    if ! aws s3 ls "s3://${S3_BUCKET}" &>/dev/null; then
        echo "ERROR: S3 bucket ${S3_BUCKET} does not exist"
        echo "Run: cd infrastructure && terraform apply"
        exit 1
    fi
    echo "  [OK] S3 bucket: ${S3_BUCKET}"

    # bc for cost calculations
    if ! command -v bc &>/dev/null; then
        echo "ERROR: 'bc' not found. Install with: brew install bc"
        exit 1
    fi
    echo "  [OK] bc available"

    # Python3
    if ! command -v python3 &>/dev/null; then
        echo "ERROR: python3 not found"
        exit 1
    fi
    echo "  [OK] python3 available"

    # Check email config (warn, don't block)
    if [ -z "$NOTIFY_EMAIL" ]; then
        echo "  [WARN] FORGERL_NOTIFY_EMAIL not set -- no email notification"
    else
        echo "  [OK] Email notification -> ${NOTIFY_EMAIL}"
    fi
}

# ============================================================================
# Phase 1: Data Collection
# ============================================================================

run_collection_phase() {
    echo ""
    echo "============================================================"
    echo "PHASE 1: DATA COLLECTION"
    echo "============================================================"
    echo "Instance:  ${COLLECTION_INSTANCE_TYPE} (spot @ \$${COLLECTION_SPOT_PRICE}/hr)"
    echo "Games:     ${NUM_GAMES}"
    echo "Workers:   ${COLLECTION_WORKERS}"
    echo "Output:    s3://${S3_BUCKET}/imitation_data/${RUN_ID}/"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would launch data collection instance"
        return 0
    fi

    # Use the existing deploy script
    bash "${SCRIPT_DIR}/deploy_data_collection.sh" \
        --games "$NUM_GAMES" \
        --workers "$COLLECTION_WORKERS" \
        --timeout "$GAME_TIMEOUT"

    echo ""
    echo "Collection instance launched. Waiting for completion..."

    # Poll for completion (check every 2 minutes, timeout after 3 hours)
    local max_checks=90  # 90 * 2min = 3 hours
    local check=0
    while [ $check -lt $max_checks ]; do
        sleep 120

        # Check for completion signal
        if aws s3 ls "s3://${S3_BUCKET}/imitation_data/collection_${TIMESTAMP}/collection_complete.json" &>/dev/null; then
            echo ""
            echo "Collection COMPLETE!"
            echo ""
            # List collected files
            aws s3 ls "s3://${S3_BUCKET}/imitation_data/collection_${TIMESTAMP}/" --human-readable
            return 0
        fi

        check=$((check + 1))
        echo "  Waiting for collection... (${check}/${max_checks} checks, $((check * 2)) min elapsed)"
    done

    echo "ERROR: Collection timed out after 3 hours"
    echo "Check instance status manually."
    return 1
}

# ============================================================================
# Phase 2: GPU Training
# ============================================================================

run_training_phase() {
    echo ""
    echo "============================================================"
    echo "PHASE 2: GPU TRAINING"
    echo "============================================================"
    echo "Instance:  ${TRAINING_INSTANCE_TYPE} (spot @ \$${TRAINING_SPOT_PRICE}/hr)"
    echo "Epochs:    ${TRAINING_EPOCHS}"
    echo "Batch:     ${TRAINING_BATCH_SIZE}"
    echo "LR:        ${TRAINING_LR} (warmup: ${TRAINING_WARMUP} epochs)"
    echo "AMP:       Enabled (mixed precision)"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would launch GPU training instance"
        return 0
    fi

    # Find the collection data path
    local data_path
    data_path=$(aws s3 ls "s3://${S3_BUCKET}/imitation_data/" --recursive \
        | grep "collection_complete.json" \
        | sort -r \
        | head -1 \
        | awk '{print $4}' \
        | sed 's|/collection_complete.json||')

    if [ -z "$data_path" ]; then
        echo "ERROR: No collection data found in S3"
        echo "Run collection phase first: $0 --phase collection"
        return 1
    fi

    echo "Using data: s3://${S3_BUCKET}/${data_path}/"

    # Package training code
    echo "Packaging training code..."
    local train_package="training_${TIMESTAMP}.tar.gz"
    COPYFILE_DISABLE=1 tar -czf "/tmp/${train_package}" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='forge-repo' \
        --exclude='data' \
        --exclude='checkpoints' \
        --exclude='wandb' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        -C "$PROJECT_DIR" \
        src scripts pyproject.toml 2>/dev/null || true

    aws s3 cp "/tmp/${train_package}" "s3://${S3_BUCKET}/test_packages/${train_package}" --quiet
    echo "  Code uploaded"

    # Find Deep Learning AMI
    local ami_id
    ami_id=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)

    if [ -z "$ami_id" ] || [ "$ami_id" = "None" ]; then
        echo "ERROR: Could not find Deep Learning AMI"
        return 1
    fi
    echo "Using AMI: ${ami_id}"

    # Get network info
    local vpc_id subnet_id sg_id
    vpc_id=$(aws ec2 describe-vpcs --region "$REGION" \
        --filters "Name=isDefault,Values=true" \
        --query 'Vpcs[0].VpcId' --output text)
    subnet_id=$(aws ec2 describe-subnets --region "$REGION" \
        --filters "Name=vpc-id,Values=$vpc_id" \
        --query 'Subnets[0].SubnetId' --output text)

    local sg_name="mtg-rl-training-sg"
    sg_id=$(aws ec2 describe-security-groups --region "$REGION" \
        --filters "Name=group-name,Values=$sg_name" "Name=vpc-id,Values=$vpc_id" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

    if [ "$sg_id" = "None" ] || [ -z "$sg_id" ]; then
        sg_id=$(aws ec2 create-security-group --region "$REGION" \
            --group-name "$sg_name" \
            --description "MTG RL training instances" \
            --vpc-id "$vpc_id" \
            --query 'GroupId' --output text)
        aws ec2 authorize-security-group-ingress --region "$REGION" \
            --group-id "$sg_id" --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
    fi

    # Create training userdata
    local notify_section=""
    if [ -n "$NOTIFY_EMAIL" ]; then
        notify_section="
# --- Phase 3: Email Notification ---
echo 'Sending training report email...'
export FORGERL_NOTIFY_EMAIL='${NOTIFY_EMAIL}'
export FORGERL_SMTP_HOST=\${FORGERL_SMTP_HOST:-smtp.gmail.com}
export FORGERL_SMTP_PORT=\${FORGERL_SMTP_PORT:-587}

# Try to retrieve SMTP credentials from AWS Secrets Manager
SMTP_SECRET=\$(aws secretsmanager get-secret-value \
    --region ${REGION} \
    --secret-id mtg-rl/smtp-credentials \
    --query SecretString --output text 2>/dev/null || echo '')
if [ -n \"\$SMTP_SECRET\" ]; then
    export FORGERL_SMTP_USER=\$(echo \"\$SMTP_SECRET\" | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"user\",\"\"))')
    export FORGERL_SMTP_PASS=\$(echo \"\$SMTP_SECRET\" | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"password\",\"\"))')
fi

cd /home/ubuntu/mtg
python3 -c \"
from src.utils.email_notifier import EmailNotifier
import json

with open('/home/ubuntu/training_results/metrics.json') as f:
    metrics = json.load(f)
metrics['model_name'] = 'AlphaZero Major Run ${RUN_ID}'

notifier = EmailNotifier()
notifier.send_training_complete(
    metrics=metrics,
    report_path='/home/ubuntu/training_results/training_report.pdf' if __import__('os').path.exists('/home/ubuntu/training_results/training_report.pdf') else None,
)
print('Email sent successfully')
\" || echo 'Email notification failed (non-fatal)'
"
    fi

    local user_data
    user_data=$(cat << USERDATA
#!/bin/bash
set -ex

exec > >(tee /var/log/training.log) 2>&1
echo "=========================================="
echo "MAJOR TRAINING RUN: ${RUN_ID}"
echo "=========================================="
date

cd /home/ubuntu

# Install system deps
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv unzip > /dev/null

# Install AWS CLI v2
curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install --update
rm -rf aws awscliv2.zip

# Download training code
aws s3 cp s3://${S3_BUCKET}/test_packages/${train_package} code.tar.gz
mkdir -p mtg && cd mtg
tar -xzf ../code.tar.gz

# Install Python dependencies with UV
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
uv sync --extra dev 2>/dev/null || pip install torch numpy h5py safetensors

# Download training data from S3
echo "Downloading training data..."
mkdir -p /home/ubuntu/training_data
aws s3 sync "s3://${S3_BUCKET}/${data_path}/" /home/ubuntu/training_data/ \
    --exclude "*.log" --exclude "*.txt"

# Count HDF5 files
HDF5_COUNT=\$(find /home/ubuntu/training_data -name "*.h5" -o -name "*.hdf5" | wc -l)
echo "Found \$HDF5_COUNT HDF5 data files"

# Create output directory
mkdir -p /home/ubuntu/training_results

# Run training with optimizations
echo "=========================================="
echo "STARTING GPU TRAINING"
echo "=========================================="
echo "Epochs:     ${TRAINING_EPOCHS}"
echo "Batch size: ${TRAINING_BATCH_SIZE}"
echo "LR:         ${TRAINING_LR}"
echo "Warmup:     ${TRAINING_WARMUP} epochs"
echo "AMP:        Enabled"
echo "Device:     \$(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
date

TRAIN_START=\$(date +%s)

# Run imitation training
python3 -m src.training.forge_imitation \
    --train-only \
    --epochs ${TRAINING_EPOCHS} \
    --batch-size ${TRAINING_BATCH_SIZE} \
    --lr ${TRAINING_LR} \
    --warmup-epochs ${TRAINING_WARMUP} \
    --grad-accum ${TRAINING_GRAD_ACCUM} \
    --num-workers 4 \
    --checkpoint /home/ubuntu/training_results/model.pt \
    --tensorboard \
    --tb-log-dir /home/ubuntu/training_results/tensorboard \
    2>&1 | tee /home/ubuntu/training_results/training.log

TRAIN_END=\$(date +%s)
TRAIN_DURATION=\$((TRAIN_END - TRAIN_START))

echo ""
echo "Training completed in \${TRAIN_DURATION}s (\$((TRAIN_DURATION / 3600))h \$(((TRAIN_DURATION % 3600) / 60))m)"

# Save metrics
python3 -c "
import json
metrics = {
    'run_id': '${RUN_ID}',
    'training_duration_s': \${TRAIN_DURATION},
    'epochs': ${TRAINING_EPOCHS},
    'batch_size': ${TRAINING_BATCH_SIZE},
    'learning_rate': '${TRAINING_LR}',
    'warmup_epochs': ${TRAINING_WARMUP},
    'amp_enabled': True,
    'instance_type': '${TRAINING_INSTANCE_TYPE}',
    'games_collected': ${NUM_GAMES},
}

# Try to read training history from checkpoint
try:
    import torch
    ckpt = torch.load('/home/ubuntu/training_results/model.pt', map_location='cpu', weights_only=False)
    history = ckpt.get('training_history', [])
    if history:
        metrics['final_loss'] = history[-1].get('loss', 0)
        metrics['final_accuracy'] = history[-1].get('accuracy', 0)
        metrics['final_lr'] = history[-1].get('lr', 0)
        metrics['history'] = history
except Exception as e:
    print(f'Could not read checkpoint: {e}')

with open('/home/ubuntu/training_results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, default=str)
print('Metrics saved')
"

# Upload results to S3
echo "Uploading results to S3..."
aws s3 sync /home/ubuntu/training_results/ \
    "s3://${S3_BUCKET}/training_runs/${RUN_ID}/" \
    --exclude "tensorboard/*"

# Upload TensorBoard logs separately
aws s3 sync /home/ubuntu/training_results/tensorboard/ \
    "s3://${S3_BUCKET}/tensorboard-logs/${RUN_ID}/" 2>/dev/null || true

# Upload training log
aws s3 cp /var/log/training.log \
    "s3://${S3_BUCKET}/training_runs/${RUN_ID}/instance_log.txt"

# Signal completion
echo '{"status":"complete","run_id":"${RUN_ID}","timestamp":"${TIMESTAMP}"}' | \
    aws s3 cp - "s3://${S3_BUCKET}/training_runs/${RUN_ID}/training_complete.json"

echo "Results uploaded to s3://${S3_BUCKET}/training_runs/${RUN_ID}/"

${notify_section}

echo "=========================================="
echo "TRAINING RUN COMPLETE"
echo "=========================================="
date

# Shutdown
shutdown -h now
USERDATA
)

    # Encode user data
    local user_data_b64
    user_data_b64=$(echo "$user_data" | base64)

    # Launch spot instance
    echo ""
    echo "Launching GPU training instance..."

    local launch_spec
    launch_spec=$(cat << EOF
{
    "ImageId": "$ami_id",
    "InstanceType": "$TRAINING_INSTANCE_TYPE",
    "SecurityGroupIds": ["$sg_id"],
    "SubnetId": "$subnet_id",
    "IamInstanceProfile": {"Name": "mtg-rl-training"},
    "UserData": "$user_data_b64",
    "BlockDeviceMappings": [{
        "DeviceName": "/dev/sda1",
        "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}
    }]
}
EOF
)

    local spot_response instance_id
    spot_response=$(aws ec2 request-spot-instances \
        --region "$REGION" \
        --instance-count 1 \
        --type "one-time" \
        --spot-price "$TRAINING_SPOT_PRICE" \
        --launch-specification "$launch_spec" \
        --query 'SpotInstanceRequests[0]' \
        --output json 2>/dev/null || echo "{}")

    local spot_request_id
    spot_request_id=$(echo "$spot_response" | python3 -c "import json,sys; print(json.load(sys.stdin).get('SpotInstanceRequestId',''))" 2>/dev/null || echo "")

    instance_id=""
    if [ -n "$spot_request_id" ]; then
        echo "Spot request: $spot_request_id"
        echo "Waiting for fulfillment..."

        for i in {1..30}; do
            instance_id=$(aws ec2 describe-spot-instance-requests \
                --region "$REGION" \
                --spot-instance-request-ids "$spot_request_id" \
                --query 'SpotInstanceRequests[0].InstanceId' \
                --output text 2>/dev/null || echo "None")

            if [ "$instance_id" != "None" ] && [ -n "$instance_id" ]; then
                break
            fi
            echo "  Waiting... ($i/30)"
            sleep 10
        done
    fi

    # Fallback to on-demand
    if [ -z "$instance_id" ] || [ "$instance_id" = "None" ]; then
        echo "Spot not available, using on-demand instance..."
        instance_id=$(aws ec2 run-instances \
            --region "$REGION" \
            --image-id "$ami_id" \
            --instance-type "$TRAINING_INSTANCE_TYPE" \
            --security-group-ids "$sg_id" \
            --subnet-id "$subnet_id" \
            --iam-instance-profile "Name=mtg-rl-training" \
            --user-data "$user_data" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
            --query 'Instances[0].InstanceId' \
            --output text)
    fi

    # Tag instance
    aws ec2 create-tags --region "$REGION" \
        --resources "$instance_id" \
        --tags "Key=Name,Value=major-training-${TIMESTAMP}" \
               "Key=Project,Value=mtg-rl" \
               "Key=Task,Value=major-training"

    aws ec2 wait instance-running --region "$REGION" --instance-ids "$instance_id"
    local public_ip
    public_ip=$(aws ec2 describe-instances --region "$REGION" \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo ""
    echo "GPU training instance launched!"
    echo "  Instance: ${instance_id}"
    echo "  IP:       ${public_ip}"
    echo ""
    echo "Monitor:"
    echo "  aws s3 ls s3://${S3_BUCKET}/training_runs/${RUN_ID}/training_complete.json"
    echo ""
    echo "Download results when complete:"
    echo "  aws s3 sync s3://${S3_BUCKET}/training_runs/${RUN_ID}/ training_output/${RUN_ID}/"
}

# ============================================================================
# Main
# ============================================================================

echo ""
echo "============================================================"
echo "FORGERL MAJOR TRAINING RUN"
echo "============================================================"
echo "Run ID:    ${RUN_ID}"
echo "Phase:     ${PHASE}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Prerequisites
validate_prerequisites

# Cost estimation
echo ""
echo "--- Cost Estimation ---"
read -r coll_hrs coll_cost train_hrs train_cost storage_cost total_cost <<< "$(estimate_costs)"

echo "Phase 1 (Collection): ~${coll_hrs}h @ \$${COLLECTION_SPOT_PRICE}/hr = \$${coll_cost}"
echo "Phase 2 (Training):   ~${train_hrs}h @ \$${TRAINING_SPOT_PRICE}/hr = \$${train_cost}"
echo "S3 Storage:           \$${storage_cost}"
echo "---------------------------------------"
echo "TOTAL ESTIMATED:      \$${total_cost}"
echo ""

# Budget check
if [ "$SKIP_COST_CHECK" = false ]; then
    current_spend=$(get_current_month_spend)
    remaining_budget=$(echo "scale=2; $MONTHLY_BUDGET - $current_spend" | bc)
    echo "Current month spend:  \$${current_spend}"
    echo "Remaining budget:     \$${remaining_budget} (of \$${MONTHLY_BUDGET})"
    echo ""

    would_exceed=$(echo "$total_cost > $remaining_budget" | bc)
    if [ "$would_exceed" = "1" ]; then
        echo "ERROR: Estimated cost (\$${total_cost}) exceeds remaining budget (\$${remaining_budget})"
        echo ""
        echo "Options:"
        echo "  1. Reduce --games or --epochs to lower cost"
        echo "  2. Use --skip-cost-check to override (CAREFUL!)"
        echo "  3. Wait for next billing cycle"
        exit 1
    fi
    echo "Budget check PASSED: \$${total_cost} <= \$${remaining_budget}"
fi

# Dry run summary
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "============================================================"
    echo "[DRY RUN] EXECUTION PLAN"
    echo "============================================================"
    if [ "$PHASE" = "all" ] || [ "$PHASE" = "collection" ]; then
        echo ""
        echo "Phase 1: Data Collection"
        echo "  - Launch ${COLLECTION_INSTANCE_TYPE} spot instance"
        echo "  - Collect ${NUM_GAMES} games with ${COLLECTION_WORKERS} workers"
        echo "  - Upload to s3://${S3_BUCKET}/imitation_data/${RUN_ID}/"
        echo "  - Estimated time: ~${coll_hrs}h, cost: \$${coll_cost}"
    fi
    if [ "$PHASE" = "all" ] || [ "$PHASE" = "training" ]; then
        echo ""
        echo "Phase 2: GPU Training"
        echo "  - Launch ${TRAINING_INSTANCE_TYPE} spot instance (T4 GPU)"
        echo "  - Train for ${TRAINING_EPOCHS} epochs (batch=${TRAINING_BATCH_SIZE})"
        echo "  - Mixed precision (AMP) enabled"
        echo "  - LR: ${TRAINING_LR} with ${TRAINING_WARMUP}-epoch cosine warmup"
        echo "  - Gradient clipping: max_norm=1.0"
        echo "  - Save to s3://${S3_BUCKET}/training_runs/${RUN_ID}/"
        echo "  - Estimated time: ~${train_hrs}h, cost: \$${train_cost}"
    fi
    echo ""
    echo "Phase 3: Notification"
    if [ -n "$NOTIFY_EMAIL" ]; then
        echo "  - Send HIGH priority email to ${NOTIFY_EMAIL}"
        echo "  - Attach training report PDF"
    else
        echo "  - No email (set FORGERL_NOTIFY_EMAIL to enable)"
    fi
    echo ""
    echo "Total estimated cost: \$${total_cost}"
    echo ""
    echo "To launch for real, remove --dry-run flag."
    echo "============================================================"
    exit 0
fi

# Confirmation
echo ""
echo "Ready to launch major training run."
echo "Estimated cost: \$${total_cost}"
read -p "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Track costs
update_cost_tracking "$total_cost" "$RUN_ID"

# Execute phases
if [ "$PHASE" = "all" ] || [ "$PHASE" = "collection" ]; then
    run_collection_phase
fi

if [ "$PHASE" = "all" ] || [ "$PHASE" = "training" ]; then
    run_training_phase
fi

echo ""
echo "============================================================"
echo "MAJOR TRAINING RUN LAUNCHED SUCCESSFULLY"
echo "============================================================"
echo "Run ID: ${RUN_ID}"
echo "Estimated total cost: \$${total_cost}"
echo ""
echo "Monitor progress:"
echo "  # Check collection:"
echo "  aws s3 ls s3://${S3_BUCKET}/imitation_data/collection_${TIMESTAMP}/collection_complete.json"
echo ""
echo "  # Check training:"
echo "  aws s3 ls s3://${S3_BUCKET}/training_runs/${RUN_ID}/training_complete.json"
echo ""
echo "  # Download results:"
echo "  aws s3 sync s3://${S3_BUCKET}/training_runs/${RUN_ID}/ training_output/${RUN_ID}/"
echo ""
if [ -n "$NOTIFY_EMAIL" ]; then
    echo "You will receive a HIGH PRIORITY email at ${NOTIFY_EMAIL} when training completes."
fi
echo "============================================================"
