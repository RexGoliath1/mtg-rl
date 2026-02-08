"""
EC2 userdata script generation for ForgeRL deployments.

Generates bash scripts that run on instance boot for both data
collection and GPU training phases.

Uses string.Template for variable substitution (no Jinja2 dependency).

Key GOTCHAs from project memory:
  - Forge daemon needs Xvfb (X11) even in daemon mode (Java Swing)
  - CWD must be forge-gui-desktop/ so ../forge-gui/res/ resolves
  - Deck paths must be absolute (Forge loadDeck() fails with relative)
  - Forge CLI uses positional args: daemon -p 17171 (NOT --daemon --port)
"""

from __future__ import annotations

from string import Template

from src.deploy.config import CollectionConfig, DeployConfig, TrainingConfig


# --------------------------------------------------------------------------
# Collection userdata template
# --------------------------------------------------------------------------

_COLLECTION_TEMPLATE = Template(r"""#!/bin/bash
set -ex

exec > >(tee /var/log/data-collection.log) 2>&1
echo "=========================================="
echo "DATA COLLECTION SETUP"
echo "=========================================="
date

cd /home/ubuntu

# Install dependencies
apt-get update -qq
apt-get install -y -qq openjdk-17-jdk maven python3-pip python3-venv unzip xvfb > /dev/null

# Install AWS CLI v2
curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install --update
rm -rf aws awscliv2.zip

# Download code package
aws s3 cp s3://${s3_bucket}/test_packages/${code_package} code.tar.gz
mkdir -p mtg && cd mtg
tar -xzf ../code.tar.gz

# Install Python dependencies
python3 -m pip install -q numpy h5py requests

# Check for pre-built Forge JAR
if aws s3 ls s3://${s3_bucket}/test_packages/${forge_package} &>/dev/null; then
    echo "Using pre-built Forge JAR..."
    aws s3 cp s3://${s3_bucket}/test_packages/${forge_package} forge_jar.tar.gz
    mkdir -p forge-repo
    tar -xzf forge_jar.tar.gz -C forge-repo
    FORGE_JAR=$$(find forge-repo -name "*jar-with-dependencies.jar" ! -name "._*" | head -1)
else
    echo "Building Forge from source..."
    git clone --depth 1 -b ${forge_branch} https://github.com/RexGoliath1/forge.git forge-repo
    cd forge-repo
    mvn package -DskipTests -pl forge-gui-desktop -am -q
    cd ..
    FORGE_JAR=$$(find forge-repo -name "*jar-with-dependencies.jar" ! -name "._*" | head -1)
fi

echo "Forge JAR: $$FORGE_JAR"

# Start virtual display (Forge GUI needs X11 even in daemon mode)
echo "Starting Xvfb..."
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Convert to absolute path
FORGE_JAR="$$(pwd)/$$FORGE_JAR"

# Verify Forge JAR is valid
if [ ! -f "$$FORGE_JAR" ]; then
    echo "ERROR: Forge JAR not found at: $$FORGE_JAR"
    exit 1
fi
echo "JAR size: $$(du -h "$$FORGE_JAR" | cut -f1)"

# Forge GuiHeadless expects CWD=forge-gui-desktop/ so ../forge-gui/res/ resolves
FORGE_WORKDIR="$$(pwd)/forge-repo/forge-gui-desktop"
echo "Forge working dir: $$FORGE_WORKDIR"

# Start Forge daemon
echo "Starting Forge daemon..."
export JAVA_OPTS="-Xmx${forge_java_heap}"
cd "$$FORGE_WORKDIR"
java -jar "$$FORGE_JAR" daemon -p ${forge_port} > /var/log/forge-daemon.log 2>&1 &
FORGE_PID=$$!
cd /home/ubuntu/mtg

# Wait for daemon to start (cold JVM + loading card data)
echo "Waiting for Forge daemon to initialize..."
for i in $$(seq 1 12); do
    sleep 10
    if ! kill -0 $$FORGE_PID 2>/dev/null; then
        echo "ERROR: Forge daemon crashed. Last 30 lines of log:"
        tail -30 /var/log/forge-daemon.log
        aws s3 cp /var/log/forge-daemon.log \
            s3://${s3_bucket}/imitation_data/${run_id}/forge_crash.log || true
        exit 1
    fi
    echo "  Still starting... ($$((i*10))s)"
done
echo "Forge daemon running (PID: $$FORGE_PID)"

# Run data collection
echo "=========================================="
echo "STARTING DATA COLLECTION"
echo "=========================================="
echo "Games: ${num_games}"
echo "Workers: ${num_workers}"
echo "Timeout: ${game_timeout}"
date

python3 scripts/collect_ai_training_data.py \
    --games ${num_games} \
    --output /home/ubuntu/training_data \
    --host localhost \
    --port ${forge_port} \
    --workers ${num_workers} \
    --timeout ${game_timeout} \
    --save-interval ${save_interval}

echo "=========================================="
echo "COLLECTION COMPLETE"
echo "=========================================="
date

# Upload results to S3
echo "Uploading results to S3..."
aws s3 sync /home/ubuntu/training_data/ \
    s3://${s3_bucket}/imitation_data/${run_id}/ \
    --exclude "*.tex"

# Upload logs
aws s3 cp /var/log/data-collection.log \
    s3://${s3_bucket}/imitation_data/${run_id}/collection_log.txt
aws s3 cp /var/log/forge-daemon.log \
    s3://${s3_bucket}/imitation_data/${run_id}/forge_daemon.log 2>/dev/null || true

echo "Results uploaded to s3://${s3_bucket}/imitation_data/${run_id}/"

# Signal completion
echo '{"status":"complete","timestamp":"${timestamp}","games":${num_games}}' | \
    aws s3 cp - s3://${s3_bucket}/imitation_data/${run_id}/collection_complete.json

# Stop Forge daemon
kill $$FORGE_PID 2>/dev/null || true

echo "Shutting down..."
shutdown -h now
""")


# --------------------------------------------------------------------------
# Training userdata template
# --------------------------------------------------------------------------

_TRAINING_TEMPLATE = Template(r"""#!/bin/bash
set -ex

exec > >(tee /var/log/training.log) 2>&1
echo "=========================================="
echo "MAJOR TRAINING RUN: ${run_id}"
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
aws s3 cp s3://${s3_bucket}/test_packages/${code_package} code.tar.gz
mkdir -p mtg && cd mtg
tar -xzf ../code.tar.gz

# Install Python dependencies with UV
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$$HOME/.local/bin:$$PATH"

# Weights & Biases setup
WANDB_FLAG=""
${wandb_setup}

uv sync --extra dev 2>/dev/null || pip install torch numpy h5py safetensors

# Download training data from S3
echo "Downloading training data..."
mkdir -p /home/ubuntu/training_data
aws s3 sync "s3://${s3_bucket}/${data_path}/" /home/ubuntu/training_data/ \
    --exclude "*.log" --exclude "*.txt"

# Count HDF5 files
HDF5_COUNT=$$(find /home/ubuntu/training_data -name "*.h5" -o -name "*.hdf5" | wc -l)
echo "Found $$HDF5_COUNT HDF5 data files"

# Create output directory
mkdir -p /home/ubuntu/training_results

# Run training with optimizations
echo "=========================================="
echo "STARTING GPU TRAINING"
echo "=========================================="
echo "Epochs:     ${epochs}"
echo "Batch size: ${batch_size}"
echo "LR:         ${learning_rate}"
echo "Warmup:     ${warmup_epochs} epochs"
echo "AMP:        Enabled"
echo "Device:     $$(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
date

TRAIN_START=$$(date +%s)

# Run imitation training
python3 -m src.training.forge_imitation \
    --train-only \
    --epochs ${epochs} \
    --batch-size ${batch_size} \
    --lr ${learning_rate} \
    --warmup-epochs ${warmup_epochs} \
    --grad-accum ${grad_accum} \
    --num-workers ${num_data_workers} \
    --checkpoint /home/ubuntu/training_results/model.pt \
    --tensorboard \
    --tb-log-dir /home/ubuntu/training_results/tensorboard \
    $$WANDB_FLAG \
    2>&1 | tee /home/ubuntu/training_results/training.log

TRAIN_END=$$(date +%s)
TRAIN_DURATION=$$((TRAIN_END - TRAIN_START))

echo ""
echo "Training completed in $${TRAIN_DURATION}s ($$((TRAIN_DURATION / 3600))h $$(((TRAIN_DURATION % 3600) / 60))m)"

# Save metrics
python3 -c "
import json
metrics = {
    'run_id': '${run_id}',
    'training_duration_s': $${TRAIN_DURATION},
    'epochs': ${epochs},
    'batch_size': ${batch_size},
    'learning_rate': '${learning_rate}',
    'warmup_epochs': ${warmup_epochs},
    'amp_enabled': True,
    'instance_type': '${instance_type}',
}
try:
    import torch
    ckpt = torch.load('/home/ubuntu/training_results/model.pt', map_location='cpu', weights_only=False)
    history = ckpt.get('training_history', [])
    if history:
        metrics['final_loss'] = history[-1].get('loss', 0)
        metrics['final_accuracy'] = history[-1].get('accuracy', 0)
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
    "s3://${s3_bucket}/training_runs/${run_id}/" \
    --exclude "tensorboard/*"

# Upload TensorBoard logs separately
aws s3 sync /home/ubuntu/training_results/tensorboard/ \
    "s3://${s3_bucket}/tensorboard-logs/${run_id}/" 2>/dev/null || true

# Upload training log
aws s3 cp /var/log/training.log \
    "s3://${s3_bucket}/training_runs/${run_id}/instance_log.txt"

# Signal completion
echo '{"status":"complete","run_id":"${run_id}","timestamp":"${timestamp}"}' | \
    aws s3 cp - "s3://${s3_bucket}/training_runs/${run_id}/training_complete.json"

echo "Results uploaded to s3://${s3_bucket}/training_runs/${run_id}/"

${notification_section}

echo "=========================================="
echo "TRAINING RUN COMPLETE"
echo "=========================================="
date

# Shutdown
shutdown -h now
""")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_collection_userdata(
    deploy_config: DeployConfig,
    collection_config: CollectionConfig,
    run_id: str,
    timestamp: str,
    code_package: str,
    forge_package: str = "",
) -> str:
    """
    Generate bash userdata script for a data collection EC2 instance.

    Args:
        deploy_config: Top-level deploy settings.
        collection_config: Collection-specific settings.
        run_id: Unique run identifier (e.g. "collection_20260208_143000").
        timestamp: Timestamp string for S3 paths.
        code_package: Filename of the code tarball in S3.
        forge_package: Filename of the Forge tarball in S3 (empty to build on instance).

    Returns:
        Complete bash script as a string.
    """
    return _COLLECTION_TEMPLATE.substitute(
        s3_bucket=deploy_config.s3_bucket,
        code_package=code_package,
        forge_package=forge_package,
        forge_branch=collection_config.forge_branch,
        forge_port=collection_config.forge_port,
        forge_java_heap=collection_config.forge_java_heap,
        num_games=collection_config.num_games,
        num_workers=collection_config.num_workers,
        game_timeout=collection_config.game_timeout,
        save_interval=collection_config.save_interval,
        run_id=run_id,
        timestamp=timestamp,
    )


def generate_training_userdata(
    deploy_config: DeployConfig,
    training_config: TrainingConfig,
    run_id: str,
    timestamp: str,
    code_package: str,
    data_path: str,
) -> str:
    """
    Generate bash userdata script for a GPU training EC2 instance.

    Args:
        deploy_config: Top-level deploy settings.
        training_config: Training-specific settings.
        run_id: Unique run identifier.
        timestamp: Timestamp string.
        code_package: Filename of the code tarball in S3.
        data_path: S3 key prefix where training data resides.

    Returns:
        Complete bash script as a string.
    """
    # Build W&B setup section
    wandb_setup = ""
    if deploy_config.wandb_api_key:
        wandb_setup = (
            f"export WANDB_API_KEY='{deploy_config.wandb_api_key}'\n"
            f"export WANDB_ENTITY='{deploy_config.wandb_entity}'\n"
            f"export WANDB_PROJECT='{deploy_config.wandb_project}'\n"
            "pip install wandb 2>/dev/null || true\n"
            'WANDB_FLAG="--wandb"\n'
            f'echo "W&B tracking enabled (entity: {deploy_config.wandb_entity},'
            f' project: {deploy_config.wandb_project})"'
        )

    notification = generate_notification_section(deploy_config, run_id)

    return _TRAINING_TEMPLATE.substitute(
        s3_bucket=deploy_config.s3_bucket,
        code_package=code_package,
        data_path=data_path,
        run_id=run_id,
        timestamp=timestamp,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        warmup_epochs=training_config.warmup_epochs,
        grad_accum=training_config.grad_accum,
        num_data_workers=training_config.num_data_workers,
        instance_type=training_config.instance_type,
        wandb_setup=wandb_setup,
        notification_section=notification,
    )


def generate_notification_section(
    deploy_config: DeployConfig,
    run_id: str,
) -> str:
    """
    Generate the email notification bash snippet for training userdata.

    Returns an empty string if no notification email is configured.

    Args:
        deploy_config: Deploy settings (checks notify_email).
        run_id: Run identifier for the email subject.

    Returns:
        Bash script snippet for email notification.
    """
    if not deploy_config.notify_email:
        return ""

    return f"""
# --- Phase 3: Email Notification ---
echo 'Sending training report email...'
export FORGERL_NOTIFY_EMAIL='{deploy_config.notify_email}'
export FORGERL_SMTP_HOST=${{FORGERL_SMTP_HOST:-smtp.gmail.com}}
export FORGERL_SMTP_PORT=${{FORGERL_SMTP_PORT:-587}}

# Try to retrieve SMTP credentials from AWS Secrets Manager
SMTP_SECRET=$(aws secretsmanager get-secret-value \
    --region {deploy_config.region} \
    --secret-id mtg-rl/smtp-credentials \
    --query SecretString --output text 2>/dev/null || echo '')
if [ -n "$SMTP_SECRET" ]; then
    export FORGERL_SMTP_USER=$(echo "$SMTP_SECRET" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("user",""))')
    export FORGERL_SMTP_PASS=$(echo "$SMTP_SECRET" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("password",""))')
fi

cd /home/ubuntu/mtg
python3 -c "
from src.utils.email_notifier import EmailNotifier
import json

with open('/home/ubuntu/training_results/metrics.json') as f:
    metrics = json.load(f)
metrics['model_name'] = 'AlphaZero Major Run {run_id}'

notifier = EmailNotifier()
notifier.send_training_complete(
    metrics=metrics,
    report_path='/home/ubuntu/training_results/training_report.pdf' if __import__('os').path.exists('/home/ubuntu/training_results/training_report.pdf') else None,
)
print('Email sent successfully')
" || echo 'Email notification failed (non-fatal)'
"""
