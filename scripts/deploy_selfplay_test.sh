#!/bin/bash
# =============================================================================
# Deploy Parallel Self-Play Test to AWS
# =============================================================================
#
# Deploys a 10-minute test run of parallel self-play training to verify
# the infrastructure works before adding Forge integration.
#
# Usage:
#   ./scripts/deploy_selfplay_test.sh [--duration MINUTES] [--actors N]
#
# =============================================================================

set -e

# Configuration
S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"
SPOT_PRICE="0.25"
DURATION_MINUTES="${1:-10}"
NUM_ACTORS="${2:-8}"

echo "============================================================"
echo "MTG RL - Parallel Self-Play Test Deployment"
echo "============================================================"
echo "S3 Bucket: $S3_BUCKET"
echo "Region: $REGION"
echo "Instance: $INSTANCE_TYPE (spot)"
echo "Duration: $DURATION_MINUTES minutes"
echo "Actors: $NUM_ACTORS"
echo ""

# Check AWS credentials
echo "[1/5] Checking AWS credentials..."
if ! aws sts get-caller-identity --query Account --output text >/dev/null 2>&1; then
    echo "ERROR: AWS credentials not configured"
    exit 1
fi
echo "  AWS Account: $(aws sts get-caller-identity --query Account --output text)"

# Package code
echo ""
echo "[2/5] Packaging code..."
cd "$(dirname "$0")/.."

# Create tarball excluding large/unnecessary files
tar -czf training-code.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='forge-repo' \
    --exclude='data/17lands' \
    --exclude='checkpoints' \
    --exclude='training_output' \
    --exclude='wandb' \
    --exclude='*.tar.gz' \
    --exclude='.terraform' \
    --exclude='terraform.tfstate*' \
    --exclude='reports' \
    --exclude='*.h5' \
    .

echo "  Package size: $(du -h training-code.tar.gz | cut -f1)"

# Upload to S3
echo ""
echo "[3/5] Uploading to S3..."
aws s3 cp training-code.tar.gz "s3://$S3_BUCKET/code/selfplay-test-code.tar.gz"
aws s3 cp data/card_mechanics_commander.h5 "s3://$S3_BUCKET/data/card_mechanics_commander.h5" 2>/dev/null || echo "  (H5 file already uploaded or not found)"
rm training-code.tar.gz
echo "  Uploaded to s3://$S3_BUCKET/code/selfplay-test-code.tar.gz"

# Create userdata script
echo ""
echo "[4/5] Creating userdata script..."

cat > /tmp/selfplay_userdata.sh << 'USERDATA'
#!/bin/bash
exec > >(tee /var/log/selfplay-setup.log) 2>&1

echo "============================================================"
echo "MTG RL - Parallel Self-Play Test Setup"
echo "Started at: $(date)"
echo "============================================================"

# Configuration
S3_BUCKET="__S3_BUCKET__"
DURATION_MINUTES="__DURATION__"
NUM_ACTORS="__ACTORS__"

echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Duration: $DURATION_MINUTES minutes"
echo "  Actors: $NUM_ACTORS"

# Install dependencies
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -y
apt-get install -y python3-pip python3-venv htop nvtop unzip

# Setup working directory
echo ""
echo "[2/6] Setting up working directory..."
mkdir -p /home/ubuntu/mtg-rl
cd /home/ubuntu/mtg-rl

# Download code
echo ""
echo "[3/6] Downloading code from S3..."
aws s3 cp "s3://$S3_BUCKET/code/selfplay-test-code.tar.gz" .
tar -xzf selfplay-test-code.tar.gz
rm selfplay-test-code.tar.gz

# Download H5 file
echo ""
echo "[4/6] Downloading card mechanics data..."
mkdir -p data
aws s3 cp "s3://$S3_BUCKET/data/card_mechanics_commander.h5" data/ || echo "H5 not found, will use default"

# Setup Python environment
echo ""
echo "[5/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy h5py

# Run the test
echo ""
echo "[6/6] Running parallel self-play test..."
cd /home/ubuntu/mtg-rl

# Create the test script
cat > run_selfplay_test.py << 'TESTSCRIPT'
#!/usr/bin/env python3
"""
Cloud Parallel Self-Play Test

Runs a timed test of the parallel self-play infrastructure.
"""

import os
import sys
import time
import json
import torch

sys.path.insert(0, os.getcwd())

from src.training.profiler import TrainingProfiler, get_gpu_info, compare_configurations

# Simple test network (same as local test)
import torch.nn as nn
from typing import Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np


class SimpleTestNetwork(nn.Module):
    def __init__(self, state_dim=512, num_actions=153, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, state, action_mask):
        h = self.encoder(state)
        logits = self.policy_head(h)
        logits = logits.masked_fill(action_mask == 0, float('-inf'))
        policy = torch.softmax(logits, dim=-1)
        value = self.value_head(h)
        return policy, value


@dataclass
class Config:
    num_actors: int = 8
    games_per_iteration: int = 32
    state_dim: int = 512
    num_actions: int = 153
    batch_size: int = 64
    buffer_size: int = 100000
    min_buffer_size: int = 100


def simulate_game(network, config, device, profiler):
    samples = []
    game_length = np.random.randint(30, 50)

    for move in range(game_length):
        state = torch.randn(config.state_dim)
        action_mask = torch.ones(config.num_actions)
        num_legal = np.random.randint(5, 30)
        invalid = np.random.choice(config.num_actions, config.num_actions - num_legal, replace=False)
        action_mask[invalid] = 0

        with profiler.measure("forward_pass"):
            with torch.no_grad():
                state_batch = state.unsqueeze(0).to(device)
                mask_batch = action_mask.unsqueeze(0).to(device)
                policy, value = network(state_batch, mask_batch)
                policy = policy.squeeze(0).cpu()

        noise = torch.from_numpy(np.random.dirichlet([0.3] * config.num_actions)).float()
        noisy_policy = 0.75 * policy + 0.25 * noise
        noisy_policy = noisy_policy * action_mask
        noisy_policy = noisy_policy / (noisy_policy.sum() + 1e-8)

        samples.append({
            "state": state,
            "policy": noisy_policy,
            "action_mask": action_mask,
        })

    winner = np.random.choice([0, 1])
    for i, sample in enumerate(samples):
        player = i % 2
        sample["value"] = torch.tensor([1.0 if player == winner else -1.0])

    profiler.increment("games", 1)
    profiler.increment("samples", len(samples))
    return samples


def train_step(network, optimizer, replay_buffer, config, device, profiler):
    if len(replay_buffer) < config.min_buffer_size:
        return 0.0

    indices = np.random.choice(len(replay_buffer), min(config.batch_size, len(replay_buffer)), replace=False)
    states = torch.stack([replay_buffer[i]["state"] for i in indices]).to(device)
    policies = torch.stack([replay_buffer[i]["policy"] for i in indices]).to(device)
    values = torch.stack([replay_buffer[i]["value"] for i in indices]).to(device)
    masks = torch.stack([replay_buffer[i]["action_mask"] for i in indices]).to(device)

    network.train()
    with profiler.measure("forward_train"):
        pred_policy, pred_value = network(states, masks)

    with profiler.measure("loss"):
        policy_loss = -torch.sum(policies * torch.log(pred_policy + 1e-8), dim=-1).mean()
        value_loss = torch.nn.functional.mse_loss(pred_value, values)
        loss = policy_loss + value_loss

    with profiler.measure("backward"):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    duration_minutes = int(os.environ.get("DURATION_MINUTES", 10))
    num_actors = int(os.environ.get("NUM_ACTORS", 8))
    s3_bucket = os.environ.get("S3_BUCKET", "")

    print("=" * 70)
    print("PARALLEL SELF-PLAY TEST (CLOUD)")
    print("=" * 70)

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU Information:")
    print(f"  CUDA available: {gpu_info['cuda_available']}")
    if gpu_info['devices']:
        for dev in gpu_info['devices']:
            print(f"  GPU {dev['index']}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")

    print(f"\nConfiguration:")
    print(f"  Duration: {duration_minutes} minutes")
    print(f"  Actors: {num_actors}")

    # Configuration comparison
    print(f"\n{compare_configurations()}")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(num_actors=num_actors, games_per_iteration=num_actors * 4)
    profiler = TrainingProfiler()

    print(f"\nCreating network...")
    network = SimpleTestNetwork().to(device)
    print(f"Network: {sum(p.numel() for p in network.parameters()):,} params")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    replay_buffer = deque(maxlen=config.buffer_size)

    # Warmup
    print(f"\nWarming up GPU...")
    with torch.no_grad():
        dummy = torch.randn(64, config.state_dim).to(device)
        mask = torch.ones(64, config.num_actions).to(device)
        for _ in range(10):
            _ = network(dummy, mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed training
    print(f"\nRunning for {duration_minutes} minutes...")
    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    iteration = 0
    total_games = 0
    total_samples = 0

    while time.time() < end_time:
        iteration += 1
        iter_start = time.time()

        # Self-play
        with profiler.measure("selfplay_total"):
            for actor in range(config.num_actors):
                for _ in range(config.games_per_iteration // config.num_actors):
                    samples = simulate_game(network, config, device, profiler)
                    for s in samples:
                        replay_buffer.append(s)
                    total_games += 1
                    total_samples += len(samples)

        # Training
        with profiler.measure("training_total"):
            for _ in range(2):
                loss = train_step(network, optimizer, replay_buffer, config, device, profiler)

        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        remaining = end_time - time.time()

        if iteration % 10 == 0:
            print(f"[Iter {iteration}] Games: {total_games} | Samples: {total_samples} | "
                  f"Loss: {loss:.4f} | Buffer: {len(replay_buffer)} | "
                  f"Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s")

    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Total games: {total_games}")
    print(f"Total samples: {total_samples}")
    print(f"Games/second: {total_games/elapsed:.2f}")
    print(f"Samples/second: {total_samples/elapsed:.1f}")
    print(f"Iterations: {iteration}")

    # Projections
    samples_per_hour = total_samples / elapsed * 3600
    hours_to_1m = 1_000_000 / samples_per_hour
    print(f"\nProjections:")
    print(f"  Samples/hour: {samples_per_hour:,.0f}")
    print(f"  Time to 1M samples: {hours_to_1m:.2f} hours ({hours_to_1m/24:.2f} days)")
    print(f"  Cost to 1M samples: ${hours_to_1m * 0.16:.2f} (spot @ $0.16/hr)")

    # GPU memory
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1e6:.1f} MB")

    # Profiling report
    print(f"\n{profiler.report()}")

    # Save results
    results = {
        "duration_seconds": elapsed,
        "total_games": total_games,
        "total_samples": total_samples,
        "games_per_second": total_games / elapsed,
        "samples_per_second": total_samples / elapsed,
        "samples_per_hour": samples_per_hour,
        "hours_to_1m": hours_to_1m,
        "cost_to_1m_usd": hours_to_1m * 0.16,
        "device": str(device),
        "num_actors": num_actors,
        "gpu_info": gpu_info,
        "profiler_stats": profiler.get_stats(),
    }

    with open("selfplay_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Upload to S3
    if s3_bucket:
        import subprocess
        subprocess.run([
            "aws", "s3", "cp", "selfplay_test_results.json",
            f"s3://{s3_bucket}/results/selfplay_test_results.json"
        ])
        print(f"\nResults uploaded to s3://{s3_bucket}/results/selfplay_test_results.json")


if __name__ == "__main__":
    main()
TESTSCRIPT

# Run the test
source venv/bin/activate
export S3_BUCKET="$S3_BUCKET"
export DURATION_MINUTES="$DURATION_MINUTES"
export NUM_ACTORS="$NUM_ACTORS"

python run_selfplay_test.py 2>&1 | tee /var/log/selfplay-test.log

# Upload logs
aws s3 cp /var/log/selfplay-test.log "s3://$S3_BUCKET/logs/selfplay-test.log"
aws s3 cp /var/log/selfplay-setup.log "s3://$S3_BUCKET/logs/selfplay-setup.log"

echo ""
echo "============================================================"
echo "Test complete! Results uploaded to S3."
echo "Shutting down in 60 seconds..."
echo "============================================================"
sleep 60
shutdown -h now
USERDATA

# Replace placeholders
sed -i.bak "s|__S3_BUCKET__|$S3_BUCKET|g" /tmp/selfplay_userdata.sh
sed -i.bak "s|__DURATION__|$DURATION_MINUTES|g" /tmp/selfplay_userdata.sh
sed -i.bak "s|__ACTORS__|$NUM_ACTORS|g" /tmp/selfplay_userdata.sh
rm /tmp/selfplay_userdata.sh.bak

echo "  Userdata script created"

# Launch spot instance
echo ""
echo "[5/5] Launching spot instance..."

# Get latest Deep Learning AMI (PyTorch, Ubuntu 22.04)
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text \
    --region $REGION)

echo "  AMI: $AMI_ID"

# Get subnet
SUBNET_ID=$(aws ec2 describe-subnets \
    --filters "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' \
    --output text \
    --region $REGION)

echo "  Subnet: $SUBNET_ID"

# Get security group (or create one)
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=mtg-rl-training" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region $REGION 2>/dev/null || echo "")

if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
    echo "  Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name mtg-rl-training \
        --description "MTG RL Training" \
        --region $REGION \
        --query 'GroupId' \
        --output text)

    # Allow SSH (for debugging)
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION
fi

echo "  Security Group: $SG_ID"

# Get IAM instance profile
PROFILE_ARN=$(aws iam get-instance-profile \
    --instance-profile-name mtg-rl-training-profile \
    --query 'InstanceProfile.Arn' \
    --output text 2>/dev/null || echo "")

if [ -z "$PROFILE_ARN" ]; then
    echo "  WARNING: Instance profile 'mtg-rl-training-profile' not found"
    echo "  Run 'cd infrastructure && terraform apply' first"
    exit 1
fi

echo "  Instance Profile: $PROFILE_ARN"

# Launch spot instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --subnet-id $SUBNET_ID \
    --security-group-ids $SG_ID \
    --iam-instance-profile Name=mtg-rl-training-profile \
    --instance-market-options "MarketType=spot,SpotOptions={MaxPrice=$SPOT_PRICE,SpotInstanceType=one-time}" \
    --user-data file:///tmp/selfplay_userdata.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mtg-rl-selfplay-test},{Key=Project,Value=mtg-rl}]" \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region $REGION)

echo ""
echo "============================================================"
echo "DEPLOYMENT COMPLETE"
echo "============================================================"
echo "Instance ID: $INSTANCE_ID"
echo "Instance Type: $INSTANCE_TYPE (spot)"
echo "Duration: $DURATION_MINUTES minutes"
echo ""
echo "Monitor progress:"
echo "  # Check instance status"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text --region $REGION"
echo ""
echo "  # View live logs (after instance starts)"
echo "  aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo "  # Then: tail -f /var/log/selfplay-test.log"
echo ""
echo "  # Check S3 for results (after completion)"
echo "  aws s3 ls s3://$S3_BUCKET/results/"
echo "  aws s3 cp s3://$S3_BUCKET/results/selfplay_test_results.json -"
echo ""
echo "Instance will auto-shutdown after test completes."
echo "============================================================"
