# MTG RL Deployment Guide

This consolidated guide covers all deployment options: local development, cloud training, and production deployment.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Cloud Training](#cloud-training)
3. [AWS Cost Controls](#aws-cost-controls)
4. [AWS Production Deployment](#aws-production-deployment)
5. [Spot Instance Strategy](#spot-instance-strategy)
6. [Distributed Training](#distributed-training)

---

## Local Development

### Prerequisites

```bash
# Python environment
python -m venv venv
source venv/bin/activate
pip install torch numpy tensorboard

# Java for Forge (if using daemon)
brew install openjdk@17
```

### Quick Start

```bash
# Train on 17lands data (no Forge needed)
python train_draft.py --sets FDN --epochs 10

# Or use the full pipeline
python training_pipeline.py --mode bc --sets FDN
```

### Docker Development

```bash
# Build images
docker build -f infrastructure/docker/Dockerfile.daemon -t mtg-daemon .
docker build -f infrastructure/docker/Dockerfile.training -t mtg-training .

# Run locally
docker-compose -f infrastructure/docker-compose.yml up
```

---

## Cloud Training

### Recommended Provider: RunPod

**Why**: Best price/performance for transformer training, easy setup, reliable.

| GPU | Price/hr | Availability | Best For |
|-----|----------|--------------|----------|
| A100 80GB | $1.79 | High | Full training |
| A100 40GB | $1.10 | High | Experimentation |
| H100 | $2.50+ | Medium | Fastest training |

### Cost Estimates

| Training Level | Description | GPU Hours | Cost |
|----------------|-------------|-----------|------|
| **Minimal** | 1 set BC + 10K RL | ~10 | ~$20 |
| **Standard** | 5 sets BC + 50K RL | ~30 | ~$65 |
| **Full** | 10 sets BC + 100K RL | ~65 | ~$130 |

### RunPod Setup

1. Create account at https://runpod.io
2. Add $50-100 credits
3. Deploy GPU pod:
   - GPU: NVIDIA A100 80GB SXM
   - Container: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
   - Disk: 100GB
   - Volume: 50GB persistent

```bash
# On your pod
git clone https://github.com/RexGoliath1/mtg-rl.git
cd mtg-rl
pip install -r requirements.txt

# Upload 17lands data to data/17lands/
# Then train
python train_draft.py --sets FDN DSK BLB --epochs 10
```

### Alternative Providers

| Provider | A100 Price | Notes |
|----------|------------|-------|
| Lambda Labs | $1.29/hr | Cheapest, often sold out |
| Vast.ai | $0.80-1.50/hr | Marketplace, variable quality |
| AWS (spot) | ~$1.20/hr | Reliable, more setup |
| GCP | $2.21/hr | Good K8s support |

---

## AWS Cost Controls

**IMPORTANT**: Set up cost controls BEFORE deploying any resources.

### Quick Setup (CLI)

```bash
# Set up budget and billing alerts
./scripts/setup_aws_cost_controls.sh your@email.com 50
```

This creates:
- Monthly budget of $50 with alerts at 50%, 80%, 100%
- Forecast alert if projected to exceed budget
- CloudWatch billing alarm

### Manual Console Setup

If you prefer the AWS Console:

1. **Enable Billing Alerts** (one-time):
   - Go to: Billing Dashboard > Billing Preferences
   - Check: "Receive Billing Alerts"
   - Save

2. **Create Budget**:
   - Go to: Billing > Budgets > Create budget
   - Type: Cost budget
   - Name: `mtg-rl-monthly-budget`
   - Amount: $50 (or your limit)
   - Alerts: Add at 50%, 80%, 100% thresholds
   - Email: Your email address

3. **Create Billing Alarm**:
   - Go to: CloudWatch > Alarms > Create alarm
   - Select metric: Billing > Total Estimated Charge
   - Threshold: $50 (or your limit)
   - Notification: Create/select SNS topic with your email

### Additional Manual Limits

For extra protection, you can set these in AWS Console:

1. **Service Quotas** (Billing > Service Quotas):
   - EC2: Limit vCPU count for p/g instance types
   - Example: Set "Running On-Demand G instances" to 4 vCPUs

2. **IAM Permissions** (if using sub-accounts):
   - Create IAM policy that denies expensive instance types
   - Attach to IAM user/role

3. **AWS Organizations (if available)**:
   - Set Service Control Policies to limit spending

### Check Current Spending

```bash
# View current month's costs
aws ce get-cost-and-usage \
    --time-period Start=$(date -v-30d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics UnblendedCost \
    --query 'ResultsByTime[0].Total.UnblendedCost'

# View costs by service
aws ce get-cost-and-usage \
    --time-period Start=$(date -v-30d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity MONTHLY \
    --metrics UnblendedCost \
    --group-by Type=DIMENSION,Key=SERVICE
```

### Terraform Setup

For infrastructure-as-code approach:

```bash
cd infrastructure
terraform init
terraform apply -var="alert_email=your@email.com" -var="monthly_budget=50"
```

See `infrastructure/cost_controls.tf` for details.

---

## AWS Production Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         CloudFront                           │
│                    (CDN + DDoS protection)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      AWS WAF                                 │
│              (Rate limiting: 1000 req/IP/5min)              │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   API Gateway                                │
│           (Rate limiting: 100 req/sec burst)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     ECS Fargate                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  API Service    │  │  Worker Service │                   │
│  │  (Auto-scaling) │  │  (Inference)    │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Instance Recommendations

**Training Instance (GPU)**

| Instance | GPU | vCPU | RAM | Spot Price |
|----------|-----|------|-----|------------|
| g4dn.xlarge | T4 (16GB) | 4 | 16GB | ~$0.16/hr |
| g4dn.2xlarge | T4 (16GB) | 8 | 32GB | ~$0.23/hr |
| g5.xlarge | A10G (24GB) | 4 | 16GB | ~$0.30/hr |

**Recommendation**: Start with g4dn.xlarge

**Daemon Instance (CPU only)**

| Instance | vCPU | RAM | Spot Price |
|----------|------|-----|------------|
| c6i.xlarge | 4 | 8GB | ~$0.05/hr |
| c6i.2xlarge | 8 | 16GB | ~$0.10/hr |

### Terraform Deployment

```bash
cd infrastructure

# Initialize
terraform init

# Plan
terraform plan -var="environment=prod"

# Deploy
terraform apply -var="environment=prod"
```

### Multi-Layer Rate Limiting

1. **WAF Layer** (1000 requests/IP/5min)
```hcl
resource "aws_wafv2_rate_based_statement" {
  limit = 1000
  aggregate_key_type = "IP"
}
```

2. **API Gateway** (100 requests/sec burst)
```hcl
resource "aws_api_gateway_usage_plan" {
  throttle_settings {
    rate_limit  = 100
    burst_limit = 200
  }
}
```

3. **Application Layer** (Redis-backed sliding window)
```python
# See rate_limiter.py for implementation
```

---

## Spot Instance Strategy

### How Spot Instances Work

- Bid on unused AWS capacity at 60-90% discount
- AWS can reclaim with 2-minute warning (SIGTERM)
- Price varies by instance type, AZ, and time

### Cost Savings

| Scenario | On-Demand | Spot | Savings |
|----------|-----------|------|---------|
| g4dn.xlarge 24/7 | $378/mo | ~$115/mo | 70% |
| g4dn.xlarge 12hr/day | $189/mo | ~$58/mo | 70% |

### Termination Handling

```python
import requests
import signal

def check_spot_termination():
    """Check if spot instance is being terminated (2 min warning)."""
    try:
        r = requests.get(
            'http://169.254.169.254/latest/meta-data/spot/instance-action',
            timeout=1
        )
        return r.status_code == 200
    except:
        return False

# In training loop
if check_spot_termination():
    checkpoint_manager.save(checkpoint, 'emergency_checkpoint.pt')
    sys.exit(0)
```

### Checkpointing Strategy

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metrics,
    'timestamp': datetime.now().isoformat(),
}

# Save every 30 minutes AND every 1000 games
if time.time() - last_save > 1800 or episode % 1000 == 0:
    torch.save(checkpoint, f'checkpoints/model_ep{episode}.pt')
    s3.upload_file(local_path, bucket, f'checkpoints/model_ep{episode}.pt')
```

---

## Distributed Training

### Architecture Options

**Option 1: Single Machine (Simplest)**
```
┌─────────────────────────────────┐
│  g4dn.xlarge                    │
│  ┌─────────────┐ ┌────────────┐ │
│  │ RL Training │ │ Forge      │ │
│  │ (GPU)       │ │ Daemon     │ │
│  └─────────────┘ └────────────┘ │
└─────────────────────────────────┘
```
- Cost: ~$0.16/hr spot
- Best for: Getting started

**Option 2: Horizontal Scaling**
```
                ┌─────────────────────┐
                │  Training Server    │
                │  (GPU)              │
                └──────────┬──────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
    │ Daemon 1│       │ Daemon 2│       │ Daemon 3│
    │ (CPU)   │       │ (CPU)   │       │ (CPU)   │
    └─────────┘       └─────────┘       └─────────┘
```
- Cost: ~$0.26/hr spot (GPU + CPU instances)
- Best for: High throughput

**Option 3: AlphaStar-style (Maximum Scale)**
```
    ┌─────────────────────────────────────────┐
    │        Shared Model Storage (S3)         │
    └─────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌───────┐       ┌───────┐       ┌───────┐
│Actor 1│       │Actor 2│       │Actor N│
│+Daemon│       │+Daemon│       │+Daemon│
└───────┘       └───────┘       └───────┘
        \           │           /
         \          │          /
          ▼         ▼         ▼
        ┌─────────────────────┐
        │   Learner (GPU)     │
        │   Aggregates data   │
        │   Updates model     │
        └─────────────────────┘
```
- Cost: Scales with actor count
- Best for: 10M+ games

### Scaling Projections

| Setup | Games/Hour | 10M Games | Est. Cost |
|-------|------------|-----------|-----------|
| 1 machine, 4 daemons | 24,000 | 17 days | $0 (local) |
| 4 machines, 16 daemons | 96,000 | 4.3 days | $260 |
| K8s cluster, 160 daemons | 960,000 | 10 hours | $200 (spot) |

---

## Quick Reference

### Commands

```bash
# Local training
python train_draft.py --sets FDN --epochs 10

# Cloud training (on RunPod/Lambda)
python train_draft.py --sets FDN DSK BLB MKM LCI --epochs 10

# AWS deployment
cd infrastructure && terraform apply

# Docker
docker-compose up
```

### Environment Variables

```bash
export DAEMON_HOST=localhost
export DAEMON_PORT=17171
export S3_BUCKET=mtg-training-checkpoints
export AWS_REGION=us-west-2
```

### Monitoring

```bash
# TensorBoard
tensorboard --logdir logs/ --port 6006

# GPU utilization
watch -n 1 nvidia-smi
```

---

## Sources

- [RunPod Pricing](https://www.runpod.io/pricing)
- [AWS Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/)
- [Lambda Labs Cloud](https://lambdalabs.com)
