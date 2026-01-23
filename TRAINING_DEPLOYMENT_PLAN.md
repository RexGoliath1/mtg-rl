# MTG RL Training Deployment Plan

## Overview

This document outlines the plan for deploying MTG reinforcement learning training on AWS, including competitive decks, infrastructure, checkpointing, and spot instance strategies.

---

## 1. Competitive Standard Decks

### Current State
We have basic test decks (`red_aggro.dck`, `white_weenie.dck`) but need competitive Standard decks for meaningful training.

### Recommended Approach
1. **Source decks from MTGGoldfish/MTGTop8** - Get current Standard meta decks
2. **Convert to Forge format** - Forge uses `.dck` files with specific format
3. **Start with 4-6 archetypes** covering different strategies:
   - Aggro (Red Deck Wins, Mono-White)
   - Midrange (Golgari, Gruul)
   - Control (Azorius, Esper)
   - Combo (if present in meta)

### Deck Diversity for Training
- Training against diverse archetypes prevents overfitting to one strategy
- Self-play with deck randomization
- Curriculum: start with simpler aggro mirrors, gradually add complexity

---

## 2. AWS Instance Recommendations

### Training Instance (GPU)

| Instance | GPU | vCPU | RAM | Cost/hr (on-demand) | Cost/hr (spot) | Recommendation |
|----------|-----|------|-----|---------------------|----------------|----------------|
| g4dn.xlarge | T4 (16GB) | 4 | 16GB | $0.526 | ~$0.16 | **Best starting point** |
| g4dn.2xlarge | T4 (16GB) | 8 | 32GB | $0.752 | ~$0.23 | Good for parallel envs |
| g5.xlarge | A10G (24GB) | 4 | 16GB | $1.006 | ~$0.30 | Better GPU, higher cost |
| p3.2xlarge | V100 (16GB) | 8 | 61GB | $3.06 | ~$0.92 | Overkill for this task |

**Recommendation: Start with g4dn.xlarge**
- T4 GPU is sufficient for PPO/policy networks
- 16GB GPU memory handles batch sizes we need
- Spot price ~70% cheaper than on-demand
- Can scale to g4dn.2xlarge if CPU-bound

### Forge Daemon Instance (CPU only)

| Instance | vCPU | RAM | Cost/hr (on-demand) | Cost/hr (spot) |
|----------|------|-----|---------------------|----------------|
| c6i.xlarge | 4 | 8GB | $0.17 | ~$0.05 |
| c6i.2xlarge | 8 | 16GB | $0.34 | ~$0.10 |
| c6i.4xlarge | 16 | 32GB | $0.68 | ~$0.20 |

**Recommendation: Start with c6i.2xlarge**
- Daemon is CPU-bound (game simulation)
- Can run 8-16 concurrent games per daemon
- Scale horizontally by adding more daemon instances

### Architecture Options

**Option A: Single Instance (Simplest)**
```
┌─────────────────────────────────┐
│  g4dn.xlarge                    │
│  ┌─────────────┐ ┌────────────┐ │
│  │ RL Training │ │ Forge      │ │
│  │ (GPU)       │ │ Daemon     │ │
│  │             │ │ (CPU)      │ │
│  └─────────────┘ └────────────┘ │
└─────────────────────────────────┘
```
- Pros: Simple, no network latency
- Cons: GPU underutilized while waiting for games
- Cost: ~$0.16/hr spot

**Option B: Separate Instances (Scalable)**
```
┌────────────────┐     ┌────────────────┐
│ g4dn.xlarge    │────▶│ c6i.2xlarge    │
│ RL Training    │     │ Forge Daemon   │
│ (GPU)          │     │ (CPU)          │
└────────────────┘     └────────────────┘
                              │
                       ┌──────┴──────┐
                       ▼             ▼
                ┌───────────┐ ┌───────────┐
                │ Daemon 2  │ │ Daemon 3  │
                └───────────┘ └───────────┘
```
- Pros: Scale daemons independently, better resource utilization
- Cons: Network latency (~1ms), more complex
- Cost: ~$0.16 + $0.10 = $0.26/hr spot

**Recommendation: Start with Option A**, migrate to Option B when you need more game throughput.

---

## 3. Model Checkpointing Strategy

### What to Save
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': policy.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'episode_rewards': episode_rewards,
    'win_rates': win_rates,
    'config': training_config,
    'timestamp': datetime.now().isoformat(),
}
torch.save(checkpoint, f's3://bucket/checkpoints/model_epoch_{epoch}.pt')
```

### Checkpointing Frequency
- **Every N episodes**: Save every 1000-5000 games
- **Every N hours**: Save every 1-2 hours regardless of progress
- **On interrupt**: Save immediately on SIGTERM (spot termination warning)

### Storage
- **S3 Bucket**: Primary storage for checkpoints
- **Local EBS**: Fast access during training
- **Sync strategy**: Write locally, async upload to S3

### Implementation
```python
import boto3
import signal

class CheckpointManager:
    def __init__(self, s3_bucket, local_dir='/tmp/checkpoints'):
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
        self.local_dir = local_dir

        # Handle spot termination
        signal.signal(signal.SIGTERM, self._emergency_save)

    def save(self, checkpoint, name):
        local_path = f'{self.local_dir}/{name}'
        torch.save(checkpoint, local_path)
        self.s3.upload_file(local_path, self.bucket, f'checkpoints/{name}')

    def load_latest(self):
        # List S3, find latest, download
        ...

    def _emergency_save(self, signum, frame):
        # Called on SIGTERM (spot termination gives 2 min warning)
        self.save(current_checkpoint, 'emergency_checkpoint.pt')
        sys.exit(0)
```

---

## 4. Spot Instance Strategy

### How Spot Instances Work
- Bid on unused AWS capacity at 60-90% discount
- AWS can reclaim with 2-minute warning (SIGTERM)
- Price varies by instance type, AZ, and time

### Spot Interruption Handling

1. **Termination Notice Detection**
```python
import requests

def check_spot_termination():
    """Check if spot instance is being terminated (2 min warning)."""
    try:
        r = requests.get(
            'http://169.254.169.254/latest/meta-data/spot/instance-action',
            timeout=1
        )
        if r.status_code == 200:
            return True  # Termination imminent
    except:
        pass
    return False
```

2. **Graceful Shutdown**
```python
# In training loop
if check_spot_termination() or episode % 1000 == 0:
    checkpoint_manager.save(checkpoint, f'checkpoint_{episode}.pt')
    if check_spot_termination():
        print("Spot termination - saved checkpoint, exiting")
        sys.exit(0)
```

### Spot Fleet Configuration

```yaml
# spot_fleet_config.yaml
SpotFleetRequestConfig:
  IamFleetRole: arn:aws:iam::xxx:role/spot-fleet-role
  TargetCapacity: 1
  TerminateInstancesWithExpiration: true

  LaunchSpecifications:
    - InstanceType: g4dn.xlarge
      ImageId: ami-xxx  # Deep Learning AMI
      SubnetId: subnet-xxx
      SecurityGroups:
        - GroupId: sg-xxx
      SpotPrice: "0.20"  # Max bid

  # Fallback to on-demand if no spot available
  OnDemandTargetCapacity: 0
  OnDemandAllocationStrategy: lowestPrice
```

### Spot Limitations for Our Setup

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| 2-min termination warning | Training interrupted | Frequent checkpoints, fast save |
| Variable availability | May not get instance | Use multiple AZs, instance types |
| Price spikes | Cost unpredictable | Set max bid, use on-demand fallback |
| State loss | Lose in-memory data | Checkpoint everything important |

### Cost Comparison (Monthly Estimate)

| Scenario | On-Demand | Spot | Savings |
|----------|-----------|------|---------|
| g4dn.xlarge 24/7 | $378 | ~$115 | 70% |
| g4dn.xlarge 12hr/day | $189 | ~$58 | 70% |
| With c6i.2xlarge daemon | +$245 | +$72 | 70% |

---

## 5. Deployment Architecture

### Docker Setup

```dockerfile
# Dockerfile.daemon
FROM eclipse-temurin:17-jre
COPY forge-gui-desktop/target/*.jar /app/forge.jar
COPY decks/ /app/decks/
EXPOSE 17171
CMD ["java", "-Xmx2g", "-jar", "/app/forge.jar", "daemon", "-p", "17171"]

# Dockerfile.training
FROM pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime
RUN pip install gymnasium numpy boto3
COPY *.py /app/
COPY decks/ /app/decks/
CMD ["python", "/app/train.py"]
```

### Docker Compose (Local Dev)
```yaml
version: '3.8'
services:
  daemon:
    build:
      context: .
      dockerfile: Dockerfile.daemon
    ports:
      - "17171:17171"

  training:
    build:
      context: .
      dockerfile: Dockerfile.training
    depends_on:
      - daemon
    environment:
      - DAEMON_HOST=daemon
      - S3_BUCKET=mtg-training-checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### AWS Deployment

```bash
# 1. Build and push images
docker build -f Dockerfile.daemon -t mtg-daemon .
docker build -f Dockerfile.training -t mtg-training .
aws ecr push ...

# 2. Launch spot instance with user-data
aws ec2 run-instances \
  --image-id ami-xxx \
  --instance-type g4dn.xlarge \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.20"}}' \
  --user-data file://startup.sh

# startup.sh
#!/bin/bash
docker pull xxx.ecr.aws/mtg-daemon
docker pull xxx.ecr.aws/mtg-training
docker-compose up
```

---

## 6. Training Curriculum

### Phase 1: Basic Learning (Week 1)
- Red aggro mirror matches
- Simple reward: +1 win, -1 loss
- Goal: Learn basic sequencing (land, creature, attack)

### Phase 2: Diverse Matchups (Week 2-3)
- Add 4-6 competitive decks
- Random matchup selection
- Add reward shaping (life differential, board state)

### Phase 3: Self-Play Improvement (Week 4+)
- Policy plays against past versions
- ELO-style rating tracking
- Focus on improving against own weaknesses

---

## 7. Immediate Next Steps

1. **Get competitive decks** (1-2 hours)
   - Download from MTGGoldfish
   - Convert to Forge format
   - Test in daemon

2. **Create training script with checkpointing** (2-3 hours)
   - Implement CheckpointManager
   - Add spot termination handling
   - S3 integration

3. **Docker setup** (1-2 hours)
   - Dockerize daemon
   - Dockerize training
   - Test locally

4. **AWS deployment** (2-3 hours)
   - Create ECR repos
   - Set up S3 bucket
   - Launch first spot instance
   - Verify training runs

5. **Monitor and iterate**
   - Watch training metrics
   - Adjust hyperparameters
   - Scale as needed

---

## 8. Files to Create

- [ ] `Dockerfile.daemon` - Forge daemon container
- [ ] `Dockerfile.training` - Training container
- [ ] `docker-compose.yml` - Local development
- [ ] `train.py` - Main training script with checkpointing
- [ ] `checkpoint_manager.py` - S3 checkpointing
- [ ] `spot_handler.py` - Spot termination handling
- [ ] `aws/startup.sh` - EC2 user-data script
- [ ] `aws/spot_fleet.json` - Spot fleet config
- [ ] `decks/competitive/` - Meta decks

