# Cloud Training Recommendations for ForgeRL

## Quick Recommendation

**For this project, I recommend: [RunPod](https://www.runpod.io) with an A100 80GB**

- **Cost**: ~$1.79-2.00/hour
- **Why**: Best price/performance for transformer training, ample VRAM for large batches
- **Alternative**: [Lambda Labs](https://lambdalabs.com) A100 at ~$1.29/hour (often sold out)

---

## Training Cost Estimates

### Phase 1: Behavioral Cloning on 17lands (Primary Cost)

| Data Size | GPU | Time | Cost |
|-----------|-----|------|------|
| 10M picks (1 set) | A100 80GB | ~2 hours | ~$4 |
| 50M picks (5 sets) | A100 80GB | ~8 hours | ~$16 |
| 100M picks (10 sets) | A100 80GB | ~15 hours | ~$30 |
| 100M picks (10 sets) | H100 | ~8 hours | ~$24 |

### Phase 2: RL Fine-tuning on Forge

| Drafts | GPU | Time | Cost |
|--------|-----|------|------|
| 10,000 drafts | A100 80GB | ~6 hours | ~$12 |
| 50,000 drafts | A100 80GB | ~24 hours | ~$48 |
| 100,000 drafts | A100 80GB | ~48 hours | ~$96 |

### Total Estimated Cost

| Training Level | Description | Total Cost |
|----------------|-------------|------------|
| **Minimal** | 1 set BC + 10K RL | ~$20 |
| **Standard** | 5 sets BC + 50K RL | ~$65 |
| **Full** | 10 sets BC + 100K RL | ~$130 |

---

## Cloud Provider Comparison (January 2025)

### Tier 1: Best Value (Recommended)

| Provider | GPU | Price/hr | Availability | Notes |
|----------|-----|----------|--------------|-------|
| **[RunPod](https://runpod.io)** | A100 80GB | $1.79 | High | Best overall, easy setup |
| **[Lambda Labs](https://lambdalabs.com)** | A100 | $1.29 | Medium | Cheapest, often sold out |
| **[Vast.ai](https://vast.ai)** | A100 | $0.80-1.50 | Variable | Marketplace, variable quality |
| **[DataCrunch](https://datacrunch.io)** | H100 | $1.99 | Medium | Good H100 pricing |

### Tier 2: Reliable but Pricier

| Provider | GPU | Price/hr | Availability | Notes |
|----------|-----|----------|--------------|-------|
| [Paperspace](https://paperspace.com) | A100 | $3.09 | High | Gradient notebooks |
| [JarvisLabs](https://jarvislabs.ai) | A100 | $1.29-2.29 | High | Good for experimentation |
| [CoreWeave](https://coreweave.com) | A100 | $2.21 | High | Enterprise-focused |

### Tier 3: Hyperscalers (Most Expensive)

| Provider | GPU | Price/hr | Notes |
|----------|-----|----------|-------|
| AWS (p4d.24xlarge) | 8x A100 | $32.77 | Full instance only |
| GCP (a2-highgpu-1g) | A100 | $3.67 | Spot instances cheaper |
| Azure | A100 | $3.67 | Good for enterprise |

---

## Recommended Setup: RunPod

### 1. Create Account
```
https://www.runpod.io/console/signup
```

### 2. Add Credits
- Start with $50-100 for experimentation
- Training runs: budget $130-150 for full pipeline

### 3. Deploy GPU Pod

**Recommended Configuration:**
- GPU: NVIDIA A100 80GB SXM
- Container: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- Disk: 100GB (for data + checkpoints)
- Volume: 50GB persistent (for model saves)

### 4. Setup Script

```bash
# SSH into your pod, then:

# Clone repo
git clone https://github.com/RexGoliath1/mtg-rl.git
cd mtg-rl

# Install dependencies
pip install torch numpy tensorboard

# Download 17lands data (manual step - see instructions)
mkdir -p data/17lands
# Upload your downloaded CSV files here

# Start training
python draft_training.py --mode bc --bc-epochs 10

# Monitor with TensorBoard
tensorboard --logdir logs/ --port 6006
```

### 5. Port Forwarding for TensorBoard
```bash
# On your local machine
ssh -L 6006:localhost:6006 root@<pod-ip>
# Then open http://localhost:6006
```

---

## Alternative: Lambda Labs (Cheapest)

### Pricing
- A100 40GB: $1.10/hour
- A100 80GB: $1.29/hour (when available)
- H100: $2.49/hour

### Setup
```bash
# Lambda Cloud CLI
pip install lambda-cloud

# List instances
lambda cloud instances list

# Launch instance
lambda cloud instances launch \
  --instance-type gpu_1x_a100 \
  --region us-west-1 \
  --ssh-key-name my-key

# SSH in
ssh ubuntu@<instance-ip>
```

### Availability Issues
Lambda Labs often has limited availability. Check the [availability dashboard](https://lambdalabs.com/service/gpu-cloud) before planning long training runs.

---

## Alternative: Vast.ai (Marketplace)

### Pros
- Cheapest prices possible ($0.40-0.80/hr for A100)
- Wide variety of GPUs

### Cons
- Variable reliability
- Community machines (not always enterprise-grade)
- More setup required

### Best for
- Experimentation on a budget
- Non-critical training runs

---

## GPU Selection Guide

### A100 40GB vs 80GB

| Aspect | 40GB | 80GB |
|--------|------|------|
| Price | ~$1.10/hr | ~$1.79/hr |
| Max Batch Size | 32-64 | 64-128 |
| Recommended | Small experiments | Full training |

**Recommendation**: Use 80GB for serious training. The larger batch sizes significantly improve throughput.

### A100 vs H100

| Aspect | A100 80GB | H100 |
|--------|-----------|------|
| Price | ~$1.79/hr | ~$2.50-3.00/hr |
| FP16 Performance | 312 TFLOPS | 989 TFLOPS |
| Training Speed | 1x | ~2-3x |
| Cost Efficiency | Higher | Slightly lower |

**Recommendation**: A100 for budget-conscious training. H100 if you need faster iteration.

---

## Training Tips

### 1. Use Spot/Preemptible Instances
- 60-80% cheaper than on-demand
- Set up checkpointing every 30 minutes
- Use `checkpoint_manager.py` for auto-save

### 2. Efficient Data Loading
```python
# Use multiple workers
DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

### 3. Mixed Precision Training
```python
# Automatic mixed precision (AMP)
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Gradient Accumulation for Larger Batches
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Monitor GPU Utilization
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5 > gpu_log.csv
```

---

## Checkpointing Strategy

### Automatic S3 Backup
```python
# In training config
config = TrainingConfig(
    s3_bucket="your-bucket",
    s3_prefix="mtg-rl/checkpoints",
    save_interval_episodes=1000,
    save_interval_seconds=1800,  # 30 minutes
)
```

### Manual Backup Script
```bash
#!/bin/bash
# backup_checkpoints.sh

BUCKET="s3://your-bucket/mtg-rl"
LOCAL_DIR="checkpoints/"

# Sync to S3
aws s3 sync $LOCAL_DIR $BUCKET/checkpoints/ \
    --exclude "*.tmp" \
    --storage-class STANDARD_IA

echo "Backup complete: $(date)"
```

---

## Quick Start Commands

```bash
# 1. Setup (on cloud instance)
git clone https://github.com/RexGoliath1/mtg-rl.git
cd mtg-rl
pip install -r requirements.txt

# 2. Download 17lands data (manual)
python scripts/download_17lands.py --list
# Then download from https://www.17lands.com/public_datasets

# 3. Run BC training
python draft_training.py --mode bc --bc-epochs 10

# 4. Run RL training (requires Forge daemon)
# First, start Forge on a separate machine or container
python draft_training.py --mode rl --rl-drafts 50000

# 5. Monitor
tensorboard --logdir logs/
```

---

## Cost Optimization Tips

1. **Start small**: Test with 1 set before training on all data
2. **Use spot instances**: Save 60-80% on compute costs
3. **Checkpoint frequently**: Avoid losing work on preemption
4. **Off-peak hours**: Some providers have lower demand at night
5. **Multi-GPU only if needed**: Single A100 80GB is often sufficient
6. **Download data locally first**: Avoid cloud egress fees

---

## Summary

| Stage | Provider | GPU | Estimated Time | Estimated Cost |
|-------|----------|-----|----------------|----------------|
| BC Pre-training | RunPod | A100 80GB | 15 hours | $30 |
| RL Fine-tuning | RunPod | A100 80GB | 48 hours | $96 |
| **Total** | | | **~63 hours** | **~$126** |

**Bottom line**: You can train a competitive draft AI for about $130 on cloud GPUs. Start with RunPod for the best balance of price, reliability, and ease of use.

---

## Sources

- [RunPod Pricing](https://www.runpod.io/pricing)
- [Cloud GPU Pricing Comparison 2025](https://gpuvec.com/)
- [A100 GPU Pricing Showdown](https://www.thundercompute.com/blog/a100-gpu-pricing-showdown-2025-who-s-the-cheapest-for-deep-learning-workloads)
- [Best Cloud GPU Platforms 2025](https://www.digitalocean.com/resources/articles/best-cloud-gpu-platforms)
