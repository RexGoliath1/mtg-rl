# MTG RL Training Analysis

## Benchmark Results (Native macOS)

| Metric | Value |
|--------|-------|
| Test Scale | 1,000 games |
| Parallel Workers | 10 |
| Total Time | 609.7 seconds |
| Success Rate | 100% |
| **Throughput** | **1.64 games/sec** |
| **Games/Hour** | **5,904** |

### Game Duration Distribution
- Min: 812ms
- Mean: 6,072ms (6.0s)
- Median: 4,553ms (4.5s)
- P95: 16,043ms (16s)
- P99: 21,884ms (22s)
- Max: 30,795ms (31s)

## Docker Overhead Estimate

Based on typical Docker on macOS performance characteristics:

| Component | Overhead | Impact |
|-----------|----------|--------|
| VM layer (Docker Desktop) | 5-10% | CPU virtualization |
| Network (NAT) | ~1ms/call | Socket communication |
| File I/O (bind mounts) | 10-30% | Card DB loading (one-time) |
| **Total Expected** | **10-20%** | **~4,700-5,300 games/hour** |

For production Linux servers (no VM overhead):
- Expected: **~5,600-5,800 games/hour** (similar to native)

## Training Requirements for Competitive Play

### Reference Points from Game AI Research

| System | Games for Mastery | Notes |
|--------|-------------------|-------|
| AlphaGo | 30 million | Self-play games |
| AlphaZero (Chess) | 44 million | 9 hours on 5000 TPUs |
| AlphaZero (Go) | 29 million | 40 hours on 5000 TPUs |
| OpenAI Five (Dota) | ~180 years | Of Dota gameplay |
| Poker AI (Libratus) | ~15 million | Hands in self-play |

### MTG Complexity Analysis

MTG is significantly more complex than chess but simpler than Go in some dimensions:

| Factor | MTG | Chess | Go |
|--------|-----|-------|-----|
| Legal positions | ~10^60 | ~10^44 | ~10^170 |
| Average moves/turn | 5-20 | 30 | 250 |
| Hidden information | Yes | No | No |
| Randomness | Yes (draws) | No | No |
| Deck variety | ~32K cards | N/A | N/A |

### Estimated Training Scale for Competitive MTG AI

**Conservative Estimate (Modern Competitive):**
- Target: Consistent wins against top AI (similar to MTGO)
- Required games: **5-10 million self-play games**
- Focus: Single format (Modern/Standard), limited deck pool

**Aggressive Estimate (Human-Competitive):**
- Target: Win rate similar to strong human players
- Required games: **20-50 million games**
- With deck variety and format adaptation

## Training Time Projections

### Single Machine (Current Setup)

| Scale | Games | Time at 5,904/hr |
|-------|-------|------------------|
| Initial testing | 10,000 | 1.7 hours |
| Early training | 100,000 | 17 hours |
| Serious training | 1,000,000 | 7 days |
| Competitive | 10,000,000 | 71 days |
| Human-level | 50,000,000 | 353 days |

### Scaling Options

**Horizontal Scaling (Multiple Daemons):**
- Each daemon: ~5,900 games/hour
- 8 daemons on one machine: ~47,000 games/hour
- Memory: ~2GB per daemon + shared OS

**Cloud Scaling (AWS/GCP):**
- c5.4xlarge (16 vCPU): ~4 daemons = 23,600 games/hour
- Cost: ~$0.68/hour → $0.000029/game
- 10M games: ~$290
- 50M games: ~$1,450

**Multi-Machine Cluster:**

| Machines | Daemons | Games/Hour | 10M Games |
|----------|---------|------------|-----------|
| 1 | 4 | 23,600 | 18 days |
| 4 | 16 | 94,400 | 4.4 days |
| 10 | 40 | 236,000 | 1.8 days |
| 20 | 80 | 472,000 | 21 hours |

## Recommended Training Strategy

### Phase 1: Proof of Concept (1 week)
- Games: 100,000
- Setup: Single machine, 4 daemons
- Goal: Verify learning signal, debug pipeline
- Hardware: Local Mac or single cloud instance

### Phase 2: Initial Training (2-4 weeks)
- Games: 1-5 million
- Setup: 4-10 cloud instances
- Goal: Beat random baseline, learn basic strategy
- Hardware: c5.4xlarge or similar

### Phase 3: Competitive Training (1-3 months)
- Games: 10-30 million
- Setup: 10-20 cloud instances
- Goal: Consistent wins vs MTGO AI
- Hardware: Spot instances for cost reduction

### Phase 4: Human-Level (3-6 months)
- Games: 30-100 million
- Setup: 20-50 instances + regular evaluation
- Goal: Competitive with strong human players
- Hardware: Mix of spot and on-demand

## Training Dashboard Requirements

For multi-day training runs, monitor:

1. **Training Metrics**
   - Games played (total, per-hour)
   - Win rate vs baseline
   - Policy loss / Value loss
   - Entropy of action distribution

2. **Performance Metrics**
   - Games per second
   - Queue depth
   - Memory usage
   - GPU utilization (if applicable)

3. **Game Quality Metrics**
   - Average game length (turns)
   - Win margin (life difference)
   - Card advantage at end

4. **Checkpointing**
   - Model checkpoints every N games
   - Best model by win rate
   - Automatic evaluation vs baseline

## Recommended Tools

| Purpose | Tool | Notes |
|---------|------|-------|
| Dashboard | Weights & Biases (wandb) | Best for ML experiments |
| Alternative | TensorBoard | Works with PyTorch |
| Alerts | Slack/Discord webhooks | Training failure alerts |
| Logging | Structured JSON logs | For post-hoc analysis |
| Checkpoints | S3/GCS | Cloud storage for models |
