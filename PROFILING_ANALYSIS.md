# 10,000 Game Profiling Analysis

## Executive Summary

Profiled 10,000 MTG games using the Forge daemon with detailed timing breakdown to identify performance bottlenecks and speedup opportunities.

**Key Finding: Server processing is 99.99% of game time. Network overhead is negligible (<1ms).**

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Total Games | 10,000 |
| Parallel Workers | 10 |
| Game Timeout | 120 seconds |
| Daemon Host | localhost:17171 |
| Decks | test_red.dck vs test_blue.dck |

## Results Summary

### Overall Performance

| Metric | Value |
|--------|-------|
| Total Runtime | 130.8 minutes (2.18 hours) |
| Throughput | **1.27 games/sec** |
| Hourly Rate | 4,589 games/hour |
| Success Rate | **100%** (0 failures) |

### Game Duration Distribution

```
Min:    649 ms    (fastest game - likely early concession)
P50:    6,379 ms  (median - typical game)
Mean:   7,838 ms  (average)
P90:    15,622 ms (90th percentile)
P95:    19,245 ms (95th percentile - slow games)
P99:    27,706 ms (99th percentile - very slow games)
Max:    53,623 ms (slowest game - likely complex board state)
StdDev: 5,809 ms  (high variance indicates game complexity varies)
```

**Distribution Insight:** Median (6.4s) < Mean (7.8s) indicates right-skewed distribution with some very long games pulling the average up.

### Time Breakdown by Phase

| Phase | Time (ms) | Percentage |
|-------|-----------|------------|
| Connect | 0.54 | 0.007% |
| Send Command | 0.02 | 0.000% |
| Server Processing | **7,837** | **99.99%** |
| Receive Response | 0.04 | 0.001% |
| Network Total | 0.56 | 0.007% |

**Critical Insight:** Network overhead is sub-millisecond. The daemon's game simulation is the sole bottleneck.

### Win Distribution

| Player | Wins | Percentage |
|--------|------|------------|
| Red | 8,947 | **89.5%** |
| Blue | 1,053 | 10.5% |

**Note:** Significant deck imbalance. For RL training, this provides strong signal for learning.

## Detailed Analysis

### 1. Server Processing Dominance

The server processes each game in ~7.8 seconds on average. This time includes:
- **Deck shuffling and initialization**
- **Game state management**
- **AI decision making** (both players)
- **Rule engine execution**
- **Combat resolution**
- **Effect processing**

Since each daemon handles games sequentially, parallelism requires multiple daemon instances.

### 2. Variance Analysis

High standard deviation (5.8s) relative to mean (7.8s) indicates:
- Simple games finish in ~1-3 seconds
- Complex games with many decisions take 20-50+ seconds
- Board state complexity drives game duration

### 3. Network Efficiency

With only 0.56ms network overhead:
- Connection pooling provides minimal benefit (saves ~0.5ms)
- Batching saves connection overhead but negligible overall
- Focus optimization efforts on server, not network

### 4. Parallelism Bottleneck

With 10 workers hitting 1 daemon:
- Effective throughput: 1.27 games/sec
- Workers spend most time waiting for daemon
- Adding workers without daemons doesn't help

## Speedup Opportunities

### Actionable Optimizations

| Optimization | Expected Speedup | Effort |
|--------------|------------------|--------|
| **Multi-daemon (N daemons)** | **Nx (linear)** | Low |
| JVM G1GC tuning | 1.1-1.2x | Low |
| Quiet mode (-q flag) | 1.05x | None |
| Persistent connections | 1.001x | Medium |

### Multi-Daemon Scaling Projections

With current ~7.8s average game time:

| Daemons | Games/sec | Games/hour | Time for 10M |
|---------|-----------|------------|--------------|
| 1 | 0.13 | 462 | 901 days |
| 4 | 0.51 | 1,848 | 225 days |
| 10 | 1.28 | 4,608 | 90 days |
| 16 | 2.05 | 7,373 | 56 days |
| 40 | 5.13 | 18,432 | 23 days |
| 80 | 10.26 | 36,864 | 11 days |
| 160 | 20.51 | 73,728 | 5.6 days |

### Cloud Scaling Recommendations

**For 10M games in ~2 days:**
- Need ~200 games/sec sustained
- Requires ~1,500 daemon-seconds per second
- With 7.8s/game average: **~1,500 / 7.8 = 192 daemons**

**AWS Configuration:**
```
12x c5.4xlarge (16 vCPU each)
- 16 daemons per machine = 192 total
- ~$0.68/hr each = $8.16/hr total
- 10M games in ~48 hours = ~$392
- With spot: ~$150
```

## Theoretical Limits

### Maximum Speedup

If server could respond instantly (0ms):
- Theoretical max: **77.9x** faster
- Limited only by network (0.56ms round trip)
- This is unachievable but shows the ceiling

### Practical Maximum

With optimized game engine (hypothetical 50% speedup):
- Game time: 3.9s instead of 7.8s
- Doubles effective throughput
- Would require Forge core optimization (significant effort)

## Recommendations

### Immediate Actions (1 day)

1. **Start 8-16 daemons locally**
   ```bash
   for port in $(seq 17171 17186); do
       java -Xmx2g -Djava.awt.headless=true \
           -jar forge-daemon.jar daemon --port $port &
   done
   ```

2. **Update training code to use daemon pool**
   - Round-robin across daemons
   - Handle daemon failures gracefully

### Short-term (1 week)

3. **Deploy to cloud for 10M+ training**
   - Use c5.4xlarge instances (16 vCPU)
   - 16 daemons per machine
   - Start with 4 machines (64 daemons) = ~8 games/sec

4. **Implement async training pipeline**
   - Decouple game collection from PPO updates
   - Use replay buffer

### Long-term (1 month)

5. **Kubernetes deployment**
   - Auto-scaling daemon pods
   - Prometheus monitoring
   - Graceful failure handling

6. **Investigate Forge optimization**
   - Profile with JFR for hot spots
   - Consider contributing optimizations upstream

## Files Generated

- `/tmp/profiling_10k_results.json` - Full profiling data
- `PROFILING_ANALYSIS.md` - This document
- `TRAINING_SPEEDUP.md` - Optimization strategies
- `CLOUD_ARCHITECTURE.md` - Multi-server deployment

## Conclusion

**The path to 10M+ games is clear: horizontal scaling with multiple daemons.**

Single-threaded game simulation at ~7.8s/game means:
- 1 daemon = 462 games/hour
- Need 200+ daemons for practical 10M training
- Cloud deployment is essential for competitive training
- Network is not a factor; pure compute scaling

With 12 AWS c5.4xlarge instances ($150 spot), 10M games is achievable in ~48 hours.
