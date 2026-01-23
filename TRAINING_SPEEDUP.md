# Training Speedup Strategies for MTG RL

## Executive Summary

Based on profiling 10,000 games with detailed timing breakdown, this document outlines optimization strategies to accelerate training from the current ~1.4 games/sec to target 100+ games/sec for competitive 10M+ game training runs.

## Current Performance Baseline

| Metric | Current Value |
|--------|---------------|
| Games/sec (10 workers) | ~1.4 |
| Avg game duration | ~7.3 seconds |
| Network overhead | ~2-5% |
| Server processing | ~95% |
| Games/hour | ~5,000 |
| Time for 10M games | ~83 days |

## Bottleneck Analysis

### 1. Server-Side Processing (Primary Bottleneck)

The Forge daemon spends most time in:
- **Game rule engine** - Card effect resolution, state updates
- **AI decision making** - Action enumeration and selection
- **Memory allocation** - Object creation for game states

### 2. Single-Threaded Daemon

Each daemon instance is single-threaded, limiting parallelism within a process.

### 3. Network I/O

Socket operations are blocking and add latency between games.

## Optimization Strategies

### Tier 1: JVM Optimizations (Expected: 1.3-1.5x speedup)

```bash
# Optimized JVM flags for game simulation
java \
  -Xmx4g -Xms4g \
  -XX:+UseG1GC \
  -XX:MaxGCPauseMillis=50 \
  -XX:+ParallelRefProcEnabled \
  -XX:+UseStringDeduplication \
  -XX:+OptimizeStringConcat \
  -Djava.awt.headless=true \
  -jar forge-daemon.jar daemon --port 17171
```

**Key flags explained:**
- `-Xmx4g -Xms4g`: Fixed heap avoids resize pauses
- `G1GC`: Better for large heaps with mixed allocation
- `MaxGCPauseMillis=50`: Limits GC pauses to 50ms
- `StringDeduplication`: Reduces memory for card names/text
- `headless=true`: Avoids AWT initialization overhead

### Tier 2: Multi-Daemon Parallelism (Expected: Linear scaling)

```bash
# Start 8 daemons on ports 17171-17178
for port in $(seq 17171 17178); do
    java -Xmx2g -Djava.awt.headless=true \
        -jar forge-daemon.jar daemon --port $port &
done
```

**Optimal daemon count:**
- Per-machine: `num_cpus - 2` (leave headroom for OS/Python)
- With 16 cores: 14 daemons
- Expected throughput: ~14 games/sec per machine

### Tier 3: Connection Pooling (Expected: 1.1-1.2x speedup)

```python
class DaemonConnectionPool:
    """Persistent connection pool to avoid reconnect overhead."""

    def __init__(self, endpoints: list, pool_size_per_endpoint: int = 2):
        self.pools = {}
        for host, port in endpoints:
            self.pools[(host, port)] = queue.Queue()
            for _ in range(pool_size_per_endpoint):
                sock = self._create_connection(host, port)
                self.pools[(host, port)].put(sock)

    def get_connection(self, host, port):
        return self.pools[(host, port)].get()

    def return_connection(self, host, port, sock):
        self.pools[(host, port)].put(sock)

    def play_game(self, host, port, deck1, deck2):
        sock = self.get_connection(host, port)
        try:
            sock.sendall(f"NEWGAME {deck1} {deck2} -q\n".encode())
            result = self._recv_until_complete(sock)
            return result
        finally:
            self.return_connection(host, port, sock)
```

### Tier 4: Batch Game Requests (Expected: 1.2-1.3x speedup)

Modify daemon to accept batch requests:

```python
# Client-side batching
def batch_games(daemon, games: list) -> list:
    """Play multiple games over single connection."""
    results = []
    with socket.socket() as s:
        s.connect(daemon)
        for deck1, deck2 in games:
            s.sendall(f"NEWGAME {deck1} {deck2} -q\n".encode())
            result = recv_until(s, b"GAME_RESULT")
            results.append(result)
    return results
```

### Tier 5: Async Training Pipeline (Expected: 1.5-2x speedup)

Decouple game playing from neural network training:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Actor Workers  │ --> │  Replay Buffer  │ --> │     Learner     │
│  (play games)   │     │  (transitions)  │     │  (PPO updates)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ↑                                                │
        └────────────────────────────────────────────────┘
                        (model sync)
```

```python
class AsyncTrainingPipeline:
    def __init__(self, n_actors: int, replay_size: int = 100000):
        self.replay_buffer = collections.deque(maxlen=replay_size)
        self.model_lock = threading.Lock()
        self.current_model = None

    def actor_loop(self, daemon):
        """Actors continuously play games."""
        while True:
            # Get latest model (non-blocking)
            with self.model_lock:
                model = copy.deepcopy(self.current_model)

            # Play game and collect transitions
            transitions = self.play_game(daemon, model)

            # Add to buffer (non-blocking)
            self.replay_buffer.extend(transitions)

    def learner_loop(self, batch_size: int = 256):
        """Learner continuously trains on buffer."""
        while True:
            if len(self.replay_buffer) >= batch_size:
                batch = random.sample(self.replay_buffer, batch_size)
                loss = self.ppo_update(batch)

                # Update shared model
                with self.model_lock:
                    self.current_model = self.model.state_dict()
```

### Tier 6: Mixed Precision Training (Expected: 1.5-2x on GPU)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_step(batch):
    optimizer.zero_grad()

    with autocast():
        # Forward pass in FP16
        policy_logits, value = model(batch['states'])
        loss = compute_ppo_loss(policy_logits, value, batch)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
```

### Tier 7: Gradient Accumulation (For large batches)

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, mini_batch in enumerate(chunks(batch, batch_size // accumulation_steps)):
    with autocast():
        loss = compute_loss(mini_batch) / accumulation_steps
    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Cloud Scaling Strategy

### Single Machine Maximum

| CPUs | Daemons | Games/sec | Games/hour |
|------|---------|-----------|------------|
| 8 | 6 | 6 | 21,600 |
| 16 | 14 | 14 | 50,400 |
| 32 | 30 | 30 | 108,000 |
| 64 | 60 | 60 | 216,000 |

### Multi-Machine Scaling

| Machines | Total Daemons | Games/sec | Games/hour | 10M Games |
|----------|---------------|-----------|------------|-----------|
| 1 (16-core) | 14 | 14 | 50,400 | 8.3 days |
| 4 (16-core) | 56 | 56 | 201,600 | 2.1 days |
| 10 (16-core) | 140 | 140 | 504,000 | 20 hours |
| 20 (16-core) | 280 | 280 | 1,008,000 | 10 hours |

### Recommended Cloud Configuration

**AWS (Best cost/performance):**
```
4x c5.4xlarge (16 vCPU, 32GB)
- 14 daemons per machine = 56 total
- ~$0.68/hr each = $2.72/hr total
- 10M games in ~50 hours = ~$136
```

**With Spot Instances:**
```
- Spot price: ~$0.27/hr (60% discount)
- 10M games cost: ~$55
```

## Implementation Roadmap

### Phase 1: Quick Wins (1 day)
1. Apply JVM optimizations
2. Start multiple local daemons
3. Implement connection pooling

**Expected improvement: 5-8x (7 → 35-50 games/sec)**

### Phase 2: Architecture (3 days)
1. Implement async training pipeline
2. Add replay buffer
3. Decouple actors from learner

**Expected improvement: 2x additional (50 → 100 games/sec)**

### Phase 3: Cloud Scale (1 week)
1. Docker containerization
2. Multi-machine deployment
3. Distributed coordination

**Expected improvement: Linear with machines (100 → 400+ games/sec)**

## Profiling Commands

### JVM Profiling with JFR
```bash
java -XX:StartFlightRecording=filename=game.jfr,duration=60s \
    -jar forge-daemon.jar daemon --port 17171
```

### Analyze JFR
```bash
jfr print --events jdk.ExecutionSample game.jfr | head -100
```

### Python Profiling
```bash
python -m cProfile -s cumtime training_script.py
```

## Monitoring Metrics

### Key Performance Indicators
1. **Games/second** - Primary throughput
2. **P95 game duration** - Tail latency
3. **Daemon CPU utilization** - Resource efficiency
4. **Replay buffer size** - Training throughput
5. **Model sync latency** - Actor freshness

### Prometheus Metrics
```python
from prometheus_client import Counter, Gauge, Histogram

games_total = Counter('mtg_games_total', 'Total games played')
game_duration = Histogram('mtg_game_duration_seconds', 'Game duration',
                          buckets=[1, 2, 5, 10, 20, 30, 60, 120])
daemon_utilization = Gauge('mtg_daemon_utilization', 'Daemon busy percentage',
                           ['daemon_id'])
buffer_size = Gauge('mtg_buffer_size', 'Replay buffer size')
```

## Summary

| Optimization | Speedup | Effort | Priority |
|--------------|---------|--------|----------|
| JVM flags | 1.3-1.5x | Low | High |
| Multi-daemon | Linear | Low | High |
| Connection pool | 1.1-1.2x | Medium | Medium |
| Batch requests | 1.2-1.3x | Medium | Medium |
| Async pipeline | 1.5-2x | High | High |
| Mixed precision | 1.5-2x | Low | Medium |
| Cloud scaling | Linear | High | For 10M+ |

**Bottom line:** With all optimizations on a single 16-core machine, expect ~50-100 games/sec. With 4 cloud machines, expect ~200+ games/sec, making 10M games achievable in ~2 days for ~$150.
