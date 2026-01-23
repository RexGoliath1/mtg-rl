# Multi-Server Cloud Architecture for MTG RL Training

## Overview

This document describes how to scale MTG RL training across multiple cloud servers for competitive training at 10M+ games.

## Architecture Options

### Option 1: Horizontal Daemon Scaling (Recommended)

```
                    ┌─────────────────────────────────────┐
                    │         Training Coordinator         │
                    │  (Python - orchestrates training)    │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼─────────┐ ┌───────▼───────┐ ┌─────────▼─────────┐
    │   Daemon Pool 1    │ │ Daemon Pool 2 │ │   Daemon Pool 3    │
    │   (4 daemons)      │ │  (4 daemons)  │ │   (4 daemons)      │
    │   Server 1         │ │   Server 2    │ │   Server 3         │
    └───────────────────┘ └───────────────┘ └───────────────────┘
```

**How it works:**
- Each server runs multiple Forge daemon instances (4-8 per machine)
- Training coordinator distributes games across all daemons
- Results collected centrally for PPO updates
- Model checkpoints synced via shared storage (S3/GCS)

**Pros:**
- Simple to implement
- Each daemon is independent
- Easy to add/remove capacity
- No GPU required for game simulation

**Cons:**
- Network latency between coordinator and daemons
- Centralized bottleneck at coordinator

### Option 2: Distributed Self-Play (AlphaStar-style)

```
    ┌─────────────────────────────────────────────────────────┐
    │                    Shared Model Storage                  │
    │                    (S3/GCS + Redis)                     │
    └─────────────────────────────────────────────────────────┘
                    │               │               │
        ┌───────────▼───────┐ ┌────▼────┐ ┌───────▼───────────┐
        │   Actor Worker 1   │ │Actor 2  │ │   Actor Worker N   │
        │  ┌─────┐ ┌─────┐  │ │  ...    │ │  ┌─────┐ ┌─────┐  │
        │  │Daemon│ │Agent│  │ │         │ │  │Daemon│ │Agent│  │
        │  └─────┘ └─────┘  │ │         │ │  └─────┘ └─────┘  │
        │  Plays games      │ │         │ │  Plays games      │
        │  Collects samples │ │         │ │  Collects samples │
        └───────────────────┘ └─────────┘ └───────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │        Learner (GPU)          │
                    │    Aggregates & trains        │
                    │    Updates shared model       │
                    └───────────────────────────────┘
```

**How it works:**
- Actor workers independently play games and collect transitions
- Transitions stored in shared replay buffer (Redis/RabbitMQ)
- Learner pulls batches and performs PPO updates
- Updated model published to shared storage
- Actors periodically sync new model

**Pros:**
- Fully distributed, no single bottleneck
- Scales to 100+ actors
- Actors can use different opponent strategies
- GPU learner can be separate from CPU actors

**Cons:**
- More complex infrastructure
- Requires distributed storage/messaging
- Model sync latency

### Option 3: Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forge-daemon
spec:
  replicas: 20
  selector:
    matchLabels:
      app: forge-daemon
  template:
    spec:
      containers:
      - name: daemon
        image: mtg-forge-daemon:latest
        ports:
        - containerPort: 17171
        resources:
          requests:
            memory: "2Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: forge-daemon-lb
spec:
  type: LoadBalancer
  selector:
    app: forge-daemon
  ports:
  - port: 17171
```

## Cloud Provider Recommendations

### AWS (Recommended for Cost)

| Component | Instance Type | Specs | Cost/hr |
|-----------|--------------|-------|---------|
| Daemon Server | c5.4xlarge | 16 vCPU, 32GB | $0.68 |
| Learner (GPU) | g4dn.xlarge | 4 vCPU, 16GB, T4 | $0.526 |
| Redis Cache | cache.r5.large | 2 vCPU, 13GB | $0.166 |

**10M games estimate:**
- 4 daemon servers (4 daemons each) = ~100k games/hour
- Training time: ~100 hours
- Cost: ~$400 (with spot instances: ~$150)

### GCP (Better for K8s)

| Component | Instance Type | Cost/hr |
|-----------|--------------|---------|
| Daemon Server | n2-standard-8 | $0.39 |
| Learner (GPU) | n1-standard-4 + T4 | $0.55 |
| Redis | Memorystore | $0.049/GB |

### Azure

| Component | Instance Type | Cost/hr |
|-----------|--------------|---------|
| Daemon Server | Standard_D8s_v3 | $0.384 |
| Learner (GPU) | NC4as_T4_v3 | $0.526 |

## Implementation: Distributed Training

### distributed_trainer.py

```python
import redis
import threading
from concurrent.futures import ThreadPoolExecutor
import socket
import json

class DistributedTrainer:
    def __init__(self, daemon_endpoints: list, redis_host: str):
        self.daemons = daemon_endpoints
        self.redis = redis.Redis(host=redis_host)
        self.transition_queue = "mtg:transitions"
        self.model_key = "mtg:current_model"

    def actor_loop(self, daemon_idx: int):
        """Actor worker - plays games and collects transitions."""
        host, port = self.daemons[daemon_idx]

        while True:
            # Get current model
            model_bytes = self.redis.get(self.model_key)
            if model_bytes:
                self.load_model(model_bytes)

            # Play game
            transitions = self.play_game(host, port)

            # Push to queue
            self.redis.lpush(
                self.transition_queue,
                json.dumps(transitions)
            )

    def learner_loop(self, batch_size: int = 256):
        """Learner - trains on collected transitions."""
        while True:
            # Collect batch
            batch = []
            for _ in range(batch_size):
                item = self.redis.brpop(self.transition_queue, timeout=10)
                if item:
                    batch.append(json.loads(item[1]))

            if len(batch) >= batch_size // 2:
                # Train step
                self.ppo_update(batch)

                # Publish updated model
                model_bytes = self.serialize_model()
                self.redis.set(self.model_key, model_bytes)

    def run(self, n_actors: int):
        """Run distributed training."""
        # Start actor threads
        with ThreadPoolExecutor(max_workers=n_actors + 1) as executor:
            # Actors
            for i in range(n_actors):
                daemon_idx = i % len(self.daemons)
                executor.submit(self.actor_loop, daemon_idx)

            # Learner
            executor.submit(self.learner_loop)
```

### Multi-Machine Daemon Manager

```python
class DaemonCluster:
    """Manages daemons across multiple machines."""

    def __init__(self, machines: list):
        """
        Args:
            machines: List of (host, ssh_user, num_daemons)
        """
        self.machines = machines
        self.daemon_ports = []

    def start_cluster(self):
        """Start daemons on all machines."""
        import paramiko

        for host, user, n_daemons in self.machines:
            ssh = paramiko.SSHClient()
            ssh.connect(host, username=user)

            for i in range(n_daemons):
                port = 17171 + i
                cmd = f"""
                    nohup java -Xmx2g -Djava.awt.headless=true \
                        -jar /opt/forge/forge-daemon.jar daemon \
                        --port {port} > /var/log/daemon_{port}.log 2>&1 &
                """
                ssh.exec_command(cmd)
                self.daemon_ports.append((host, port))

            ssh.close()

        return self.daemon_ports

    def get_load_balanced_endpoint(self):
        """Simple round-robin load balancing."""
        import itertools
        endpoints = itertools.cycle(self.daemon_ports)
        return lambda: next(endpoints)
```

## Speedup Strategies

### 1. Batch Game Requests

Instead of one game per connection, batch multiple:

```python
def batch_games(daemon, n_games=10):
    """Request multiple games in one connection."""
    results = []
    with socket.socket() as s:
        s.connect(daemon)
        for _ in range(n_games):
            s.sendall(b"NEWGAME deck1 deck2 -q\n")
            result = recv_until(s, b"GAME_RESULT")
            results.append(result)
    return results
```

**Expected speedup: 1.2-1.5x** (reduced connection overhead)

### 2. Persistent Connections

Keep connections open between games:

```python
class PersistentDaemonConnection:
    def __init__(self, host, port):
        self.sock = socket.socket()
        self.sock.connect((host, port))

    def play_game(self, deck1, deck2):
        self.sock.sendall(f"NEWGAME {deck1} {deck2} -q\n".encode())
        return self.recv_result()
```

**Expected speedup: 1.1x**

### 3. Reduce Game Output

Run with minimal logging:

```bash
java -jar forge-daemon.jar daemon -q -c 30  # Quiet mode, 30s timeout
```

**Expected speedup: 1.1x**

### 4. JVM Optimization

```bash
java -Xmx4g -Xms4g \
     -XX:+UseG1GC \
     -XX:MaxGCPauseMillis=50 \
     -XX:+ParallelRefProcEnabled \
     -jar forge-daemon.jar daemon
```

**Expected speedup: 1.1-1.2x**

### 5. Async Training

Separate game playing from PPO updates:

```
[Actors play games] --> [Replay Buffer] --> [Learner updates]
```

Actors never wait for training to complete.

**Expected speedup: 1.5-2x** (removes training from critical path)

### 6. Mixed Precision Training

Use FP16 for neural network:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = compute_ppo_loss(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Expected speedup: 1.5-2x on GPU**

## Scaling Projections

| Setup | Games/Hour | 10M Games | Est. Cost |
|-------|------------|-----------|-----------|
| 1 machine, 4 daemons | 24,000 | 17 days | $0 (local) |
| 4 machines, 16 daemons | 96,000 | 4.3 days | $260 |
| 10 machines, 40 daemons | 240,000 | 1.7 days | $300 |
| 20 machines, 80 daemons | 480,000 | 21 hours | $340 |
| K8s cluster, 160 daemons | 960,000 | 10 hours | $200 (spot) |

## Quick Start

### Local Multi-Daemon

```bash
# Start 4 daemons locally
for port in 17171 17172 17173 17174; do
    java -Xmx2g -Djava.awt.headless=true \
        -jar forge-daemon.jar daemon --port $port &
done

# Run distributed training
python distributed_trainer.py \
    --daemons localhost:17171,localhost:17172,localhost:17173,localhost:17174 \
    --games 100000
```

### AWS Quick Deploy

```bash
# Terraform for AWS
terraform init
terraform apply -var="num_daemon_servers=4"

# Get endpoints
DAEMONS=$(terraform output daemon_endpoints)

# Run training
python distributed_trainer.py --daemons $DAEMONS --games 10000000
```

## Monitoring

### Metrics to Track

1. **Games per second** - Primary throughput metric
2. **Daemon utilization** - % time each daemon is busy
3. **Queue depth** - Transitions waiting for training
4. **Model sync latency** - Time for actors to get new model
5. **Win rate vs baseline** - Training progress indicator

### Prometheus Metrics

```python
from prometheus_client import Counter, Gauge, Histogram

games_played = Counter('mtg_games_played_total', 'Total games played')
game_duration = Histogram('mtg_game_duration_seconds', 'Game duration')
queue_depth = Gauge('mtg_transition_queue_depth', 'Pending transitions')
win_rate = Gauge('mtg_win_rate', 'Win rate vs baseline')
```

## Failure Handling

1. **Daemon crash**: Automatically restart via systemd/K8s
2. **Network partition**: Actors buffer locally, retry
3. **Learner crash**: Resume from last checkpoint
4. **Stale model**: Actors check model version, skip old games
