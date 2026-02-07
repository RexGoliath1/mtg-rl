#!/usr/bin/env python3
"""
Checkpoint Manager for MTG RL Training

Handles saving/loading model checkpoints with S3 support and
spot instance termination handling.
"""

import os
import sys
import signal
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import requests

import torch


@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    episode: int
    total_steps: int
    total_games: int
    wins: int
    losses: int
    episode_rewards: list
    win_rates: list
    best_win_rate: float
    training_config: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingState':
        return cls(**data)


class CheckpointManager:
    """
    Manages model checkpoints with local and S3 storage.

    Features:
    - Automatic periodic checkpointing
    - Emergency save on spot termination
    - S3 upload with async background sync
    - Resume from latest checkpoint
    """

    def __init__(
        self,
        local_dir: str = "checkpoints",
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "mtg-rl/checkpoints",
        save_interval_episodes: int = 1000,
        save_interval_seconds: int = 3600,  # 1 hour
        keep_last_n: int = 3,
    ):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.s3_bucket = s3_bucket or os.environ.get('S3_BUCKET')
        self.s3_prefix = s3_prefix
        self.save_interval_episodes = save_interval_episodes
        self.save_interval_seconds = save_interval_seconds
        self.keep_last_n = keep_last_n

        self.last_save_time = time.time()
        self.last_save_episode = 0

        # S3 client (lazy init)
        self._s3_client = None

        # Current state for emergency save
        self._current_checkpoint = None
        self._model = None
        self._optimizer = None

        # Set up spot termination handler
        self._setup_spot_handler()

        # Background upload thread
        self._upload_queue = []
        self._upload_lock = threading.Lock()
        self._start_upload_thread()

    @property
    def s3_client(self):
        if self._s3_client is None and self.s3_bucket:
            import boto3
            self._s3_client = boto3.client('s3')
        return self._s3_client

    def _setup_spot_handler(self):
        """Set up handler for spot instance termination."""
        signal.signal(signal.SIGTERM, self._handle_termination)
        signal.signal(signal.SIGINT, self._handle_termination)

        # Start spot check thread
        self._spot_check_thread = threading.Thread(
            target=self._spot_check_loop,
            daemon=True
        )
        self._spot_check_thread.start()

    def _spot_check_loop(self):
        """Continuously check for spot termination notice."""
        while True:
            if self._check_spot_termination():
                print("SPOT TERMINATION NOTICE DETECTED!")
                self._emergency_save()
                sys.exit(0)
            time.sleep(5)  # Check every 5 seconds

    def _check_spot_termination(self) -> bool:
        """Check AWS metadata for spot termination notice."""
        try:
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/spot/instance-action',
                timeout=1
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass
        return False

    def _handle_termination(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        print(f"\nReceived signal {signum}, saving checkpoint...")
        self._emergency_save()
        sys.exit(0)

    def _emergency_save(self):
        """Emergency save current state."""
        if self._current_checkpoint is None:
            print("No checkpoint data available for emergency save")
            return

        checkpoint_name = f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        self.save(
            self._current_checkpoint,
            self._model,
            self._optimizer,
            checkpoint_name,
            is_emergency=True
        )
        print(f"Emergency checkpoint saved: {checkpoint_name}")

    def _start_upload_thread(self):
        """Start background S3 upload thread."""
        def upload_worker():
            while True:
                with self._upload_lock:
                    if self._upload_queue:
                        local_path, s3_key = self._upload_queue.pop(0)
                    else:
                        local_path = None

                if local_path:
                    try:
                        self._upload_to_s3(local_path, s3_key)
                    except Exception as e:
                        print(f"S3 upload failed: {e}")
                else:
                    time.sleep(1)

        thread = threading.Thread(target=upload_worker, daemon=True)
        thread.start()

    def _upload_to_s3(self, local_path: Path, s3_key: str):
        """Upload file to S3."""
        if not self.s3_client:
            return

        try:
            self.s3_client.upload_file(
                str(local_path),
                self.s3_bucket,
                s3_key
            )
            print(f"Uploaded to s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"Failed to upload to S3: {e}")

    def should_save(self, episode: int) -> bool:
        """Check if we should save based on episode count or time."""
        episodes_since_save = episode - self.last_save_episode
        time_since_save = time.time() - self.last_save_time

        return (
            episodes_since_save >= self.save_interval_episodes or
            time_since_save >= self.save_interval_seconds
        )

    def save(
        self,
        training_state: TrainingState,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_name: Optional[str] = None,
        is_emergency: bool = False,
        is_best: bool = False,
    ):
        """
        Save a checkpoint.

        Args:
            training_state: Current training metrics
            model: PyTorch model
            optimizer: PyTorch optimizer
            checkpoint_name: Custom name (default: auto-generated)
            is_emergency: If True, save synchronously
            is_best: If True, also save as 'best_model.pt'
        """
        # Update current state for emergency saves
        self._current_checkpoint = training_state
        self._model = model
        self._optimizer = optimizer

        # Generate checkpoint name
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_ep{training_state.episode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

        # Build checkpoint dict
        checkpoint = {
            'training_state': training_state.to_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        # Save locally
        local_path = self.local_dir / checkpoint_name
        torch.save(checkpoint, local_path)
        print(f"Saved checkpoint: {local_path}")

        # Save best model
        if is_best:
            best_path = self.local_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        # Queue S3 upload
        if self.s3_bucket and not is_emergency:
            s3_key = f"{self.s3_prefix}/{checkpoint_name}"
            with self._upload_lock:
                self._upload_queue.append((local_path, s3_key))

            if is_best:
                best_s3_key = f"{self.s3_prefix}/best_model.pt"
                with self._upload_lock:
                    self._upload_queue.append((best_path, best_s3_key))
        elif is_emergency and self.s3_bucket:
            # Synchronous upload for emergency
            s3_key = f"{self.s3_prefix}/{checkpoint_name}"
            self._upload_to_s3(local_path, s3_key)

        # Update tracking
        self.last_save_time = time.time()
        self.last_save_episode = training_state.episode

        # Prune old checkpoints (keep last N + best + emergency)
        if not is_emergency:
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        """Delete old checkpoints, keeping last N + best + emergency."""
        regular = sorted(
            self.local_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if len(regular) <= self.keep_last_n:
            return

        to_delete = regular[self.keep_last_n:]
        for path in to_delete:
            path.unlink()
            print(f"Pruned old checkpoint: {path.name}")

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[TrainingState]:
        """
        Load the latest checkpoint.

        First checks S3 for newer checkpoints, then falls back to local.

        Returns:
            TrainingState if checkpoint found, None otherwise
        """

        latest_path = None

        # Check local checkpoints
        local_checkpoints = list(self.local_dir.glob("checkpoint_*.pt"))
        if local_checkpoints:
            # Sort by modification time
            local_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest_path = local_checkpoints[0]

        # Check S3 for newer checkpoints
        if self.s3_bucket and self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=f"{self.s3_prefix}/checkpoint_"
                )

                if 'Contents' in response:
                    # Sort by last modified
                    s3_objects = sorted(
                        response['Contents'],
                        key=lambda x: x['LastModified'],
                        reverse=True
                    )

                    if s3_objects:
                        s3_latest = s3_objects[0]

                        # Check if S3 is newer than local
                        s3_time = s3_latest['LastModified'].timestamp()
                        local_time = latest_path.stat().st_mtime if latest_path else 0

                        if s3_time > local_time:
                            # Download from S3
                            s3_key = s3_latest['Key']
                            local_name = Path(s3_key).name
                            latest_path = self.local_dir / local_name

                            self.s3_client.download_file(
                                self.s3_bucket,
                                s3_key,
                                str(latest_path)
                            )
                            print(f"Downloaded newer checkpoint from S3: {s3_key}")
            except Exception as e:
                print(f"Failed to check S3 for checkpoints: {e}")

        if latest_path is None:
            print("No checkpoint found")
            return None

        # Load checkpoint
        print(f"Loading checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        training_state = TrainingState.from_dict(checkpoint['training_state'])

        print(f"Resumed from episode {training_state.episode}, "
              f"win rate: {training_state.best_win_rate:.1%}")

        return training_state

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = []

        # Local checkpoints
        for path in self.local_dir.glob("checkpoint_*.pt"):
            checkpoints.append({
                'source': 'local',
                'path': str(path),
                'modified': datetime.fromtimestamp(path.stat().st_mtime),
            })

        # S3 checkpoints
        if self.s3_bucket and self.s3_client:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=f"{self.s3_prefix}/checkpoint_"
                )

                if 'Contents' in response:
                    for obj in response['Contents']:
                        checkpoints.append({
                            'source': 's3',
                            'path': f"s3://{self.s3_bucket}/{obj['Key']}",
                            'modified': obj['LastModified'],
                        })
            except Exception as e:
                print(f"Failed to list S3 checkpoints: {e}")

        # Sort by modified time
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)

        return checkpoints


# Example usage
if __name__ == "__main__":
    import torch.nn as nn

    # Simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters())

    manager = CheckpointManager(
        local_dir="test_checkpoints",
        s3_bucket=None,  # Set to test with S3
    )

    # Simulate training
    state = TrainingState(
        episode=100,
        total_steps=10000,
        total_games=100,
        wins=55,
        losses=45,
        episode_rewards=[1.0, 0.5, -0.5],
        win_rates=[0.5, 0.52, 0.55],
        best_win_rate=0.55,
        training_config={'lr': 0.001},
        timestamp=datetime.now().isoformat(),
    )

    # Save checkpoint
    manager.save(state, model, optimizer, is_best=True)

    # List checkpoints
    print("\nAvailable checkpoints:")
    for cp in manager.list_checkpoints():
        print(f"  {cp['source']}: {cp['path']}")

    # Load checkpoint
    new_model = TestModel()
    new_optimizer = torch.optim.Adam(new_model.parameters())
    loaded_state = manager.load_latest(new_model, new_optimizer)

    if loaded_state:
        print(f"\nLoaded state: episode={loaded_state.episode}, wins={loaded_state.wins}")
