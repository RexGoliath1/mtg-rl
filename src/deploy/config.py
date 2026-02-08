"""
Centralized deployment configuration for ForgeRL AWS operations.

All hardcoded values from shell scripts are consolidated here as
validated dataclasses with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# Spot pricing reference (us-east-1, as of 2026-02)
SPOT_PRICES: dict[str, float] = {
    "c5.2xlarge": 0.17,
    "g4dn.xlarge": 0.20,
    "g5.xlarge": 0.35,
    "p3.2xlarge": 1.00,
}

# Valid instance types per task
COLLECTION_INSTANCE_TYPES = {"c5.xlarge", "c5.2xlarge", "c5.4xlarge", "m5.2xlarge"}
TRAINING_INSTANCE_TYPES = {"g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "g5.2xlarge", "p3.2xlarge"}


@dataclass
class DeployConfig:
    """Top-level AWS deployment configuration."""

    region: str = "us-east-1"
    s3_bucket: str = "mtg-rl-checkpoints-20260124190118616600000001"
    security_group_name: str = "mtg-rl-training-sg"
    iam_instance_profile: str = "mtg-rl-training"
    ecr_registry: str = ""  # e.g. "123456789.dkr.ecr.us-east-1.amazonaws.com"
    key_name: Optional[str] = None
    monthly_budget: float = 100.0
    notify_email: str = field(default_factory=lambda: os.environ.get("FORGERL_NOTIFY_EMAIL", ""))
    wandb_api_key: str = field(default_factory=lambda: os.environ.get("WANDB_API_KEY", ""))
    wandb_entity: str = field(default_factory=lambda: os.environ.get("WANDB_ENTITY", ""))
    wandb_project: str = field(default_factory=lambda: os.environ.get("WANDB_PROJECT", "forgerl"))

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if not self.s3_bucket:
            errors.append("s3_bucket is required")
        if self.monthly_budget <= 0:
            errors.append("monthly_budget must be positive")
        if not self.region:
            errors.append("region is required")
        # Check boto3 availability
        try:
            import boto3  # noqa: F401
        except ImportError:
            errors.append("boto3 is not installed. Run: uv sync --extra data")
        return errors


@dataclass
class CollectionConfig:
    """Data collection phase configuration."""

    num_games: int = 10000
    num_workers: int = 8
    game_timeout: int = 60
    instance_type: str = "c5.2xlarge"
    spot_price: float = 0.17
    volume_size_gb: int = 50
    save_interval: int = 500
    forge_port: int = 17171
    forge_branch: str = "feature/rl-daemon-mode"
    forge_java_heap: str = "6g"

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if self.num_games <= 0:
            errors.append("num_games must be positive")
        if self.num_workers <= 0:
            errors.append("num_workers must be positive")
        if self.game_timeout <= 0:
            errors.append("game_timeout must be positive")
        if self.instance_type not in COLLECTION_INSTANCE_TYPES:
            errors.append(
                f"instance_type '{self.instance_type}' not in allowed types: "
                f"{sorted(COLLECTION_INSTANCE_TYPES)}"
            )
        if self.spot_price <= 0:
            errors.append("spot_price must be positive")
        return errors


@dataclass
class TrainingConfig:
    """GPU training phase configuration."""

    epochs: int = 30
    batch_size: int = 256
    learning_rate: str = "1e-3"
    warmup_epochs: int = 5
    grad_accum: int = 1
    num_data_workers: int = 4
    instance_type: str = "g4dn.xlarge"
    spot_price: float = 0.20
    volume_size_gb: int = 100
    amp_enabled: bool = True
    tensorboard: bool = True

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.warmup_epochs < 0:
            errors.append("warmup_epochs must be non-negative")
        if self.warmup_epochs >= self.epochs:
            errors.append("warmup_epochs must be less than epochs")
        if self.instance_type not in TRAINING_INSTANCE_TYPES:
            errors.append(
                f"instance_type '{self.instance_type}' not in allowed types: "
                f"{sorted(TRAINING_INSTANCE_TYPES)}"
            )
        if self.spot_price <= 0:
            errors.append("spot_price must be positive")
        # Validate learning rate is parseable
        try:
            float(self.learning_rate)
        except ValueError:
            errors.append(f"learning_rate '{self.learning_rate}' is not a valid float")
        return errors


@dataclass
class CostEstimate:
    """Breakdown of estimated AWS costs for a run."""

    collection_hours: float = 0.0
    collection_cost: float = 0.0
    training_hours: float = 0.0
    training_cost: float = 0.0
    storage_cost: float = 0.01  # Negligible for <100MB
    total: float = 0.0

    def summary(self) -> str:
        """Human-readable cost summary."""
        lines = [
            "--- Cost Estimate ---",
            f"Collection: ~{self.collection_hours:.1f}h = ${self.collection_cost:.2f}",
            f"Training:   ~{self.training_hours:.1f}h = ${self.training_cost:.2f}",
            f"Storage:    ${self.storage_cost:.2f}",
            f"{'─' * 35}",
            f"TOTAL:      ${self.total:.2f}",
        ]
        return "\n".join(lines)
