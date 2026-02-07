#!/usr/bin/env python3
"""
Weights & Biases Integration for MTG-RL

Provides:
- Experiment tracking (metrics, hyperparameters)
- Model versioning (artifacts)
- Learning curve visualization
- Model registry for production deployment

Setup:
    pip install wandb
    wandb login  # One-time setup with API key

Usage:
    # In training script:
    from wandb_integration import WandbTracker

    tracker = WandbTracker(
        project="mtg-draft",
        config={"epochs": 50, "batch_size": 256, ...}
    )

    for epoch in range(epochs):
        metrics = train_epoch(...)
        tracker.log_epoch(epoch, metrics)

    tracker.save_model("checkpoints/best.pt", metadata={...})
    tracker.finish()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' for experiment tracking.")


class WandbTracker:
    """Weights & Biases experiment tracker for MTG-RL."""

    def __init__(
        self,
        project: str = "mtg-draft",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize W&B tracking.

        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Hyperparameters and configuration
            tags: Tags for filtering runs
            notes: Run description
            enabled: Whether to actually log to W&B
        """
        self.enabled = enabled and WANDB_AVAILABLE

        if not self.enabled:
            self.run = None
            return

        # Generate run name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sets = config.get('sets', ['unknown']) if config else ['unknown']
            name = f"draft_{'-'.join(sets)}_{timestamp}"

        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags or ["behavioral-cloning"],
            notes=notes,
            reinit=True,
        )

        # Define metrics for better visualization
        if self.run:
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ):
        """
        Log metrics for an epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics (loss, accuracy, etc.)
            val_metrics: Validation metrics
            extra: Additional metrics to log
        """
        if not self.enabled:
            return

        log_dict = {"epoch": epoch}

        # Add training metrics with prefix
        for key, value in train_metrics.items():
            log_dict[f"train/{key}"] = value

        # Add validation metrics with prefix
        for key, value in val_metrics.items():
            log_dict[f"val/{key}"] = value

        # Add extra metrics
        if extra:
            log_dict.update(extra)

        wandb.log(log_dict)

    def log_test_results(self, test_metrics: Dict[str, float]):
        """Log final test set results."""
        if not self.enabled:
            return

        # Log as summary metrics
        for key, value in test_metrics.items():
            wandb.run.summary[f"test/{key}"] = value

        # Also log as regular metrics for visibility
        wandb.log({f"test/{key}": value for key, value in test_metrics.items()})

    def save_model(
        self,
        model_path: str,
        name: str = "model",
        metadata: Optional[Dict] = None,
        aliases: Optional[list] = None,
    ):
        """
        Save model as W&B artifact (model versioning).

        Args:
            model_path: Path to the model checkpoint
            name: Artifact name
            metadata: Additional metadata (accuracy, config, etc.)
            aliases: Aliases like "best", "latest", "production"
        """
        if not self.enabled:
            return

        # Create artifact
        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata or {},
        )

        # Add the model file
        artifact.add_file(model_path)

        # Log the artifact
        aliases = aliases or ["latest"]
        wandb.log_artifact(artifact, aliases=aliases)

        print(f"  [W&B] Saved model artifact: {name} with aliases {aliases}")

    def save_dataset(
        self,
        data_dir: str,
        name: str = "training-data",
        metadata: Optional[Dict] = None,
    ):
        """Save dataset as W&B artifact for reproducibility."""
        if not self.enabled:
            return

        artifact = wandb.Artifact(
            name=name,
            type="dataset",
            metadata=metadata or {},
        )

        artifact.add_dir(data_dir)
        wandb.log_artifact(artifact)

        print(f"  [W&B] Saved dataset artifact: {name}")

    def log_learning_curves(self, epochs_data: list):
        """Log learning curves as W&B table for detailed analysis."""
        if not self.enabled:
            return

        # Create a table
        columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "gap"]
        data = []

        for e in epochs_data:
            gap = e['train_acc'] - e['val_acc']
            data.append([
                e['epoch'],
                e['train_loss'],
                e['train_acc'],
                e['val_loss'],
                e['val_acc'],
                gap,
            ])

        table = wandb.Table(columns=columns, data=data)
        wandb.log({"learning_curves": table})

    def alert(self, title: str, text: str, level: str = "INFO"):
        """Send an alert (useful for training completion or errors)."""
        if not self.enabled:
            return

        wandb.alert(
            title=title,
            text=text,
            level=getattr(wandb.AlertLevel, level, wandb.AlertLevel.INFO),
        )

    def finish(self, quiet: bool = False):
        """Finish the W&B run."""
        if self.enabled and self.run:
            wandb.finish(quiet=quiet)


class ModelRegistry:
    """
    Model registry for managing model versions.

    Works with or without W&B - falls back to local JSON registry.
    """

    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load existing registry or create new one."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": [], "production": None}

    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        name: str,
        version: str,
        path: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        notes: str = "",
    ):
        """
        Register a new model version.

        Args:
            name: Model name (e.g., "draft-bc")
            version: Version string (e.g., "v1.0.0" or "20260124_200017")
            path: Path to checkpoint (local or S3)
            metrics: Performance metrics
            config: Training configuration
            notes: Additional notes
        """
        entry = {
            "name": name,
            "version": version,
            "path": path,
            "metrics": metrics,
            "config": config,
            "notes": notes,
            "registered_at": datetime.now().isoformat(),
        }

        self.registry["models"].append(entry)
        self._save_registry()

        print(f"Registered model: {name} v{version}")
        print(f"  Path: {path}")
        print(f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")

    def promote_to_production(self, name: str, version: str):
        """Promote a model version to production."""
        # Find the model
        for model in self.registry["models"]:
            if model["name"] == name and model["version"] == version:
                self.registry["production"] = {
                    "name": name,
                    "version": version,
                    "promoted_at": datetime.now().isoformat(),
                }
                self._save_registry()
                print(f"Promoted {name} v{version} to production")
                return

        raise ValueError(f"Model {name} v{version} not found in registry")

    def get_production_model(self) -> Optional[Dict]:
        """Get the current production model."""
        prod = self.registry.get("production")
        if not prod:
            return None

        for model in self.registry["models"]:
            if model["name"] == prod["name"] and model["version"] == prod["version"]:
                return model

        return None

    def list_models(self, name: Optional[str] = None) -> list:
        """List all registered models, optionally filtered by name."""
        models = self.registry["models"]
        if name:
            models = [m for m in models if m["name"] == name]
        return models

    def get_best_model(self, name: str, metric: str = "test_accuracy") -> Optional[Dict]:
        """Get the best model by a specific metric."""
        models = self.list_models(name)
        if not models:
            return None

        return max(models, key=lambda m: m["metrics"].get(metric, 0))


# Example integration with training script
def create_training_tracker(config: Dict) -> WandbTracker:
    """Create a W&B tracker for training."""
    return WandbTracker(
        project="mtg-draft",
        config=config,
        tags=[
            "behavioral-cloning",
            f"sets-{'-'.join(config.get('sets', []))}",
            f"samples-{config.get('train_size', 0)}",
        ],
        notes=f"BC training on {config.get('sets', [])} with {config.get('train_size', 0)} samples",
    )


if __name__ == "__main__":
    # Example usage
    print("Weights & Biases Integration for MTG-RL")
    print("=" * 50)

    if not WANDB_AVAILABLE:
        print("\nTo enable W&B tracking:")
        print("  pip install wandb")
        print("  wandb login")
    else:
        print("\nW&B is available!")
        print("\nExample usage in training:")
        print("""
    from wandb_integration import WandbTracker, ModelRegistry

    # Initialize tracker
    tracker = WandbTracker(
        project="mtg-draft",
        config={"epochs": 50, "batch_size": 256, "sets": ["FDN"]}
    )

    # During training
    for epoch in range(epochs):
        train_metrics = {"loss": 0.5, "accuracy": 0.65}
        val_metrics = {"loss": 0.6, "accuracy": 0.64}
        tracker.log_epoch(epoch, train_metrics, val_metrics)

    # Save model
    tracker.save_model(
        "checkpoints/best.pt",
        metadata={"test_accuracy": 0.68},
        aliases=["best", "latest"]
    )

    # Register in local registry
    registry = ModelRegistry()
    registry.register_model(
        name="draft-bc",
        version="v1.0.0",
        path="s3://bucket/checkpoints/best.pt",
        metrics={"test_accuracy": 0.68},
        config={"sets": ["FDN"]},
    )

    tracker.finish()
        """)
