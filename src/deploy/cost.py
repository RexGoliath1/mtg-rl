"""
Budget checking and cost estimation for ForgeRL AWS deployments.

Mirrors the cost estimation logic from launch_major_training.sh
with proper Python math (no dependency on bc).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.deploy.config import (
    CollectionConfig,
    CostEstimate,
    SPOT_PRICES,
    TrainingConfig,
)


# Default local cost tracking file
DEFAULT_COST_TRACKING_FILE = Path.home() / ".mtg_rl_cost_tracking.json"


def estimate_costs(
    collection_config: Optional[CollectionConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> CostEstimate:
    """
    Estimate total AWS costs for a training run.

    Cost model (from launch_major_training.sh):
      - Collection: ~1.5 hours per 10K games on c5.2xlarge
      - Training: ~24 hours per 30 epochs on g4dn.xlarge for 4M decisions
      - Storage: ~$0.01 (negligible for <100MB)

    Args:
        collection_config: Data collection settings. None to skip.
        training_config: GPU training settings. None to skip.

    Returns:
        CostEstimate with per-phase and total breakdown.
    """
    estimate = CostEstimate()

    if collection_config is not None:
        # ~1.5 hours per 10K games
        estimate.collection_hours = (collection_config.num_games / 10000) * 1.5
        price = SPOT_PRICES.get(
            collection_config.instance_type,
            collection_config.spot_price,
        )
        estimate.collection_cost = round(estimate.collection_hours * price, 2)

    if training_config is not None:
        # ~24 hours per 30 epochs
        estimate.training_hours = (training_config.epochs / 30) * 24
        price = SPOT_PRICES.get(
            training_config.instance_type,
            training_config.spot_price,
        )
        estimate.training_cost = round(estimate.training_hours * price, 2)

    estimate.total = round(
        estimate.collection_cost + estimate.training_cost + estimate.storage_cost,
        2,
    )
    return estimate


def get_current_month_spend(
    region: str = "us-east-1",
    cost_tracking_file: Optional[Path] = None,
) -> float:
    """
    Get current month's AWS spend via Cost Explorer API with local fallback.

    Tries AWS Cost Explorer first (requires ce:GetCostAndUsage permission).
    Falls back to local cost tracking file if API is unavailable.

    Args:
        region: AWS region for API calls.
        cost_tracking_file: Local JSON file tracking costs. Defaults to ~/.mtg_rl_cost_tracking.json.

    Returns:
        Current month spend in USD.
    """
    if cost_tracking_file is None:
        cost_tracking_file = DEFAULT_COST_TRACKING_FILE

    # Try AWS Cost Explorer
    spend = _try_cost_explorer(region)
    if spend is not None:
        return spend

    # Fallback: local tracking file
    return _read_local_spend(cost_tracking_file)


def _try_cost_explorer(region: str) -> Optional[float]:
    """Query AWS Cost Explorer for current month spend. Returns None on failure."""
    try:
        import boto3
        ce = boto3.client("ce", region_name=region)

        now = datetime.now(timezone.utc)
        start = now.strftime("%Y-%m-01")
        end = now.strftime("%Y-%m-%d")

        response = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
        )

        results = response.get("ResultsByTime", [])
        if results:
            amount_str = results[0]["Total"]["UnblendedCost"]["Amount"]
            return round(float(amount_str), 2)
    except Exception:
        pass
    return None


def _read_local_spend(cost_tracking_file: Path) -> float:
    """Read current month spend from local JSON tracking file."""
    if not cost_tracking_file.exists():
        return 0.0
    try:
        data = json.loads(cost_tracking_file.read_text())
        month_key = datetime.now().strftime("%Y-%m")
        return float(data.get(month_key, {}).get("total_spent", 0.0))
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


def check_budget(
    estimated_cost: float,
    budget_limit: float = 100.0,
    region: str = "us-east-1",
    cost_tracking_file: Optional[Path] = None,
) -> bool:
    """
    Check if estimated cost fits within remaining monthly budget.

    Args:
        estimated_cost: Estimated cost for the run in USD.
        budget_limit: Monthly budget cap in USD.
        region: AWS region for cost lookups.
        cost_tracking_file: Local cost tracking file path.

    Returns:
        True if within budget, False if would exceed.
    """
    current_spend = get_current_month_spend(region, cost_tracking_file)
    remaining = budget_limit - current_spend

    print(f"  Current month spend:  ${current_spend:.2f}")
    print(f"  Remaining budget:     ${remaining:.2f} (of ${budget_limit:.2f})")
    print(f"  Estimated run cost:   ${estimated_cost:.2f}")

    if estimated_cost > remaining:
        print(f"  [FAIL] Estimated cost ${estimated_cost:.2f} exceeds remaining ${remaining:.2f}")
        return False

    print("  [OK] Budget check passed")
    return True


def update_cost_tracking(
    estimated_cost: float,
    run_id: str,
    cost_tracking_file: Optional[Path] = None,
) -> None:
    """
    Record an estimated cost in the local tracking file.

    Args:
        estimated_cost: Estimated cost in USD.
        run_id: Unique run identifier.
        cost_tracking_file: Local JSON file for tracking.
    """
    if cost_tracking_file is None:
        cost_tracking_file = DEFAULT_COST_TRACKING_FILE

    try:
        data = json.loads(cost_tracking_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    month_key = datetime.now().strftime("%Y-%m")
    if month_key not in data:
        data[month_key] = {"total_spent": 0.0, "runs": []}

    data[month_key]["runs"].append({
        "run_id": run_id,
        "estimated_cost": estimated_cost,
        "timestamp": datetime.now().isoformat(),
        "status": "launched",
    })
    data[month_key]["total_spent"] += estimated_cost

    cost_tracking_file.write_text(json.dumps(data, indent=2))
    print(f"  [OK] Cost tracking updated: +${estimated_cost:.2f} for {run_id}")
