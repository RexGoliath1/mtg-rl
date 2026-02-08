"""
src.deploy - Unified AWS deployment module for ForgeRL training orchestration.

Replaces scattered shell scripts with a single Python module for
data collection, GPU training, cost tracking, and monitoring.
"""

from src.deploy.config import (
    DeployConfig,
    CollectionConfig,
    TrainingConfig,
    CostEstimate,
)
from src.deploy.aws import (
    validate_credentials,
    find_ami,
    get_vpc_info,
    launch_spot_instance,
    wait_for_instance,
    get_running_instances,
    terminate_instances,
)
from src.deploy.cost import (
    estimate_costs,
    get_current_month_spend,
    check_budget,
)
from src.deploy.package import (
    create_code_tarball,
    create_forge_tarball,
    upload_to_s3,
    download_from_s3,
    s3_key_exists,
)
from src.deploy.userdata import (
    generate_collection_userdata,
    generate_docker_collection_userdata,
    generate_training_userdata,
    generate_notification_section,
)
from src.deploy.monitor import (
    poll_s3_completion,
    stream_s3_logs,
    get_instance_status,
    download_results,
)

__all__ = [
    # Config
    "DeployConfig",
    "CollectionConfig",
    "TrainingConfig",
    "CostEstimate",
    # AWS
    "validate_credentials",
    "find_ami",
    "get_vpc_info",
    "launch_spot_instance",
    "wait_for_instance",
    "get_running_instances",
    "terminate_instances",
    # Cost
    "estimate_costs",
    "get_current_month_spend",
    "check_budget",
    # Packaging
    "create_code_tarball",
    "create_forge_tarball",
    "upload_to_s3",
    "download_from_s3",
    "s3_key_exists",
    # Userdata
    "generate_collection_userdata",
    "generate_docker_collection_userdata",
    "generate_training_userdata",
    "generate_notification_section",
    # Monitor
    "poll_s3_completion",
    "stream_s3_logs",
    "get_instance_status",
    "download_results",
]
