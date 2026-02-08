"""
Monitoring and result retrieval for ForgeRL AWS deployments.

Provides S3 polling for completion signals, log streaming,
instance status checks, and result downloading.

The ``tail_live_log`` function reads the periodically-uploaded live log
from S3 (pushed every 5 min by the userdata background uploader) so you
can see progress without SSH access.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional


def poll_s3_completion(
    s3_bucket: str,
    s3_key: str,
    interval: int = 60,
    timeout: int = 7200,
    region: str = "us-east-1",
) -> bool:
    """
    Poll S3 for a completion signal file.

    Checks for the existence of s3_key every `interval` seconds
    until found or timeout is reached.

    Args:
        s3_bucket: S3 bucket name.
        s3_key: S3 key to check for (e.g. "imitation_data/run_id/collection_complete.json").
        interval: Seconds between checks.
        timeout: Maximum seconds to wait.
        region: AWS region.

    Returns:
        True if the completion signal was found, False if timed out.
    """
    import boto3
    s3 = boto3.client("s3", region_name=region)

    max_checks = timeout // interval
    for check in range(1, max_checks + 1):
        try:
            s3.head_object(Bucket=s3_bucket, Key=s3_key)
            elapsed = check * interval
            print(f"  [OK] Completion signal found after {elapsed // 60}m {elapsed % 60}s")
            return True
        except Exception:
            pass

        elapsed = check * interval
        print(
            f"  Waiting for completion... "
            f"({check}/{max_checks} checks, {elapsed // 60}m elapsed)"
        )
        time.sleep(interval)

    print(f"  [TIMEOUT] Completion signal not found after {timeout // 60}m")
    return False


def stream_s3_logs(
    s3_bucket: str,
    log_key: str,
    region: str = "us-east-1",
    tail_lines: int = 50,
) -> str:
    """
    Download and return the tail of an S3 log file.

    Args:
        s3_bucket: S3 bucket name.
        log_key: S3 key of the log file.
        region: AWS region.
        tail_lines: Number of lines from the end to return.

    Returns:
        String containing the last N lines, or an error message.
    """
    import boto3
    s3 = boto3.client("s3", region_name=region)

    try:
        response = s3.get_object(Bucket=s3_bucket, Key=log_key)
        content = response["Body"].read().decode("utf-8", errors="replace")
        lines = content.splitlines()
        tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
        return "\n".join(tail)
    except s3.exceptions.NoSuchKey:
        return f"Log file not found: s3://{s3_bucket}/{log_key}"
    except Exception as e:
        return f"Error reading log: {e}"


def get_instance_status(
    instance_id: str,
    region: str = "us-east-1",
) -> dict[str, str]:
    """
    Get the current status of an EC2 instance.

    Args:
        instance_id: EC2 instance ID.
        region: AWS region.

    Returns:
        Dict with keys: instance_id, state, public_ip, instance_type, launch_time.
    """
    import boto3
    ec2 = boto3.client("ec2", region_name=region)

    try:
        response = ec2.describe_instances(InstanceIds=[instance_id])
        inst = response["Reservations"][0]["Instances"][0]
        return {
            "instance_id": instance_id,
            "state": inst["State"]["Name"],
            "public_ip": inst.get("PublicIpAddress", "N/A"),
            "instance_type": inst.get("InstanceType", "unknown"),
            "launch_time": str(inst.get("LaunchTime", "")),
        }
    except Exception as e:
        return {
            "instance_id": instance_id,
            "state": "error",
            "public_ip": "N/A",
            "instance_type": "unknown",
            "launch_time": "",
            "error": str(e),
        }


def download_results(
    s3_bucket: str,
    s3_prefix: str,
    local_dir: Path,
    region: str = "us-east-1",
    include_tensorboard: bool = False,
) -> list[Path]:
    """
    Download training results from S3 to a local directory.

    Args:
        s3_bucket: S3 bucket name.
        s3_prefix: S3 key prefix (e.g. "training_runs/major_20260208_143000/").
        local_dir: Local directory to download into.
        region: AWS region.
        include_tensorboard: Whether to include tensorboard/ files.

    Returns:
        List of downloaded file paths.
    """
    from src.deploy.package import download_from_s3

    exclude = []
    if not include_tensorboard:
        exclude.append("tensorboard/")

    return download_from_s3(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        local_dir=local_dir,
        region=region,
        exclude_patterns=exclude,
    )


def tail_live_log(
    s3_bucket: str,
    run_prefix: str,
    region: str = "us-east-1",
    tail_lines: int = 40,
    log_name: str = "live_log.txt",
) -> str:
    """
    Read the live log that the background uploader pushes every 5 minutes.

    This is the primary way to check on a running EC2 job without SSH.
    The userdata template uploads logs to ``<run_prefix>/live_log.txt``
    every 5 minutes.

    Args:
        s3_bucket: S3 bucket name.
        run_prefix: S3 prefix for the run (e.g. "imitation_data/collection_20260208/").
        region: AWS region.
        tail_lines: Number of lines from the end to return.
        log_name: Name of the live log file.

    Returns:
        String containing the last N lines, or an error/status message.
    """
    log_key = f"{run_prefix.rstrip('/')}/{log_name}"
    content = stream_s3_logs(s3_bucket, log_key, region=region, tail_lines=tail_lines)
    return content


def parse_progress_from_log(log_content: str) -> Optional[dict]:
    """
    Parse the most recent PROGRESS line from a structured log.

    Extracts game count, success rate, throughput, ETA, etc. from lines
    matching the ``PROGRESS game=X/Y ...`` format emitted by
    ``collect_ai_training_data.py``.

    Args:
        log_content: Raw log text (e.g. from ``tail_live_log``).

    Returns:
        Dict with parsed progress fields, or None if no PROGRESS line found.
    """
    # Find all PROGRESS lines and take the last one
    progress_lines = [
        line for line in log_content.splitlines()
        if "PROGRESS game=" in line
    ]
    if not progress_lines:
        return None

    last = progress_lines[-1]
    result = {}

    # Extract key=value pairs using regex
    patterns = {
        "game": r"game=(\d+)/(\d+)",
        "success": r"success=([\d.]+)%",
        "throughput": r"throughput=([\d.]+)",
        "elapsed": r"elapsed=([\d.]+)s",
        "eta": r"ETA=([\d.]+) min",
        "decisions": r"decisions=(\d+)",
        "errors": r"errors=(\d+)",
        "timeouts": r"timeouts=(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, last)
        if match:
            if key == "game":
                result["games_done"] = int(match.group(1))
                result["games_total"] = int(match.group(2))
                result["pct"] = result["games_done"] / result["games_total"] * 100
            elif key in ("success", "throughput", "elapsed", "eta"):
                result[key] = float(match.group(1))
            else:
                result[key] = int(match.group(1))

    return result if result else None


def find_latest_collection(
    s3_bucket: str,
    region: str = "us-east-1",
) -> Optional[str]:
    """
    Find the S3 prefix of the most recent completed data collection.

    Searches for collection_complete.json files and returns the
    parent prefix of the most recent one.

    Args:
        s3_bucket: S3 bucket name.
        region: AWS region.

    Returns:
        S3 prefix string (e.g. "imitation_data/collection_20260208/"), or None.
    """
    import boto3
    s3 = boto3.client("s3", region_name=region)

    paginator = s3.get_paginator("list_objects_v2")
    completion_keys: list[str] = []

    for page in paginator.paginate(
        Bucket=s3_bucket,
        Prefix="imitation_data/",
    ):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("collection_complete.json"):
                completion_keys.append(obj["Key"])

    if not completion_keys:
        return None

    # Sort by key name (timestamps in path ensure chronological order)
    completion_keys.sort(reverse=True)
    latest = completion_keys[0]
    # Strip the filename to get the prefix
    prefix = latest.rsplit("/", 1)[0] + "/"
    print(f"  [OK] Latest collection: s3://{s3_bucket}/{prefix}")
    return prefix
