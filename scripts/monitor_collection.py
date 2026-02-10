#!/usr/bin/env python3
"""
Monitor data collection progress by polling S3.

Tracks: file count, total size, growth rate, per-instance status.
Alerts if growth stalls (collection may be stuck).

Usage:
    python3 scripts/monitor_collection.py
    python3 scripts/monitor_collection.py --fleet-id fleet_20260210_120000
    python3 scripts/monitor_collection.py --poll-interval 30
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

S3_BUCKET = "mtg-rl-checkpoints-20260124190118616600000001"
S3_PREFIX = "imitation_data/"


def run_aws(args: list[str]) -> str:
    """Run an AWS CLI command and return stdout."""
    result = subprocess.run(
        ["aws"] + args,
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"AWS CLI error: {result.stderr.strip()}", file=sys.stderr)
        return ""
    return result.stdout


def get_s3_objects(prefix: str) -> list[dict]:
    """List S3 objects under a prefix."""
    output = run_aws([
        "s3api", "list-objects-v2",
        "--bucket", S3_BUCKET,
        "--prefix", prefix,
        "--output", "json",
    ])
    if not output:
        return []
    data = json.loads(output)
    return data.get("Contents", [])


def format_size(bytes_val: int) -> str:
    """Format bytes to human-readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def monitor(fleet_id: str | None, poll_interval: int):
    """Poll S3 and display collection progress."""
    prefix = S3_PREFIX
    if fleet_id:
        prefix = f"{S3_PREFIX}{fleet_id}/"

    prev_total_size = 0
    prev_file_count = 0
    prev_time = time.time()
    stall_count = 0

    print(f"Monitoring S3: s3://{S3_BUCKET}/{prefix}")
    print(f"Poll interval: {poll_interval}s")
    print("=" * 70)

    while True:
        objects = get_s3_objects(prefix)
        now = time.time()
        elapsed = now - prev_time

        # Aggregate stats
        h5_files = [o for o in objects if o["Key"].endswith(".h5")]
        json_files = [o for o in objects if o["Key"].endswith(".json")]
        log_files = [o for o in objects if o["Key"].endswith(".log")]

        total_size = sum(o.get("Size", 0) for o in objects)
        h5_size = sum(o.get("Size", 0) for o in h5_files)

        # Growth rate
        size_delta = total_size - prev_total_size
        file_delta = len(h5_files) - prev_file_count
        rate_mbs = (size_delta / (1024 * 1024)) / max(elapsed, 1) if size_delta > 0 else 0

        # Per-instance breakdown (if fleet)
        instances: dict[str, int] = {}
        for obj in h5_files:
            parts = obj["Key"].split("/")
            for p in parts:
                if p.startswith("instance_"):
                    instances[p] = instances.get(p, 0) + 1
                    break

        # Display
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] Collection Status")
        print(f"  HDF5 files: {len(h5_files)} (+{file_delta})")
        print(f"  HDF5 size:  {format_size(h5_size)}")
        print(f"  Total size: {format_size(total_size)} (+{format_size(size_delta)})")
        print(f"  Growth:     {rate_mbs:.2f} MB/s")
        print(f"  JSON:       {len(json_files)}  Logs: {len(log_files)}")

        if instances:
            print(f"  Instances:  {len(instances)} active")
            for inst, count in sorted(instances.items()):
                print(f"    {inst}: {count} files")

        # Stall detection
        if size_delta == 0 and prev_total_size > 0:
            stall_count += 1
            if stall_count >= 3:
                print(f"\n  *** ALERT: No growth for {stall_count * poll_interval}s — collection may be stuck! ***")
        else:
            stall_count = 0

        # Check for completion signals (summary JSON)
        summaries = [o for o in json_files if "summary" in o["Key"]]
        if summaries:
            print(f"\n  Completed: {len(summaries)} summary file(s) found")
            if fleet_id and len(summaries) >= len(instances):
                print("  All instances complete!")
                break

        prev_total_size = total_size
        prev_file_count = len(h5_files)
        prev_time = now

        try:
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break


def main():
    parser = argparse.ArgumentParser(description="Monitor S3 data collection progress")
    parser.add_argument("--fleet-id", type=str, default=None,
                        help="Fleet ID to monitor (e.g., fleet_20260210_120000)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between polls (default: 60)")
    args = parser.parse_args()

    monitor(args.fleet_id, args.poll_interval)


if __name__ == "__main__":
    main()
