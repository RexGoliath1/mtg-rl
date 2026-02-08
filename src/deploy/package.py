"""
Code packaging and S3 operations for ForgeRL deployments.

Handles creating tarballs of project code and Forge JAR+resources,
uploading/downloading from S3, and checking for existing artifacts.

GOTCHA (from project memory): macOS tar creates ._* AppleDouble files.
All tar operations set COPYFILE_DISABLE=1 to prevent this.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# Default patterns to exclude when packaging project code
CODE_EXCLUDE_PATTERNS = [
    "*.pyc",
    "__pycache__",
    ".git",
    "forge-repo",
    "data",
    "checkpoints",
    "wandb",
    "training_output",
    "*.pt",
    "*.pth",
    "*.h5",
    "*.hdf5",
    ".venv",
    "node_modules",
]

# Files/dirs to include in the code tarball
CODE_INCLUDE = ["src", "scripts", "decks", "pyproject.toml"]


def create_code_tarball(
    project_root: Path,
    exclude_patterns: Optional[list[str]] = None,
    include_paths: Optional[list[str]] = None,
) -> Path:
    """
    Create a tarball of project code for deployment.

    Args:
        project_root: Root directory of the project.
        exclude_patterns: Glob patterns to exclude. Defaults to CODE_EXCLUDE_PATTERNS.
        include_paths: Relative paths within project_root to include. Defaults to CODE_INCLUDE.

    Returns:
        Path to the created tarball in a temp directory.

    Raises:
        RuntimeError: If tar command fails.
    """
    if exclude_patterns is None:
        exclude_patterns = CODE_EXCLUDE_PATTERNS
    if include_paths is None:
        include_paths = CODE_INCLUDE

    # Filter to only paths that exist
    existing_paths = [p for p in include_paths if (project_root / p).exists()]
    if not existing_paths:
        raise RuntimeError(f"No includable paths found in {project_root}: {include_paths}")

    tarball_path = Path(tempfile.mkdtemp()) / "forgerl_code.tar.gz"

    exclude_args = []
    for pattern in exclude_patterns:
        exclude_args.extend(["--exclude", pattern])

    cmd = [
        "tar", "-czf", str(tarball_path),
        *exclude_args,
        "-C", str(project_root),
        *existing_paths,
    ]

    # macOS GOTCHA: prevent ._* AppleDouble files
    env = os.environ.copy()
    env["COPYFILE_DISABLE"] = "1"

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    # tar may emit warnings but still succeed (exit code 0 or 1 on some systems)
    if result.returncode > 1:
        raise RuntimeError(f"tar failed (exit {result.returncode}): {result.stderr}")

    size_mb = tarball_path.stat().st_size / (1024 * 1024)
    print(f"  [OK] Code tarball: {tarball_path.name} ({size_mb:.1f} MB)")
    return tarball_path


def create_forge_tarball(forge_dir: Path) -> Optional[Path]:
    """
    Create a tarball of the Forge JAR and resources.

    Looks for the jar-with-dependencies JAR and the forge-gui/res directory.

    Args:
        forge_dir: Path to forge-repo directory.

    Returns:
        Path to the created tarball, or None if Forge is not built.
    """
    # Find the JAR
    target_dir = forge_dir / "forge-gui-desktop" / "target"
    if not target_dir.exists():
        print("  [SKIP] Forge not built (no target directory)")
        return None

    jar_files = list(target_dir.glob("*jar-with-dependencies.jar"))
    # Exclude macOS ._* files
    jar_files = [j for j in jar_files if not j.name.startswith("._")]
    if not jar_files:
        print("  [SKIP] Forge JAR not found in target directory")
        return None

    jar_file = jar_files[0]
    jar_rel = str(jar_file.relative_to(forge_dir))

    # Verify resources exist
    res_dir = forge_dir / "forge-gui" / "res"
    if not res_dir.exists():
        print("  [WARN] Forge resources directory not found, packaging JAR only")

    tarball_path = Path(tempfile.mkdtemp()) / "forge_jar.tar.gz"

    include_paths = [jar_rel]
    if res_dir.exists():
        include_paths.append("forge-gui/res")

    cmd = [
        "tar", "-czf", str(tarball_path),
        "-C", str(forge_dir),
        *include_paths,
    ]

    env = os.environ.copy()
    env["COPYFILE_DISABLE"] = "1"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode > 1:
        raise RuntimeError(f"Forge tar failed: {result.stderr}")

    size_mb = tarball_path.stat().st_size / (1024 * 1024)
    print(f"  [OK] Forge tarball: {tarball_path.name} ({size_mb:.1f} MB)")
    return tarball_path


def _get_s3_client(region: str = "us-east-1"):
    """Create a boto3 S3 client."""
    import boto3
    return boto3.client("s3", region_name=region)


def upload_to_s3(
    local_path: Path,
    s3_bucket: str,
    s3_key: str,
    region: str = "us-east-1",
) -> None:
    """
    Upload a local file to S3.

    Args:
        local_path: Path to local file.
        s3_bucket: S3 bucket name.
        s3_key: S3 object key (path within bucket).
        region: AWS region.
    """
    s3 = _get_s3_client(region)
    file_size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  Uploading {local_path.name} ({file_size_mb:.1f} MB) to s3://{s3_bucket}/{s3_key}...")
    s3.upload_file(str(local_path), s3_bucket, s3_key)
    print("  [OK] Upload complete")


def download_from_s3(
    s3_bucket: str,
    s3_prefix: str,
    local_dir: Path,
    region: str = "us-east-1",
    exclude_patterns: Optional[list[str]] = None,
) -> list[Path]:
    """
    Download all files under an S3 prefix to a local directory.

    Args:
        s3_bucket: S3 bucket name.
        s3_prefix: S3 key prefix to download from.
        local_dir: Local directory to download into.
        region: AWS region.
        exclude_patterns: Filename patterns to skip (simple substring match).

    Returns:
        List of downloaded local file paths.
    """
    import boto3
    s3 = boto3.client("s3", region_name=region)

    if exclude_patterns is None:
        exclude_patterns = []

    local_dir.mkdir(parents=True, exist_ok=True)

    # List objects under prefix
    paginator = s3.get_paginator("list_objects_v2")
    downloaded: list[Path] = []

    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(s3_prefix):].lstrip("/")
            if not relative:
                continue

            # Check exclude patterns
            if any(pat in relative for pat in exclude_patterns):
                continue

            local_file = local_dir / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(s3_bucket, key, str(local_file))
            downloaded.append(local_file)

    print(f"  [OK] Downloaded {len(downloaded)} files to {local_dir}")
    return downloaded


def s3_key_exists(
    s3_bucket: str,
    s3_key: str,
    region: str = "us-east-1",
) -> bool:
    """
    Check if an S3 key exists.

    Args:
        s3_bucket: S3 bucket name.
        s3_key: S3 object key.
        region: AWS region.

    Returns:
        True if the key exists, False otherwise.
    """
    s3 = _get_s3_client(region)
    try:
        s3.head_object(Bucket=s3_bucket, Key=s3_key)
        return True
    except Exception:
        return False
