#!/bin/bash
set -e

# ============================================================================
# Merge Fleet Collection Data
# ============================================================================
# Downloads HDF5 files from all fleet instance subdirectories, merges them
# into a single training dataset, and uploads the merged file back to S3.
#
# Usage:
#   ./scripts/merge_collection_data.sh --fleet-prefix fleet_20260208_143000
#   ./scripts/merge_collection_data.sh --fleet-prefix fleet_20260208_143000 --output-dir /tmp/merged
#   ./scripts/merge_collection_data.sh --fleet-prefix fleet_20260208_143000 --skip-upload
# ============================================================================

S3_BUCKET="mtg-rl-checkpoints-20260124190118616600000001"
REGION="${REGION:-us-east-1}"
FLEET_PREFIX=""
OUTPUT_DIR=""
SKIP_UPLOAD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fleet-prefix)
            FLEET_PREFIX="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        --bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --fleet-prefix FLEET_ID [--output-dir DIR] [--skip-upload] [--bucket BUCKET]"
            exit 1
            ;;
    esac
done

if [ -z "$FLEET_PREFIX" ]; then
    echo "ERROR: --fleet-prefix is required"
    echo "Usage: $0 --fleet-prefix fleet_20260208_143000"
    exit 1
fi

# Default output dir
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/tmp/fleet_merge_${FLEET_PREFIX}"
fi

S3_BASE="s3://${S3_BUCKET}/imitation_data/${FLEET_PREFIX}"

echo "============================================================"
echo "FLEET DATA MERGE"
echo "============================================================"
echo "Fleet Prefix: $FLEET_PREFIX"
echo "S3 Base:      $S3_BASE"
echo "Output Dir:   $OUTPUT_DIR"
echo ""

# Check for fleet manifest
echo "Checking fleet manifest..."
MANIFEST=$(aws s3 cp "${S3_BASE}/fleet_manifest.json" - 2>/dev/null || echo "")
if [ -n "$MANIFEST" ]; then
    NUM_INSTANCES=$(echo "$MANIFEST" | python3 -c "import sys,json; print(json.load(sys.stdin)['num_instances'])")
    TOTAL_GAMES=$(echo "$MANIFEST" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_games'])")
    echo "Fleet manifest found: $NUM_INSTANCES instances, $TOTAL_GAMES total games"
else
    echo "WARNING: No fleet manifest found. Discovering instance directories..."
    NUM_INSTANCES=""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download all HDF5 files from all instance subdirectories
echo ""
echo "Downloading HDF5 files..."
DOWNLOAD_COUNT=0

# List all instance subdirectories
INSTANCE_DIRS=$(aws s3 ls "${S3_BASE}/" | grep "PRE" | awk '{print $2}' | grep "^instance_")

if [ -z "$INSTANCE_DIRS" ]; then
    echo "ERROR: No instance_* subdirectories found in ${S3_BASE}/"
    echo "Available contents:"
    aws s3 ls "${S3_BASE}/"
    exit 1
fi

for INST_DIR in $INSTANCE_DIRS; do
    INST_NAME="${INST_DIR%/}"
    INST_S3="${S3_BASE}/${INST_NAME}"
    INST_LOCAL="${OUTPUT_DIR}/${INST_NAME}"
    mkdir -p "$INST_LOCAL"

    echo "  Downloading from ${INST_NAME}..."

    # Check completion status
    if aws s3 ls "${INST_S3}/collection_complete.json" &>/dev/null; then
        echo "    Status: COMPLETE"
    else
        echo "    WARNING: Instance not yet complete (no collection_complete.json)"
        echo "    Downloading available data anyway..."
    fi

    # Download only HDF5 files (final + any checkpoints)
    H5_FILES=$(aws s3 ls "${INST_S3}/" 2>/dev/null | grep "\.h5$" | awk '{print $4}' || true)
    if [ -z "$H5_FILES" ]; then
        echo "    No HDF5 files found - skipping"
        continue
    fi

    for H5_FILE in $H5_FILES; do
        aws s3 cp "${INST_S3}/${H5_FILE}" "${INST_LOCAL}/${H5_FILE}" --quiet
        DOWNLOAD_COUNT=$((DOWNLOAD_COUNT + 1))
        echo "    Downloaded: ${H5_FILE}"
    done
done

if [ $DOWNLOAD_COUNT -eq 0 ]; then
    echo ""
    echo "ERROR: No HDF5 files found across any instances"
    exit 1
fi

echo ""
echo "Downloaded $DOWNLOAD_COUNT HDF5 files total"

# Merge using Python (picks only *_final.h5 from each instance, falls back to newest checkpoint)
echo ""
echo "Merging HDF5 files..."

MERGED_FILE="${OUTPUT_DIR}/merged_${FLEET_PREFIX}.h5"

python3 << PYEOF
import h5py
import numpy as np
import json
import os
import sys
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
merged_path = Path("${MERGED_FILE}")

# Find the best HDF5 file per instance (prefer *_final.h5, else newest checkpoint)
h5_files = []
for inst_dir in sorted(output_dir.iterdir()):
    if not inst_dir.is_dir() or not inst_dir.name.startswith("instance_"):
        continue
    finals = sorted(inst_dir.glob("*_final.h5"))
    if finals:
        h5_files.append(finals[-1])  # newest final
    else:
        checkpoints = sorted(inst_dir.glob("*.h5"))
        if checkpoints:
            h5_files.append(checkpoints[-1])  # newest checkpoint
            print(f"  WARNING: {inst_dir.name} has no final.h5, using checkpoint: {checkpoints[-1].name}")

if not h5_files:
    print("ERROR: No usable HDF5 files found")
    sys.exit(1)

print(f"  Merging {len(h5_files)} files:")
for f in h5_files:
    print(f"    {f.parent.name}/{f.name}")

# Read and concatenate datasets
all_data = {}
total_rows = 0
dataset_names = None

for h5_path in h5_files:
    with h5py.File(h5_path, "r") as f:
        if dataset_names is None:
            dataset_names = list(f.keys())
        rows = f["states"].shape[0]
        total_rows += rows
        print(f"    {h5_path.parent.name}: {rows:,} decisions")
        for name in dataset_names:
            if name not in all_data:
                all_data[name] = []
            all_data[name].append(f[name][:])

# Concatenate
print(f"\n  Total decisions: {total_rows:,}")

with h5py.File(merged_path, "w") as out:
    for name in dataset_names:
        if len(all_data[name]) == 0:
            continue
        if all_data[name][0].dtype.kind in ('U', 'S', 'O'):
            # String data (game_state_json) - concatenate as list
            merged = []
            for arr in all_data[name]:
                merged.extend(arr.tolist())
            dt = h5py.special_dtype(vlen=str)
            out.create_dataset(name, data=merged, dtype=dt,
                             compression="gzip", compression_opts=4)
        else:
            merged = np.concatenate(all_data[name], axis=0)
            out.create_dataset(name, data=merged,
                             compression="gzip", compression_opts=4)

    # Metadata
    out.attrs["encoding_version"] = 2
    out.attrs["fleet_id"] = "${FLEET_PREFIX}"
    out.attrs["num_instances"] = len(h5_files)
    out.attrs["total_decisions"] = total_rows
    out.attrs["source_files"] = json.dumps([str(f) for f in h5_files])

size_mb = merged_path.stat().st_size / (1024 * 1024)
print(f"\n  Merged file: {merged_path.name}")
print(f"  Size: {size_mb:.2f} MB")
print(f"  Rows: {total_rows:,}")

# Print per-dataset shapes
with h5py.File(merged_path, "r") as f:
    print(f"\n  Datasets:")
    for name in f.keys():
        ds = f[name]
        print(f"    {name}: shape={ds.shape}, dtype={ds.dtype}")
PYEOF

MERGE_EXIT=$?
if [ $MERGE_EXIT -ne 0 ]; then
    echo "ERROR: Merge failed (exit code: $MERGE_EXIT)"
    exit 1
fi

echo ""
echo "Merge complete: $MERGED_FILE"

# Print summary stats
echo ""
echo "============================================================"
echo "MERGE SUMMARY"
echo "============================================================"
python3 << PYEOF
import h5py
import numpy as np

with h5py.File("${MERGED_FILE}", "r") as f:
    total = f["states"].shape[0]
    print(f"Total decisions:  {total:,}")
    print(f"State dimensions: {f['states'].shape[1]}")
    print(f"Fleet ID:         {f.attrs.get('fleet_id', 'unknown')}")
    print(f"Source instances:  {f.attrs.get('num_instances', 'unknown')}")

    # Decision type breakdown
    if "decision_types" in f:
        dtypes = f["decision_types"][:]
        type_names = {0: "choose_action", 1: "declare_attackers", 2: "declare_blockers", 3: "unknown"}
        print(f"\nDecision types:")
        for code, name in sorted(type_names.items()):
            count = int(np.sum(dtypes == code))
            if count > 0:
                pct = count / total * 100
                print(f"  {name}: {count:,} ({pct:.1f}%)")

    # Turn distribution
    if "turns" in f:
        turns = f["turns"][:]
        print(f"\nTurn distribution:")
        print(f"  Mean: {np.mean(turns):.1f}")
        print(f"  Max:  {np.max(turns)}")
        print(f"  Median: {np.median(turns):.1f}")
PYEOF

# Upload merged file to S3
if [ "$SKIP_UPLOAD" = false ]; then
    echo ""
    echo "Uploading merged file to S3..."
    aws s3 cp "$MERGED_FILE" "${S3_BASE}/merged_${FLEET_PREFIX}.h5"
    echo "Uploaded to: ${S3_BASE}/merged_${FLEET_PREFIX}.h5"

    # Upload merge complete signal
    echo "{\"status\":\"merged\",\"fleet_id\":\"${FLEET_PREFIX}\",\"merged_file\":\"merged_${FLEET_PREFIX}.h5\"}" | \
        aws s3 cp - "${S3_BASE}/merge_complete.json"
    echo "Merge complete signal uploaded"
else
    echo ""
    echo "Skipping S3 upload (--skip-upload)"
fi

echo ""
echo "============================================================"
echo "NEXT STEPS"
echo "============================================================"
echo ""
echo "# Download merged data for training:"
echo "aws s3 cp ${S3_BASE}/merged_${FLEET_PREFIX}.h5 data/training/"
echo ""
echo "# Train on merged data:"
echo "uv run python3 scripts/training_pipeline.py --mode bc --data data/training/merged_${FLEET_PREFIX}.h5"
echo ""
echo "Local files in: $OUTPUT_DIR"
echo "============================================================"
