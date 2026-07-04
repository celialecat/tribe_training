#!/usr/bin/env bash
# ==============================================================================
# Run the end-to-end pipeline on the instance. All outputs (Hydra run dirs,
# TensorBoard logs, checkpoints) are written under $MOUNT_POINT so they persist
# on the block storage after the instance is destroyed.
#
# Env in: MOUNT_POINT, DATASET_SOURCE, DATASET_LIMIT, TRAIN_OVERRIDES, PIPELINE_SKIP
# ==============================================================================
set -euo pipefail
MOUNT_POINT="${MOUNT_POINT:-/mnt/ysp-data}"

# shellcheck disable=SC1090
source "${MOUNT_POINT}/ysp.env"
# shellcheck disable=SC1091
source "${YSP_VENV}/bin/activate"

# Run from the block storage so relative Hydra outputs (outputs/, runs/) persist.
cd "${MOUNT_POINT}"

args=()
[ -n "${DATASET_SOURCE:-}" ] && args+=(--source "${DATASET_SOURCE}")
[ -n "${DATASET_LIMIT:-}" ]  && args+=(--limit "${DATASET_LIMIT}")
[ -n "${PIPELINE_SKIP:-}" ]  && args+=(--skip ${PIPELINE_SKIP})

echo "==[pipeline]== ysp-pipeline ${args[*]} -- ${TRAIN_OVERRIDES:-}"
# TRAIN_OVERRIDES are Hydra key=value pairs forwarded after the positional break.
# shellcheck disable=SC2086
python -m app.pipeline.cli "${args[@]}" ${TRAIN_OVERRIDES:-}

echo "==[pipeline]== checkpoints + logs are under ${MOUNT_POINT}/outputs and ${MOUNT_POINT}/runs"
