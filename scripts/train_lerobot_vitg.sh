#!/usr/bin/env bash
# Train lerobot ViT-g/16 (256px, 8 frames)
#
# Local (all GPUs auto-detected):
#   bash scripts/train_lerobot_vitg.sh
#
# Local with WandB:
#   WANDB_API_KEY=xxx bash scripts/train_lerobot_vitg.sh
#
# SLURM distributed (via submitit):
#   bash scripts/train_lerobot_vitg.sh --distributed --account my_account --partition learn
#
# Override config:
#   CONFIG=configs/train/vitg16/lerobot-256px-8f.yaml bash scripts/train_lerobot_vitg.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/train/vitg16/lerobot-vitg-256px-8f.yaml}"
DISTRIBUTED=false
EXTRA_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--distributed" ]]; then
        DISTRIBUTED=true
    else
        EXTRA_ARGS+=("$arg")
    fi
done

if $DISTRIBUTED; then
    echo "Submitting SLURM job via submitit..."
    python -m app.main_distributed \
        --fname "${CONFIG}" \
        "${EXTRA_ARGS[@]}"
else
    NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    NGPUS="${NGPUS:-1}"
    DEVICES=""
    for i in $(seq 0 $((NGPUS - 1))); do
        DEVICES="$DEVICES cuda:$i"
    done
    echo "Launching locally on ${NGPUS} GPU(s): ${DEVICES}"
    python -m app.main \
        --fname "${CONFIG}" \
        --devices ${DEVICES} \
        "${EXTRA_ARGS[@]}"
fi
