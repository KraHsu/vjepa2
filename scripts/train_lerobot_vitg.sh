#!/usr/bin/env bash
# Train lerobot ViT-g/16 (256px, 8 frames)
#
# ── Local (all GPUs auto-detected) ──
#   bash scripts/train_lerobot_vitg.sh
#
# ── Local with WandB ──
#   WANDB_API_KEY=xxx bash scripts/train_lerobot_vitg.sh
#
# ── DLC / multi-node (env vars injected by scheduler) ──
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train_lerobot_vitg.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash scripts/train_lerobot_vitg.sh
#
# ── SLURM distributed (via submitit) ──
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
    # ── GPU / Node topology (compatible with DLC and local runs) ──
    NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
    NNODES="${NNODES:-${WORLD_SIZE:-1}}"
    NODE_RANK="${NODE_RANK:-${RANK:-0}}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-29500}"

    echo "Launching on ${NNODES} node(s), ${NPROC_PER_NODE} GPU(s)/node, node rank ${NODE_RANK}"
    echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

    torchrun \
        --nnodes "${NNODES}" \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --node_rank "${NODE_RANK}" \
        --master_addr "${MASTER_ADDR}" \
        --master_port "${MASTER_PORT}" \
        scripts/train_lerobot_torchrun.py \
        --fname "${CONFIG}" \
        "${EXTRA_ARGS[@]}"
fi
