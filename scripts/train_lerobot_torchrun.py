"""
Torchrun-compatible entry point for lerobot V-JEPA2 training.

Replaces app/main.py's mp.Process approach so that torchrun handles
process spawning and injects the correct RANK/WORLD_SIZE/MASTER_ADDR
environment variables for multi-node DLC training.
"""

import os
import sys
from pathlib import Path

# Must be set before importing torch so CUDA sees only one device per process.
# train.py uses cuda:0 throughout; this maps that to the correct physical GPU.
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

# Ensure project root is on sys.path when invoked via torchrun scripts/...
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pprint

import torch.distributed as dist
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, required=True, help="path to yaml config file")
parser.add_argument("--resume_preempt", action="store_true")


def main():
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank != 0:
        logging.getLogger().setLevel(logging.ERROR)

    logger.info(f"rank={rank}/{world_size}  local_rank={_local_rank}")

    # Each process sees exactly one GPU (cuda:0 == physical GPU _local_rank).
    # device_id accepts int or torch.device on recent PyTorch; int is the
    # broader-compatible form.
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank,
                            device_id=0)

    # Load config
    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = Path(params["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "params-pretrain.yaml", "w") as f:
            yaml.dump(params, f)

    # Barrier so all ranks wait until rank-0 has written the config
    dist.barrier()

    from app.vjepa_lerobot.train import main as train_main

    train_main(args=params, resume_preempt=args.resume_preempt)


if __name__ == "__main__":
    main()
