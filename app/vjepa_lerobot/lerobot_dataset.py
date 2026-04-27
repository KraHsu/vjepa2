# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import getLogger
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu

logger = getLogger()


def init_data(
    data_root,
    datasets,
    batch_size,
    frames_per_clip=8,
    fps=4,
    rank=0,
    world_size=1,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    tubelet_size=2,
    action_dim=7,
    state_dim=7,
):
    dataset = LeRobotVideoDataset(
        data_root=data_root,
        datasets=datasets,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        frameskip=tubelet_size,
        action_dim=action_dim,
        state_dim=state_dim,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("LeRobotVideoDataset data loader created")
    return data_loader, dist_sampler


class LeRobotVideoDataset(torch.utils.data.Dataset):
    """
    Dataset for LeRobot-format egocentric video data.

    Directory layout expected:
        {data_root}/{dataset_name}/
            data/chunk-NNN/episode_NNNNNN.parquet
            videos/chunk-NNN/observation.images.ego/episode_NNNNNN.mp4

    Each parquet row is one frame and contains:
        - frame_index, episode_index, timestamp
        - action  : list[float], length = action_dim_raw (e.g. 102)
        - state   : list[float], length = state_dim_raw  (e.g. 122)

    We project action/state to the target dims expected by the AC predictor
    via a simple linear slice (first `action_dim` / `state_dim` elements).
    """

    VIDEO_SUBDIR = "observation.images.ego"

    def __init__(
        self,
        data_root,
        datasets,
        frames_per_clip=8,
        fps=4,
        frameskip=2,
        transform=None,
        action_dim=7,
        state_dim=7,
    ):
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.frameskip = frameskip
        self.transform = transform
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Build episode index: list of (parquet_path, video_path)
        self.samples = []
        for ds_name in datasets:
            ds_root = os.path.join(data_root, ds_name)
            data_dir = os.path.join(ds_root, "data")
            video_base = os.path.join(ds_root, "videos")
            for chunk in sorted(os.listdir(data_dir)):
                chunk_dir = os.path.join(data_dir, chunk)
                if not os.path.isdir(chunk_dir):
                    continue
                for fname in sorted(os.listdir(chunk_dir)):
                    if not fname.endswith(".parquet"):
                        continue
                    ep_name = fname.replace(".parquet", "")  # episode_NNNNNN
                    parquet_path = os.path.join(chunk_dir, fname)
                    video_path = os.path.join(
                        video_base, chunk, self.VIDEO_SUBDIR, ep_name + ".mp4"
                    )
                    if os.path.exists(video_path):
                        self.samples.append((parquet_path, video_path))

        logger.info(f"LeRobotVideoDataset: {len(self.samples)} episodes across {datasets}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        loaded = False
        while not loaded:
            try:
                result = self._load(index)
                loaded = True
            except Exception as e:
                logger.warning(f"Failed to load sample {index}: {e}")
                index = np.random.randint(len(self))
        return result

    def _load(self, index):
        parquet_path, video_path = self.samples[index]

        # -- load parquet metadata
        df = pd.read_parquet(parquet_path)
        n_frames_total = len(df)

        # -- load video
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        target_fps = self.fps if self.fps is not None else vfps
        fstp = max(1, round(vfps / target_fps))  # frame step
        nframes_needed = fpc * fstp
        vlen = len(vr)

        if vlen < nframes_needed:
            raise ValueError(f"Video too short: {video_path}, need {nframes_needed}, got {vlen}")

        ef = np.random.randint(nframes_needed, vlen + 1)
        sf = ef - nframes_needed
        indices = np.arange(sf, sf + nframes_needed, fstp).astype(np.int64)

        # -- load frames
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        if self.transform is not None:
            buffer = self.transform(buffer)  # -> [C, T, H, W]

        # -- load action / state from parquet
        # Map video frame indices to parquet row indices
        # parquet rows are at native video fps; we subsample by fstp
        parquet_indices = np.clip(indices, 0, n_frames_total - 1)
        T_sub = len(parquet_indices)  # == frames_per_clip

        actions_raw = np.stack([np.array(df["action"].iloc[i]) for i in parquet_indices])  # [T, D_a]
        states_raw = np.stack([np.array(df["state"].iloc[i]) for i in parquet_indices])    # [T, D_s]

        # Slice / pad to target dims.
        # ac_predictor uses action_embed_dim for both action and state encoders,
        # so both must have the same width = action_dim.
        actions = actions_raw[:, : self.action_dim].astype(np.float32)   # [T, action_dim]

        # state may have a different raw dim; slice then zero-pad to action_dim
        s_raw = states_raw[:, : self.state_dim].astype(np.float32)       # [T, state_dim]
        if self.state_dim < self.action_dim:
            pad = np.zeros((s_raw.shape[0], self.action_dim - self.state_dim), dtype=np.float32)
            states = np.concatenate([s_raw, pad], axis=1)                # [T, action_dim]
        else:
            states = s_raw[:, : self.action_dim]                         # [T, action_dim]

        # Use raw action field directly (absolute joint angles / poses).
        # state_deltas are used as the "action" signal fed to the predictor.
        state_deltas = np.diff(states, axis=0)  # [T-1, action_dim]
        # Pad to length T by repeating the last delta so all tensors share the same T dim
        state_deltas = np.concatenate([state_deltas, state_deltas[-1:]], axis=0)  # [T, action_dim]

        # extrinsics: zeros placeholder (not available in lerobot format)
        extrinsics = np.zeros((T_sub, 6), dtype=np.float32)

        return (
            buffer,                                    # [C, T, H, W] tensor
            torch.from_numpy(state_deltas),            # [T, action_dim]  actions
            torch.from_numpy(states),                  # [T, state_dim]   states
            torch.from_numpy(extrinsics),              # [T, 6]           extrinsics
        )
