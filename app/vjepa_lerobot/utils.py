# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch

import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _unwrap(m):
    """Return the underlying module if wrapped in DDP/FSDP, else m itself."""
    return m.module if (m is not None and hasattr(m, "module")) else m


def _strip_pretrain_prefixes(state):
    """Strip wrapper prefixes from a pretrain state_dict.

    Pretrain checkpoints can use any of `module.`, `backbone.`, or `module.backbone.`
    depending on how they were trained/saved. We normalize to bare param names
    (`patch_embed.proj.weight`, ...) to match an unwrapped raw nn.Module.
    """
    out = {}
    for k, v in state.items():
        nk = k
        # Repeatedly peel off either prefix in any order until neither is present.
        for _ in range(2):
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("backbone."):
                nk = nk[len("backbone.") :]
        out[nk] = v
    return out


def _load_into(name, ddp_or_module, state, epoch):
    """Strict-aware load with explicit miss reporting.

    Loads into `_unwrap(...)` so the destination keys are bare (no `module.`),
    matching the stripped state_dict. Logs match counts so silent prefix mismatches
    can never go unnoticed again.
    """
    raw = _unwrap(ddp_or_module)
    stripped = _strip_pretrain_prefixes(state)
    dst_keys = set(raw.state_dict().keys())
    src_keys = set(stripped.keys())
    matched = dst_keys & src_keys
    if len(matched) == 0:
        sample_dst = sorted(dst_keys)[:2]
        sample_src = sorted(src_keys)[:2]
        raise RuntimeError(
            f"load_pretrained({name}): 0/{len(dst_keys)} keys matched. "
            f"Pretrain prefix stripping failed.\n"
            f"  dst sample: {sample_dst}\n"
            f"  src sample: {sample_src}"
        )
    msg = raw.load_state_dict(stripped, strict=False)
    logger.info(
        f"loaded pretrained {name} from epoch {epoch}: "
        f"matched={len(matched)}/{len(dst_keys)}, "
        f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
    )


def load_pretrained(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=False,
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    if load_encoder:
        _load_into("encoder", encoder, checkpoint[context_encoder_key], epoch)

    if load_predictor:
        _load_into("predictor", predictor, checkpoint["predictor"], epoch)

    if load_encoder and target_encoder is not None:
        _load_into("target_encoder", target_encoder, checkpoint[target_encoder_key], epoch)

    del checkpoint
    return encoder, predictor, target_encoder


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt=None,
    scaler=None,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    for key, model in [("encoder", encoder), ("predictor", predictor), ("target_encoder", target_encoder)]:
        if model is None:
            continue
        _load_into(key, model, checkpoint[key], epoch)

    if opt is not None:
        opt.load_state_dict(checkpoint["opt"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"loaded optimizers from epoch {epoch}")
    del checkpoint
    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=8,
    tubelet_size=2,
    model_name="vit_giant_xformers",
    crop_size=256,
    pred_depth=24,
    pred_num_heads=16,
    pred_embed_dim=1024,
    uniform_power=False,
    use_sdpa=True,
    use_rope=True,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=True,
    action_embed_dim=7,
    state_embed_dim=7,
    use_extrinsics=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )

    predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_extrinsics=use_extrinsics,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    encoder.to(device)
    predictor.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder parameters: {count_parameters(encoder):,}")
    logger.info(f"Predictor parameters: {count_parameters(predictor):,}")
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_scaler=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    # GradScaler is meaningful only for fp16. bf16 has the same dynamic range as
    # fp32 so dynamic loss scaling does nothing useful (and the scale grows
    # unboundedly until it saturates).
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    return optimizer, scaler, scheduler, wd_scheduler
