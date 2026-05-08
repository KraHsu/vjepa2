# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import contextlib
import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_lerobot.lerobot_dataset import init_data
from app.vjepa_lerobot.utils import init_opt, init_video_model, load_checkpoint, load_pretrained
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, get_logger, gpu_timer

log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)
STATS_MAX_ELEMENTS = 1_000_000


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _sample_flat(tensor, max_elements=STATS_MAX_ELEMENTS):
    flat = tensor.detach().flatten()
    if flat.numel() > max_elements:
        stride = int(np.ceil(flat.numel() / max_elements))
        flat = flat[::stride][:max_elements]
    return flat.float()


def _sample_flat_pair(a, b, max_elements=STATS_MAX_ELEMENTS):
    a_flat = a.detach().flatten()
    b_flat = b.detach().flatten()
    if a_flat.numel() > max_elements:
        stride = int(np.ceil(a_flat.numel() / max_elements))
        a_flat = a_flat[::stride][:max_elements]
        b_flat = b_flat[::stride][:max_elements]
    return a_flat.float(), b_flat.float()


def _tensor_stats(prefix, tensor):
    t = _sample_flat(tensor)
    return {
        f"{prefix}/mean": t.mean().item(),
        f"{prefix}/std": t.std(unbiased=False).item(),
        f"{prefix}/norm": t.norm().item(),
        f"{prefix}/max_abs": t.abs().max().item(),
    }


def _param_norm(model):
    total = 0.0
    with torch.no_grad():
        for p in _unwrap(model).parameters():
            norm = torch.linalg.vector_norm(p.detach(), dtype=torch.float32)
            total += norm.item() ** 2
    return total**0.5


def _grad_norm(model):
    total = 0.0
    has_grad = False
    for p in _unwrap(model).parameters():
        if p.grad is None:
            continue
        has_grad = True
        norm = torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
        total += norm.item() ** 2
    return total**0.5 if has_grad else 0.0


def _ema_momentum(step, total_steps, ema):
    if ema is None:
        return None
    start, end = float(ema[0]), float(ema[1])
    progress = min(max(step, 0), max(1, total_steps)) / max(1, total_steps)
    return start + progress * (end - start)


@torch.no_grad()
def _update_target_encoder(encoder, target_encoder, momentum):
    params_q = [p.detach() for p in _unwrap(encoder).parameters()]
    params_k = [p.detach() for p in _unwrap(target_encoder).parameters()]
    torch._foreach_mul_(params_k, momentum)
    torch._foreach_add_(params_k, params_q, alpha=1.0 - momentum)


def _optimizer_group_lr(optimizer, group_idx):
    if group_idx >= len(optimizer.param_groups):
        return 0.0
    return optimizer.param_groups[group_idx].get("lr", 0.0)


def main(args, resume_preempt=False):
    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    # Loss scaling is only useful for fp16. bf16 keeps the fp32 dynamic range,
    # so a GradScaler would just multiply/divide by a constant 2^16 and grow
    # the scale until it saturates with no benefit.
    use_scaler = (dtype == torch.float16)

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)
    action_embed_dim = cfgs_model.get("action_embed_dim", 7)
    state_embed_dim = cfgs_model.get("state_embed_dim", 7)

    # -- DATA
    cfgs_data = args.get("data")
    data_root = cfgs_data.get("data_root")
    datasets = cfgs_data.get("datasets", [])
    max_num_frames = cfgs_data.get("frames_per_clip", 8)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    fps = cfgs_data.get("fps", 4)
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size", 16)
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 8)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug", {})
    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp", 1.0)
    normalize_reps = cfgs_loss.get("normalize_reps", True)
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)

    tokens_per_frame = int((crop_size // patch_size) ** 2)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    # -- TRAIN CONTROL / DEBUGGING
    cfgs_train_control = args.get("train_control", {})
    target_update = cfgs_train_control.get("target_update", "ema")
    if target_update not in ("ema", "fixed"):
        raise ValueError(f"Unsupported train_control.target_update={target_update}")
    ema = cfgs_train_control.get("ema", [0.996, 1.0])
    predictor_warmup_steps = int(cfgs_train_control.get("predictor_warmup_steps", 0))
    freeze_encoder_during_predictor_warmup = cfgs_train_control.get(
        "freeze_encoder_during_predictor_warmup", True
    )
    pause_ema_during_predictor_warmup = cfgs_train_control.get(
        "pause_ema_during_predictor_warmup", True
    )
    log_detailed_wandb = cfgs_train_control.get("log_detailed_wandb", True)
    detailed_log_freq = int(cfgs_train_control.get("detailed_log_freq", log_freq))
    log_grad_norms = cfgs_train_control.get("log_grad_norms", True)
    log_param_norms = cfgs_train_control.get("log_param_norms", True)
    log_latent_stats = cfgs_train_control.get("log_latent_stats", True)
    log_input_stats = cfgs_train_control.get("log_input_stats", True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    wandb_run = None
    if rank == 0 and wandb is not None and os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "vjepa2-lerobot"),
            name=os.environ.get("WANDB_RUN_NAME", os.path.basename(folder)),
            config=args,
            dir=folder,
            resume="allow",
        )

    tb_writer = SummaryWriter(log_dir=os.path.join(folder, "tb_logs")) if rank == 0 else None
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        state_embed_dim=state_embed_dim,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    transform = make_transforms(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    (unsupervised_loader, unsupervised_sampler) = init_data(
        data_root=data_root,
        datasets=datasets,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        fps=fps,
        tubelet_size=tubelet_size,
        transform=transform,
        collator=torch.utils.data.default_collate,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        action_dim=action_embed_dim,
        state_dim=state_embed_dim,
    )
    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        use_scaler=use_scaler,
        betas=betas,
        eps=eps,
    )
    encoder = DistributedDataParallel(
        encoder,
        static_graph=(predictor_warmup_steps <= 0),
        find_unused_parameters=(predictor_warmup_steps > 0),
    )
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    for p in target_encoder.parameters():
        p.requires_grad = False

    encoder, predictor, target_encoder = load_pretrained(
        r_path=p_file,
        encoder=encoder,
        predictor=predictor,
        context_encoder_key=context_encoder_key,
        target_encoder_key=target_encoder_key,
        target_encoder=target_encoder,
        load_predictor=load_predictor,
        load_encoder=load_encoder,
    )

    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        (encoder, predictor, target_encoder, optimizer, scaler, start_epoch) = load_checkpoint(
            r_path=resume_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    total_train_steps = int(num_epochs * ipe)

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    # Monotonically incrementing seed for DistributedSampler: any time we
    # rebuild the loader (epoch boundary OR mid-epoch StopIteration OR mid-skip
    # exhaustion) we bump this so shuffle order keeps changing.
    sampler_step = start_epoch
    unsupervised_sampler.set_epoch(sampler_step)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                sampler_step += 1
                unsupervised_sampler.set_epoch(sampler_step)
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        pbar = tqdm(range(ipe), desc=f"Epoch {epoch+1}/{num_epochs}", disable=(rank != 0), dynamic_ncols=True)
        for itr in pbar:
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    sampler_step += 1
                    unsupervised_sampler.set_epoch(sampler_step)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        raise e

            clips = sample[0].to(device, non_blocking=True)          # [B C T H W]
            actions = sample[1].to(device, dtype=torch.float, non_blocking=True)   # [B T action_dim]
            states = sample[2].to(device, dtype=torch.float, non_blocking=True)    # [B T state_dim]
            extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)  # [B T 6]
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                gc.collect()

            def train_step():
                global_step = epoch * ipe + itr
                step_num = global_step + 1
                in_predictor_warmup = (
                    freeze_encoder_during_predictor_warmup
                    and predictor_warmup_steps > 0
                    and global_step < predictor_warmup_steps
                )
                collect_detailed = (
                    rank == 0
                    and wandb_run is not None
                    and log_detailed_wandb
                    and detailed_log_freq > 0
                    and (global_step % detailed_log_freq == 0)
                )
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def featurize(net, c, with_grad):
                    # c: [B, C, T, H, W]. Run encoder once on the full clip (tubelet=2
                    # produces T/2 temporal tokens), then repeat_interleave the temporal
                    # axis so the output matches the predictor's per-frame layout
                    # [B, T * tokens_per_frame, D]. Frames within the same tubelet pair
                    # share identical features (same as feeding (frame, frame) tubelets).
                    grad_ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
                    with grad_ctx:
                        h = net(c)  # [B, (T/tubelet)*tokens_per_frame, D]
                        h = h.view(batch_size, max_num_frames // tubelet_size, -1, h.size(-1))
                        h = h.repeat_interleave(tubelet_size, dim=1)  # [B, T, tokens_per_frame, D]
                        h = h.flatten(1, 2)
                        if normalize_reps:
                            h = F.layer_norm(h, (h.size(-1),))
                        return h

                def forward_predictions(z):
                    def _step_predictor(_z, _a, _s, _e):
                        _z = predictor(_z, _a, _s, _e)
                        if normalize_reps:
                            _z = F.layer_norm(_z, (_z.size(-1),))
                        return _z

                    _z, _a, _s, _e = z[:, :-tokens_per_frame], actions[:, :-1], states[:, :-1], extrinsics[:, :-1]
                    z_tf = _step_predictor(_z, _a, _s, _e)

                    _z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
                    for n in range(1, auto_steps):
                        _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], extrinsics[:, : n + 1]
                        _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                        _z = torch.cat([_z, _z_nxt], dim=1)
                    z_ar = _z[:, tokens_per_frame:]

                    return z_tf, z_ar

                def loss_fn(z, h):
                    _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
                    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    # Target features provide the learning signal. With EMA enabled,
                    # target_encoder follows encoder after each optimizer step.
                    h = featurize(target_encoder, clips, with_grad=False)
                    # During predictor warmup, keep encoder fixed and train only
                    # predictor on the pretrained feature space.
                    z = featurize(encoder, clips, with_grad=not in_predictor_warmup)
                    z_tf, z_ar = forward_predictions(z)
                    jloss = loss_fn(z_tf, h)
                    sloss = loss_fn(z_ar, h)
                    loss = jloss + sloss

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                detailed_stats = {}
                if collect_detailed:
                    if log_grad_norms:
                        enc_grad_norm = _grad_norm(encoder)
                        pred_grad_norm = _grad_norm(predictor)
                        detailed_stats.update(
                            {
                                "grad/encoder_norm": enc_grad_norm,
                                "grad/predictor_norm": pred_grad_norm,
                                "grad/total_norm": (enc_grad_norm**2 + pred_grad_norm**2) ** 0.5,
                            }
                        )
                    if log_latent_stats:
                        detailed_stats.update(_tensor_stats("latent/h", h))
                        detailed_stats.update(_tensor_stats("latent/z", z))
                        detailed_stats.update(_tensor_stats("latent/z_tf", z_tf))
                        detailed_stats.update(_tensor_stats("latent/z_ar", z_ar))
                        z_det, h_det = _sample_flat_pair(z, h)
                        detailed_stats["latent/z_h_cosine"] = F.cosine_similarity(
                            z_det, h_det, dim=0
                        ).item()
                        detailed_stats["latent/z_target_l1"] = torch.mean(
                            torch.abs(z_det - h_det)
                        ).item()
                    if log_input_stats:
                        detailed_stats.update(_tensor_stats("input/clip", clips))
                        detailed_stats.update(_tensor_stats("input/action", actions))
                        detailed_stats.update(_tensor_stats("input/state", states))
                        detailed_stats.update(_tensor_stats("input/extrinsics", extrinsics))

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                ema_momentum = None
                ema_updated = False
                if target_update == "ema":
                    ema_momentum = _ema_momentum(step_num, total_train_steps, ema)
                    if not (pause_ema_during_predictor_warmup and in_predictor_warmup):
                        _update_target_encoder(encoder, target_encoder, ema_momentum)
                        ema_updated = True

                if collect_detailed:
                    detailed_stats.update(
                        {
                            "optim/encoder_lr": _optimizer_group_lr(optimizer, 0),
                            "optim/predictor_lr": _optimizer_group_lr(optimizer, 1),
                            "optim/ema_momentum": 0.0 if ema_momentum is None else ema_momentum,
                            "train/is_predictor_warmup": float(in_predictor_warmup),
                            "train/ema_updated": float(ema_updated),
                        }
                    )
                    if log_param_norms:
                        detailed_stats.update(
                            {
                                "param/encoder_norm": _param_norm(encoder),
                                "param/predictor_norm": _param_norm(predictor),
                                "param/target_encoder_norm": _param_norm(target_encoder),
                            }
                        )

                optimizer.zero_grad()

                return (
                    loss.detach().item(),
                    jloss.detach().item(),
                    sloss.detach().item(),
                    _new_lr,
                    _new_wd,
                    0.0 if ema_momentum is None else ema_momentum,
                    float(in_predictor_warmup),
                    detailed_stats,
                )

            (
                loss,
                jloss,
                sloss,
                _new_lr,
                _new_wd,
                _ema_momentum,
                _is_predictor_warmup,
                detailed_stats,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            def log_stats():
                global_step = epoch * ipe + itr
                if rank == 0:
                    pbar.set_postfix(
                        loss=f"{loss_meter.avg:.3f}",
                        jl=f"{jloss_meter.avg:.3f}",
                        sl=f"{sloss_meter.avg:.3f}",
                        lr=f"{_new_lr:.2e}",
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss", loss, global_step)
                        tb_writer.add_scalar("train/jloss", jloss, global_step)
                        tb_writer.add_scalar("train/sloss", sloss, global_step)
                        tb_writer.add_scalar("train/loss_avg", loss_meter.avg, global_step)
                        tb_writer.add_scalar("optim/lr", _new_lr, global_step)
                        tb_writer.add_scalar("optim/wd", _new_wd, global_step)
                        tb_writer.add_scalar("optim/ema_momentum", _ema_momentum, global_step)
                        tb_writer.add_scalar("train/is_predictor_warmup", _is_predictor_warmup, global_step)
                        tb_writer.add_scalar("perf/gpu_time_ms", gpu_etime_ms, global_step)
                        tb_writer.add_scalar("perf/mem_mb", torch.cuda.max_memory_allocated() / 1024.0**2, global_step)
                    if wandb_run is not None:
                        wandb_payload = {
                            "train/loss": loss,
                            "train/jloss": jloss,
                            "train/sloss": sloss,
                            "train/loss_avg": loss_meter.avg,
                            "train/is_predictor_warmup": _is_predictor_warmup,
                            "optim/lr": _new_lr,
                            "optim/wd": _new_wd,
                            "optim/ema_momentum": _ema_momentum,
                            "perf/gpu_time_ms": gpu_etime_ms,
                            "perf/mem_mb": torch.cuda.max_memory_allocated() / 1024.0**2,
                        }
                        wandb_payload.update(detailed_stats)
                        wandb_run.log(wandb_payload, step=global_step)

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        pbar.close()
        logger.info("avg. loss %.3f" % loss_meter.avg)
        if rank == 0 and tb_writer is not None:
            tb_writer.add_scalar("epoch/loss_avg", loss_meter.avg, epoch + 1)
            tb_writer.add_scalar("epoch/jloss_avg", jloss_meter.avg, epoch + 1)
            tb_writer.add_scalar("epoch/sloss_avg", sloss_meter.avg, epoch + 1)
        if rank == 0 and wandb_run is not None:
            wandb_run.log(
                {
                    "epoch/loss_avg": loss_meter.avg,
                    "epoch/jloss_avg": jloss_meter.avg,
                    "epoch/sloss_avg": sloss_meter.avg,
                },
                step=(epoch + 1) * ipe - 1,
            )
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_checkpoint(epoch + 1, os.path.join(folder, f"e{epoch}.pt"))

    if wandb_run is not None:
        wandb_run.finish()
