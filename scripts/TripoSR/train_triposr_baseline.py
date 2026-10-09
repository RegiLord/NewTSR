#!/usr/bin/env python3
"""
Baseline TripoSR training script built around the released VAST-AI-Research/TripoSR code.

The settings that come directly from the TripoSR technical report are:
  - conditioning resolution: 512x512 (from the released model config/report)
  - local supervision crop: 128x128
  - optimizer: AdamW
  - learning rate: 4e-4
  - scheduler: CosineAnnealingLR
  - warm-up: 2,000 optimizer steps
  - reconstruction loss:
        MSE + 2.0 * LPIPS + 0.05 * BCE(mask)
  - renderer: the released TripoSR renderer/config (128 samples/ray, radius 0.87,
    exp density activation, density bias -1.0)

Important: the public TripoSR repository does not include the original training code.
The report also does NOT specify the total number of optimizer steps, batch size,
number of supervision views V, exact foreground-crop probability, LPIPS backbone,
AdamW weight decay/betas, or warm-up shape. Those are explicit CLI choices here.

Expected dataset layout:

dataset_root/
  objects/
    <uid>/
      rgb/
        000.png
        ...
      depth/
        000_depth.exr
        ...
      camera/
        000.npy or 000.npz
        ...

Depth is used only to derive the GT foreground mask for normal TripoSR training.
Normal maps are deliberately unused in this baseline.
"""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import lpips
import numpy as np
import OpenEXR
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import ViTModel

from tsr.system import TSR
from tsr.utils import rays_intersect_bbox


REPORT_LR = 4e-4
REPORT_WARMUP_STEPS = 2000
REPORT_LPIPS_WEIGHT = 2.0
REPORT_MASK_WEIGHT = 0.05
REPORT_PATCH_SIZE = 128


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def index_files(directory, extension):
    if not directory.exists():
        return {}

    out = {}
    for path in directory.glob(f"*{extension}"):
        view_id = path.stem.split("_")[0]
        out[view_id] = path
    return out


def read_exr_channels(path):
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        return {name: channel.pixels.astype(np.float32) for name, channel in exr.channels().items()}


def find_exr_channel(channels, name):
    if name in channels:
        return channels[name]

    for key, value in channels.items():
        if key.endswith("." + name):
            return value

    return None


def load_depth_mask(path, clip_end):
    channels = read_exr_channels(path)
    depth = find_exr_channel(channels, "Z")

    if depth is None:
        depth = find_exr_channel(channels, "V")

    if depth is None:
        depth = find_exr_channel(channels, "R")

    if depth is None:
        candidates = [value for key, value in channels.items() if key != "A" and value.ndim == 2]
        if not candidates:
            raise RuntimeError(f"No usable depth channel in {path}")
        depth = candidates[0]

    valid = np.isfinite(depth) & (depth > 0.0) & (depth < clip_end)
    return torch.from_numpy(valid.astype(np.float32))[None]


def load_rgb(path):
    image = Image.open(path)

    if image.mode == "RGBA":
        rgba = np.asarray(image).astype(np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4])
        return torch.from_numpy(rgb).permute(2, 0, 1)

    rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


def default_intrinsics(height, width, lens_mm, sensor_width_mm):
    fx = lens_mm / sensor_width_mm * width
    fy = fx
    return torch.tensor([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def rt_to_c2w(rt):
    rt = np.asarray(rt, dtype=np.float32)

    if rt.shape == (4, 4):
        w2c = rt
    elif rt.shape == (3, 4):
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3] = rt
    else:
        raise ValueError(f"Expected RT shape 3x4 or 4x4, got {rt.shape}")

    return torch.from_numpy(np.linalg.inv(w2c).astype(np.float32))


def load_camera(path, height, width, lens_mm, sensor_width_mm):
    K = None
    c2w = None

    if path.suffix == ".npz":
        data = np.load(path)

        if "K" in data:
            K = torch.from_numpy(np.asarray(data["K"], dtype=np.float32))

        if "c2w" in data:
            c2w = torch.from_numpy(np.asarray(data["c2w"], dtype=np.float32))
        elif "RT" in data:
            c2w = rt_to_c2w(data["RT"])
        elif "rt" in data:
            c2w = rt_to_c2w(data["rt"])
        else:
            raise RuntimeError(f"{path} has no c2w/RT camera matrix.")
    else:
        c2w = rt_to_c2w(np.load(path))

    if K is None:
        K = default_intrinsics(height, width, lens_mm, sensor_width_mm)

    if c2w.shape == (3, 4):
        tmp = torch.eye(4, dtype=torch.float32)
        tmp[:3] = c2w
        c2w = tmp

    if K.shape != (3, 3) or c2w.shape != (4, 4):
        raise RuntimeError(f"Invalid camera data in {path}: K={tuple(K.shape)}, c2w={tuple(c2w.shape)}")

    return K, c2w


class ObjaverseRenderDataset(Dataset):
    def __init__(self, object_dirs, supervision_views, clip_end, lens_mm, sensor_width_mm):
        self.objects = []
        self.supervision_views = supervision_views
        self.clip_end = clip_end
        self.lens_mm = lens_mm
        self.sensor_width_mm = sensor_width_mm

        for object_dir in object_dirs:
            rgb = index_files(object_dir / "rgb", ".png")
            depth = index_files(object_dir / "depth", ".exr")

            cameras = {}
            cameras.update(index_files(object_dir / "camera", ".npy"))
            cameras.update(index_files(object_dir / "camera", ".npz"))

            if not cameras:
                cameras.update(index_files(object_dir, ".npy"))
                cameras.update(index_files(object_dir, ".npz"))

            common = sorted(set(rgb) & set(depth) & set(cameras))

            if len(common) >= max(2, supervision_views):
                self.objects.append({"uid": object_dir.name, "rgb": rgb, "depth": depth, "camera": cameras, "views": common})

        if not self.objects:
            raise RuntimeError("No complete training objects were found.")

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        item = self.objects[index]
        input_view = random.choice(item["views"])

        if self.supervision_views <= len(item["views"]):
            target_views = random.sample(item["views"], self.supervision_views)
        else:
            target_views = random.choices(item["views"], k=self.supervision_views)

        input_rgb = load_rgb(item["rgb"][input_view])
        target_rgbs = []
        target_masks = []
        target_K = []
        target_c2w = []

        for view_id in target_views:
            rgb = load_rgb(item["rgb"][view_id])
            height, width = rgb.shape[-2:]

            mask = load_depth_mask(item["depth"][view_id], self.clip_end)
            K, c2w = load_camera(item["camera"][view_id], height, width, self.lens_mm, self.sensor_width_mm)

            target_rgbs.append(rgb)
            target_masks.append(mask)
            target_K.append(K)
            target_c2w.append(c2w)

        return {
            "uid": item["uid"],
            "input_rgb": input_rgb,
            "target_rgb": torch.stack(target_rgbs),
            "target_mask": torch.stack(target_masks),
            "K": torch.stack(target_K),
            "c2w": torch.stack(target_c2w),
        }


def choose_crop(mask, patch_size, foreground_probability):
    height, width = mask.shape[-2:]

    if height < patch_size or width < patch_size:
        raise ValueError(f"Image {height}x{width} is smaller than the {patch_size}x{patch_size} supervision crop.")

    if random.random() < foreground_probability:
        ys, xs = torch.where(mask[0] > 0.5)

        if len(ys) > 0:
            i = random.randrange(len(ys))
            cy, cx = int(ys[i]), int(xs[i])

            y_min = max(0, cy - patch_size + 1)
            y_max = min(cy, height - patch_size)
            x_min = max(0, cx - patch_size + 1)
            x_max = min(cx, width - patch_size)

            y0 = random.randint(y_min, y_max)
            x0 = random.randint(x_min, x_max)
            return y0, x0

    return random.randint(0, height - patch_size), random.randint(0, width - patch_size)


def make_patch_rays(K, c2w, y0, x0, patch_size, device):
    K = K.to(device)
    c2w = c2w.to(device)

    ys, xs = torch.meshgrid(
        torch.arange(y0, y0 + patch_size, device=device, dtype=torch.float32) + 0.5,
        torch.arange(x0, x0 + patch_size, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    dirs_cam = torch.stack([(xs - cx) / fx, -(ys - cy) / fy, -torch.ones_like(xs)], dim=-1)
    dirs_cam = F.normalize(dirs_cam, dim=-1)

    rotation = c2w[:3, :3]
    dirs_world = torch.einsum("ij,hwj->hwi", rotation, dirs_cam)
    dirs_world = F.normalize(dirs_world, dim=-1)

    rays_o = c2w[:3, 3].view(1, 1, 3).expand_as(dirs_world)
    return rays_o, dirs_world


def render_rays_with_opacity(renderer, decoder, triplane, rays_o, rays_d, ray_chunk_size=64):
    """
    Differentiable TripoSR volume rendering with accumulated opacity.

    The important memory detail is that we chunk at the RAY level, before
    constructing the N_rays x 128 sample positions. Each ray chunk is
    checkpointed independently. This prevents backward from recomputing the
    entire 128x128x128 sample volume at once.
    """
    shape = rays_o.shape[:-1]
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)

    t_near, t_far, valid = rays_intersect_bbox(rays_o_flat, rays_d_flat, renderer.cfg.radius)
    t_near = t_near.reshape(-1)
    t_far = t_far.reshape(-1)
    valid = valid.reshape(-1)
    valid_indices = torch.where(valid)[0]

    # White background and zero opacity for rays that miss the bounding box.
    output_rgb = torch.ones((rays_o_flat.shape[0], 3), device=triplane.device, dtype=torch.float32)
    output_opacity = torch.zeros((rays_o_flat.shape[0],), device=triplane.device, dtype=torch.float32)

    if valid_indices.numel() == 0:
        return output_rgb.view(*shape, 3), output_opacity.view(*shape)

    n_samples = renderer.cfg.num_samples_per_ray
    edges = torch.linspace(0.0, 1.0, n_samples + 1, device=triplane.device, dtype=torch.float32)
    mids = 0.5 * (edges[:-1] + edges[1:])
    delta = edges[1:] - edges[:-1]

    def render_valid_chunk(triplane_tensor, o_chunk, d_chunk, near_chunk, far_chunk):
        # Keep ray sampling in float32 for numerical stability. query_triplane
        # will run under the surrounding autocast context where appropriate.
        z_vals = near_chunk[:, None] * (1.0 - mids[None]) + far_chunk[:, None] * mids[None]
        xyz = o_chunk[:, None, :] + z_vals[..., None] * d_chunk[:, None, :]

        field = renderer.query_triplane(decoder, xyz, triplane_tensor)
        density_act = field["density_act"]
        color = field["color"]

        alpha = 1.0 - torch.exp(-delta[None] * density_act[..., 0].float())
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1] + 1e-10], dim=-1),
            dim=-1,
        )
        weights = alpha * transmittance

        opacity = weights.sum(dim=-1)
        rgb = (weights[..., None] * color.float()).sum(dim=-2) + (1.0 - opacity[..., None])
        return rgb, opacity

    rgb_chunks = []
    opacity_chunks = []
    index_chunks = []

    for start_idx in range(0, valid_indices.numel(), ray_chunk_size):
        idx = valid_indices[start_idx:start_idx + ray_chunk_size]
        o_chunk = rays_o_flat[idx]
        d_chunk = rays_d_flat[idx]
        near_chunk = t_near[idx]
        far_chunk = t_far[idx]

        if torch.is_grad_enabled():
            rgb_chunk, opacity_chunk = torch_checkpoint(
                render_valid_chunk,
                triplane,
                o_chunk,
                d_chunk,
                near_chunk,
                far_chunk,
                use_reentrant=False,
            )
        else:
            rgb_chunk, opacity_chunk = render_valid_chunk(triplane, o_chunk, d_chunk, near_chunk, far_chunk)

        rgb_chunks.append(rgb_chunk)
        opacity_chunks.append(opacity_chunk)
        index_chunks.append(idx)

    valid_rgb = torch.cat(rgb_chunks, dim=0)
    valid_opacity = torch.cat(opacity_chunks, dim=0)
    merged_indices = torch.cat(index_chunks, dim=0)

    # Allocate using compositing dtype, matching the released renderer behavior.
    output_rgb = output_rgb.to(dtype=valid_rgb.dtype)
    output_opacity = output_opacity.to(dtype=valid_opacity.dtype)

    output_rgb = output_rgb.index_copy(0, merged_indices, valid_rgb)
    output_opacity = output_opacity.index_copy(0, merged_indices, valid_opacity)

    return output_rgb.view(*shape, 3), output_opacity.view(*shape)


def build_model(config_path, init_checkpoint, device):
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    if init_checkpoint:
        model = TSR.from_pretrained(init_checkpoint, config_name="config.yaml", weight_name="model.ckpt")
        print(f"Initialized all TripoSR weights from: {init_checkpoint}")
    else:
        model = TSR(cfg)

        # The released tokenizer class constructs the DINO architecture from its
        # config; the report states that the image encoder is initialized from
        # pretrained DINOv1 weights, so explicitly load them here.
        dino_name = cfg.image_tokenizer.pretrained_model_name_or_path
        print(f"Loading pretrained DINOv1 initialization: {dino_name}")
        pretrained_dino = ViTModel.from_pretrained(dino_name)
        model.image_tokenizer.model.load_state_dict(pretrained_dino.state_dict())
        del pretrained_dino

    return model.to(device), cfg


def split_objects(dataset_root, validation_fraction, seed):
    object_dirs = sorted((dataset_root / "objects").glob("*"))
    object_dirs = [path for path in object_dirs if path.is_dir()]

    if len(object_dirs) < 2:
        raise RuntimeError(f"Need at least two object directories under {dataset_root / 'objects'}")

    rng = random.Random(seed)
    rng.shuffle(object_dirs)

    n_val = max(1, int(round(len(object_dirs) * validation_fraction)))
    n_val = min(n_val, len(object_dirs) - 1)

    return object_dirs[n_val:], object_dirs[:n_val]


def save_checkpoint(output_dir, model, optimizer, scheduler, scaler, step, config_path, args):
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.ckpt")
    shutil.copy2(config_path, output_dir / "config.yaml")

    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
        },
        output_dir / "train_state.pt",
    )


def load_train_state(path, model, optimizer, scheduler, scaler):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    return int(state["step"])


def get_autocast(args):
    if not args.amp or not torch.cuda.is_available():
        return torch.amp.autocast("cuda", enabled=False)

    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def compute_batch_loss(model, perceptual, batch, device, args):
    # Keep conditioning images on CPU. TSR.forward() preprocesses them and
    # moves the resulting tensor to `device` itself.
    #
    # Pass a LIST of individual HWC tensors. In the released TripoSR
    # ImagePreprocessor, a 4-D CUDA tensor misses the torch.FloatTensor fast
    # path and is wrapped as one list item, which adds an extra batch dimension.
    input_rgb = batch["input_rgb"]
    target_rgb = batch["target_rgb"]
    target_mask = batch["target_mask"]
    K = batch["K"]
    c2w = batch["c2w"]

    input_images = [image.permute(1, 2, 0).contiguous() for image in input_rgb]
    scene_codes = model(input_images, device=str(device))

    total_mse = scene_codes.new_zeros(())
    total_lpips = scene_codes.new_zeros(())
    total_mask = scene_codes.new_zeros(())
    view_count = 0

    batch_size, n_views = target_rgb.shape[:2]

    for b in range(batch_size):
        for v in range(n_views):
            gt_rgb_full = target_rgb[b, v]
            gt_mask_full = target_mask[b, v]

            # A random crop may lie completely outside TripoSR's bounding sphere.
            # In that case every rendered pixel is the fixed white background and
            # the loss has no dependency on model parameters. Retry instead of
            # attempting backward on a zero-gradient sample.
            pred_rgb_hwc = None
            pred_opacity_hw = None
            gt_rgb = None
            gt_mask = None

            for _crop_attempt in range(args.crop_retries):
                y0, x0 = choose_crop(gt_mask_full, REPORT_PATCH_SIZE, args.foreground_crop_probability)

                candidate_gt_rgb = gt_rgb_full[:, y0:y0 + REPORT_PATCH_SIZE, x0:x0 + REPORT_PATCH_SIZE].to(device, non_blocking=True)
                candidate_gt_mask = gt_mask_full[:, y0:y0 + REPORT_PATCH_SIZE, x0:x0 + REPORT_PATCH_SIZE].to(device, non_blocking=True)

                rays_o, rays_d = make_patch_rays(K[b, v], c2w[b, v], y0, x0, REPORT_PATCH_SIZE, device)

                # Check whether this crop contains at least one ray that can
                # influence the TripoSR radiance field before doing the costly render.
                flat_o = rays_o.reshape(-1, 3)
                flat_d = rays_d.reshape(-1, 3)
                _, _, valid_rays = rays_intersect_bbox(flat_o, flat_d, model.renderer.cfg.radius)

                if not bool(valid_rays.any()):
                    continue

                candidate_rgb_hwc, candidate_opacity_hw = render_rays_with_opacity(
                    model.renderer, model.decoder, scene_codes[b], rays_o, rays_d, args.ray_chunk_size
                )

                # This should normally be true whenever there are valid rays,
                # but retain the check as a guard against any future renderer change.
                if not candidate_rgb_hwc.requires_grad and torch.is_grad_enabled():
                    continue

                pred_rgb_hwc = candidate_rgb_hwc
                pred_opacity_hw = candidate_opacity_hw
                gt_rgb = candidate_gt_rgb
                gt_mask = candidate_gt_mask
                break

            if pred_rgb_hwc is None:
                # No useful crop for this supervision view. With multiple views,
                # simply ignore this one; if all views fail, the caller handles it.
                continue

            pred_rgb = pred_rgb_hwc.permute(2, 0, 1)
            pred_mask = pred_opacity_hw.unsqueeze(0)

            mse = F.mse_loss(pred_rgb, gt_rgb)

            # `pred_mask` is already accumulated opacity in [0, 1], not logits.
            # BCE on probabilities is unsafe under CUDA autocast, so evaluate
            # this term explicitly in float32 with autocast disabled.
            with torch.amp.autocast("cuda", enabled=False):
                mask_bce = F.binary_cross_entropy(
                    pred_mask.float().clamp(1e-6, 1.0 - 1e-6),
                    gt_mask.float(),
                )

            pred_lpips = pred_rgb.unsqueeze(0) * 2.0 - 1.0
            gt_lpips = gt_rgb.unsqueeze(0) * 2.0 - 1.0
            perceptual_loss = perceptual(pred_lpips, gt_lpips).mean()

            total_mse = total_mse + mse
            total_lpips = total_lpips + perceptual_loss
            total_mask = total_mask + mask_bce
            view_count += 1

    if view_count == 0:
        return None, None

    total_mse = total_mse / view_count
    total_lpips = total_lpips / view_count
    total_mask = total_mask / view_count

    total = total_mse + REPORT_LPIPS_WEIGHT * total_lpips + REPORT_MASK_WEIGHT * total_mask
    return total, {"mse": total_mse.detach(), "lpips": total_lpips.detach(), "mask": total_mask.detach()}


@torch.no_grad()
def evaluate(model, perceptual, loader, device, args, max_batches):
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "lpips": 0.0, "mask": 0.0}
    count = 0

    for batch in loader:
        if count >= max_batches:
            break

        with get_autocast(args):
            loss, parts = compute_batch_loss(model, perceptual, batch, device, args)

        if loss is None:
            continue

        totals["loss"] += float(loss)
        totals["mse"] += float(parts["mse"])
        totals["lpips"] += float(parts["lpips"])
        totals["mask"] += float(parts["mask"])
        count += 1

    model.train()

    if count == 0:
        return None

    return {key: value / count for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/objaverse_pilot"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs/triposr_baseline"))
    parser.add_argument("--steps", type=int, required=True, help="Total optimizer steps. The TripoSR report does not publish this value.")
    parser.add_argument("--batch-size", type=int, default=1, help="Not specified by the report.")
    #supervision views-> how many crops are counted in the loss
    parser.add_argument("--supervision-views", type=int, default=1, help="V in the report loss. Exact V is not published.")
    parser.add_argument("--foreground-crop-probability", type=float, default=0.8, help="Report says foreground-biased crops but does not publish the probability.")
    parser.add_argument("--crop-retries", type=int, default=8, help="Retry crop sampling when every ray misses TripoSR's bounding sphere.")

    #validation set
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--clip-end", type=float, default=100.0)
    parser.add_argument("--lens-mm", type=float, default=35.0)
    parser.add_argument("--sensor-width-mm", type=float, default=32.0)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex", help="LPIPS backbone is not specified by the report.")
    parser.add_argument("--init-checkpoint", default=None, help="Optional HF repo/local TripoSR checkpoint. Omit to train the baseline from scratch with pretrained DINO initialization.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--renderer-chunk-size", type=int, default=512, help="Point-query chunk size inside TripoSR query_triplane.")
    parser.add_argument("--ray-chunk-size", type=int, default=64, help="Number of camera rays rendered/checkpointed together. Lower this to reduce backward VRAM.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu-offload-saved-tensors", action="store_true", help="Move tensors saved for backward to CPU. Much slower, but can make full TripoSR fine-tuning fit on ~12 GB GPUs.")
    parser.add_argument("--freeze-dino", action="store_true", help="Not specified by the report; default is to train DINO end-to-end.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    if args.steps <= REPORT_WARMUP_STEPS:
        raise ValueError(f"--steps must be greater than the report's {REPORT_WARMUP_STEPS} warm-up steps.")

    if not 0.0 <= args.foreground_crop_probability <= 1.0:
        raise ValueError("--foreground-crop-probability must be in [0, 1].")

    if args.ray_chunk_size <= 0:
        raise ValueError("--ray-chunk-size must be greater than 0.")

    if args.crop_retries <= 0:
        raise ValueError("--crop-retries must be greater than 0.")

    if not args.config.exists():
        raise FileNotFoundError(
            f"{args.config} does not exist. Place the official stabilityai/TripoSR config.yaml in the repository root "
            "or pass --config /path/to/config.yaml."
        )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dirs, val_dirs = split_objects(args.dataset_root, args.validation_fraction, args.seed)

    train_dataset = ObjaverseRenderDataset(train_dirs, args.supervision_views, args.clip_end, args.lens_mm, args.sensor_width_mm)
    val_dataset = ObjaverseRenderDataset(val_dirs, args.supervision_views, args.clip_end, args.lens_mm, args.sensor_width_mm)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=max(0, args.num_workers // 2), pin_memory=True)

    model, cfg = build_model(args.config, args.init_checkpoint, device)
    model.renderer.set_chunk_size(args.renderer_chunk_size)

    if args.gradient_checkpointing:
        # TripoSR's custom Transformer1D implements checkpointing with its own
        # boolean flag. Hugging Face ViT must be enabled through the public
        # PreTrainedModel API so that `_gradient_checkpointing_func` is also
        # installed on ViTEncoder.
        model.backbone.gradient_checkpointing = True
        model.image_tokenizer.model.gradient_checkpointing_enable()

    if args.freeze_dino:
        for parameter in model.image_tokenizer.parameters():
            parameter.requires_grad = False

    perceptual = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # foreach=False avoids the CUDA foreach optimizer implementation's extra
    # peak memory of roughly one additional parameter-sized tensor list.
    optimizer = torch.optim.AdamW(trainable, lr=REPORT_LR, foreach=False)

    # The report publishes 2,000 warm-up steps but not the warm-up shape.
    # Linear warm-up is used here, followed by the explicitly reported CosineAnnealingLR.
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / REPORT_WARMUP_STEPS, end_factor=1.0, total_iters=REPORT_WARMUP_STEPS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps - REPORT_WARMUP_STEPS)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[REPORT_WARMUP_STEPS])

    scaler = None
    if args.amp and args.amp_dtype == "fp16" and torch.cuda.is_available():
        scaler = torch.amp.GradScaler("cuda")

    start_step = 0
    if args.resume is not None:
        start_step = load_train_state(args.resume, model, optimizer, scheduler, scaler)
        print(f"Resumed from optimizer step {start_step}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / "run_config.json", "w") as f:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, f, indent=2)

    print(f"Device: {device}")
    print(f"Train objects: {len(train_dataset)}")
    print(f"Validation objects: {len(val_dataset)}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    print(f"CPU saved-tensor offload: {args.cpu_offload_saved_tensors}")
    print("AdamW foreach: False (lower peak optimizer memory)")
    print(f"Report loss: MSE + {REPORT_LPIPS_WEIGHT}*LPIPS + {REPORT_MASK_WEIGHT}*BCE(mask)")
    print(f"Report optimizer: AdamW, lr={REPORT_LR}")
    print(f"Report crop: {REPORT_PATCH_SIZE}x{REPORT_PATCH_SIZE} from 512x512 supervision images")
    print(f"Report warm-up: {REPORT_WARMUP_STEPS} steps; scheduler after warm-up: CosineAnnealingLR")

    model.train()
    train_iter = iter(train_loader)

    for step in range(start_step, args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad(set_to_none=True)

        if args.cpu_offload_saved_tensors:
            # PyTorch moves intermediates saved for backward to host RAM and
            # copies them back when backward needs them. This preserves the
            # numerical training objective but trades substantial speed/PCIe
            # traffic for GPU memory.
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                with get_autocast(args):
                    loss, parts = compute_batch_loss(model, perceptual, batch, device, args)
        else:
            with get_autocast(args):
                loss, parts = compute_batch_loss(model, perceptual, batch, device, args)

        if loss is None:
            print(f"step={step + 1:07d} skipped: no renderable crop found")
            continue

        optimizer_stepped = True

        if scaler is not None:
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()

            # If the scale decreased, GradScaler found inf/NaN gradients and
            # skipped optimizer.step(). Do not advance the LR schedule in that case.
            optimizer_stepped = scale_after >= scale_before
        else:
            loss.backward()
            optimizer.step()

        if optimizer_stepped:
            scheduler.step()

        current_step = step + 1

        if current_step % 10 == 0 or current_step == 1:
            lr = optimizer.param_groups[0]["lr"]
            memory_text = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                memory_text = f" vram={allocated:.2f}G/{reserved:.2f}G"

            print(
                f"step={current_step:07d} "
                f"loss={float(loss):.6f} "
                f"mse={float(parts['mse']):.6f} "
                f"lpips={float(parts['lpips']):.6f} "
                f"mask={float(parts['mask']):.6f} "
                f"lr={lr:.8g}"
                f"{memory_text}"
            )

        if args.eval_every > 0 and current_step % args.eval_every == 0:
            val = evaluate(model, perceptual, val_loader, device, args, args.eval_batches)
            if val is not None:
                print(
                    f"VAL step={current_step:07d} "
                    f"loss={val['loss']:.6f} "
                    f"mse={val['mse']:.6f} "
                    f"lpips={val['lpips']:.6f} "
                    f"mask={val['mask']:.6f}"
                )

        if args.save_every > 0 and current_step % args.save_every == 0:
            checkpoint_dir = args.output_dir / f"step_{current_step:07d}"
            save_checkpoint(checkpoint_dir, model, optimizer, scheduler, scaler, current_step, args.config, args)
            print(f"Saved checkpoint: {checkpoint_dir}")

    final_dir = args.output_dir / "final"
    save_checkpoint(final_dir, model, optimizer, scheduler, scaler, args.steps, args.config, args)
    print(f"Training complete. Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
