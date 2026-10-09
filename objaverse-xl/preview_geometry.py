import argparse
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image


def read_exr_channels(path):
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        channels = {name: channel.pixels.astype(np.float32) for name, channel in exr.channels().items()}

    print(f"{path.name} channels: {list(channels.keys())}")
    return channels


def find_channel(channels, name):
    if name in channels:
        return channels[name]

    for key, value in channels.items():
        if key.endswith("." + name):
            return value

    return None


def preview_depth(input_path, output_path, clip_end=100.0):
    channels = read_exr_channels(input_path)

    depth = find_channel(channels, "Z")

    if depth is None:
        depth = find_channel(channels, "V")

    if depth is None:
        depth = find_channel(channels, "R")

    if depth is None:
        candidates = [v for k, v in channels.items() if k != "A" and v.ndim == 2]

        if not candidates:
            raise RuntimeError("Could not find a usable depth channel.")

        depth = candidates[0]

    valid = np.isfinite(depth) & (depth > 0.0) & (depth < clip_end)

    if not valid.any():
        raise RuntimeError(
            f"No valid depth values found. "
            f"Raw range: {np.nanmin(depth)} -> {np.nanmax(depth)}"
        )

    print(f"Raw depth range: {np.nanmin(depth)} -> {np.nanmax(depth)}")
    print(f"Valid depth range: {depth[valid].min()} -> {depth[valid].max()}")

    near = np.percentile(depth[valid], 1)
    far = np.percentile(depth[valid], 99)

    preview = np.zeros_like(depth, dtype=np.float32)

    if far > near:
        preview[valid] = 1.0 - (depth[valid] - near) / (far - near)

    preview = np.clip(preview, 0.0, 1.0)
    preview = (preview * 255.0).astype(np.uint8)

    Image.fromarray(preview, mode="L").save(output_path)

    print(f"Saved depth preview: {output_path}")


def preview_normal(input_path, output_path):
    channels = read_exr_channels(input_path)

    x = find_channel(channels, "X")
    y = find_channel(channels, "Y")
    z = find_channel(channels, "Z")

    if x is None or y is None or z is None:
        raise RuntimeError(
            f"Could not find X/Y/Z normal channels. "
            f"Available channels: {list(channels.keys())}"
        )

    normal = np.stack([x, y, z], axis=-1)

    print(f"Normal X range: {x.min()} -> {x.max()}")
    print(f"Normal Y range: {y.min()} -> {y.max()}")
    print(f"Normal Z range: {z.min()} -> {z.max()}")

    # Convert normal components from [-1, 1] to [0, 1].
    preview = (normal + 1.0) * 0.5
    preview = np.clip(preview, 0.0, 1.0)

    preview = (preview * 255.0).astype(np.uint8)

    Image.fromarray(preview, mode="RGB").save(output_path)

    print(f"Saved normal preview: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("previews"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    preview_depth(args.depth, args.output_dir / "depth_preview.png")
    preview_normal(args.normal, args.output_dir / "normal_preview.png")


if __name__ == "__main__":
    main()