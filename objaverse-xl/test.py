import argparse
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image
import numpy as np

import objaverse


REPO_ROOT = Path(__file__).resolve().parent
BLENDER_SCRIPT = REPO_ROOT / "scripts" / "rendering" / "blender_script.py"
OUTPUT_ROOT = REPO_ROOT / "test_renders"



def render_is_viable(render_path, min_coverage=0.03):
    image = Image.open(render_path).convert("RGBA")
    arr = np.asarray(image)

    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3]

    print(f"Alpha range: {alpha.min()} - {alpha.max()}")

    # If the image actually has transparency, use it.
    if alpha.min() < 250:
        mask = alpha > 10
        print("Using alpha channel for foreground detection.")

    # Otherwise estimate background from image corners.
    else:
        print("Alpha is fully opaque; estimating background from corners.")

        h, w = rgb.shape[:2]
        patch = max(4, min(h, w) // 20)

        corners = np.concatenate([
            rgb[:patch, :patch].reshape(-1, 3),
            rgb[:patch, -patch:].reshape(-1, 3),
            rgb[-patch:, :patch].reshape(-1, 3),
            rgb[-patch:, -patch:].reshape(-1, 3),
        ])

        background = np.median(corners, axis=0)

        difference = np.linalg.norm(rgb - background, axis=2)

        # Pixels sufficiently different from the estimated background.
        mask = difference > 15

    coverage = mask.mean()

    print(f"Estimated foreground coverage: {coverage * 100:.2f}%")

    if coverage < min_coverage:
        print("Rejected: foreground too small.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", default="blender", help="Path to Blender executable")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=20)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not BLENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find Blender rendering script: {BLENDER_SCRIPT}")

    blender = shutil.which(args.blender)
    if blender is None:
        raise RuntimeError(f"Could not find Blender executable: {args.blender}")

    print("Loading Objaverse UIDs...")
    uids = objaverse.load_uids()

    for attempt in range(1, args.max_attempts + 1):
        uid = random.choice(uids)

        print()
        print(f"Attempt {attempt}/{args.max_attempts}")
        print(f"Selected object: {uid}")

        try:
            objects = objaverse.load_objects(uids=[uid], download_processes=1)
            object_path = Path(objects[uid]).resolve()

            print(f"Object path: {object_path}")

            output_dir = OUTPUT_ROOT / uid
            output_dir.mkdir(parents=True, exist_ok=True)

            command = [
                blender,
                "--background",
                "--factory-startup",
                "--python",
                str(BLENDER_SCRIPT),
                "--",
                "--object_path",
                str(object_path),
                "--output_dir",
                str(output_dir),
                "--engine",
                "BLENDER_EEVEE",
                "--num_renders",
                "1",
                "--only_northern_hemisphere",
            ]

            subprocess.run(command, check=True)

            render_path = output_dir / "rgb" / "000.png"

            if not render_path.exists():
                print("Rejected: no render produced")
                shutil.rmtree(output_dir, ignore_errors=True)
                continue

            if not render_is_viable(render_path):
                print("Rejected: object is too small, invisible, or fills the frame")
                shutil.rmtree(output_dir, ignore_errors=True)
                continue

            print()
            print("Viable object found.")
            print(f"UID: {uid}")
            print(f"Render: {render_path}")
            return

        except Exception as exc:
            print(f"Rejected because rendering failed: {exc}")

    raise RuntimeError(f"Could not find a viable object after {args.max_attempts} attempts")


if __name__ == "__main__":
    main()