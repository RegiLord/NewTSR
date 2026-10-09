import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import objaverse


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RENDER_SCRIPT = REPO_ROOT / "scripts" / "rendering" / "blender_script.py"
DEFAULT_OUTPUT = Path("/home/leusteanstefan/Documents/Practica/datasets/objaverse_pilot")


def count_files(directory, patterns):
    if not directory.exists():
        return 0

    found = set()

    for pattern in patterns:
        found.update(directory.glob(pattern))

    return len(found)


def count_camera_files(object_dir):
    camera_dir = object_dir / "camera"

    count = count_files(camera_dir, ["*.npy", "*.npz"])

    # Also tolerate the older layout where camera matrices were written
    # directly into the object's directory.
    if count == 0:
        count = count_files(object_dir, ["*.npy", "*.npz"])

    return count


def get_render_counts(object_dir):
    return {
        "rgb": count_files(object_dir / "rgb", ["*.png"]),
        "depth": count_files(object_dir / "gt" / "depth", ["*.exr"]),
        "normal": count_files(object_dir / "gt" / "normal", ["*.exr"]),
        "camera": count_camera_files(object_dir),
    }


def is_complete(object_dir, views):
    counts = get_render_counts(object_dir)

    return (
        counts["rgb"] >= views
        and counts["depth"] >= views
        and counts["normal"] >= views
        and counts["camera"] >= views
    )


def load_uid_file(path):
    if not path.exists():
        return []

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def append_line(path, value):
    with open(path, "a") as f:
        f.write(value + "\n")


def log_failure(path, uid, reason, counts=None):
    record = {
        "uid": uid,
        "reason": reason,
    }

    if counts is not None:
        record["counts"] = counts

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def render_object(blender, render_script, object_path, output_dir, views, engine):
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(render_script),
        "--",
        "--object_path",
        str(object_path),
        "--output_dir",
        str(output_dir),
        "--engine",
        engine,
        "--num_renders",
        str(views),
    ]

    return subprocess.run(command, check=False).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=700)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--engine", default="BLENDER_EEVEE")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render-script", type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--max-attempts", type=int, default=2000)
    args = parser.parse_args()

    blender = shutil.which(args.blender)

    if blender is None:
        raise RuntimeError(f"Could not find Blender executable: {args.blender}")

    if not args.render_script.exists():
        raise FileNotFoundError(f"Could not find renderer: {args.render_script}")

    args.output.mkdir(parents=True, exist_ok=True)

    objects_dir = args.output / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    success_file = args.output / "successful_uids.txt"
    attempted_file = args.output / "attempted_uids.txt"
    failure_file = args.output / "failures.jsonl"
    config_file = args.output / "dataset_config.json"

    config = {
        "target_objects": args.target,
        "views_per_object": args.views,
        "seed": args.seed,
        "engine": args.engine,
        "render_script": str(args.render_script.resolve()),
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    successful = set(load_uid_file(success_file))
    attempted = set(load_uid_file(attempted_file))

    # Recover successfully rendered objects if the script previously stopped
    # after rendering but before recording the UID.
    for object_dir in objects_dir.iterdir():
        if object_dir.is_dir() and is_complete(object_dir, args.views):
            uid = object_dir.name

            if uid not in successful:
                append_line(success_file, uid)
                successful.add(uid)

    print(f"Existing complete objects: {len(successful)}/{args.target}")

    if len(successful) >= args.target:
        print("Dataset is already complete.")
        return

    print("Loading Objaverse UID list...")
    uids = list(objaverse.load_uids())

    rng = random.Random(args.seed)
    rng.shuffle(uids)

    attempts_this_run = 0

    for uid in uids:
        if len(successful) >= args.target:
            break

        if attempts_this_run >= args.max_attempts:
            break

        if uid in successful or uid in attempted:
            continue

        attempts_this_run += 1
        attempted.add(uid)
        append_line(attempted_file, uid)

        print()
        print("=" * 70)
        print(f"Dataset progress: {len(successful)}/{args.target}")
        print(f"Attempt: {attempts_this_run}/{args.max_attempts}")
        print(f"UID: {uid}")
        print("=" * 70)

        try:
            objects = objaverse.load_objects(uids=[uid], download_processes=1)

            if uid not in objects:
                print("Download failed.")
                log_failure(failure_file, uid, "download_failed")
                continue

            object_path = Path(objects[uid]).resolve()
            object_dir = objects_dir / uid

            # Remove leftovers from a previous incomplete attempt.
            if object_dir.exists():
                shutil.rmtree(object_dir)

            object_dir.mkdir(parents=True, exist_ok=True)

            print(f"Object: {object_path}")
            print(f"Rendering {args.views} views...")

            # gt_dir = object_dir / "gt"
            # if gt_dir.exists():
            #     shutil.rmtree(gt_dir)       
            # gt_dir.mkdir(parents=True, exist_ok=True)

            return_code = render_object(blender, args.render_script, object_path, object_dir, args.views, args.engine)

            if return_code != 0:
                print(f"Blender failed with return code {return_code}.")
                log_failure(failure_file, uid, f"blender_return_code_{return_code}")
                shutil.rmtree(object_dir, ignore_errors=True)
                continue

            counts = get_render_counts(object_dir)

            print(
                f"Produced: RGB={counts['rgb']}, "
                f"Depth={counts['depth']}, "
                f"Normal={counts['normal']}, "
                f"Camera={counts['camera']}"
            )

            if not is_complete(object_dir, args.views):
                print("Rejected: incomplete render set.")
                log_failure(failure_file, uid, "incomplete_render_set", counts)
                shutil.rmtree(object_dir, ignore_errors=True)
                continue

            successful.add(uid)
            append_line(success_file, uid)

            print(f"Accepted. Dataset progress: {len(successful)}/{args.target}")

        except Exception as exc:
            print(f"Failed: {exc}")
            log_failure(failure_file, uid, f"{type(exc).__name__}: {exc}")

    print()
    print("=" * 70)
    print(f"Finished with {len(successful)}/{args.target} complete objects.")
    print(f"Expected views: {len(successful) * args.views}")
    print(f"Dataset: {args.output}")
    print("=" * 70)

    if len(successful) < args.target:
        print(
            "Target was not reached. Run the script again to resume, "
            "or increase --max-attempts."
        )


if __name__ == "__main__":
    main()