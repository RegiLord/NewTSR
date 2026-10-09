import argparse
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
import OpenEXR
from PIL import Image, ImageTk


def read_exr_channels(path):
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        return {name: channel.pixels.astype(np.float32) for name, channel in exr.channels().items()}


def find_channel(channels, name):
    if name in channels:
        return channels[name]

    for key, value in channels.items():
        if key.endswith("." + name):
            return value

    return None


def read_depth(path, clip_end=100.0):
    channels = read_exr_channels(path)

    depth = find_channel(channels, "Z")

    if depth is None:
        depth = find_channel(channels, "V")

    if depth is None:
        depth = find_channel(channels, "R")

    if depth is None:
        candidates = [value for key, value in channels.items() if key != "A" and value.ndim == 2]
        if not candidates:
            raise RuntimeError(f"Could not find a depth channel in {path}")
        depth = candidates[0]

    valid = np.isfinite(depth) & (depth > 0.0) & (depth < clip_end)
    return depth, valid


def create_depth_preview(depth_path, output_path, clip_end=100.0):
    depth, valid = read_depth(depth_path, clip_end)

    if not valid.any():
        raise RuntimeError(f"No valid depth pixels found in {depth_path}")

    near = np.percentile(depth[valid], 1)
    far = np.percentile(depth[valid], 99)

    preview = np.zeros_like(depth, dtype=np.float32)

    if far > near:
        preview[valid] = 1.0 - (depth[valid] - near) / (far - near)
    else:
        preview[valid] = 1.0

    preview = np.clip(preview, 0.0, 1.0)
    preview = (preview * 255.0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="L").save(output_path)


def create_normal_preview(normal_path, output_path):
    channels = read_exr_channels(normal_path)

    x = find_channel(channels, "X")
    y = find_channel(channels, "Y")
    z = find_channel(channels, "Z")

    if x is None or y is None or z is None:
        raise RuntimeError(
            f"Could not find X/Y/Z normal channels in {normal_path}. "
            f"Available channels: {list(channels.keys())}"
        )

    normal = np.stack([x, y, z], axis=-1)

    preview = (normal + 1.0) * 0.5
    preview = np.clip(preview, 0.0, 1.0)
    preview = (preview * 255.0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="RGB").save(output_path)


def view_number_from_rgb(rgb_path):
    return rgb_path.stem


def find_matching_file(directory, view_number, suffix, extension):
    exact = directory / f"{view_number}_{suffix}{extension}"

    if exact.exists():
        return exact

    matches = sorted(directory.glob(f"{view_number}*{extension}"))

    if matches:
        return matches[0]

    return None


class DatasetPreviewer:
    def __init__(self, root, dataset_root, image_size=420, clip_end=100.0):
        self.root = root
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.clip_end = clip_end

        self.samples = sorted(self.dataset_root.glob("objects/*/rgb/*.png"))

        if not self.samples:
            raise RuntimeError(
                f"No RGB images found under:\n"
                f"{self.dataset_root / 'objects/*/rgb/*.png'}"
            )

        self.index = 0
        self.photo_refs = []

        self.root.title("Objaverse Dataset Previewer")
        self.root.minsize(900, 500)

        self.build_gui()
        self.show_current()

        self.root.bind("<Left>", lambda event: self.previous_image())
        self.root.bind("<Right>", lambda event: self.next_image())

    def build_gui(self):
        self.info_label = tk.Label(self.root, text="", font=("TkDefaultFont", 11))
        self.info_label.pack(pady=(10, 4))

        self.path_label = tk.Label(self.root, text="", fg="gray")
        self.path_label.pack(pady=(0, 10))

        image_frame = tk.Frame(self.root)
        image_frame.pack(fill="both", expand=True, padx=10)

        self.panels = []

        for title in ("RGB", "Depth", "Normal"):
            frame = tk.Frame(image_frame)
            frame.pack(side="left", fill="both", expand=True, padx=5)

            title_label = tk.Label(frame, text=title, font=("TkDefaultFont", 12, "bold"))
            title_label.pack(pady=(0, 5))

            image_label = tk.Label(frame, bd=1, relief="solid", anchor="center")
            image_label.pack(fill="both", expand=True)

            self.panels.append(image_label)

        controls = tk.Frame(self.root)
        controls.pack(fill="x", pady=12)

        self.previous_button = tk.Button(
            controls,
            text="← Previous",
            width=16,
            command=self.previous_image,
        )
        self.previous_button.pack(side="left", padx=20)

        self.counter_label = tk.Label(controls, text="")
        self.counter_label.pack(side="left", expand=True)

        self.delete_button = tk.Button(controls, text="Delete Object", width=16, command=self.delete_current_object)
        self.delete_button.pack(side="left", padx=8)

        self.next_button = tk.Button(
            controls,
            text="Next →",
            width=16,
            command=self.next_image,
        )
        self.next_button.pack(side="right", padx=20)

    def get_paths(self, rgb_path):
        object_dir = rgb_path.parent.parent
        uid = object_dir.name
        view_number = view_number_from_rgb(rgb_path)

        depth_path = find_matching_file(
            object_dir / "depth",
            view_number,
            "depth",
            ".exr",
        )

        normal_path = find_matching_file(
            object_dir / "normal",
            view_number,
            "normal",
            ".exr",
        )

        preview_dir = object_dir / "previews"

        depth_preview = preview_dir / "depth" / f"{view_number}.png"
        normal_preview = preview_dir / "normal" / f"{view_number}.png"

        return uid, view_number, depth_path, normal_path, depth_preview, normal_preview

    def ensure_previews(self, depth_path, normal_path, depth_preview, normal_preview):
        if depth_path is None:
            raise RuntimeError("Matching depth EXR was not found.")

        if normal_path is None:
            raise RuntimeError("Matching normal EXR was not found.")

        if not depth_preview.exists() or depth_preview.stat().st_mtime < depth_path.stat().st_mtime:
            create_depth_preview(depth_path, depth_preview, self.clip_end)

        if not normal_preview.exists() or normal_preview.stat().st_mtime < normal_path.stat().st_mtime:
            create_normal_preview(normal_path, normal_preview)

    def load_display_image(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)

            canvas = Image.new("RGB", (self.image_size, self.image_size), (32, 32, 32))
            x = (self.image_size - image.width) // 2
            y = (self.image_size - image.height) // 2
            canvas.paste(image, (x, y))

        return ImageTk.PhotoImage(canvas)

    def show_current(self):
        rgb_path = self.samples[self.index]
        uid, view_number, depth_path, normal_path, depth_preview, normal_preview = self.get_paths(rgb_path)

        try:
            self.ensure_previews(depth_path, normal_path, depth_preview, normal_preview)

            images = [
                self.load_display_image(rgb_path),
                self.load_display_image(depth_preview),
                self.load_display_image(normal_preview),
            ]

            self.photo_refs = images

            for label, image in zip(self.panels, images):
                label.configure(image=image, text="")

            self.info_label.configure(text=f"Object: {uid}    |    View: {view_number}")
            self.path_label.configure(text=str(rgb_path))
            self.counter_label.configure(text=f"{self.index + 1} / {len(self.samples)}")

            self.previous_button.configure(state="normal" if self.index > 0 else "disabled")
            self.next_button.configure(
                state="normal" if self.index < len(self.samples) - 1 else "disabled"
            )

        except Exception as exc:
            for label in self.panels:
                label.configure(image="", text="Could not display sample")

            self.photo_refs = []
            self.info_label.configure(text=f"Failed: {uid} / {view_number}")
            self.path_label.configure(text=str(exc))
            self.counter_label.configure(text=f"{self.index + 1} / {len(self.samples)}")

    def previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_current()

    def next_image(self):
        if self.index < len(self.samples) - 1:
            self.index += 1
            self.show_current()

    def remove_uid_from_success_file(self, uid):
        success_file = self.dataset_root / "successful_uids.txt"

        if not success_file.exists():
            return

        lines = success_file.read_text().splitlines()
        remaining = [line for line in lines if line.strip() and line.strip() != uid]

        content = "\n".join(remaining)
        if content:
            content += "\n"

        success_file.write_text(content)

    def delete_current_object(self):
        if not self.samples:
            return

        rgb_path = self.samples[self.index]
        object_dir = rgb_path.parent.parent
        uid = object_dir.name

        confirmed = messagebox.askyesno(
            "Delete Object",
            f"Delete object {uid} and all of its files?\n\n{object_dir}",
        )

        if not confirmed:
            return

        old_index = self.index

        try:
            shutil.rmtree(object_dir)
            self.remove_uid_from_success_file(uid)
        except Exception as exc:
            messagebox.showerror("Delete Failed", f"Could not delete {uid}:\n{exc}")
            return

        self.samples = [sample for sample in self.samples if sample.parent.parent.name != uid]

        if not self.samples:
            self.photo_refs = []

            for label in self.panels:
                label.configure(image="", text="No images remaining")

            self.info_label.configure(text="Dataset is empty")
            self.path_label.configure(text="")
            self.counter_label.configure(text="0 / 0")
            self.previous_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
            self.delete_button.configure(state="disabled")
            return

        # Keeping the same numeric index lands on the first view of the next
        # object, because all views belonging to the deleted UID were removed.
        self.index = min(old_index, len(self.samples) - 1)
        self.show_current()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("datasets/objaverse_pilot"),
        help="Dataset root containing objects/<uid>/rgb, depth and normal.",
    )
    parser.add_argument("--size", type=int, default=420, help="Displayed size of each image.")
    parser.add_argument("--clip-end", type=float, default=100.0, help="Maximum valid depth.")
    args = parser.parse_args()

    root = tk.Tk()

    try:
        DatasetPreviewer(root, args.dataset, args.size, args.clip_end)
    except Exception as exc:
        messagebox.showerror("Dataset Previewer", str(exc))
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    main()
