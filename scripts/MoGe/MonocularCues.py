import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel
import numpy as np
from PIL import Image

device = torch.device("cuda")
MODEL_PATH = "/home/leusteanstefan/Documents/Practica/MoGe/moge-2-vitb-normal/model.pt"

# Load the model
model = MoGeModel.from_pretrained(MODEL_PATH).to(device)

"""
`output` contains the final prediction. Pass `return_per_step=True` to also return every refinement step.
All maps have the same height and width as the input image.
{
  "points": (H, W, 3),                  # final metric point map in OpenCV camera coordinates (x right, y down, z forward)
  "depth": (H, W),                      # final metric depth map
  "intrinsics": (3, 3),                 # normalized camera intrinsics for the final prediction
  "mask": (H, W),                       # binary mask for valid pixels
  "normal": (H, W, 3),                 # normal map in OpenCV camera coordinates (optional)
}
With `return_per_step=True`, `points_per_step`, `depth_per_step`, and `intrinsics_per_step`
contain `refine_steps + 1` entries, including the initial prediction.
"""

def get_depth_image(model_output):
  depth = model_output["depth"].detach().cpu().numpy()

  valid = np.isfinite(depth)

  d_min = np.min(depth[valid])
  d_max = np.max(depth[valid])

  depth_png = np.zeros_like(depth, dtype=np.uint8)
  depth_png[valid] = ((depth[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)

  return depth_png

def get_normal_image(model_output):
    normal = model_output["normal"].detach().cpu().numpy()
    normal = normal.astype(np.float32)

    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(norm, 1e-8)

    encoded = (normal + 1.0) * 0.5
    encoded = np.clip(encoded, 0.0, 1.0)
    encoded = np.round(encoded * 255.0).astype(np.uint8)

    return encoded


def get_monocular_cues(image):

    if isinstance(image, str):
      # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
      c_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)                       
      c_image = torch.tensor(c_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)   
    elif isinstance(image, np.ndarray):
      c_image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    output = model.infer(c_image)

    return get_depth_image(output), get_normal_image(output)