import math
import os
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from moge.model.v2 import MoGeModel

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
)


class NewTSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        depth_normal_cls: str
        depth_normal: dict

        fusioner_cls: str 
        fusioner: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        result = model.load_state_dict(ckpt, strict=False)

        model.freeze_all_except_fusioner()
        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

        # So that i can have no problems i'll simply use the moge 
        self.depth_normal = MoGeModel.from_pretrained(
            self.cfg.depth_normal["model_path"]
        ).to("cuda")        

        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.fusioner = find_class(self.cfg.fusioner_cls)(self.cfg.fusioner)

        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

        self.freeze_all_except_fusioner()

    def freeze_all_except_fusioner(self):
        # Freeze all parameters except fusioner
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith("fusioner."))

        for name, module in self.named_children():
            if name != "fusioner":
                module.eval()

    def train(self, mode: bool = True):
        # Keep modules in evaluation mode even when training

        super().train(mode)

        for name, module in self.named_children():
            if name != "fusioner":
                module.eval()

        return self

    def forward(self, image, device: str) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(device)

        # Convert RGB to image tokenizer format
        rgb_img = rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1)

        # Convert RGB to MoGe input format
        moge_input = rgb_img[:, 0, :3]

        with torch.no_grad():
            # Generate depth and normal predictions
            moge_output = self.depth_normal.infer(moge_input)

            # Convert both predictions to DINO-compatible tensors
            depth_img = self.get_depth_image(moge_output)
            normal_img = self.get_normal_image(moge_output)

            print("RGB:", rgb_img.shape)
            print("Depth:", depth_img.shape)
            print("Normal:", normal_img.shape)

            if rgb_img.shape != depth_img.shape or rgb_img.shape != normal_img.shape:
                raise RuntimeError("RGB, depth, and normal tensors must have identical shapes")

            input_image_tokens: torch.Tensor = self.image_tokenizer(rgb_img)
            input_depth_tokens: torch.Tensor = self.image_tokenizer(depth_img)
            input_normal_tokens: torch.Tensor = self.image_tokenizer(normal_img)

            input_image_tokens = rearrange(
                input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
            )
            input_depth_tokens = rearrange(
                input_depth_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
            )
            input_normal_tokens = rearrange(
                input_normal_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
            )
            
            print("RGB_tokens:", input_image_tokens.shape)
            print("Depth_tokens:", input_depth_tokens.shape)
            print("Normal_tokens:", input_normal_tokens.shape)
            

        fused_tokens = self.fusioner(
            input_image_tokens,
            input_depth_tokens,
            input_normal_tokens
        )
        
        batch_size = rgb_img.shape[0]

        tokens: torch.Tensor = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=fused_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))

        return scene_codes  

    def render(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)

        return images

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, scene_codes, has_vertex_color, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            color = None
            if has_vertex_color:
                with torch.no_grad():
                    color = self.renderer.query_triplane(
                        self.decoder,
                        v_pos,
                        scene_code,
                    )["color"]
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy() if has_vertex_color else None,
            )
            meshes.append(mesh)
        return meshes

    def get_depth_image(self, model_output: dict) -> torch.Tensor:

        depth = model_output["depth"].float()

        if depth.ndim == 2:
            depth = depth.unsqueeze(0)

        if depth.ndim != 3:
            raise ValueError(f"Expected depth shape (B,H,W) or (H,W), got {tuple(depth.shape)}")

        # Identify valid depth values
        valid = torch.isfinite(depth)

        # Calculate minimum and maximum depth separately for each image
        d_min = torch.where(valid, depth, torch.inf).amin(dim=(1, 2), keepdim=True)
        d_max = torch.where(valid, depth, -torch.inf).amax(dim=(1, 2), keepdim=True)

        # Handle images with no valid depth measurements
        has_valid = valid.any(dim=(1, 2), keepdim=True)

        d_min = torch.where(has_valid, d_min, torch.zeros_like(d_min))
        d_max = torch.where(has_valid, d_max, torch.zeros_like(d_max))

        # Normalize each depth image to [0, 1]
        safe_depth = torch.where(valid, depth, torch.zeros_like(depth))

        normalized = ((safe_depth - d_min) / (d_max - d_min).clamp_min(1e-8)).clamp(0.0, 1.0)

        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))

        # Convert (B,H,W) -> (B,1,3,H,W)
        return normalized[:, None, None].expand(-1, 1, 3, -1, -1)

    def get_normal_image(self, model_output: dict) -> torch.Tensor:

        normal = model_output["normal"].float()

        if normal.ndim == 3:
            normal = normal.unsqueeze(0)

        if normal.ndim != 4 or normal.shape[-1] != 3:
            raise ValueError(f"Expected normal shape (B,H,W,3) or (H,W,3), got {tuple(normal.shape)}")

        # Handle invalid values
        normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize each normal vector
        normal = F.normalize(normal, dim=-1, eps=1e-8)

        # Convert [-1, 1] to [0, 1]
        encoded = ((normal + 1.0) * 0.5).clamp(0.0, 1.0)

        # Convert (B,H,W,3) -> (B,1,3,H,W)
        return rearrange(encoded, "B H W C -> B 1 C H W")
