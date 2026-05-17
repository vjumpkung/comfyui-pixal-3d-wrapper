"""Vendored, transformers>=5 compatible inference subset of
``pixal3d.trainers.flow_matching.mixins.image_conditioned_proj``.

Only the pieces actually used by Pixal3D inference are kept:

- :func:`project_points_to_image_batch`
- :func:`sample_features`
- :class:`ProjGrid`
- :class:`DinoV3ProjFeatureExtractor`

Upstream additionally ships ``DinoV3VaeProjFeatureExtractor`` and
``ImageConditionedProjMixin``; both are training/Flux-VAE-only and are
omitted here.

DINOv3 encoder iteration goes through :mod:`._dinov3`, which encapsulates
the transformers-version dynamics so we don't touch unstable private APIs
inline.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import DINOv3ViTModel

from ._dinov3 import extract_dinov3_features


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------


def project_points_to_image_batch(
    points_3d: torch.Tensor,
    transform_matrix: torch.Tensor,
    camera_angle_x: torch.Tensor,
    resolution: int = 518,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D points to 2D image coordinates (batch).

    Args:
        points_3d: shape [N, 3] or [B, N, 3], in [-1, 1].
        transform_matrix: shape [B, 4, 4], camera-to-world transform.
        camera_angle_x: shape [B], horizontal FOV in radians.
        resolution: output image side length in pixels.

    Returns:
        ``(points_2d, depth, valid_mask)``:
        - ``points_2d``: [B, N, 2] pixel coordinates.
        - ``depth``: [B, N] depth values.
        - ``valid_mask``: [B, N] in-frame / in-front-of-camera mask.
    """
    device = points_3d.device
    B = transform_matrix.shape[0]

    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(
            transform_matrix, dtype=torch.float32, device=device
        )
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
    if not isinstance(camera_angle_x, torch.Tensor):
        camera_angle_x = torch.tensor(
            camera_angle_x, dtype=torch.float32, device=device
        )

    if points_3d.dim() == 2:
        points_3d_batch = points_3d.unsqueeze(0).expand(B, -1, -1)
    else:
        points_3d_batch = points_3d
    N = points_3d_batch.shape[1]

    ones = torch.ones(B, N, 1, device=device, dtype=points_3d_batch.dtype)
    points_homogeneous = torch.cat([points_3d_batch, ones], dim=-1)

    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(
        transform_matrix.dtype
    )

    points_camera = torch.bmm(
        points_homogeneous, world_to_camera.transpose(-2, -1)
    )[..., :3]

    x_cam = points_camera[..., 0]
    y_cam = points_camera[..., 1]
    z_cam = points_camera[..., 2]

    depth = -z_cam

    sensor_width = 32.0
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)
    focal_length_pixels = focal_length * resolution / sensor_width
    focal_length_pixels = focal_length_pixels.unsqueeze(1)

    x_ndc = focal_length_pixels * x_cam / (-z_cam + 1e-8)
    y_ndc = focal_length_pixels * y_cam / (-z_cam + 1e-8)

    x_pixel = x_ndc + resolution / 2.0
    y_pixel = -y_ndc + resolution / 2.0

    valid_mask = (
        (x_pixel >= 0)
        & (x_pixel < resolution)
        & (y_pixel >= 0)
        & (y_pixel < resolution)
        & (depth > 0)
    )

    points_2d = torch.stack([x_pixel, y_pixel], dim=-1)
    return points_2d, depth, valid_mask


def sample_features(
    fmap: torch.Tensor, queries_ndc: torch.Tensor
) -> torch.Tensor:
    """Bilinear-sample ``fmap`` at NDC coordinates ``queries_ndc``.

    Args:
        fmap: [B, C, H, W].
        queries_ndc: [B, K, 2], normalized device coordinates in [-1, 1].

    Returns:
        [B, C, K] sampled features.
    """
    B, C, H, W = fmap.shape
    Bq, K, _ = queries_ndc.shape
    assert Bq == B, "Batch size mismatch"

    grid = queries_ndc.view(B, K, 1, 2)
    feat = F.grid_sample(
        fmap,
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return feat.squeeze(-1)


# ---------------------------------------------------------------------------
# Projection grid
# ---------------------------------------------------------------------------


class ProjGrid(nn.Module):
    """3D grid → image-feature projection module.

    Generates a fixed ``grid_resolution``³ point grid in [-1, 1]³, projects
    it to the image plane via :func:`project_points_to_image_batch`, then
    bilinear-samples ``features_map`` at those locations. This is the core
    view-aligned feature extraction used by Pixal3D's image conditioning.
    """

    def __init__(self, grid_resolution: int = 16, image_resolution: int = 518):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution

        one_dim = torch.linspace(-1, 1, grid_resolution)
        x, y, z = torch.meshgrid(one_dim, one_dim, one_dim, indexing="ij")
        grid_points = torch.stack((x, y, z), dim=-1)

        rotation_matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        grid_points = torch.matmul(grid_points, rotation_matrix.T)
        grid_points = grid_points.reshape(-1, 3)
        self.register_buffer("grid_points", grid_points)

        front_view_transform_matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.register_buffer(
            "front_view_transform_matrix", front_view_transform_matrix
        )

    def forward(
        self,
        features_map: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        BHWC: bool = True,
    ) -> torch.Tensor:
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape

        grid_points = self.grid_points
        grid_points = grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance

        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points,
            transform_matrix,
            camera_angle_x,
            self.image_resolution,
        )

        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1

        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)

        x = sample_features(features_map, image_points_norm)
        x = x.permute(0, 2, 1)
        return x

    def visualize_projection(
        self,
        image: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        prefix: str = "proj_vis",
    ) -> List[Image.Image]:
        B = image.shape[0]

        grid_points = self.grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance

        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points,
            transform_matrix,
            camera_angle_x,
            self.image_resolution,
        )

        vis_images = []
        for b in range(B):
            img_np = image[b].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            pil_img = Image.fromarray(img_np)
            if pil_img.size != (self.image_resolution, self.image_resolution):
                pil_img = pil_img.resize(
                    (self.image_resolution, self.image_resolution), Image.LANCZOS
                )

            vis_img = pil_img.copy()
            draw = ImageDraw.Draw(vis_img)

            pts = image_points[b].cpu().numpy()
            depths = depth[b].cpu().numpy()
            mask = valid_mask[b].cpu().numpy()

            valid_depths = depths[mask]
            if len(valid_depths) > 0:
                d_min, d_max = valid_depths.min(), valid_depths.max()
                if d_max - d_min > 1e-6:
                    depths_norm = (depths - d_min) / (d_max - d_min)
                else:
                    depths_norm = np.ones_like(depths) * 0.5
            else:
                depths_norm = np.ones_like(depths) * 0.5

            for i, (pt, d, m, dn) in enumerate(
                zip(pts, depths, mask, depths_norm)
            ):
                if not m:
                    continue
                x, y = pt
                r = int(255 * dn)
                g = int(255 * (1 - abs(2 * dn - 1)))
                b_color = int(255 * (1 - dn))
                color = (r, g, b_color)
                radius = 2
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=color,
                    outline=color,
                )

            vis_images.append(vis_img)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prefix}_batch{b}.png")
                vis_img.save(save_path)
                print(f"Saved projection visualization to: {save_path}")

        return vis_images


# ---------------------------------------------------------------------------
# DINOv3 feature extractor with projection
# ---------------------------------------------------------------------------


class DinoV3ProjFeatureExtractor(nn.Module):
    """DINOv3 feature extractor with view-aligned projection.

    Produces:

    1. Global features (CLS + register tokens): ``[B, 1+num_reg, embed_dim]``
    2. View-aligned projected features:

       - without NAF: ``[B, grid_resolution³, embed_dim]``
       - with NAF: ``[B, grid_resolution³, embed_dim * 2]`` (lr ⊕ hr)

    Args:
        model_name: DINOv3 HF repo or local path.
        image_size: Square input image side.
        grid_resolution: 3D grid resolution per axis.
        use_naf_upsample: Whether to upsample DINOv3 features with NAF and
            concatenate to the projected features.
        naf_target_size: NAF output spatial size (int or [H, W]).
    """

    def __init__(
        self,
        model_name: str,
        image_size: int = 512,
        grid_resolution: int = 16,
        use_naf_upsample: bool = False,
        naf_target_size: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.use_naf_upsample = use_naf_upsample
        if naf_target_size is None:
            self.naf_target_size = (128, 128)
        elif isinstance(naf_target_size, int):
            self.naf_target_size = (naf_target_size, naf_target_size)
        else:
            self.naf_target_size = tuple(naf_target_size)

        # Resolve via module globals so callers can monkey-patch
        # ``DINOv3ViTModel`` to share frozen backbones across instances.
        self.model = globals()["DINOv3ViTModel"].from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)

        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.patch_size = self.model.config.patch_size
        self.patch_number = image_size // self.patch_size
        self.embed_dim = self.model.config.hidden_size

        self.proj_grid = ProjGrid(
            grid_resolution=grid_resolution,
            image_resolution=image_size,
        )

        self.naf_model = None  # lazy-loaded

        self.proj_channels = (
            self.embed_dim * 2 if use_naf_upsample else self.embed_dim
        )

    def _load_naf(self):
        """Lazy-load the pretrained NAF upsampler."""
        if self.naf_model is None:
            import torch.hub

            device = next(self.model.parameters()).device
            self.naf_model = torch.hub.load(
                "valeoai/NAF",
                "naf",
                pretrained=True,
                device=device,
                trust_repo=True,
            )
            self.naf_model.eval()
            self.naf_model.requires_grad_(False)

    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.proj_grid.to(device)
        if self.naf_model is not None:
            self.naf_model.to(device)
        return self

    def cuda(self):
        super().cuda()
        self.model.cuda()
        self.proj_grid.cuda()
        if self.naf_model is not None:
            self.naf_model.cuda()
        return self

    def cpu(self):
        super().cpu()
        self.model.cpu()
        self.proj_grid.cpu()
        if self.naf_model is not None:
            self.naf_model.cpu()
        return self

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Transformers>=5-compatible DINOv3 pre-norm encoder pass."""
        return extract_dinov3_features(self.model, image)

    def forward(
        self,
        image: Union[torch.Tensor, List[Image.Image]],
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            image = [
                i.resize((self.image_size, self.image_size), Image.LANCZOS)
                for i in image
            ]
            image = [
                np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image
            ]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        B = image.shape[0]

        if self.use_naf_upsample:
            image_for_naf = image.clone()

        image = self.transform(image)

        with torch.no_grad():
            z = self.extract_features(image)

            z_clstoken = z[:, 0:1]
            num_reg = getattr(self.model.config, "num_register_tokens", 4)
            z_regtokens = z[:, 1 : 1 + num_reg]
            z_patchtokens = z[:, 1 + num_reg :]

            z_patchtokens_spatial = z_patchtokens.reshape(
                B, self.patch_number, self.patch_number, -1
            )

            if camera_angle_x is None or distance is None or mesh_scale is None:
                raise ValueError(
                    "camera_angle_x, distance, and mesh_scale must be provided"
                )

            z_proj_lr = self.proj_grid(
                z_patchtokens_spatial,
                camera_angle_x,
                distance,
                mesh_scale,
                transform_matrix,
            )

            if self.use_naf_upsample:
                self._load_naf()
                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)
                hr_features = self.naf_model(
                    image_for_naf, lr_features_bchw, self.naf_target_size
                )
                z_proj_hr = self.proj_grid(
                    hr_features,
                    camera_angle_x,
                    distance,
                    mesh_scale,
                    transform_matrix,
                    BHWC=False,
                )
                z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            else:
                z_proj = z_proj_lr

            z_global = torch.cat([z_clstoken, z_regtokens], dim=1)

        return z_global, z_proj

    @torch.no_grad()
    def visualize_projection(
        self,
        image: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        prefix: str = "proj_vis",
    ) -> List[Image.Image]:
        return self.proj_grid.visualize_projection(
            image=image,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            transform_matrix=transform_matrix,
            save_dir=save_dir,
            prefix=prefix,
        )


__all__ = [
    "project_points_to_image_batch",
    "sample_features",
    "ProjGrid",
    "DinoV3ProjFeatureExtractor",
    "DINOv3ViTModel",
]
