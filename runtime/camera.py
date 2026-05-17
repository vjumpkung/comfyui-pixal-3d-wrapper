from typing import Dict

import numpy as np
import torch
from PIL import Image


def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(
    camera_angle_x: float,
    grid_point: torch.Tensor,
    target_point: torch.Tensor,
    mesh_scale: float,
    image_resolution: int,
) -> Dict[str, float]:
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, _zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt = float(target_point[0].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def estimate_camera_params(
    image: Image.Image,
    moge_model: object,
    device: str = "cuda",
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
) -> Dict[str, float]:
    pil_image = image.convert("RGB")
    width, _height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)

    with torch.inference_mode():
        output = moge_model.infer(image_tensor)

    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx = intrinsics[0, 0] * width
    camera_angle_x = 2 * np.arctan(width / (2 * fx))

    distance = distance_from_fov(
        float(camera_angle_x),
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        float(mesh_scale),
        int(image_resolution),
    )["distance_from_x"]

    return {
        "camera_angle_x": float(camera_angle_x),
        "distance": float(distance),
        "mesh_scale": float(mesh_scale),
    }
