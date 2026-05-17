from typing import Tuple

import numpy as np
import torch
from PIL import Image


def image_tensor_to_pil(image: torch.Tensor, batch_index: int = 0) -> Image.Image:
    if image.ndim == 3:
        frame = image
    elif image.ndim == 4:
        index = max(0, min(int(batch_index), image.shape[0] - 1))
        frame = image[index]
    else:
        raise ValueError(
            f"Expected IMAGE tensor with 3 or 4 dimensions, got {image.shape}."
        )

    array = frame.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)

    if array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA")
    if array.shape[-1] == 3:
        return Image.fromarray(array, mode="RGB")
    raise ValueError(
        f"Expected IMAGE tensor channels to be RGB/RGBA, got {array.shape[-1]}."
    )


def pil_to_image_tensor(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB")
    array = np.asarray(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def parse_rgb_color(value: str) -> Tuple[int, int, int]:
    color = value.strip()
    if color.startswith("#"):
        color = color[1:]
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in color):
        raise ValueError("background_color must be a hex RGB color like #000000.")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
