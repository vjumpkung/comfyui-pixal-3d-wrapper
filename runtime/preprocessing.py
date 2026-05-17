import threading
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image

from .attention import require_cuda_device
from .constants import DEFAULT_REMBG_MODEL, PACKAGE_ROOT
from .devices import (
    _first_floating_tensor_dtype,
    _first_tensor_device,
    _warn_if_not_on_device,
)
from .hf import _resolve_hf_snapshot
from .image_utils import parse_rgb_color
from .profiling import _log_load_timing, _profile_load_enabled
from .types import Pixal3DBackgroundRemoverContext


_REMBG_CACHE: Dict[Tuple[str, str, str, bool], Pixal3DBackgroundRemoverContext] = {}
_REMBG_CACHE_LOCK = threading.Lock()


class BackgroundRemover:
    def __init__(self, model_name: str = DEFAULT_REMBG_MODEL):
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        self.model_name = model_name
        self.model = AutoModelForImageSegmentation.from_pretrained(
            _resolve_hf_snapshot(model_name),
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._to_pil_image = transforms.ToPILImage()

    def to(self, device: Any) -> "BackgroundRemover":
        self.model.to(device)
        return self

    def cuda(self) -> "BackgroundRemover":
        self.model.cuda()
        return self

    def cpu(self) -> "BackgroundRemover":
        self.model.cpu()
        return self

    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        device = _first_tensor_device(self.model)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _first_floating_tensor_dtype(self.model)
        input_images = (
            self.transform_image(image.convert("RGB"))
            .unsqueeze(0)
            .to(
                device=device,
                dtype=dtype,
            )
        )
        with torch.inference_mode():
            preds = self.model(input_images)[-1].sigmoid().float().cpu()
        mask = self._to_pil_image(preds[0].squeeze()).resize(image_size)
        output = image.convert("RGBA")
        output.putalpha(mask)
        return output


def load_pixal3d_background_remover_context(
    model_name: str = DEFAULT_REMBG_MODEL,
    device: str = "cuda",
    low_vram: bool = False,
    force_reload: bool = False,
) -> Pixal3DBackgroundRemoverContext:
    root = str(PACKAGE_ROOT)
    require_cuda_device(device)
    key = (root, model_name.strip(), device.strip(), bool(low_vram))
    profile_load = _profile_load_enabled()

    with _REMBG_CACHE_LOCK:
        if force_reload:
            _REMBG_CACHE.pop(key, None)
            torch.cuda.empty_cache()
        if key in _REMBG_CACHE:
            if profile_load:
                print("[Pixal3D] load: background remover cache hit")  # noqa: T201
            return _REMBG_CACHE[key]

        started_at = time.perf_counter()
        model = BackgroundRemover(model_name.strip())
        model.to(device)
        if low_vram:
            model.cpu()
        _log_load_timing("background remover load", started_at, profile_load, device)
        if not low_vram:
            _warn_if_not_on_device("background remover", model, device.strip())

        context = Pixal3DBackgroundRemoverContext(
            root=root,
            model_name=model_name.strip(),
            device=device.strip(),
            low_vram=bool(low_vram),
            model=model,
            lock=threading.Lock(),
        )
        _REMBG_CACHE[key] = context
        return context


def preprocess_image_with_background_remover(
    context: Pixal3DBackgroundRemoverContext,
    image: Image.Image,
    background_color: str = "#000000",
) -> Image.Image:
    bg_color = parse_rgb_color(background_color)

    with context.lock:
        input_image = image
        has_alpha = False
        if input_image.mode == "RGBA":
            alpha = np.array(input_image)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True

        max_size = max(input_image.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input_image = input_image.resize(
                (int(input_image.width * scale), int(input_image.height * scale)),
                Image.Resampling.LANCZOS,
            )

        if has_alpha:
            output = input_image
        else:
            input_image = input_image.convert("RGB")
            if context.low_vram:
                context.model.to(context.device)
            output = context.model(input_image)
            if context.low_vram:
                context.model.cpu()

        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox_points = np.argwhere(alpha > 0.8 * 255)
        if bbox_points.size == 0:
            raise RuntimeError("Background remover did not find a foreground object.")
        bbox = (
            np.min(bbox_points[:, 1]),
            np.min(bbox_points[:, 0]),
            np.max(bbox_points[:, 1]),
            np.max(bbox_points[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.1)
        crop_box = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        output = output.crop(crop_box)
        output = np.array(output).astype(np.float32) / 255
        rgb = output[:, :, :3]
        alpha = output[:, :, 3:4]
        bg = np.array(bg_color, dtype=np.float32) / 255.0
        composited = rgb * alpha + bg * (1.0 - alpha)
        return Image.fromarray((np.clip(composited, 0, 1) * 255).astype(np.uint8))
