import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_PIXAL3D_ROOT = PACKAGE_ROOT / "vendor" / "Pixal3D"
DEFAULT_MODEL_PATH = "TencentARC/Pixal3D"
DEFAULT_MOGE_MODEL = "Ruicheng/moge-2-vitl"
MAX_SEED = np.iinfo(np.int32).max

IMAGE_COND_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}


@dataclass
class Pixal3DContext:
    root: str
    model_path: str
    moge_model_name: str
    device: str
    low_vram: bool
    pipeline: Any
    moge_model: Any
    lock: threading.Lock


@dataclass
class Pixal3DResult:
    glb_path: str
    preprocessed_image: Image.Image
    camera_params: Dict[str, float]
    resolution: int


_MODEL_CACHE: Dict[Tuple[str, str, str, str, bool], Pixal3DContext] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def resolve_pixal3d_root(pixal3d_root: Optional[str] = None) -> str:
    configured_root = pixal3d_root or os.environ.get("PIXAL3D_ROOT") or BUNDLED_PIXAL3D_ROOT
    return os.path.abspath(os.path.expanduser(str(configured_root).strip()))


def configure_pixal3d_environment(pixal3d_root: str) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ATTN_BACKEND", "flash_attn_3")
    os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(
        pixal3d_root, "autotune_cache.json"
    )
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

    if pixal3d_root not in sys.path:
        sys.path.insert(0, pixal3d_root)


def require_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        raise RuntimeError("Pixal3D inference requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("Pixal3D inference requires CUDA, but torch.cuda is not available.")


def build_image_cond_model(config: Dict[str, Any]) -> Any:
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def _move_image_cond_models(pipeline: Any, device: str, low_vram: bool) -> None:
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        model = getattr(pipeline, attr, None)
        if model is None:
            continue
        if low_vram:
            model.cpu()
        else:
            model.to(torch.device(device))


def _preload_naf_models(pipeline: Any, device: str, low_vram: bool) -> None:
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        model = getattr(pipeline, attr, None)
        if model is None or not getattr(model, "use_naf_upsample", False):
            continue
        if low_vram:
            model.to(torch.device(device))
        model._load_naf()
        if low_vram:
            model.cpu()


def load_pixal3d_context(
    pixal3d_root: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    moge_model_name: str = DEFAULT_MOGE_MODEL,
    device: str = "cuda",
    low_vram: bool = True,
    preload_naf: bool = True,
    force_reload: bool = False,
) -> Pixal3DContext:
    root = resolve_pixal3d_root(pixal3d_root)
    package_init = os.path.join(root, "pixal3d", "__init__.py")
    if not os.path.isdir(root) or not os.path.isfile(package_init):
        raise RuntimeError(
            "Bundled Pixal3D source was not found. Expected a pixal3d package at: "
            f"{root}"
        )

    require_cuda_device(device)
    key = (root, model_path.strip(), moge_model_name.strip(), device.strip(), bool(low_vram))

    with _MODEL_CACHE_LOCK:
        if force_reload:
            _MODEL_CACHE.pop(key, None)
            torch.cuda.empty_cache()
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        configure_pixal3d_environment(root)

        try:
            from moge.model.v2 import MoGeModel
            from pixal3d.pipelines import Pixal3DImageTo3DPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import Pixal3D dependencies. Install Pixal3D/TRELLIS.2 "
                "dependencies in the same Python environment ComfyUI uses."
            ) from exc

        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path.strip())
        pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
        pipeline.image_cond_model_shape_512 = build_image_cond_model(
            IMAGE_COND_CONFIGS["shape_512"]
        )
        pipeline.image_cond_model_shape_1024 = build_image_cond_model(
            IMAGE_COND_CONFIGS["shape_1024"]
        )
        pipeline.image_cond_model_tex_1024 = build_image_cond_model(
            IMAGE_COND_CONFIGS["tex_1024"]
        )
        pipeline.low_vram = bool(low_vram)
        pipeline.to(torch.device(device))
        _move_image_cond_models(pipeline, device, bool(low_vram))

        if preload_naf:
            _preload_naf_models(pipeline, device, bool(low_vram))

        moge_model = MoGeModel.from_pretrained(moge_model_name.strip()).to(device)
        moge_model.eval()

        context = Pixal3DContext(
            root=root,
            model_path=model_path.strip(),
            moge_model_name=moge_model_name.strip(),
            device=device.strip(),
            low_vram=bool(low_vram),
            pipeline=pipeline,
            moge_model=moge_model,
            lock=threading.Lock(),
        )
        _MODEL_CACHE[key] = context
        return context


def image_tensor_to_pil(image: torch.Tensor, batch_index: int = 0) -> Image.Image:
    if image.ndim == 3:
        frame = image
    elif image.ndim == 4:
        index = max(0, min(int(batch_index), image.shape[0] - 1))
        frame = image[index]
    else:
        raise ValueError(f"Expected IMAGE tensor with 3 or 4 dimensions, got {image.shape}.")

    array = frame.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)

    if array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA")
    if array.shape[-1] == 3:
        return Image.fromarray(array, mode="RGB")
    raise ValueError(f"Expected IMAGE tensor channels to be RGB/RGBA, got {array.shape[-1]}.")


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
    rotation_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
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
    moge_model: Any,
    device: str = "cuda",
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
) -> Dict[str, float]:
    pil_image = image.convert("RGB")
    width, _height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)

    with torch.no_grad():
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


def sanitize_output_prefix(prefix: str) -> str:
    basename = os.path.basename(prefix.strip() or "pixal3d")
    basename = os.path.splitext(basename)[0]
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._-")
    return basename or "pixal3d"


def make_output_path(output_dir: str, output_prefix: str, seed: int) -> str:
    out_dir = Path(output_dir) / "pixal3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{sanitize_output_prefix(output_prefix)}_{timestamp}_seed{int(seed)}"
    candidate = out_dir / f"{stem}.glb"
    counter = 1
    while candidate.exists():
        candidate = out_dir / f"{stem}_{counter:02d}.glb"
        counter += 1
    return str(candidate)


def default_sampler_settings() -> Dict[str, Dict[str, float]]:
    return {
        "sparse_structure": {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        "shape_slat": {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        "tex_slat": {
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
    }


def make_sampler_settings(
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
) -> Dict[str, Dict[str, float]]:
    return {
        "sparse_structure": {
            "steps": int(ss_sampling_steps),
            "guidance_strength": float(ss_guidance_strength),
            "guidance_rescale": float(ss_guidance_rescale),
            "rescale_t": float(ss_rescale_t),
        },
        "shape_slat": {
            "steps": int(shape_slat_sampling_steps),
            "guidance_strength": float(shape_slat_guidance_strength),
            "guidance_rescale": float(shape_slat_guidance_rescale),
            "rescale_t": float(shape_slat_rescale_t),
        },
        "tex_slat": {
            "steps": int(tex_slat_sampling_steps),
            "guidance_strength": float(tex_slat_guidance_strength),
            "guidance_rescale": float(tex_slat_guidance_rescale),
            "rescale_t": float(tex_slat_rescale_t),
        },
    }


def run_pixal3d_to_glb(
    context: Pixal3DContext,
    image: Image.Image,
    output_path: str,
    sampler_settings: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
    preprocess_image: bool = True,
    background_color: str = "#000000",
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    decimation_target: int = 200000,
    texture_size: int = 2048,
    remesh: bool = True,
    remesh_band: int = 1,
    remesh_project: int = 0,
    extension_webp: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Pixal3DResult:
    import o_voxel

    settings = sampler_settings or default_sampler_settings()
    pipeline = context.pipeline

    def update(label: str, step: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(label, step, total)

    with context.lock:
        torch.manual_seed(int(seed))

        update("preprocess", 0, 4)
        if preprocess_image:
            bg_color = parse_rgb_color(background_color)
            image_preprocessed = pipeline.preprocess_image(image, bg_color=bg_color)
        else:
            image_preprocessed = image.convert("RGB")

        update("camera", 1, 4)
        camera_params = estimate_camera_params(
            image_preprocessed,
            context.moge_model,
            device=context.device,
            mesh_scale=float(mesh_scale),
            extend_pixel=int(extend_pixel),
            image_resolution=int(image_resolution),
        )

        update("generate", 2, 4)
        with torch.no_grad():
            mesh_list, (_shape_slat, _tex_slat, res) = pipeline.run(
                image_preprocessed,
                camera_params=camera_params,
                seed=int(seed),
                sparse_structure_sampler_params=settings["sparse_structure"],
                shape_slat_sampler_params=settings["shape_slat"],
                tex_slat_sampler_params=settings["tex_slat"],
                preprocess_image=False,
                return_latent=True,
                pipeline_type=pipeline_type,
                max_num_tokens=int(max_num_tokens),
            )

            mesh = mesh_list[0]

            update("export", 3, 4)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(decimation_target),
                texture_size=int(texture_size),
                remesh=bool(remesh),
                remesh_band=int(remesh_band),
                remesh_project=int(remesh_project),
                use_tqdm=True,
            )

            rotation = np.array(
                [
                    [-1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            glb.apply_transform(rotation)

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            glb.export(output_path, extension_webp=bool(extension_webp))

        torch.cuda.empty_cache()
        update("done", 4, 4)

    return Pixal3DResult(
        glb_path=os.path.abspath(output_path),
        preprocessed_image=image_preprocessed,
        camera_params=camera_params,
        resolution=int(res),
    )


def camera_params_to_json(camera_params: Dict[str, float], resolution: int) -> str:
    payload = {**camera_params, "resolution": int(resolution)}
    return json.dumps(payload, sort_keys=True)
