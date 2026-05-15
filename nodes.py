from io import BytesIO
from typing import Any, Dict, Optional

import torch

from .pixal3d_runtime import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MOGE_MODEL,
    DEFAULT_REMBG_MODEL,
    MAX_SEED,
    NAF_ATTENTION_BACKENDS,
    PIXAL3D_ATTENTION_BACKENDS,
    PIXAL3D_SPARSE_ATTENTION_BACKENDS,
    camera_params_to_json,
    default_sampler_settings,
    image_tensor_to_pil,
    load_pixal3d_background_remover_context,
    load_pixal3d_context,
    make_sampler_settings,
    pil_to_image_tensor,
    preprocess_image_with_background_remover,
    run_pixal3d_to_3d,
)

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


def _progress_callback() -> Any:
    if ProgressBar is None:
        return None
    pbar = ProgressBar(3)

    def update(_label: str, step: int, _total: int) -> None:
        if step > 0:
            pbar.update(1)

    return update


def _glb_file_output(glb_data: bytes) -> Any:
    try:
        from comfy_api.latest import Types
    except Exception:
        return BytesIO(glb_data)
    return Types.File3D(BytesIO(glb_data), "glb")


class Pixal3DModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL_PATH,
                        "placeholder": "TencentARC/Pixal3D or local model folder",
                    },
                ),
                "moge_model_name": (
                    "STRING",
                    {
                        "default": DEFAULT_MOGE_MODEL,
                        "placeholder": "Ruicheng/moge-2-vitl",
                    },
                ),
                "device": (
                    "STRING",
                    {
                        "default": "cuda",
                        "placeholder": "cuda",
                    },
                ),
                "low_vram": ("BOOLEAN", {"default": False}),
                "preload_naf": ("BOOLEAN", {"default": True}),
                "attention_backend": (
                    list(PIXAL3D_ATTENTION_BACKENDS),
                    {"default": "flash_attn_3"},
                ),
                "sparse_attention_backend": (
                    list(PIXAL3D_SPARSE_ATTENTION_BACKENDS),
                    {"default": "auto"},
                ),
                "naf_attention_backend": (
                    list(NAF_ATTENTION_BACKENDS),
                    {"default": "auto"},
                ),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("PIXAL3D_MODEL",)
    RETURN_NAMES = ("pixal3d_model",)
    FUNCTION = "load"
    CATEGORY = "Pixal3D"

    def load(
        self,
        model_path: str,
        moge_model_name: str,
        device: str,
        low_vram: bool,
        preload_naf: bool,
        attention_backend: str,
        sparse_attention_backend: str,
        naf_attention_backend: str,
        force_reload: bool,
    ):
        context = load_pixal3d_context(
            model_path=model_path,
            moge_model_name=moge_model_name,
            device=device,
            low_vram=low_vram,
            preload_naf=preload_naf,
            attention_backend=attention_backend,
            sparse_attention_backend=sparse_attention_backend,
            naf_attention_backend=naf_attention_backend,
            force_reload=force_reload,
        )
        return (context,)


class Pixal3DBackgroundRemoverLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {
                        "default": DEFAULT_REMBG_MODEL,
                        "placeholder": "ZhengPeng7/BiRefNet",
                    },
                ),
                "device": (
                    "STRING",
                    {
                        "default": "cuda",
                        "placeholder": "cuda",
                    },
                ),
                "low_vram": ("BOOLEAN", {"default": False}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("PIXAL3D_REMBG_MODEL",)
    RETURN_NAMES = ("rembg_model",)
    FUNCTION = "load"
    CATEGORY = "Pixal3D"

    def load(
        self,
        model_name: str,
        device: str,
        low_vram: bool,
        force_reload: bool,
    ):
        context = load_pixal3d_background_remover_context(
            model_name=model_name,
            device=device,
            low_vram=low_vram,
            force_reload=force_reload,
        )
        return (context,)


class Pixal3DSamplerSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ss_guidance_strength": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "ss_guidance_rescale": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "ss_sampling_steps": (
                    "INT",
                    {"default": 12, "min": 1, "max": 100},
                ),
                "ss_rescale_t": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "shape_slat_guidance_strength": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "shape_slat_guidance_rescale": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "shape_slat_sampling_steps": (
                    "INT",
                    {"default": 12, "min": 1, "max": 100},
                ),
                "shape_slat_rescale_t": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "tex_slat_guidance_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "tex_slat_guidance_rescale": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "tex_slat_sampling_steps": (
                    "INT",
                    {"default": 12, "min": 1, "max": 100},
                ),
                "tex_slat_rescale_t": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("PIXAL3D_SAMPLER_SETTINGS",)
    RETURN_NAMES = ("sampler_settings",)
    FUNCTION = "settings"
    CATEGORY = "Pixal3D"

    def settings(
        self,
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
    ):
        return (
            make_sampler_settings(
                ss_guidance_strength,
                ss_guidance_rescale,
                ss_sampling_steps,
                ss_rescale_t,
                shape_slat_guidance_strength,
                shape_slat_guidance_rescale,
                shape_slat_sampling_steps,
                shape_slat_rescale_t,
                tex_slat_guidance_strength,
                tex_slat_guidance_rescale,
                tex_slat_sampling_steps,
                tex_slat_rescale_t,
            ),
        )


class Pixal3DPreprocessImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rembg_model": ("PIXAL3D_REMBG_MODEL", {"forceInput": True}),
                "image": ("IMAGE", {}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "background_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preprocessed_image",)
    FUNCTION = "preprocess"
    CATEGORY = "Pixal3D"

    def preprocess(
        self,
        rembg_model: Any,
        image: torch.Tensor,
        batch_index: int,
        background_color: str,
    ):
        pil_image = image_tensor_to_pil(image, batch_index)
        preprocessed = preprocess_image_with_background_remover(
            rembg_model,
            pil_image,
            background_color=background_color,
        )
        return (pil_to_image_tensor(preprocessed),)


class Pixal3DImageTo3D:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixal3d_model": ("PIXAL3D_MODEL", {"forceInput": True}),
                "image": ("IMAGE", {}),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": int(MAX_SEED)},
                ),
                "pipeline_type": (["1024_cascade", "1536_cascade"],),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "mesh_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "extend_pixel": ("INT", {"default": 0, "min": -256, "max": 256}),
                "image_resolution": ("INT", {"default": 512, "min": 128, "max": 2048}),
                "max_num_tokens": (
                    "INT",
                    {"default": 49152, "min": 1024, "max": 262144},
                ),
                "decimation_target": (
                    "INT",
                    {"default": 200000, "min": 1000, "max": 2000000},
                ),
                "texture_size": (
                    "INT",
                    {"default": 2048, "min": 256, "max": 8192},
                ),
                "remesh": ("BOOLEAN", {"default": True}),
                "remesh_band": ("INT", {"default": 1, "min": 0, "max": 8}),
                "remesh_project": ("INT", {"default": 0, "min": 0, "max": 8}),
                "extension_webp": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sampler_settings": (
                    "PIXAL3D_SAMPLER_SETTINGS",
                    {"forceInput": True},
                ),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB", "STRING")
    RETURN_NAMES = ("model_3d", "camera_json")
    FUNCTION = "generate"
    CATEGORY = "Pixal3D"

    def generate(
        self,
        pixal3d_model: Any,
        image: torch.Tensor,
        seed: int,
        pipeline_type: str,
        batch_index: int,
        mesh_scale: float,
        extend_pixel: int,
        image_resolution: int,
        max_num_tokens: int,
        decimation_target: int,
        texture_size: int,
        remesh: bool,
        remesh_band: int,
        remesh_project: int,
        extension_webp: bool,
        sampler_settings: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        pil_image = image_tensor_to_pil(image, batch_index)
        settings = sampler_settings or default_sampler_settings()

        result = run_pixal3d_to_3d(
            pixal3d_model,
            pil_image,
            settings,
            seed=seed,
            pipeline_type=pipeline_type,
            mesh_scale=mesh_scale,
            extend_pixel=extend_pixel,
            image_resolution=image_resolution,
            max_num_tokens=max_num_tokens,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=remesh_band,
            remesh_project=remesh_project,
            extension_webp=extension_webp,
            progress_callback=_progress_callback(),
        )
        camera_json = camera_params_to_json(result.camera_params, result.resolution)
        return (_glb_file_output(result.glb_data), camera_json)


NODE_CLASS_MAPPINGS = {
    "Pixal3DModelLoader": Pixal3DModelLoader,
    "Pixal3DBackgroundRemoverLoader": Pixal3DBackgroundRemoverLoader,
    "Pixal3DSamplerSettings": Pixal3DSamplerSettings,
    "Pixal3DPreprocessImage": Pixal3DPreprocessImage,
    "Pixal3DImageTo3D": Pixal3DImageTo3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DModelLoader": "Pixal3D Model Loader",
    "Pixal3DBackgroundRemoverLoader": "Pixal3D Background Remover Loader",
    "Pixal3DSamplerSettings": "Pixal3D Sampler Settings",
    "Pixal3DPreprocessImage": "Pixal3D Preprocess Image",
    "Pixal3DImageTo3D": "Pixal3D Image to 3D",
}
