import os
from typing import Any, Dict, Optional

import torch

from .pixal3d_runtime import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MOGE_MODEL,
    MAX_SEED,
    camera_params_to_json,
    default_sampler_settings,
    image_tensor_to_pil,
    load_pixal3d_context,
    make_output_path,
    make_sampler_settings,
    pil_to_image_tensor,
    run_pixal3d_to_glb,
)

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


def _output_directory() -> str:
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.join(os.getcwd(), "output")


def _progress_callback() -> Any:
    if ProgressBar is None:
        return None
    pbar = ProgressBar(4)

    def update(_label: str, step: int, _total: int) -> None:
        if step > 0:
            pbar.update(1)

    return update


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
                "low_vram": ("BOOLEAN", {"default": True}),
                "preload_naf": ("BOOLEAN", {"default": True}),
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
        force_reload: bool,
    ):
        context = load_pixal3d_context(
            model_path=model_path,
            moge_model_name=moge_model_name,
            device=device,
            low_vram=low_vram,
            preload_naf=preload_naf,
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
                "pixal3d_model": ("PIXAL3D_MODEL", {"forceInput": True}),
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
        pixal3d_model: Any,
        image: torch.Tensor,
        batch_index: int,
        background_color: str,
    ):
        from .pixal3d_runtime import parse_rgb_color

        with pixal3d_model.lock:
            pil_image = image_tensor_to_pil(image, batch_index)
            bg_color = parse_rgb_color(background_color)
            preprocessed = pixal3d_model.pipeline.preprocess_image(pil_image, bg_color=bg_color)
        return (pil_to_image_tensor(preprocessed),)


class Pixal3DImageToGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixal3d_model": ("PIXAL3D_MODEL", {"forceInput": True}),
                "image": ("IMAGE", {}),
                "output_prefix": ("STRING", {"default": "pixal3d"}),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": int(MAX_SEED)},
                ),
                "pipeline_type": (["1024_cascade", "1536_cascade"],),
                "preprocess_image": ("BOOLEAN", {"default": True}),
                "background_color": ("STRING", {"default": "#000000"}),
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

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("glb_path", "preprocessed_image", "camera_json")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "Pixal3D"

    def generate(
        self,
        pixal3d_model: Any,
        image: torch.Tensor,
        output_prefix: str,
        seed: int,
        pipeline_type: str,
        preprocess_image: bool,
        background_color: str,
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
        output_path = make_output_path(_output_directory(), output_prefix, seed)
        settings = sampler_settings or default_sampler_settings()

        result = run_pixal3d_to_glb(
            pixal3d_model,
            pil_image,
            output_path,
            settings,
            seed=seed,
            pipeline_type=pipeline_type,
            preprocess_image=preprocess_image,
            background_color=background_color,
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
        preprocessed_tensor = pil_to_image_tensor(result.preprocessed_image)
        return {
            "ui": {"text": [result.glb_path, camera_json]},
            "result": (result.glb_path, preprocessed_tensor, camera_json),
        }


NODE_CLASS_MAPPINGS = {
    "Pixal3DModelLoader": Pixal3DModelLoader,
    "Pixal3DSamplerSettings": Pixal3DSamplerSettings,
    "Pixal3DPreprocessImage": Pixal3DPreprocessImage,
    "Pixal3DImageToGLB": Pixal3DImageToGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DModelLoader": "Pixal3D Model Loader",
    "Pixal3DSamplerSettings": "Pixal3D Sampler Settings",
    "Pixal3DPreprocessImage": "Pixal3D Preprocess Image",
    "Pixal3DImageToGLB": "Pixal3D Image to GLB",
}
