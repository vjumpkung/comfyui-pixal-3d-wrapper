from io import BytesIO
from typing import Any, Dict, Optional

import torch
from comfy_api.latest import ComfyExtension, Types, io

from .pixal3d_runtime import (
    DEFAULT_ATTENTION_BACKEND,
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


PIXAL3D_MODEL = io.Custom("PIXAL3D_MODEL")
PIXAL3D_REMBG_MODEL = io.Custom("PIXAL3D_REMBG_MODEL")
PIXAL3D_SAMPLER_SETTINGS = io.Custom("PIXAL3D_SAMPLER_SETTINGS")


def _progress_callback() -> Any:
    if ProgressBar is None:
        return None
    pbar = ProgressBar(3)

    def update(_label: str, step: int, _total: int) -> None:
        if step > 0:
            pbar.update(1)

    return update


def _glb_file_output(glb_data: bytes) -> Types.File3D:
    return Types.File3D(BytesIO(glb_data), "glb")


class Pixal3DModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Pixal3DModelLoader",
            display_name="Pixal3D Model Loader",
            category="Pixal3D",
            description="Load and cache the bundled Pixal3D pipeline, MoGe, and DINO/NAF conditioning models.",
            inputs=[
                io.String.Input(
                    "model_path",
                    default=DEFAULT_MODEL_PATH,
                    placeholder="TencentARC/Pixal3D or local model folder",
                ),
                io.String.Input(
                    "moge_model_name",
                    default=DEFAULT_MOGE_MODEL,
                    placeholder="Ruicheng/moge-2-vitl",
                ),
                io.String.Input("device", default="cuda", placeholder="cuda"),
                io.Boolean.Input("low_vram", default=False),
                io.Boolean.Input("preload_naf", default=True),
                io.Combo.Input(
                    "attention_backend",
                    options=list(PIXAL3D_ATTENTION_BACKENDS),
                    default=DEFAULT_ATTENTION_BACKEND,
                ),
                io.Combo.Input(
                    "sparse_attention_backend",
                    options=list(PIXAL3D_SPARSE_ATTENTION_BACKENDS),
                    default="auto",
                ),
                io.Combo.Input(
                    "naf_attention_backend",
                    options=list(NAF_ATTENTION_BACKENDS),
                    default="auto",
                ),
                io.Boolean.Input("force_reload", default=False),
            ],
            outputs=[PIXAL3D_MODEL.Output("pixal3d_model")],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        model_path: str,
        moge_model_name: str,
        device: str,
        low_vram: bool,
        preload_naf: bool,
        attention_backend: str,
        sparse_attention_backend: str,
        naf_attention_backend: str,
        force_reload: bool,
    ) -> io.NodeOutput:
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
        return io.NodeOutput(context)


class Pixal3DBackgroundRemoverLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Pixal3DBackgroundRemoverLoader",
            display_name="Pixal3D Background Remover Loader",
            category="Pixal3D",
            description="Load and cache the optional Pixal3D background remover used by preprocessing.",
            inputs=[
                io.String.Input(
                    "model_name",
                    default=DEFAULT_REMBG_MODEL,
                    placeholder="ZhengPeng7/BiRefNet",
                ),
                io.String.Input("device", default="cuda", placeholder="cuda"),
                io.Boolean.Input("low_vram", default=False),
                io.Boolean.Input("force_reload", default=False),
            ],
            outputs=[PIXAL3D_REMBG_MODEL.Output("rembg_model")],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        device: str,
        low_vram: bool,
        force_reload: bool,
    ) -> io.NodeOutput:
        context = load_pixal3d_background_remover_context(
            model_name=model_name,
            device=device,
            low_vram=low_vram,
            force_reload=force_reload,
        )
        return io.NodeOutput(context)


class Pixal3DSamplerSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Pixal3DSamplerSettings",
            display_name="Pixal3D Sampler Settings",
            category="Pixal3D",
            description="Override Pixal3D sampler parameters while keeping upstream defaults as the baseline.",
            inputs=[
                io.Float.Input("ss_guidance_strength", default=7.5, min=0.0, max=30.0, step=0.1),
                io.Float.Input("ss_guidance_rescale", default=0.7, min=0.0, max=1.0, step=0.05),
                io.Int.Input("ss_sampling_steps", default=12, min=1, max=100),
                io.Float.Input("ss_rescale_t", default=5.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input(
                    "shape_slat_guidance_strength",
                    default=7.5,
                    min=0.0,
                    max=30.0,
                    step=0.1,
                ),
                io.Float.Input(
                    "shape_slat_guidance_rescale",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                io.Int.Input("shape_slat_sampling_steps", default=12, min=1, max=100),
                io.Float.Input("shape_slat_rescale_t", default=3.0, min=0.0, max=20.0, step=0.1),
                io.Float.Input(
                    "tex_slat_guidance_strength",
                    default=1.0,
                    min=0.0,
                    max=30.0,
                    step=0.1,
                ),
                io.Float.Input(
                    "tex_slat_guidance_rescale",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                io.Int.Input("tex_slat_sampling_steps", default=12, min=1, max=100),
                io.Float.Input("tex_slat_rescale_t", default=3.0, min=0.0, max=20.0, step=0.1),
            ],
            outputs=[PIXAL3D_SAMPLER_SETTINGS.Output("sampler_settings")],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
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
    ) -> io.NodeOutput:
        return io.NodeOutput(
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
            )
        )


class Pixal3DPreprocessImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Pixal3DPreprocessImage",
            display_name="Pixal3D Preprocess Image",
            category="Pixal3D",
            description="Run Pixal3D background removal and crop preprocessing on a ComfyUI image.",
            inputs=[
                PIXAL3D_REMBG_MODEL.Input("rembg_model"),
                io.Image.Input("image"),
                io.Int.Input("batch_index", default=0, min=0, max=4096),
                io.String.Input("background_color", default="#000000"),
            ],
            outputs=[io.Image.Output("preprocessed_image")],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
        rembg_model: Any,
        image: torch.Tensor,
        batch_index: int,
        background_color: str,
    ) -> io.NodeOutput:
        pil_image = image_tensor_to_pil(image, batch_index)
        preprocessed = preprocess_image_with_background_remover(
            rembg_model,
            pil_image,
            background_color=background_color,
        )
        return io.NodeOutput(pil_to_image_tensor(preprocessed))


class Pixal3DImageTo3D(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Pixal3DImageTo3D",
            display_name="Pixal3D Image to 3D",
            category="Pixal3D",
            description="Generate an in-memory GLB and camera metadata JSON from one ComfyUI image.",
            inputs=[
                PIXAL3D_MODEL.Input("pixal3d_model"),
                io.Image.Input("image"),
                io.Int.Input("seed", default=42, min=0, max=int(MAX_SEED)),
                io.Combo.Input(
                    "pipeline_type",
                    options=["1024_cascade", "1536_cascade"],
                    default="1024_cascade",
                ),
                io.Int.Input("batch_index", default=0, min=0, max=4096),
                io.Float.Input("mesh_scale", default=1.0, min=0.01, max=10.0, step=0.01),
                io.Int.Input("extend_pixel", default=0, min=-256, max=256),
                io.Int.Input("image_resolution", default=512, min=128, max=2048),
                io.Int.Input("max_num_tokens", default=49152, min=1024, max=262144),
                io.Int.Input("decimation_target", default=200000, min=1000, max=2000000),
                io.Int.Input("texture_size", default=2048, min=256, max=8192),
                io.Boolean.Input("remesh", default=True),
                io.Int.Input("remesh_band", default=1, min=0, max=8),
                io.Int.Input("remesh_project", default=0, min=0, max=8),
                io.Boolean.Input("extension_webp", default=True),
                PIXAL3D_SAMPLER_SETTINGS.Input("sampler_settings", optional=True),
            ],
            outputs=[
                io.File3DGLB.Output("model_3d"),
                io.String.Output("camera_json"),
            ],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls,
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
    ) -> io.NodeOutput:
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
        return io.NodeOutput(_glb_file_output(result.glb_data), camera_json)


class Pixal3DExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Pixal3DModelLoader,
            Pixal3DBackgroundRemoverLoader,
            Pixal3DSamplerSettings,
            Pixal3DPreprocessImage,
            Pixal3DImageTo3D,
        ]


async def comfy_entrypoint() -> Pixal3DExtension:
    return Pixal3DExtension()


__all__ = [
    "Pixal3DModelLoader",
    "Pixal3DBackgroundRemoverLoader",
    "Pixal3DSamplerSettings",
    "Pixal3DPreprocessImage",
    "Pixal3DImageTo3D",
    "Pixal3DExtension",
    "comfy_entrypoint",
]
