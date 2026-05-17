"""Public runtime API for the Pixal3D ComfyUI wrapper.

Implementation lives in focused modules under :mod:`runtime` so preprocessing,
model loading, source resolution, conditioning, and generation can evolve
independently while node imports remain stable.
"""

from .runtime.attention import (
    apply_pixal3d_attention_backends,
    configure_pixal3d_environment,
    normalize_attention_backend,
    normalize_naf_attention_backend,
    normalize_sparse_attention_backend,
    require_cuda_device,
)
from .runtime.camera import (
    compute_f_pixels,
    distance_from_fov,
    estimate_camera_params,
)
from .runtime.constants import (
    DEFAULT_ATTENTION_BACKEND,
    DEFAULT_MODEL_PATH,
    DEFAULT_MOGE_MODEL,
    DEFAULT_PIXAL3D_GIT_URL,
    DEFAULT_PIXAL3D_SOURCE_CACHE,
    DEFAULT_REMBG_MODEL,
    DEFAULT_SPARSE_ATTENTION_BACKEND,
    IMAGE_COND_CONFIGS,
    MAX_SEED,
    NAF_ATTENTION_BACKENDS,
    PACKAGE_ROOT,
    PIXAL3D_AUTO_CLONE_ENV,
    PIXAL3D_ATTENTION_BACKENDS,
    PIXAL3D_GIT_REF_ENV,
    PIXAL3D_GIT_URL_ENV,
    PIXAL3D_SOURCE_CACHE_ENV,
    PIXAL3D_SOURCE_PATH_ENV,
    PIXAL3D_SPARSE_ATTENTION_BACKENDS,
)
from .runtime.generation import camera_params_to_json, run_pixal3d_to_3d
from .runtime.hf import resolve_moge_model_path, resolve_pixal3d_model_path
from .runtime.image_utils import (
    image_tensor_to_pil,
    parse_rgb_color,
    pil_to_image_tensor,
)
from .runtime.model_loader import load_pixal3d_context
from .runtime.preprocessing import (
    BackgroundRemover,
    load_pixal3d_background_remover_context,
    preprocess_image_with_background_remover,
)
from .runtime.sampler import default_sampler_settings, make_sampler_settings
from .runtime.source import configure_pixal3d_source_path, resolve_pixal3d_source
from .runtime.types import (
    Pixal3DBackgroundRemoverContext,
    Pixal3DContext,
    Pixal3DResult,
    Pixal3DSource,
)


__all__ = [
    "BackgroundRemover",
    "DEFAULT_ATTENTION_BACKEND",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_MOGE_MODEL",
    "DEFAULT_PIXAL3D_GIT_URL",
    "DEFAULT_PIXAL3D_SOURCE_CACHE",
    "DEFAULT_REMBG_MODEL",
    "DEFAULT_SPARSE_ATTENTION_BACKEND",
    "IMAGE_COND_CONFIGS",
    "MAX_SEED",
    "NAF_ATTENTION_BACKENDS",
    "PACKAGE_ROOT",
    "PIXAL3D_AUTO_CLONE_ENV",
    "PIXAL3D_ATTENTION_BACKENDS",
    "PIXAL3D_GIT_REF_ENV",
    "PIXAL3D_GIT_URL_ENV",
    "PIXAL3D_SOURCE_CACHE_ENV",
    "PIXAL3D_SOURCE_PATH_ENV",
    "PIXAL3D_SPARSE_ATTENTION_BACKENDS",
    "Pixal3DBackgroundRemoverContext",
    "Pixal3DContext",
    "Pixal3DResult",
    "Pixal3DSource",
    "apply_pixal3d_attention_backends",
    "camera_params_to_json",
    "compute_f_pixels",
    "configure_pixal3d_environment",
    "configure_pixal3d_source_path",
    "default_sampler_settings",
    "distance_from_fov",
    "estimate_camera_params",
    "image_tensor_to_pil",
    "load_pixal3d_background_remover_context",
    "load_pixal3d_context",
    "make_sampler_settings",
    "normalize_attention_backend",
    "normalize_naf_attention_backend",
    "normalize_sparse_attention_backend",
    "parse_rgb_color",
    "pil_to_image_tensor",
    "preprocess_image_with_background_remover",
    "require_cuda_device",
    "resolve_moge_model_path",
    "resolve_pixal3d_model_path",
    "resolve_pixal3d_source",
    "run_pixal3d_to_3d",
]
