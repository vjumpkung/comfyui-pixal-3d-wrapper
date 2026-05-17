import threading
import time
from typing import Any, Dict, Tuple

import torch
from PIL import Image

from .attention import (
    apply_pixal3d_attention_backends,
    configure_pixal3d_environment,
    normalize_attention_backend,
    normalize_naf_attention_backend,
    normalize_sparse_attention_backend,
    require_cuda_device,
)
from .conditioning import _configure_image_cond_naf_attention, build_image_cond_model
from .constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MOGE_MODEL,
    IMAGE_COND_CONFIGS,
)
from .devices import _warn_nonresident_pipeline_models
from .hf import resolve_moge_model_path, resolve_pixal3d_model_path
from .profiling import (
    _LoadProgress,
    _load_progress_enabled,
    _log_load_timing,
    _profile_load_enabled,
)
from .source import configure_pixal3d_source_path
from .types import Pixal3DContext


_MODEL_CACHE: Dict[Tuple[str, str, str, str, bool, str, str, str], Pixal3DContext] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class _DisabledBackgroundRemover:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, _device: Any) -> "_DisabledBackgroundRemover":
        return self

    def cuda(self) -> "_DisabledBackgroundRemover":
        return self

    def cpu(self) -> "_DisabledBackgroundRemover":
        return self

    def __call__(self, _image: Image.Image) -> Image.Image:
        raise RuntimeError(
            "Pixal3D background removal is disabled in the 3D model loader. "
            "Use Pixal3D Background Remover Loader with Pixal3D Preprocess Image."
        )


def _load_pipeline_without_background_remover(
    pipeline_cls: Any, model_path: str
) -> Any:
    from pixal3d.pipelines import rembg

    original_birefnet = getattr(rembg, "BiRefNet", None)
    if original_birefnet is None:
        return pipeline_cls.from_pretrained(model_path)

    rembg.BiRefNet = _DisabledBackgroundRemover
    try:
        pipeline = pipeline_cls.from_pretrained(model_path)
    finally:
        rembg.BiRefNet = original_birefnet
    pipeline.rembg_model = None
    return pipeline


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


def _preload_naf_models(
    pipeline: Any,
    device: str,
    low_vram: bool,
    naf_attention_backend: str,
) -> None:
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
        _configure_image_cond_naf_attention(model, naf_attention_backend)
        model._load_naf()
        if low_vram:
            model.cpu()


def load_pixal3d_context(
    model_path: str = DEFAULT_MODEL_PATH,
    moge_model_name: str = DEFAULT_MOGE_MODEL,
    device: str = "cuda",
    low_vram: bool = False,
    preload_naf: bool = True,
    attention_backend: str | None = None,
    sparse_attention_backend: str | None = None,
    naf_attention_backend: str = "auto",
    force_reload: bool = False,
) -> Pixal3DContext:
    root = configure_pixal3d_source_path()
    require_cuda_device(device)
    attention_backend = normalize_attention_backend(attention_backend)
    sparse_attention_backend = normalize_sparse_attention_backend(
        sparse_attention_backend,
        attention_backend,
    )
    naf_attention_backend = normalize_naf_attention_backend(naf_attention_backend)
    profile_load = _profile_load_enabled()
    progress_load = _load_progress_enabled()
    key = (
        root,
        model_path.strip(),
        moge_model_name.strip(),
        device.strip(),
        bool(low_vram),
        attention_backend,
        sparse_attention_backend,
        naf_attention_backend,
    )

    with _MODEL_CACHE_LOCK:
        if force_reload:
            _MODEL_CACHE.pop(key, None)
            torch.cuda.empty_cache()
        if key in _MODEL_CACHE:
            context = _MODEL_CACHE[key]
            apply_pixal3d_attention_backends(
                context.attention_backend,
                context.sparse_attention_backend,
            )
            if profile_load:
                print("[Pixal3D] load: Pixal3D context cache hit")  # noqa: T201
            return context

        configure_pixal3d_environment(attention_backend, sparse_attention_backend)

        try:
            from moge.model.v2 import MoGeModel
            from pixal3d.pipelines import Pixal3DImageTo3DPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import Pixal3D dependencies. Install Pixal3D/TRELLIS.2 "
                "dependencies in the same Python environment ComfyUI uses."
            ) from exc

        progress_total = 9 + int(bool(preload_naf))
        with _LoadProgress(
            "Pixal3D model load", progress_total, progress_load
        ) as progress:
            progress.step("resolving Pixal3D checkpoint")
            started_at = time.perf_counter()
            resolved_model_path = resolve_pixal3d_model_path(model_path.strip())
            _log_load_timing(
                "Pixal3D model snapshot resolution", started_at, profile_load
            )
            progress.advance("Pixal3D checkpoint resolved")

            progress.step("loading Pixal3D checkpoint")
            started_at = time.perf_counter()
            pipeline = _load_pipeline_without_background_remover(
                Pixal3DImageTo3DPipeline,
                resolved_model_path,
            )
            _log_load_timing(
                "Pixal3D pipeline checkpoint load", started_at, profile_load
            )
            progress.advance("Pixal3D checkpoint loaded")

            started_at = time.perf_counter()
            progress.step("building sparse-structure conditioning")
            pipeline.image_cond_model_ss = build_image_cond_model(
                IMAGE_COND_CONFIGS["ss"],
                naf_attention_backend,
                profile_load,
                "ss image conditioning",
            )
            progress.advance("sparse-structure conditioning ready")

            progress.step("building shape_512 conditioning")
            pipeline.image_cond_model_shape_512 = build_image_cond_model(
                IMAGE_COND_CONFIGS["shape_512"],
                naf_attention_backend,
                profile_load,
                "shape_512 image conditioning",
            )
            progress.advance("shape_512 conditioning ready")

            progress.step("building shape_1024 conditioning")
            pipeline.image_cond_model_shape_1024 = build_image_cond_model(
                IMAGE_COND_CONFIGS["shape_1024"],
                naf_attention_backend,
                profile_load,
                "shape_1024 image conditioning",
            )
            progress.advance("shape_1024 conditioning ready")

            progress.step("building texture conditioning")
            pipeline.image_cond_model_tex_1024 = build_image_cond_model(
                IMAGE_COND_CONFIGS["tex_1024"],
                naf_attention_backend,
                profile_load,
                "tex_1024 image conditioning",
            )
            _log_load_timing(
                "image conditioning model construction", started_at, profile_load
            )
            progress.advance("texture conditioning ready")

            progress.step("moving Pixal3D pipeline to device")
            started_at = time.perf_counter()
            pipeline.low_vram = bool(low_vram)
            pipeline.to(torch.device(device))
            _move_image_cond_models(pipeline, device, bool(low_vram))
            _log_load_timing("pipeline CUDA move", started_at, profile_load, device)
            progress.advance("Pixal3D pipeline moved")

            if preload_naf:
                progress.step("preloading shared NAF")
                started_at = time.perf_counter()
                _preload_naf_models(
                    pipeline, device, bool(low_vram), naf_attention_backend
                )
                _log_load_timing("NAF preload", started_at, profile_load, device)
                progress.advance("NAF ready")

            progress.step("resolving MoGe checkpoint")
            started_at = time.perf_counter()
            resolved_moge_model_name = resolve_moge_model_path(moge_model_name.strip())
            _log_load_timing("MoGe checkpoint resolution", started_at, profile_load)
            progress.advance("MoGe checkpoint resolved")

            progress.step("loading MoGe")
            started_at = time.perf_counter()
            moge_model = MoGeModel.from_pretrained(resolved_moge_model_name).to(device)
            moge_model.eval()
            _log_load_timing("MoGe load", started_at, profile_load, device)
            progress.advance("MoGe loaded")
        if not low_vram:
            _warn_nonresident_pipeline_models(pipeline, moge_model, device.strip())

        context = Pixal3DContext(
            root=root,
            model_path=model_path.strip(),
            moge_model_name=moge_model_name.strip(),
            device=device.strip(),
            low_vram=bool(low_vram),
            attention_backend=attention_backend,
            sparse_attention_backend=sparse_attention_backend,
            naf_attention_backend=naf_attention_backend,
            pipeline=pipeline,
            moge_model=moge_model,
            lock=threading.Lock(),
        )
        _MODEL_CACHE[key] = context
        return context
