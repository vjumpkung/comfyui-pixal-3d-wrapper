import json
import os
import re
import sys
import threading
import time
import types
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_PIXAL3D_ROOT = PACKAGE_ROOT / "vendor" / "Pixal3D"
DEFAULT_MODEL_PATH = "TencentARC/Pixal3D"
DEFAULT_MOGE_MODEL = "Ruicheng/moge-2-vitl"
DEFAULT_REMBG_MODEL = "ZhengPeng7/BiRefNet"
MAX_SEED = np.iinfo(np.int32).max
DEFAULT_ATTENTION_BACKEND = "flash_attn_3"
DEFAULT_SPARSE_ATTENTION_BACKEND = "flash_attn"
PIXAL3D_ATTENTION_BACKENDS = (
    "flash_attn_3",
    "flash_attn",
    "sdpa",
    "xformers",
    "naive",
    "flash_attn_4",
)
PIXAL3D_SPARSE_ATTENTION_BACKENDS = (
    "auto",
    "flash_attn_3",
    "flash_attn",
    "xformers",
    "flash_attn_4",
)
NAF_ATTENTION_BACKENDS = (
    "auto",
    "torch",
    "flex-fna",
    "cutlass-fna",
    "hopper-fna",
    "blackwell-fna",
)

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
    attention_backend: str
    sparse_attention_backend: str
    naf_attention_backend: str
    pipeline: Any
    moge_model: Any
    lock: threading.Lock


@dataclass
class Pixal3DBackgroundRemoverContext:
    root: str
    model_name: str
    device: str
    low_vram: bool
    model: Any
    lock: threading.Lock


@dataclass
class Pixal3DResult:
    glb_data: bytes
    camera_params: Dict[str, float]
    resolution: int


_MODEL_CACHE: Dict[Tuple[str, str, str, str, bool, str, str, str], Pixal3DContext] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_PIXAL3D_RUN_LOCK = threading.Lock()
_REMBG_CACHE: Dict[Tuple[str, str, str, bool], Pixal3DBackgroundRemoverContext] = {}
_REMBG_CACHE_LOCK = threading.Lock()
_HF_SNAPSHOT_CACHE: Dict[Tuple[str, Tuple[str, ...]], str] = {}
_HF_SNAPSHOT_CACHE_LOCK = threading.Lock()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _profile_load_enabled() -> bool:
    return _env_flag("PIXAL3D_PROFILE_LOAD")


def _sync_cuda_for_timing(device: Optional[str]) -> None:
    if not device:
        return
    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def _log_load_timing(
    label: str,
    started_at: float,
    enabled: bool,
    device: Optional[str] = None,
) -> None:
    if not enabled:
        return
    _sync_cuda_for_timing(device)
    elapsed = time.perf_counter() - started_at
    print(f"[Pixal3D] load: {label} took {elapsed:.2f}s")


def _first_tensor_device(module: Any) -> Optional[torch.device]:
    for accessor_name in ("parameters", "buffers"):
        accessor = getattr(module, accessor_name, None)
        if not callable(accessor):
            continue
        try:
            for tensor in accessor():
                return tensor.device
        except Exception:
            continue

    try:
        device = getattr(module, "device", None)
    except Exception:
        return None
    if device is None:
        return None
    try:
        return torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return None


def _first_floating_tensor_dtype(module: Any) -> torch.dtype:
    for accessor_name in ("parameters", "buffers"):
        accessor = getattr(module, accessor_name, None)
        if not callable(accessor):
            continue
        try:
            for tensor in accessor():
                if tensor.is_floating_point():
                    return tensor.dtype
        except Exception:
            continue
    return torch.float32


def _device_matches(actual: torch.device, expected: str) -> bool:
    expected_device = torch.device(expected)
    if actual.type != expected_device.type:
        return False
    if actual.type != "cuda" or expected_device.index is None:
        return True
    return actual.index == expected_device.index


def _warn_if_not_on_device(label: str, module: Any, expected_device: str) -> None:
    actual_device = _first_tensor_device(module)
    if actual_device is None or _device_matches(actual_device, expected_device):
        return
    print(
        "[Pixal3D] warning: expected "
        f"{label} on {expected_device}, found {actual_device}. "
        "This can add CPU/CUDA transfer time during generation."
    )


def _warn_nonresident_pipeline_models(
    pipeline: Any,
    moge_model: Any,
    expected_device: str,
) -> None:
    for name, model in getattr(pipeline, "models", {}).items():
        _warn_if_not_on_device(f"pipeline.models[{name!r}]", model, expected_device)

    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        model = getattr(pipeline, attr, None)
        if model is None:
            continue
        _warn_if_not_on_device(f"pipeline.{attr}", model, expected_device)
        naf_model = getattr(model, "naf_model", None)
        if naf_model is not None:
            _warn_if_not_on_device(f"pipeline.{attr}.naf_model", naf_model, expected_device)

    _warn_if_not_on_device("MoGe model", moge_model, expected_device)


def _path_exists(path: str) -> bool:
    return Path(os.path.expanduser(path)).exists()


def _looks_like_hf_repo_id(value: str) -> bool:
    model_id = value.strip()
    if not model_id or "\\" in model_id or ":" in model_id or _path_exists(model_id):
        return False
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", model_id))


def _snapshot_has_files(snapshot_dir: str, filenames: Tuple[str, ...]) -> bool:
    root = Path(snapshot_dir)
    return all((root / filename).is_file() for filename in filenames)


def _cached_snapshot_from_file(repo_id: str, filename: str) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to load Pixal3D model repositories."
        ) from exc

    cached = try_to_load_from_cache(repo_id, filename)
    if not isinstance(cached, str):
        return None

    file_path = Path(cached)
    if not file_path.is_file():
        return None

    filename_depth = len(Path(filename).parts)
    return str(file_path.parents[filename_depth - 1])


def _resolve_hf_snapshot(
    repo_id: str,
    required_files: Tuple[str, ...] = (),
) -> str:
    model_id = repo_id.strip()
    if not _looks_like_hf_repo_id(model_id):
        if _path_exists(model_id):
            return os.path.abspath(os.path.expanduser(model_id))
        return model_id

    cache_key = (model_id, tuple(sorted(required_files)))
    with _HF_SNAPSHOT_CACHE_LOCK:
        cached = _HF_SNAPSHOT_CACHE.get(cache_key)
        if cached and _snapshot_has_files(cached, required_files):
            return cached

        snapshot_dir = None
        if required_files:
            snapshot_dir = _cached_snapshot_from_file(model_id, required_files[0])
            if snapshot_dir and not _snapshot_has_files(snapshot_dir, required_files):
                snapshot_dir = None
        else:
            try:
                from huggingface_hub import snapshot_download

                snapshot_dir = snapshot_download(model_id, local_files_only=True)
            except Exception:
                snapshot_dir = None

        if snapshot_dir is None:
            from huggingface_hub import snapshot_download

            snapshot_dir = snapshot_download(model_id)

        if required_files and not _snapshot_has_files(snapshot_dir, required_files):
            missing = [
                filename
                for filename in required_files
                if not (Path(snapshot_dir) / filename).is_file()
            ]
            raise RuntimeError(
                f"Hugging Face cache for {model_id} is missing required file(s): "
                f"{', '.join(missing)}"
            )

        _HF_SNAPSHOT_CACHE[cache_key] = snapshot_dir
        return snapshot_dir


def _pixal3d_required_files(snapshot_dir: str) -> Tuple[str, ...]:
    pipeline_config = Path(snapshot_dir) / "pipeline.json"
    if not pipeline_config.is_file():
        return ("pipeline.json",)

    with pipeline_config.open("r", encoding="utf-8") as f:
        args = json.load(f).get("args", {})

    required = ["pipeline.json"]
    for model_name in args.get("models", {}).values():
        model_stem = str(model_name).strip().strip("/")
        if not model_stem:
            continue
        required.append(f"{model_stem}.json")
        required.append(f"{model_stem}.safetensors")
    return tuple(required)


def _resolve_pixal3d_model_path(model_path: str) -> str:
    source = model_path.strip()
    if not _looks_like_hf_repo_id(source):
        if _path_exists(source):
            return os.path.abspath(os.path.expanduser(source))
        return source

    snapshot_dir = _resolve_hf_snapshot(source, ("pipeline.json",))
    required_files = _pixal3d_required_files(snapshot_dir)
    if not _snapshot_has_files(snapshot_dir, required_files):
        snapshot_dir = _resolve_hf_snapshot(source, required_files)
    return snapshot_dir


def resolve_pixal3d_root(pixal3d_root: Optional[str] = None) -> str:
    configured_root = pixal3d_root or os.environ.get("PIXAL3D_ROOT") or BUNDLED_PIXAL3D_ROOT
    return os.path.abspath(os.path.expanduser(str(configured_root).strip()))


def configure_pixal3d_source_path(pixal3d_root: str) -> None:
    if pixal3d_root not in sys.path:
        sys.path.insert(0, pixal3d_root)


def normalize_attention_backend(value: Optional[str]) -> str:
    backend = (
        value
        or os.environ.get("PIXAL3D_ATTENTION_BACKEND")
        or os.environ.get("ATTN_BACKEND")
        or DEFAULT_ATTENTION_BACKEND
    ).strip()
    if backend not in PIXAL3D_ATTENTION_BACKENDS:
        choices = ", ".join(PIXAL3D_ATTENTION_BACKENDS)
        raise ValueError(
            f"Invalid Pixal3D attention backend '{backend}'. Expected one of: {choices}."
        )
    return backend


def normalize_sparse_attention_backend(
    value: Optional[str],
    attention_backend: Optional[str],
) -> str:
    backend = (
        value
        or os.environ.get("PIXAL3D_SPARSE_ATTENTION_BACKEND")
        or os.environ.get("SPARSE_ATTN_BACKEND")
        or "auto"
    ).strip()
    valid_sparse_backends = PIXAL3D_SPARSE_ATTENTION_BACKENDS[1:]
    if backend == "auto":
        dense_backend = normalize_attention_backend(attention_backend)
        if dense_backend in valid_sparse_backends:
            return dense_backend
        return DEFAULT_SPARSE_ATTENTION_BACKEND
    if backend not in valid_sparse_backends:
        choices = ", ".join(PIXAL3D_SPARSE_ATTENTION_BACKENDS)
        raise ValueError(
            f"Invalid Pixal3D sparse attention backend '{backend}'. Expected one of: {choices}."
        )
    return backend


def apply_pixal3d_attention_backends(
    attention_backend: str,
    sparse_attention_backend: str,
) -> None:
    os.environ["ATTN_BACKEND"] = attention_backend
    os.environ["SPARSE_ATTN_BACKEND"] = sparse_attention_backend

    attention_config = sys.modules.get("pixal3d.modules.attention.config")
    if attention_config is not None:
        attention_config.set_backend(attention_backend)

    sparse_config = sys.modules.get("pixal3d.modules.sparse.config")
    if sparse_config is not None:
        sparse_config.set_attn_backend(sparse_attention_backend)


def configure_pixal3d_environment(
    pixal3d_root: str,
    attention_backend: str,
    sparse_attention_backend: str,
) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    apply_pixal3d_attention_backends(attention_backend, sparse_attention_backend)
    os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(
        pixal3d_root, "autotune_cache.json"
    )
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")
    configure_pixal3d_source_path(pixal3d_root)


def require_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        raise RuntimeError("Pixal3D inference requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("Pixal3D inference requires CUDA, but torch.cuda is not available.")


def normalize_naf_attention_backend(value: Optional[str]) -> str:
    backend = (value or os.environ.get("PIXAL3D_NAF_ATTENTION_BACKEND") or "auto").strip()
    if backend not in NAF_ATTENTION_BACKENDS:
        choices = ", ".join(NAF_ATTENTION_BACKENDS)
        raise ValueError(f"Invalid NAF attention backend '{backend}'. Expected one of: {choices}.")
    return backend


def _pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _torch_neighborhood_attention_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: Any,
    dilation: Any,
    scale: float,
) -> torch.Tensor:
    b, h, w, heads, qk_dim = q.shape
    value_dim = v.shape[-1]
    kernel_h, kernel_w = _pair(kernel_size)
    dilation_h, dilation_w = _pair(dilation)
    center_h = kernel_h // 2
    center_w = kernel_w // 2
    chunk_rows = int(os.environ.get("PIXAL3D_NAF_TORCH_CHUNK_ROWS", "16"))
    chunk_rows = max(1, min(chunk_rows, h))

    def offset_slice(
        x: torch.Tensor,
        row_start: int,
        row_end: int,
        rel_h: int,
        rel_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = row_end - row_start
        out = torch.zeros(
            b,
            rows,
            w,
            heads,
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        valid = torch.zeros(1, rows, w, 1, device=x.device, dtype=torch.bool)

        q_row_start = max(row_start, -rel_h)
        q_row_end = min(row_end, h - rel_h)
        q_col_start = max(0, -rel_w)
        q_col_end = min(w, w - rel_w)
        if q_row_start >= q_row_end or q_col_start >= q_col_end:
            return out, valid

        dst_row_start = q_row_start - row_start
        dst_row_end = q_row_end - row_start
        src_row_start = q_row_start + rel_h
        src_row_end = q_row_end + rel_h
        src_col_start = q_col_start + rel_w
        src_col_end = q_col_end + rel_w

        out[:, dst_row_start:dst_row_end, q_col_start:q_col_end] = x[
            :,
            src_row_start:src_row_end,
            src_col_start:src_col_end,
        ]
        valid[:, dst_row_start:dst_row_end, q_col_start:q_col_end] = True
        return out, valid

    outputs = []
    for row_start in range(0, h, chunk_rows):
        row_end = min(row_start + chunk_rows, h)
        rows = row_end - row_start
        q_chunk = q[:, row_start:row_end]

        score_parts = []
        offset_slices = []
        for kh in range(kernel_h):
            rel_h = (kh - center_h) * dilation_h
            for kw in range(kernel_w):
                rel_w = (kw - center_w) * dilation_w
                k_slice, valid = offset_slice(k, row_start, row_end, rel_h, rel_w)
                score = (q_chunk.float() * k_slice.float()).sum(dim=-1) * float(scale)
                score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
                score_parts.append(score)
                offset_slices.append((rel_h, rel_w))

        weights = torch.softmax(torch.stack(score_parts, dim=-1), dim=-1).to(v.dtype)
        chunk_out = torch.zeros(
            b,
            rows,
            w,
            heads,
            value_dim,
            device=v.device,
            dtype=v.dtype,
        )
        for offset_index, (rel_h, rel_w) in enumerate(offset_slices):
            v_slice, _valid = offset_slice(v, row_start, row_end, rel_h, rel_w)
            chunk_out.add_(weights[..., offset_index].unsqueeze(-1) * v_slice)
        outputs.append(chunk_out)

    return torch.cat(outputs, dim=1)


def _patch_naf_model_attention(naf_model: Any, naf_attention_backend: str) -> None:
    upsampler = getattr(naf_model, "upsampler", None)
    if upsampler is None:
        return

    backend = normalize_naf_attention_backend(naf_attention_backend)
    upsampler._pixal3d_naf_attention_backend = backend
    if getattr(upsampler, "_pixal3d_attention_patched", False):
        return

    original_forward = upsampler.forward

    def forward_with_backend(self, q, k, v, image=None, return_weights=False, **kwargs):
        attn_module = sys.modules.get(self.__class__.__module__)
        natten_recent = getattr(attn_module, "NATTEN_RECENT", True)
        legacy_attention = getattr(attn_module, "legacy_attention", None)

        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        dilation = (hq // hk, wq // wk)
        self.dilation = dilation

        from einops import rearrange

        q = rearrange(q, "b (n d) h w -> b h w n d", n=self.num_heads)
        k = self._resize(k, size=(hq, wq), dtype=q.dtype)
        v = self._resize(v, size=(hq, wq), dtype=q.dtype)

        if return_weights:
            if natten_recent or legacy_attention is None:
                raise RuntimeError("NAF return_weights is not supported with this NATTEN version.")
            out, attn_weights = legacy_attention(
                q,
                k,
                v,
                self.kernel_size,
                dilation,
                scale=self.scale,
                return_weights=True,
            )
            return rearrange(out, "b h w n d -> b (n d) h w"), attn_weights

        if not natten_recent and legacy_attention is not None:
            out = legacy_attention(q, k, v, self.kernel_size, dilation, scale=self.scale)
            return rearrange(out, "b h w n d -> b (n d) h w")

        selected_backend = getattr(self, "_pixal3d_naf_attention_backend", "auto")
        # The torch fallback follows the tensor device; it is not a CPU-only path.
        if selected_backend == "torch" or (
            selected_backend in {"auto", "flex-fna"} and q.shape[-1] != v.shape[-1]
        ):
            out = _torch_neighborhood_attention_2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=dilation,
                scale=self.scale,
            )
            return rearrange(out, "b h w n d -> b (n d) h w")

        try:
            from natten import na2d
        except ImportError:
            from natten.functional import na2d

        if selected_backend == "auto":
            selected_backend = None
        try:
            out = na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=dilation,
                stride=1,
                backend=selected_backend,
            )
        except (RuntimeError, ValueError) as exc:
            if q.shape[-1] == v.shape[-1]:
                raise
            message = str(exc)
            if "different head dims" not in message and "head dim" not in message:
                raise
            out = _torch_neighborhood_attention_2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=dilation,
                scale=self.scale,
            )
        return rearrange(out, "b h w n d -> b (n d) h w")

    upsampler._pixal3d_original_forward = original_forward
    upsampler.forward = types.MethodType(forward_with_backend, upsampler)
    upsampler._pixal3d_attention_patched = True


def _configure_image_cond_naf_attention(model: Any, naf_attention_backend: str) -> None:
    backend = normalize_naf_attention_backend(naf_attention_backend)
    model._pixal3d_naf_attention_backend = backend

    if getattr(model, "_pixal3d_load_naf_patched", False):
        if getattr(model, "naf_model", None) is not None:
            _patch_naf_model_attention(model.naf_model, backend)
        return

    original_load_naf = model._load_naf

    def load_naf_with_backend(*args, **kwargs):
        result = original_load_naf(*args, **kwargs)
        if getattr(model, "naf_model", None) is not None:
            _patch_naf_model_attention(
                model.naf_model,
                getattr(model, "_pixal3d_naf_attention_backend", "auto"),
            )
        return result

    model._pixal3d_original_load_naf = original_load_naf
    model._load_naf = load_naf_with_backend
    model._pixal3d_load_naf_patched = True
    if getattr(model, "naf_model", None) is not None:
        _patch_naf_model_attention(model.naf_model, backend)


def build_image_cond_model(config: Dict[str, Any], naf_attention_backend: str) -> Any:
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    resolved_config = dict(config)
    model_name = resolved_config.get("model_name")
    if isinstance(model_name, str):
        resolved_config["model_name"] = _resolve_hf_snapshot(model_name)

    model = DinoV3ProjFeatureExtractor(**resolved_config)
    if getattr(model, "use_naf_upsample", False):
        _configure_image_cond_naf_attention(model, naf_attention_backend)
    model.eval()
    return model


def _patch_rembg_cache_resolution() -> None:
    from pixal3d.pipelines import rembg

    original = getattr(rembg, "BiRefNet", None)
    if original is None or getattr(original, "_pixal3d_dtype_safe", False):
        return

    class CachedBiRefNet(original):
        _pixal3d_cache_wrapped = True
        _pixal3d_dtype_safe = True

        def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
            if getattr(original, "_pixal3d_cache_wrapped", False):
                super().__init__(model_name)
            else:
                super().__init__(_resolve_hf_snapshot(model_name))

        def __call__(self, image: Image.Image) -> Image.Image:
            from torchvision import transforms

            image_size = image.size
            device = _first_tensor_device(self.model)
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = _first_floating_tensor_dtype(self.model)
            input_images = self.transform_image(image).unsqueeze(0).to(
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                preds = self.model(input_images)[-1].sigmoid().float().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image_size)
            image.putalpha(mask)
            return image

    CachedBiRefNet.__name__ = original.__name__
    CachedBiRefNet.__qualname__ = original.__qualname__
    rembg.BiRefNet = CachedBiRefNet


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


def _load_pipeline_without_background_remover(pipeline_cls: Any, model_path: str) -> Any:
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


def load_pixal3d_background_remover_context(
    pixal3d_root: Optional[str] = None,
    model_name: str = DEFAULT_REMBG_MODEL,
    device: str = "cuda",
    low_vram: bool = False,
    force_reload: bool = False,
) -> Pixal3DBackgroundRemoverContext:
    root = resolve_pixal3d_root(pixal3d_root)
    package_init = os.path.join(root, "pixal3d", "__init__.py")
    if not os.path.isdir(root) or not os.path.isfile(package_init):
        raise RuntimeError(
            "Bundled Pixal3D source was not found. Expected a pixal3d package at: "
            f"{root}"
        )

    require_cuda_device(device)
    key = (root, model_name.strip(), device.strip(), bool(low_vram))
    profile_load = _profile_load_enabled()

    with _REMBG_CACHE_LOCK:
        if force_reload:
            _REMBG_CACHE.pop(key, None)
            torch.cuda.empty_cache()
        if key in _REMBG_CACHE:
            if profile_load:
                print("[Pixal3D] load: background remover cache hit")
            return _REMBG_CACHE[key]

        configure_pixal3d_source_path(root)
        _patch_rembg_cache_resolution()

        from pixal3d.pipelines import rembg

        started_at = time.perf_counter()
        model = rembg.BiRefNet(model_name.strip())
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
    pixal3d_root: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    moge_model_name: str = DEFAULT_MOGE_MODEL,
    device: str = "cuda",
    low_vram: bool = False,
    preload_naf: bool = True,
    attention_backend: Optional[str] = None,
    sparse_attention_backend: Optional[str] = None,
    naf_attention_backend: str = "auto",
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
    attention_backend = normalize_attention_backend(attention_backend)
    sparse_attention_backend = normalize_sparse_attention_backend(
        sparse_attention_backend,
        attention_backend,
    )
    naf_attention_backend = normalize_naf_attention_backend(naf_attention_backend)
    profile_load = _profile_load_enabled()
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
                print("[Pixal3D] load: Pixal3D context cache hit")
            return context

        configure_pixal3d_environment(root, attention_backend, sparse_attention_backend)

        try:
            from moge.model.v2 import MoGeModel
            from pixal3d.pipelines import Pixal3DImageTo3DPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import Pixal3D dependencies. Install Pixal3D/TRELLIS.2 "
                "dependencies in the same Python environment ComfyUI uses."
            ) from exc

        _patch_rembg_cache_resolution()

        started_at = time.perf_counter()
        resolved_model_path = _resolve_pixal3d_model_path(model_path.strip())
        pipeline = _load_pipeline_without_background_remover(
            Pixal3DImageTo3DPipeline,
            resolved_model_path,
        )
        _log_load_timing("Pixal3D pipeline checkpoint load", started_at, profile_load)

        started_at = time.perf_counter()
        pipeline.image_cond_model_ss = build_image_cond_model(
            IMAGE_COND_CONFIGS["ss"],
            naf_attention_backend,
        )
        pipeline.image_cond_model_shape_512 = build_image_cond_model(
            IMAGE_COND_CONFIGS["shape_512"],
            naf_attention_backend,
        )
        pipeline.image_cond_model_shape_1024 = build_image_cond_model(
            IMAGE_COND_CONFIGS["shape_1024"],
            naf_attention_backend,
        )
        pipeline.image_cond_model_tex_1024 = build_image_cond_model(
            IMAGE_COND_CONFIGS["tex_1024"],
            naf_attention_backend,
        )
        _log_load_timing("image conditioning model construction", started_at, profile_load)

        started_at = time.perf_counter()
        pipeline.low_vram = bool(low_vram)
        pipeline.to(torch.device(device))
        _move_image_cond_models(pipeline, device, bool(low_vram))
        _log_load_timing("pipeline CUDA move", started_at, profile_load, device)

        if preload_naf:
            started_at = time.perf_counter()
            _preload_naf_models(pipeline, device, bool(low_vram), naf_attention_backend)
            _log_load_timing("NAF preload", started_at, profile_load, device)

        started_at = time.perf_counter()
        resolved_moge_model_name = _resolve_hf_snapshot(moge_model_name.strip())
        try:
            moge_model = MoGeModel.from_pretrained(resolved_moge_model_name).to(device)
        except Exception:
            if resolved_moge_model_name == moge_model_name.strip():
                raise
            moge_model = MoGeModel.from_pretrained(moge_model_name.strip()).to(device)
        moge_model.eval()
        _log_load_timing("MoGe load", started_at, profile_load, device)
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


def _export_glb_to_bytes(glb: Any, extension_webp: bool) -> bytes:
    buffer = BytesIO()
    try:
        exported = glb.export(
            buffer,
            file_type="glb",
            extension_webp=bool(extension_webp),
        )
    except TypeError:
        try:
            exported = glb.export(file_type="glb", extension_webp=bool(extension_webp))
        except TypeError:
            exported = glb.export(file_type="glb")

    if isinstance(exported, bytes):
        return exported
    if isinstance(exported, bytearray):
        return bytes(exported)

    data = buffer.getvalue()
    if data:
        return data

    if exported is None:
        raise RuntimeError("GLB export produced no data.")
    if isinstance(exported, str):
        return exported.encode("utf-8")
    raise RuntimeError(f"Unexpected GLB export result: {type(exported).__name__}.")


def run_pixal3d_to_3d(
    context: Pixal3DContext,
    image: Image.Image,
    sampler_settings: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
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

    with _PIXAL3D_RUN_LOCK, context.lock:
        apply_pixal3d_attention_backends(
            context.attention_backend,
            context.sparse_attention_backend,
        )
        torch.manual_seed(int(seed))

        image_for_generation = image.convert("RGB")

        update("camera", 0, 3)
        camera_params = estimate_camera_params(
            image_for_generation,
            context.moge_model,
            device=context.device,
            mesh_scale=float(mesh_scale),
            extend_pixel=int(extend_pixel),
            image_resolution=int(image_resolution),
        )

        update("generate", 1, 3)
        with torch.no_grad():
            mesh_list, (_shape_slat, _tex_slat, res) = pipeline.run(
                image_for_generation,
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

            update("export", 2, 3)
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
            glb_data = _export_glb_to_bytes(glb, bool(extension_webp))

        torch.cuda.empty_cache()
        update("done", 3, 3)

    return Pixal3DResult(
        glb_data=glb_data,
        camera_params=camera_params,
        resolution=int(res),
    )



def camera_params_to_json(camera_params: Dict[str, float], resolution: int) -> str:
    payload = {**camera_params, "resolution": int(resolution)}
    return json.dumps(payload, sort_keys=True)
