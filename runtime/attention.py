import os
import sys
from typing import Optional

import torch

from .constants import (
    DEFAULT_ATTENTION_BACKEND,
    DEFAULT_SPARSE_ATTENTION_BACKEND,
    NAF_ATTENTION_BACKENDS,
    PACKAGE_ROOT,
    PIXAL3D_ATTENTION_BACKENDS,
    PIXAL3D_SPARSE_ATTENTION_BACKENDS,
)
from .source import configure_pixal3d_source_path


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
    attention_backend: str,
    sparse_attention_backend: str,
) -> None:
    configure_pixal3d_source_path()
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    apply_pixal3d_attention_backends(attention_backend, sparse_attention_backend)
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(PACKAGE_ROOT / "autotune_cache.json"),
    )
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")


def require_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        raise RuntimeError("Pixal3D inference requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Pixal3D inference requires CUDA, but torch.cuda is not available."
        )


def normalize_naf_attention_backend(value: Optional[str]) -> str:
    backend = (
        value or os.environ.get("PIXAL3D_NAF_ATTENTION_BACKEND") or "auto"
    ).strip()
    if backend not in NAF_ATTENTION_BACKENDS:
        choices = ", ".join(NAF_ATTENTION_BACKENDS)
        raise ValueError(
            f"Invalid NAF attention backend '{backend}'. Expected one of: {choices}."
        )
    return backend
