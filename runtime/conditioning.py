import os
import sys
import threading
import time
import types
from typing import Any, Dict, Tuple

import torch

from .attention import normalize_naf_attention_backend
from .devices import _device_matches, _first_tensor_device
from .hf import _resolve_hf_snapshot
from .profiling import _log_load_timing, _profile_load_enabled


_DINO_MODEL_CACHE: Dict[str, Any] = {}
_DINO_MODEL_CACHE_LOCK = threading.RLock()
_NAF_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_NAF_MODEL_CACHE_LOCK = threading.Lock()


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
                raise RuntimeError(
                    "NAF return_weights is not supported with this NATTEN version."
                )
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
            out = legacy_attention(
                q, k, v, self.kernel_size, dilation, scale=self.scale
            )
            return rearrange(out, "b h w n d -> b (n d) h w")

        selected_backend = getattr(self, "_pixal3d_naf_attention_backend", "auto")
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


def _load_shared_dinov3_model(model_name: str, profile_load: bool) -> Any:
    with _DINO_MODEL_CACHE_LOCK:
        cached = _DINO_MODEL_CACHE.get(model_name)
        started_at = time.perf_counter()
        if cached is not None:
            _log_load_timing(
                f"DINO shared cache hit for {model_name}",
                started_at,
                profile_load,
            )
            return cached

        from ._compat import image_conditioned_proj

        model = image_conditioned_proj.DINOv3ViTModel.from_pretrained(model_name)
        model.eval()
        model.requires_grad_(False)
        _DINO_MODEL_CACHE[model_name] = model
        _log_load_timing(
            f"DINO shared load for {model_name}",
            started_at,
            profile_load,
        )
        return model


def _device_cache_key(device: Any) -> str:
    try:
        return str(torch.device(device))
    except (RuntimeError, TypeError, ValueError):
        return str(device)


def _load_shared_naf_model(
    device: Any,
    naf_attention_backend: str,
    profile_load: bool,
) -> Any:
    backend = normalize_naf_attention_backend(naf_attention_backend)
    device_key = _device_cache_key(device)
    target_device = torch.device(device)
    key = (device_key, backend)

    with _NAF_MODEL_CACHE_LOCK:
        cached = _NAF_MODEL_CACHE.get(key)
        started_at = time.perf_counter()
        if cached is not None:
            _log_load_timing(
                f"NAF shared cache hit for {device_key}/{backend}",
                started_at,
                profile_load,
                device_key,
            )
            actual_device = _first_tensor_device(cached)
            if actual_device is not None and not _device_matches(
                actual_device, device_key
            ):
                move_started_at = time.perf_counter()
                cached.to(target_device)
                move_label = (
                    "NAF shared CUDA move"
                    if target_device.type == "cuda"
                    else "NAF shared device move"
                )
                _log_load_timing(move_label, move_started_at, profile_load, device_key)
            _patch_naf_model_attention(cached, backend)
            return cached

        from torch import hub as torch_hub

        naf_model = torch_hub.load(
            "valeoai/NAF",
            "naf",
            pretrained=True,
            device=target_device,
            trust_repo=True,
        )
        naf_model.eval()
        naf_model.requires_grad_(False)
        _patch_naf_model_attention(naf_model, backend)
        _NAF_MODEL_CACHE[key] = naf_model
        _log_load_timing(
            f"NAF shared load for {device_key}/{backend}",
            started_at,
            profile_load,
            device_key,
        )
        return naf_model


def _configure_image_cond_naf_attention(model: Any, naf_attention_backend: str) -> None:
    backend = normalize_naf_attention_backend(naf_attention_backend)
    model._pixal3d_naf_attention_backend = backend

    if getattr(model, "_pixal3d_load_naf_patched", False):
        if getattr(model, "naf_model", None) is not None:
            _patch_naf_model_attention(model.naf_model, backend)
        return

    original_load_naf = model._load_naf

    def load_naf_with_backend(*args, **kwargs):
        dino_model = getattr(model, "model", None)
        device = _first_tensor_device(dino_model) if dino_model is not None else None
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.naf_model = _load_shared_naf_model(
            device,
            getattr(model, "_pixal3d_naf_attention_backend", "auto"),
            _profile_load_enabled(),
        )
        if getattr(model, "naf_model", None) is not None:
            _patch_naf_model_attention(
                model.naf_model,
                getattr(model, "_pixal3d_naf_attention_backend", "auto"),
            )
        return None

    model._pixal3d_original_load_naf = original_load_naf
    model._load_naf = load_naf_with_backend
    model._pixal3d_load_naf_patched = True
    if getattr(model, "naf_model", None) is not None:
        _patch_naf_model_attention(model.naf_model, backend)


def build_image_cond_model(
    config: Dict[str, Any],
    naf_attention_backend: str,
    profile_load: bool = False,
    label: str = "image conditioning",
) -> Any:
    from ._compat import image_conditioned_proj

    resolved_config = dict(config)
    model_name = resolved_config.get("model_name")
    shared_model = None
    if isinstance(model_name, str):
        started_at = time.perf_counter()
        resolved_config["model_name"] = _resolve_hf_snapshot(model_name)
        _log_load_timing(
            f"{label} DINO snapshot resolution",
            started_at,
            profile_load,
        )
        shared_model = _load_shared_dinov3_model(
            resolved_config["model_name"],
            profile_load,
        )

    started_at = time.perf_counter()
    if shared_model is None:
        model = image_conditioned_proj.DinoV3ProjFeatureExtractor(**resolved_config)
    else:
        with _DINO_MODEL_CACHE_LOCK:
            original_dinov3_cls = image_conditioned_proj.DINOv3ViTModel

            def from_pretrained_cached(requested_model_name, *args, **kwargs):
                if str(requested_model_name) == str(resolved_config["model_name"]):
                    return shared_model
                return original_dinov3_cls.from_pretrained(
                    requested_model_name,
                    *args,
                    **kwargs,
                )

            image_conditioned_proj.DINOv3ViTModel = types.SimpleNamespace(
                from_pretrained=from_pretrained_cached
            )
            try:
                model = image_conditioned_proj.DinoV3ProjFeatureExtractor(
                    **resolved_config
                )
            finally:
                image_conditioned_proj.DINOv3ViTModel = original_dinov3_cls
    _log_load_timing(
        f"{label} wrapper construction",
        started_at,
        profile_load,
    )
    if getattr(model, "use_naf_upsample", False):
        _configure_image_cond_naf_attention(model, naf_attention_backend)
    model.eval()
    return model
