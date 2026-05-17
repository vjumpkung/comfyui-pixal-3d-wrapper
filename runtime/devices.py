from typing import Any, Optional

import torch


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
    print(  # noqa: T201
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
            _warn_if_not_on_device(
                f"pipeline.{attr}.naf_model", naf_model, expected_device
            )

    _warn_if_not_on_device("MoGe model", moge_model, expected_device)
