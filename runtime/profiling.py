import os
import time
from typing import Any, Optional

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _profile_load_enabled() -> bool:
    return _env_flag("PIXAL3D_PROFILE_LOAD", default=True)


def _load_progress_enabled() -> bool:
    if "PIXAL3D_LOAD_PROGRESS" in os.environ:
        return _env_flag("PIXAL3D_LOAD_PROGRESS")
    return _profile_load_enabled()


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
    print(f"[Pixal3D] load: {label} took {elapsed:.2f}s")  # noqa: T201


class _LoadProgress:
    def __init__(self, label: str, total: int, enabled: bool):
        self.label = label
        self.total = total
        self.enabled = enabled
        self._bar = None

    def __enter__(self) -> "_LoadProgress":
        if not self.enabled:
            return self
        try:
            from tqdm.auto import tqdm
        except Exception:
            return self
        self._bar = tqdm(
            total=self.total,
            desc=self.label,
            unit="stage",
            dynamic_ncols=True,
            leave=True,
        )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self._bar is not None:
            self._bar.close()

    def step(self, label: str) -> None:
        if self._bar is not None:
            self._bar.set_description_str(f"{self.label}: {label}")

    def advance(self, label: Optional[str] = None) -> None:
        if self._bar is None:
            return
        if label:
            self._bar.set_postfix_str(label)
        self._bar.update(1)
