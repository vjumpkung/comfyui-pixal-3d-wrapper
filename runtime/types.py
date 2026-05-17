import threading
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Pixal3DSource:
    root: str
    package_dir: str


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
