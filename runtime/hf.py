import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from .profiling import _log_load_timing, _profile_load_enabled


_HF_SNAPSHOT_CACHE: Dict[Tuple[str, Tuple[str, ...]], str] = {}
_HF_SNAPSHOT_CACHE_LOCK = threading.Lock()


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


def _hf_cache_root() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        configured = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if configured:
            return Path(os.path.expanduser(configured))
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(os.path.expanduser(hf_home)) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cache_dir(repo_id: str) -> Path:
    return _hf_cache_root() / f"models--{repo_id.replace('/', '--')}"


def _scan_cached_snapshot_file(repo_id: str, filename: str) -> Optional[str]:
    repo_cache = _hf_repo_cache_dir(repo_id)
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    candidates = []
    for snapshot_dir in snapshots_dir.iterdir():
        if not snapshot_dir.is_dir():
            continue
        candidate = snapshot_dir / filename
        if candidate.is_file():
            candidates.append(candidate)
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _cached_snapshot_from_file(repo_id: str, filename: str) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to load Pixal3D model repositories."
        ) from exc

    cached = try_to_load_from_cache(repo_id, filename)
    if not isinstance(cached, str):
        cached = _scan_cached_snapshot_file(repo_id, filename)
        if cached is None:
            return None

    file_path = Path(cached)
    if not file_path.is_file():
        return None

    filename_depth = len(Path(filename).parts)
    return str(file_path.parents[filename_depth - 1])


def _cached_hf_file(repo_id: str, filename: str) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to load Pixal3D model repositories."
        ) from exc

    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str) and Path(cached).is_file():
        return cached
    return _scan_cached_snapshot_file(repo_id, filename)


def _resolve_hf_file(repo_id: str, filename: str) -> str:
    source = repo_id.strip()
    expanded = Path(os.path.expanduser(source))
    if not _looks_like_hf_repo_id(source):
        if expanded.is_dir():
            candidate = expanded / filename
            if candidate.is_file():
                return str(candidate.resolve())
            raise RuntimeError(
                f"Expected {filename} inside local model directory: {expanded}"
            )
        if expanded.exists():
            return str(expanded.resolve())
        return source

    profile_load = _profile_load_enabled()
    started_at = time.perf_counter()
    cached = _cached_hf_file(source, filename)
    if cached:
        _log_load_timing(
            f"Hugging Face file cache hit for {source}/{filename}",
            started_at,
            profile_load,
        )
        return cached

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to load Pixal3D model repositories."
        ) from exc

    path = hf_hub_download(
        repo_id=source,
        filename=filename,
        repo_type="model",
    )
    _log_load_timing(
        f"Hugging Face file download for {source}/{filename}",
        started_at,
        profile_load,
    )
    return path


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
    profile_load = _profile_load_enabled()
    started_at = time.perf_counter()
    with _HF_SNAPSHOT_CACHE_LOCK:
        cached = _HF_SNAPSHOT_CACHE.get(cache_key)
        if cached and _snapshot_has_files(cached, required_files):
            _log_load_timing(
                f"Hugging Face snapshot cache hit for {model_id}",
                started_at,
                profile_load,
            )
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
        _log_load_timing(
            f"Hugging Face snapshot resolution for {model_id}",
            started_at,
            profile_load,
        )
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


def resolve_pixal3d_model_path(model_path: str) -> str:
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


def resolve_moge_model_path(model_name: str) -> str:
    return _resolve_hf_file(model_name, "model.pt")
