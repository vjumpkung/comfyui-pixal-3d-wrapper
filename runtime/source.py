import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from .constants import (
    DEFAULT_PIXAL3D_GIT_URL,
    DEFAULT_PIXAL3D_SOURCE_CACHE,
    PIXAL3D_AUTO_CLONE_ENV,
    PIXAL3D_GIT_REF_ENV,
    PIXAL3D_GIT_URL_ENV,
    PIXAL3D_SOURCE_CACHE_ENV,
    PIXAL3D_SOURCE_PATH_ENV,
)
from .types import Pixal3DSource


_PIXAL3D_SOURCE_LOCK = threading.Lock()


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path or os.curdir))


def _pixal3d_source_from_package_dir(package_dir: Path) -> Pixal3DSource:
    package_dir = package_dir.resolve()
    return Pixal3DSource(
        root=str(package_dir.parent),
        package_dir=str(package_dir),
    )


def _pixal3d_source_from_module(module: Any) -> Optional[Pixal3DSource]:
    module_paths = getattr(module, "__path__", None)
    if module_paths:
        return _pixal3d_source_from_package_dir(Path(next(iter(module_paths))))

    module_file = getattr(module, "__file__", None)
    if module_file:
        return _pixal3d_source_from_package_dir(Path(module_file).resolve().parent)
    return None


def _resolve_explicit_pixal3d_source(source_path: str) -> Pixal3DSource:
    root = Path(os.path.expanduser(source_path)).resolve()
    package_dir = root / "pixal3d"
    if (package_dir / "__init__.py").is_file():
        return _pixal3d_source_from_package_dir(package_dir)
    if root.name == "pixal3d" and (root / "__init__.py").is_file():
        return _pixal3d_source_from_package_dir(root)
    raise RuntimeError(
        f"{PIXAL3D_SOURCE_PATH_ENV} must point to a Pixal3D source root containing "
        f"'pixal3d/__init__.py', or to the pixal3d package directory itself. Got: {root}"
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _pixal3d_source_cache_dir() -> Path:
    configured = os.environ.get(PIXAL3D_SOURCE_CACHE_ENV)
    if configured:
        return Path(os.path.expanduser(configured)).resolve()
    return DEFAULT_PIXAL3D_SOURCE_CACHE.resolve()


def _run_git(args: list[str], cwd: Optional[Path] = None) -> None:
    try:
        subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Pixal3D source was not found and automatic clone requires git. "
            f"Install git, install Pixal3D manually, or set {PIXAL3D_SOURCE_PATH_ENV}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "Pixal3D source was not found and automatic clone failed. "
            f"Set {PIXAL3D_SOURCE_PATH_ENV} to a local Pixal3D checkout. "
            f"git {' '.join(args)} failed: {details}"
        ) from exc


def _clone_pixal3d_source() -> Optional[Pixal3DSource]:
    if not _env_flag(PIXAL3D_AUTO_CLONE_ENV, default=True):
        return None

    cache_root = _pixal3d_source_cache_dir()
    checkout = cache_root / "Pixal3D"
    package_dir = checkout / "pixal3d"
    if (package_dir / "__init__.py").is_file():
        return _pixal3d_source_from_package_dir(package_dir)

    git_url = os.environ.get(PIXAL3D_GIT_URL_ENV, DEFAULT_PIXAL3D_GIT_URL).strip()
    git_ref = os.environ.get(PIXAL3D_GIT_REF_ENV, "").strip()
    cache_root.mkdir(parents=True, exist_ok=True)

    if checkout.exists() and not (checkout / ".git").is_dir():
        raise RuntimeError(
            f"Automatic Pixal3D source clone target already exists and is not a git "
            f"checkout: {checkout}. Remove it or set {PIXAL3D_SOURCE_CACHE_ENV}."
        )

    if not checkout.exists():
        print(f"[Pixal3D] source: cloning {git_url} to {checkout}")  # noqa: T201
        _run_git(["clone", "--depth", "1", git_url, str(checkout)])

    if git_ref:
        print(f"[Pixal3D] source: checking out {git_ref}")  # noqa: T201
        _run_git(["fetch", "--depth", "1", "origin", git_ref], cwd=checkout)
        _run_git(["checkout", "FETCH_HEAD"], cwd=checkout)

    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(
            "Pixal3D automatic clone completed but did not contain "
            f"pixal3d/__init__.py at: {checkout}"
        )
    return _pixal3d_source_from_package_dir(package_dir)


def _prepend_sys_path(path: str) -> None:
    normalized = _norm_path(path)
    sys.path[:] = [entry for entry in sys.path if _norm_path(entry) != normalized]
    sys.path.insert(0, path)


def _existing_pixal3d_source() -> Optional[Pixal3DSource]:
    module = sys.modules.get("pixal3d")
    if module is None:
        return None
    return _pixal3d_source_from_module(module)


def _find_importable_pixal3d_source() -> Pixal3DSource:
    import importlib.util

    spec = importlib.util.find_spec("pixal3d")
    if spec is None or spec.submodule_search_locations is None:
        cloned_source = _clone_pixal3d_source()
        if cloned_source is not None:
            return cloned_source
        raise RuntimeError(
            "Pixal3D source is not bundled with this node anymore and automatic clone "
            f"is disabled by {PIXAL3D_AUTO_CLONE_ENV}=0. Install or clone "
            "TencentARC/Pixal3D into the Python environment used by ComfyUI, or set "
            f"{PIXAL3D_SOURCE_PATH_ENV} to a checkout that contains pixal3d/__init__.py."
        )
    return _pixal3d_source_from_package_dir(
        Path(next(iter(spec.submodule_search_locations)))
    )


def resolve_pixal3d_source() -> Pixal3DSource:
    explicit_source = os.environ.get(PIXAL3D_SOURCE_PATH_ENV)
    if explicit_source:
        return _resolve_explicit_pixal3d_source(explicit_source)

    existing = _existing_pixal3d_source()
    if existing is not None:
        return existing

    return _find_importable_pixal3d_source()


def configure_pixal3d_source_path() -> str:
    with _PIXAL3D_SOURCE_LOCK:
        source = resolve_pixal3d_source()
        existing = _existing_pixal3d_source()
        if existing is not None and _norm_path(existing.root) != _norm_path(
            source.root
        ):
            raise RuntimeError(
                "A pixal3d package is already imported from a different source. "
                "Restart ComfyUI before changing PIXAL3D_SOURCE_PATH."
            )

        _prepend_sys_path(source.root)

        return source.root
