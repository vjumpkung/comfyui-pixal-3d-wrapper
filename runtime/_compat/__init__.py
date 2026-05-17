"""Transformers >=5 compat layer for Pixal3D.

This package owns the inference-side Pixal3D modules that touch unstable
private transformers APIs (DINOv3 internals). Upstream Pixal3D's copies of
these modules are version-fragile; we vendor them here, fix them for
transformers >=5, and inject them into ``sys.modules`` under their upstream
dotted names BEFORE the rest of upstream Pixal3D is imported. Upstream's
pipeline code then transparently consumes our compatible versions.

The two upstream modules replaced are:
- ``pixal3d.modules.image_feature_extractor`` (imported at module-load time
  by ``pixal3d.pipelines.pixal3d_image_to_3d``).
- ``pixal3d.trainers.flow_matching.mixins.image_conditioned_proj`` (imported
  by this wrapper to construct ``DinoV3ProjFeatureExtractor``).

Call :func:`install` exactly once before any ``pixal3d.*`` import.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Tuple

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False

_TARGET_MODULES: Tuple[Tuple[str, str], ...] = (
    ("pixal3d.modules.image_feature_extractor", "image_feature_extractor"),
    (
        "pixal3d.trainers.flow_matching.mixins.image_conditioned_proj",
        "image_conditioned_proj",
    ),
)


def _ensure_namespace_package(dotted: str) -> None:
    """Make sure each parent of ``dotted`` exists in ``sys.modules`` as a
    namespace package, so attribute lookups (e.g. ``pixal3d.modules`` after a
    submodule injection) resolve cleanly even if upstream hasn't loaded the
    real parent yet.
    """
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent in sys.modules:
            continue
        module = types.ModuleType(parent)
        module.__path__ = []
        sys.modules[parent] = module


def install() -> None:
    """Inject vendored modules under their upstream dotted names.

    Idempotent and thread-safe. Must run before the first ``import pixal3d``
    in the process. Re-importing the wrapper modules after install() is a
    no-op.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from . import image_conditioned_proj as _vendored_icp
        from . import image_feature_extractor as _vendored_ife

        local_modules = {
            "image_feature_extractor": _vendored_ife,
            "image_conditioned_proj": _vendored_icp,
        }
        for upstream_name, local_name in _TARGET_MODULES:
            _ensure_namespace_package(upstream_name)
            sys.modules[upstream_name] = local_modules[local_name]
            parent_name, _, attr = upstream_name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, attr, local_modules[local_name])
        _INSTALLED = True


def is_installed() -> bool:
    return _INSTALLED


__all__ = ["install", "is_installed"]
