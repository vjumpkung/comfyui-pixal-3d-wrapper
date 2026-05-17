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


def install() -> None:
    """Inject vendored modules under their upstream dotted names.

    Idempotent and thread-safe. Must run *after* the upstream Pixal3D root has
    been added to ``sys.path`` (so the real parent packages remain importable)
    and *before* the first ``import pixal3d.<submodule>`` that would otherwise
    load upstream's version of the targeted leaves. Python's import machinery
    checks ``sys.modules`` for fully-qualified submodule names, so injecting
    only the leaves is sufficient — the real parent packages still load
    normally off ``sys.path``. Setting parent attributes is best-effort: if
    the parent isn't loaded yet, Python's importer fills it in itself the
    next time the submodule is fetched via ``from parent import leaf``.
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
            sys.modules[upstream_name] = local_modules[local_name]
            parent_name, _, attr = upstream_name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, attr, local_modules[local_name])
        _INSTALLED = True


def is_installed() -> bool:
    return _INSTALLED


__all__ = ["install", "is_installed"]
