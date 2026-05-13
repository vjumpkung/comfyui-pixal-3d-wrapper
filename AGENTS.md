# AGENTS.md

## Project

This repository is a ComfyUI custom-node wrapper for Pixal3D. It is intended to
live under `ComfyUI/custom_nodes/comfyui-pixal-3d-wrapper`.

Pixal3D inference source is bundled under `vendor/Pixal3D`, so the ComfyUI node
does not require a separate `~/Documents/Pixal3D` checkout or a `pixal3d_root`
loader input.

Pixal3D inference requires CUDA and the upstream Pixal3D/TRELLIS.2 dependency
stack installed in the same Python environment that runs ComfyUI.

## Important Files

- `__init__.py`: exports ComfyUI node mappings.
- `nodes.py`: defines the ComfyUI nodes and UI-facing input/output contracts.
- `pixal3d_runtime.py`: lazy model loading, model cache, image conversion,
  camera estimation, Pixal3D pipeline execution, and GLB export.
- `vendor/Pixal3D`: bundled upstream Pixal3D inference source, requirements,
  and license.
- `requirements.txt`: helper dependency list for Pixal3D-side dependencies.
- `README.md`: user setup and workflow notes.

## Nodes

- `Pixal3D Model Loader`: loads and caches Pixal3D, MoGe, and DINO/NAF models.
- `Pixal3D Sampler Settings`: optional sampler overrides matching upstream
  Pixal3D defaults.
- `Pixal3D Preprocess Image`: runs Pixal3D preprocessing and returns a ComfyUI
  `IMAGE`.
- `Pixal3D Image to GLB`: output node that writes
  `ComfyUI/output/pixal3d/*.glb` and returns the GLB path, preprocessed image,
  and camera metadata JSON.

## Runtime Notes

- Keep heavyweight Pixal3D imports lazy. ComfyUI should be able to import this
  package without loading the model stack.
- The bundled Pixal3D source root is `vendor/Pixal3D`, resolved internally
  relative to `pixal3d_runtime.py`.
- `Pixal3D Model Loader` intentionally does not expose `pixal3d_root`.
- The loader defaults exposed in ComfyUI are:
  - `model_path`: `TencentARC/Pixal3D`
  - `moge_model_name`: `Ruicheng/moge-2-vitl`
  - `device`: `cuda`
- To test an external Pixal3D checkout, set `PIXAL3D_ROOT` before starting
  ComfyUI. Do not re-add a normal loader widget for this unless the user asks.
- `low_vram=True` keeps Pixal3D behavior conservative and moves conditioning
  models on demand.
- Full generation is serialized with the cached context lock to avoid concurrent
  mutation of shared model state.
- If an image has already gone through `Pixal3D Preprocess Image`, set
  `preprocess_image=false` on `Pixal3D Image to GLB`.

## Development Rules

- Follow ComfyUI backend conventions:
  - class-based nodes
  - `INPUT_TYPES`
  - `RETURN_TYPES`
  - `RETURN_NAMES`
  - `FUNCTION`
  - `CATEGORY`
  - `NODE_CLASS_MAPPINGS`
  - `NODE_DISPLAY_NAME_MAPPINGS`
- Use ComfyUI `IMAGE` tensors as `[B,H,W,C]` in float `[0,1]`.
- Return single outputs as one-item tuples.
- Avoid eager imports of Pixal3D, MoGe, `o_voxel`, or other heavy dependencies
  at module import time.
- Keep vendored Pixal3D code under `vendor/Pixal3D`. Avoid editing vendored files
  unless a compatibility patch is needed; prefer wrapper-side changes in
  `pixal3d_runtime.py`.
- Do not run full Pixal3D inference unless the active environment has CUDA and
  the Pixal3D dependency stack installed.

## Verification

Run lightweight checks from the repo root:

```bash
python -m py_compile __init__.py nodes.py pixal3d_runtime.py
```

```bash
python - <<'PY'
import importlib.util
import sys
from pathlib import Path

root = Path.cwd()
name = "comfyui-pixal-3d-wrapper"
spec = importlib.util.spec_from_file_location(
    name,
    root / "__init__.py",
    submodule_search_locations=[str(root)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)
print(len(mod.NODE_CLASS_MAPPINGS), sorted(mod.NODE_CLASS_MAPPINGS))
loader_inputs = mod.NODE_CLASS_MAPPINGS["Pixal3DModelLoader"].INPUT_TYPES()["required"]
print("pixal3d_root" in loader_inputs)
PY
```

Expected node keys:

- `Pixal3DImageToGLB`
- `Pixal3DModelLoader`
- `Pixal3DPreprocessImage`
- `Pixal3DSamplerSettings`

The second printed line from the import check should be `False`, confirming that
`Pixal3D Model Loader` does not expose `pixal3d_root`.
