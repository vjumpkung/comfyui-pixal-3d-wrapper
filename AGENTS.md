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
- `pixal3d_runtime.py`: lazy model loading, model cache, Hugging Face
  cache-first model resolution, NAF attention backend patching, image
  conversion, optional background removal preprocessing, camera estimation,
  Pixal3D pipeline execution, and in-memory GLB export.
- `vendor/Pixal3D`: bundled upstream Pixal3D inference source, requirements,
  and license.
- `requirements.txt`: helper dependency list for Pixal3D-side dependencies.
- `README.md`: user setup and workflow notes.

## Nodes

- `Pixal3D Model Loader`: loads and caches Pixal3D, MoGe, and DINO/NAF models.
- `Pixal3D Background Remover Loader`: loads and caches the optional upstream
  background remover used by preprocessing.
- `Pixal3D Sampler Settings`: optional sampler overrides matching upstream
  Pixal3D defaults.
- `Pixal3D Preprocess Image`: runs Pixal3D-style background removal/crop
  preprocessing with a `PIXAL3D_REMBG_MODEL` and returns a ComfyUI `IMAGE`.
- `Pixal3D Image to 3D`: generates an in-memory `FILE_3D_GLB` and camera
  metadata JSON. It is not an output node and does not save directly; connect
  `model_3d` to ComfyUI's built-in `Save 3D Model` node to write a file.

## Runtime Notes

- Keep heavyweight Pixal3D imports lazy. ComfyUI should be able to import this
  package without loading the model stack.
- The bundled Pixal3D source root is `vendor/Pixal3D`, resolved internally
  relative to `pixal3d_runtime.py`.
- `Pixal3D Model Loader` intentionally does not expose `pixal3d_root`.
- The Pixal3D model loader defaults exposed in ComfyUI are:
  - `model_path`: `TencentARC/Pixal3D`
  - `moge_model_name`: `Ruicheng/moge-2-vitl`
  - `device`: `cuda`
  - `low_vram`: `False`
  - `preload_naf`: `True`
  - `attention_backend`: `flash_attn_3`
  - `sparse_attention_backend`: `auto`
  - `naf_attention_backend`: `auto`
- `Pixal3D Background Remover Loader` also defaults `low_vram` to `False`, so
  the background remover stays CUDA-resident unless the user explicitly enables
  CPU offload.
- To test an external Pixal3D checkout, set `PIXAL3D_ROOT` before starting
  ComfyUI. Do not re-add a normal loader widget for this unless the user asks.
- Hugging Face repo IDs should be resolved to local snapshots before upstream
  `from_pretrained` calls. Check the local cache first, including symlinked
  snapshot files; download only on cache miss or incomplete snapshot.
- `low_vram=False` is the fast path and keeps Pixal3D stage models, DINO/NAF
  conditioning models, MoGe, and background remover on CUDA after load.
  `low_vram=True` is the conservative path and moves models to CUDA only when
  needed, then back to CPU.
- Set `PIXAL3D_PROFILE_LOAD=1` to print cache hits and coarse load timings for
  Pixal3D checkpoint load, image conditioning model construction, CUDA moves,
  NAF preload, MoGe load, and background remover load.
- The wrapper-side BiRefNet patch must keep background-remover inputs on the
  loaded model's device and floating dtype. This avoids float32 input vs fp16
  bias failures when the Hugging Face rembg model loads half-precision weights.
- `Pixal3D Image to 3D` intentionally does not call Pixal3D preprocessing.
  Preprocessing is a separate background-removal workflow using `Pixal3D
  Background Remover Loader` and `Pixal3D Preprocess Image`.
- Pixal3D dense attention backends are `flash_attn_3`, `flash_attn`, `sdpa`,
  `xformers`, `naive`, and `flash_attn_4`. Sparse attention backends are
  `auto`, `flash_attn_3`, `flash_attn`, `xformers`, and `flash_attn_4`; sparse
  attention does not support `sdpa`.
- NAF attention backends are `auto`, `torch`, `flex-fna`, `cutlass-fna`,
  `hopper-fna`, and `blackwell-fna`. On Windows, NATTEN often lacks libnatten,
  so `auto` may need the wrapper-side `torch` fallback for NAF's mismatched QK/V
  head dimensions. The `torch` fallback is slower but avoids unsupported
  NATTEN backends.
- Full generation is serialized with the cached context lock to avoid concurrent
  mutation of shared model state.

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
- Existing vendored compatibility patches support DINOv3 transformer layer
  layouts from both older and newer `transformers` versions. Preserve that
  compatibility if touching DINOv3 feature extraction.
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
print(loader_inputs["low_vram"][1]["default"])
rembg_inputs = mod.NODE_CLASS_MAPPINGS["Pixal3DBackgroundRemoverLoader"].INPUT_TYPES()["required"]
print(rembg_inputs["low_vram"][1]["default"])
PY
```

Expected node keys:

- `Pixal3DBackgroundRemoverLoader`
- `Pixal3DImageTo3D`
- `Pixal3DModelLoader`
- `Pixal3DPreprocessImage`
- `Pixal3DSamplerSettings`

The second printed line from the import check should be `False`, confirming that
`Pixal3D Model Loader` does not expose `pixal3d_root`. The final two printed
lines should also be `False`, confirming both loader `low_vram` defaults stay
GPU-resident.
