# AGENTS.md

## Project

This repository is a ComfyUI custom-node wrapper for Pixal3D. It is intended to
live under `ComfyUI/custom_nodes/comfyui-pixal-3d-wrapper`.

Pixal3D inference source is not bundled. It must be importable in the Python
environment that runs ComfyUI, or `PIXAL3D_SOURCE_PATH` must point to a local
Pixal3D checkout containing `pixal3d/__init__.py`. The ComfyUI loader still does
not expose a `pixal3d_root` input.

Pixal3D inference requires CUDA and the upstream Pixal3D/TRELLIS.2 dependency
stack installed in the same Python environment that runs ComfyUI.

## Important Files

- `__init__.py`: exports ComfyUI node mappings.
- `nodes.py`: defines the ComfyUI nodes and UI-facing input/output contracts.
- `pixal3d_runtime.py`: stable public runtime facade used by `nodes.py`.
- `runtime/`: focused runtime modules for constants/types, Hugging Face
  resolution, Pixal3D source/environment setup, DINO/NAF conditioning,
  preprocessing, model loading, image conversion, camera/sampler helpers,
  pipeline execution, and in-memory GLB export.
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
- Pixal3D source is resolved lazily from the import environment or
  `PIXAL3D_SOURCE_PATH`; do not import Pixal3D at package import time.
- `Pixal3D Model Loader` intentionally does not expose `pixal3d_root`.
- The Pixal3D model loader defaults exposed in ComfyUI are:
  - `model_path`: `TencentARC/Pixal3D`
  - `moge_model_name`: `Ruicheng/moge-2-vitl`
  - `device`: `cuda`
  - `low_vram`: `False`
  - `preload_naf`: `True`
  - `attention_backend`: `sdpa`
  - `sparse_attention_backend`: `auto`
  - `naf_attention_backend`: `auto`
- `Pixal3D Background Remover Loader` also defaults `low_vram` to `False`, so
  the background remover stays CUDA-resident unless the user explicitly enables
  CPU offload.
- Keep external Pixal3D source selection outside the node UI. `PIXAL3D_SOURCE_PATH`
  is the supported source-checkout hook for non-installed Pixal3D trees.
- Hugging Face repo IDs should be resolved to local snapshots before upstream
  `from_pretrained` calls. Check the local cache first, including symlinked
  snapshot files; download only on cache miss or incomplete snapshot.
- MoGe should resolve Hugging Face repo IDs to the cached `model.pt` file before
  calling `MoGeModel.from_pretrained`, because MoGe accepts a checkpoint file
  path and may otherwise perform its own Hub `HEAD` request.
- Image conditioning wrappers should share immutable heavyweight backbones where
  safe: reuse one frozen DINOv3 model per resolved `model_name` while keeping
  each wrapper's own `image_size`, `grid_resolution`, and `proj_grid`; reuse one
  frozen NAF upsampler per device/backend across NAF-enabled conditioning
  wrappers.
- `low_vram=False` is the fast path and keeps Pixal3D stage models, DINO/NAF
  conditioning models, MoGe, and background remover on CUDA after load.
  `low_vram=True` is the conservative path and moves models to CUDA only when
  needed, then back to CPU.
- Load debugging is shown by default: cache hits and load timings for Hugging
  Face snapshot resolution, Pixal3D checkpoint load, shared DINO load/cache
  hits, per-conditioning wrapper construction, shared NAF load/cache hits, CUDA
  moves, NAF preload, MoGe load, and background remover load.
- The tqdm stage progress bar for Pixal3D model loading is also shown by
  default. Set `PIXAL3D_PROFILE_LOAD=0` and `PIXAL3D_LOAD_PROGRESS=0` before
  starting ComfyUI to silence the timing logs and progress bar.
- The wrapper-owned BiRefNet loader must keep background-remover inputs on the
  loaded model's device and floating dtype. This avoids float32 input vs fp16
  bias failures when the Hugging Face rembg model loads half-precision weights.
- `Pixal3D Image to 3D` intentionally does not call Pixal3D preprocessing.
  Preprocessing is a separate background-removal workflow using `Pixal3D
  Background Remover Loader` and `Pixal3D Preprocess Image`.
- Pixal3D dense attention backends are `flash_attn_3`, `flash_attn`, `sdpa`,
  `xformers`, `naive`, and `flash_attn_4`. Sparse attention backends are
  `auto`, `flash_attn_3`, `flash_attn`, `sdpa`, `xformers`, and
  `flash_attn_4`; `auto` follows the dense backend when supported and otherwise
  falls back to `sdpa`.
- NAF attention backends are `auto`, `torch`, `flex-fna`, `cutlass-fna`,
  `hopper-fna`, and `blackwell-fna`. On Windows, NATTEN often lacks libnatten,
  so `auto` may need the wrapper-side `torch` fallback for NAF's mismatched QK/V
  head dimensions. The `torch` fallback is slower but avoids unsupported
  NATTEN backends.
- Full generation is serialized with the cached context lock to avoid concurrent
  mutation of shared model state.
- Non-low-VRAM generation should reassert image-conditioning model residency on
  the target CUDA device before use, since shared DINO/NAF instances may have
  been moved by a separate low-VRAM context.

## Development Rules

- Follow ComfyUI V3 backend conventions:
  - class-based nodes inheriting `io.ComfyNode`
  - `define_schema`
  - `execute`
  - `io.NodeOutput`
  - `ComfyExtension`
  - `comfy_entrypoint`
- Use ComfyUI `IMAGE` tensors as `[B,H,W,C]` in float `[0,1]`.
- Return outputs with `io.NodeOutput`.
- Avoid eager imports of Pixal3D, MoGe, `o_voxel`, or other heavy dependencies
  at module import time.
- Do not reintroduce bundled Pixal3D source into this repository. Prefer
  wrapper-side compatibility code in the focused `runtime/` modules and keep
  `pixal3d_runtime.py` as the public re-export facade.
- Wrapper-side DINOv3 compatibility code supports transformer layer layouts from
  both older and newer `transformers` versions. Preserve that compatibility if
  touching DINOv3 feature extraction.
- Do not run full Pixal3D inference unless the active environment has CUDA and
  the Pixal3D dependency stack installed.

## Verification

Run lightweight checks from the repo root:

```bash
python - <<'PY'
from pathlib import Path

for path in ("__init__.py", "nodes.py", "pixal3d_runtime.py"):
    compile(Path(path).read_text(encoding="utf-8"), path, "exec")
print("syntax ok")
PY
```

```bash
PYTHONPATH=../.. python - <<'PY'
import asyncio
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

async def main():
    extension = await mod.comfy_entrypoint()
    node_list = await extension.get_node_list()
    schemas = {node.GET_SCHEMA().node_id: node.GET_SCHEMA() for node in node_list}
    print(len(schemas), sorted(schemas))
    loader_inputs = {input.id: input for input in schemas["Pixal3DModelLoader"].inputs}
    print("pixal3d_root" in loader_inputs)
    print(loader_inputs["low_vram"].default)
    rembg_inputs = {
        input.id: input
        for input in schemas["Pixal3DBackgroundRemoverLoader"].inputs
    }
    print(rembg_inputs["low_vram"].default)

asyncio.run(main())
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
