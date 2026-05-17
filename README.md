# ComfyUI Pixal3D Wrapper

ComfyUI custom nodes for running Pixal3D image-to-3D generation. Pixal3D
inference source is not bundled in this custom node. If Pixal3D is not already
installed or pointed to by `PIXAL3D_SOURCE_PATH`, the loader clones
`TencentARC/Pixal3D` into `.pixal3d_source/Pixal3D` on first use.

## Nodes

- **Pixal3D Model Loader**: loads Pixal3D, MoGe, and DINO/NAF conditioning models.
- **Pixal3D Background Remover Loader**: loads the optional upstream background remover used by preprocessing.
- **Pixal3D Sampler Settings**: optional advanced sampler controls matching the upstream Pixal3D inference defaults.
- **Pixal3D Preprocess Image**: runs Pixal3D background removal/crop preprocessing and returns an `IMAGE`.
- **Pixal3D Image to 3D**: generates an in-memory `FILE_3D_GLB` and camera metadata JSON without saving directly.

## Setup

1. Install TRELLIS.2 and Pixal3D dependencies into the same Python environment used by ComfyUI.
2. Install this wrapper into `ComfyUI/custom_nodes/comfyui-pixal-3d-wrapper`.
3. Optional: set `PIXAL3D_SOURCE_PATH` before starting ComfyUI if you want to use a specific local Pixal3D checkout.

Pixal3D inference requires CUDA. The loader defaults to `device=cuda` and keeps
models cached on CUDA after the first load. Enable `low_vram` only when you need
CPU offload between stages; it reduces VRAM use but adds CPU/CUDA transfer time.

On Windows, if NAF fails with `NATTEN was not built with libnatten`, keep the
loader's `naf_attention_backend` at `auto` or select `torch`. The `torch`
fallback is slower, but supports NAF's mismatched QK/V head dimensions when
NATTEN Flex cannot.

The loader's `attention_backend` controls Pixal3D dense attention and defaults
to `sdpa`, which works without flash-attention installed. It can be set to
`flash_attn_3`, `flash_attn`, `sdpa`, `xformers`, `naive`, or `flash_attn_4`.
`sparse_attention_backend` controls Pixal3D sparse attention; its `auto` default
follows the dense backend when sparse attention supports it and otherwise falls
back to `sdpa`. If you install flash-attention or xformers, you can still
explicitly select `flash_attn`, `flash_attn_3`, `flash_attn_4`, or `xformers`.

Pixal3D also requires the upstream `utils3d` wheel:

```bash
pip install utils3d
```

Example external source setup:

```powershell
git clone https://github.com/TencentARC/Pixal3D.git
$env:PIXAL3D_SOURCE_PATH = "D:\path\to\Pixal3D"
python main.py
```

`PIXAL3D_SOURCE_PATH` can point either to a checkout containing
`pixal3d/__init__.py` or directly to the `pixal3d` package directory.

Automatic source clone can be controlled with:

- `PIXAL3D_AUTO_CLONE=0`: disable auto-clone and require an installed package or `PIXAL3D_SOURCE_PATH`.
- `PIXAL3D_SOURCE_CACHE=/path/to/cache`: change the clone cache directory.
- `PIXAL3D_GIT_URL=https://...`: use a fork instead of `TencentARC/Pixal3D`.
- `PIXAL3D_GIT_REF=branch-or-commit`: check out a specific ref after cloning.

## Runtime Flow

Runtime implementation is split under `runtime/` by responsibility.
`pixal3d_runtime.py` remains as the public facade imported by the ComfyUI nodes.

Two upstream Pixal3D modules that use unstable transformers DINOv3 private
APIs (`pixal3d.modules.image_feature_extractor` and
`pixal3d.trainers.flow_matching.mixins.image_conditioned_proj`) are vendored
under `runtime/_compat/` with transformers >=5 fixes baked in. The wrapper
injects these into `sys.modules` before upstream Pixal3D loads, so the rest
of the inference path (sampler, sparse-voxel ops, decoders, mesh extraction)
still comes from upstream while the version-fragile surface stays in this
repo.

**Pixal3D Preprocess Image** converts one ComfyUI `IMAGE` frame to PIL, runs the
wrapper-owned BiRefNet background remover when the image has no useful alpha
channel, crops around the alpha mask, composites over `background_color`, and
returns a ComfyUI `IMAGE`.

**Pixal3D Image to 3D** converts one ComfyUI `IMAGE` frame to PIL, estimates
camera parameters with MoGe, calls `Pixal3DImageTo3DPipeline.run` with
`preprocess_image=False`, converts the first mesh to an in-memory GLB through
`o_voxel.postprocess.to_glb`, and returns `FILE_3D_GLB` plus camera JSON.

## This Custom Nodes require like a TRELLIS.2 

- Prebuilt wheels for TRELLIS.2 can be used too.
- Example Wheels : https://github.com/visualbruno/ComfyUI-Trellis2/tree/main/wheels
- If you want to use it in runpod : https://console.runpod.io/deploy?template=o5gnyb1fzu&ref=6h6f9kga

## Debugging Load

Model-load cache hits, timings, and the tqdm stage progress bar are shown by
default. To silence them, start ComfyUI with:

```bash
PIXAL3D_PROFILE_LOAD=0 PIXAL3D_LOAD_PROGRESS=0 python main.py
```

## Basic Workflow

1. Add **Pixal3D Model Loader**.
2. Connect its output to **Pixal3D Image to 3D**.
3. Connect a ComfyUI `IMAGE` to **Pixal3D Image to 3D**.
4. Connect `model_3d` to ComfyUI's built-in **Save 3D Model** or **Preview 3D & Animation** node.
5. Queue the graph.

**Pixal3D Image to 3D** does not run background removal/crop preprocessing. To
use that optional step, add **Pixal3D Background Remover Loader**, connect it to
**Pixal3D Preprocess Image**, then connect the preprocessed image to **Pixal3D
Image to 3D**.

Use **Pixal3D Sampler Settings** only when you want to override the upstream
defaults.

## Reference

- https://github.com/TencentARC/Pixal3D
- https://huggingface.co/TencentARC/Pixal3D
