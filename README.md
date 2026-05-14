# ComfyUI Pixal3D Wrapper

ComfyUI custom nodes for running Pixal3D image-to-3D generation. The Pixal3D
inference source is bundled in this custom node under `vendor/Pixal3D`, so the
loader does not require a separate `~/Documents/Pixal3D` path.

## Nodes

- **Pixal3D Model Loader**: loads Pixal3D, MoGe, and DINO/NAF conditioning models.
- **Pixal3D Background Remover Loader**: loads the optional upstream background remover used by preprocessing.
- **Pixal3D Sampler Settings**: optional advanced sampler controls matching the upstream Pixal3D inference defaults.
- **Pixal3D Preprocess Image**: runs Pixal3D background removal/crop preprocessing and returns an `IMAGE`.
- **Pixal3D Image to 3D**: generates an in-memory `FILE_3D_GLB` and camera metadata JSON without saving directly.

## Setup

1. Install TRELLIS.2 and Pixal3D dependencies into the same Python environment used by ComfyUI.
2. Install this wrapper into `ComfyUI/custom_nodes/comfyui-pixal-3d-wrapper`.

Pixal3D inference requires CUDA. The loader defaults to `device=cuda` and keeps
models cached after the first load.

On Windows, if NAF fails with `NATTEN was not built with libnatten`, keep the
loader's `naf_attention_backend` at `auto` or select `torch`. The `torch`
fallback is slower, but supports NAF's mismatched QK/V head dimensions when
NATTEN Flex cannot.

The loader's `attention_backend` controls Pixal3D dense attention and can be set
to `flash_attn_3`, `flash_attn`, `sdpa`, `xformers`, `naive`, or
`flash_attn_4`. `sparse_attention_backend` controls Pixal3D sparse attention;
its `auto` default follows the dense backend when sparse attention supports it,
otherwise it falls back to `flash_attn`. Sparse Pixal3D attention does not have
an `sdpa` implementation, so use `flash_attn`, `flash_attn_3`, `flash_attn_4`,
or `xformers` there.

Pixal3D also requires the upstream `utils3d` wheel:

```bash
pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
```

The bundled Pixal3D source path is resolved relative to this wrapper. If you need
to test a different Pixal3D checkout, set `PIXAL3D_ROOT` in the environment
before starting ComfyUI.

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
