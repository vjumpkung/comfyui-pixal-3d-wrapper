# ComfyUI Pixal3D Wrapper

ComfyUI custom nodes for running Pixal3D image-to-GLB generation. The Pixal3D
inference source is bundled in this custom node under `vendor/Pixal3D`, so the
loader does not require a separate `~/Documents/Pixal3D` path.

## Nodes

- **Pixal3D Model Loader**: loads Pixal3D, MoGe, and DINO/NAF conditioning models.
- **Pixal3D Sampler Settings**: optional advanced sampler controls matching the upstream Pixal3D inference defaults.
- **Pixal3D Preprocess Image**: runs Pixal3D background removal/crop preprocessing and returns an `IMAGE`.
- **Pixal3D Image to GLB**: output node that generates a GLB, returns the GLB path, the preprocessed image, and camera metadata JSON.

## Setup

1. Install TRELLIS.2 and Pixal3D dependencies into the same Python environment used by ComfyUI.
2. Install this wrapper into `ComfyUI/custom_nodes/comfyui-pixal-3d-wrapper`.

Pixal3D inference requires CUDA. The loader defaults to `device=cuda` and keeps
models cached after the first load.

Pixal3D also requires the upstream `utils3d` wheel:

```bash
pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
```

The bundled Pixal3D source path is resolved relative to this wrapper. If you need
to test a different Pixal3D checkout, set `PIXAL3D_ROOT` in the environment
before starting ComfyUI.

## Basic Workflow

1. Add **Pixal3D Model Loader**.
2. Connect its output to **Pixal3D Image to GLB**.
3. Connect a ComfyUI `IMAGE` to **Pixal3D Image to GLB**.
4. Queue the graph.

Generated files are written to `ComfyUI/output/pixal3d/*.glb`.

Use **Pixal3D Sampler Settings** only when you want to override the upstream
defaults. If you pass an image that was already processed by **Pixal3D Preprocess
Image**, set `preprocess_image` to `false` on **Pixal3D Image to GLB**.
