import json
import threading
from io import BytesIO
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image

from .attention import apply_pixal3d_attention_backends
from .camera import estimate_camera_params
from .model_loader import _move_image_cond_models
from .sampler import default_sampler_settings
from .types import Pixal3DContext, Pixal3DResult


_PIXAL3D_RUN_LOCK = threading.Lock()


def _export_glb_to_bytes(glb: Any, extension_webp: bool) -> bytes:
    buffer = BytesIO()
    try:
        exported = glb.export(
            buffer,
            file_type="glb",
            extension_webp=bool(extension_webp),
        )
    except TypeError:
        try:
            exported = glb.export(file_type="glb", extension_webp=bool(extension_webp))
        except TypeError:
            exported = glb.export(file_type="glb")

    if isinstance(exported, bytes):
        return exported
    if isinstance(exported, bytearray):
        return bytes(exported)

    data = buffer.getvalue()
    if data:
        return data

    if exported is None:
        raise RuntimeError("GLB export produced no data.")
    if isinstance(exported, str):
        return exported.encode("utf-8")
    raise RuntimeError(f"Unexpected GLB export result: {type(exported).__name__}.")


def run_pixal3d_to_3d(
    context: Pixal3DContext,
    image: Image.Image,
    sampler_settings: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    decimation_target: int = 200000,
    texture_size: int = 2048,
    remesh: bool = True,
    remesh_band: int = 1,
    remesh_project: int = 0,
    extension_webp: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Pixal3DResult:
    import o_voxel

    settings = sampler_settings or default_sampler_settings()
    pipeline = context.pipeline

    def update(label: str, step: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(label, step, total)

    with _PIXAL3D_RUN_LOCK, context.lock:
        apply_pixal3d_attention_backends(
            context.attention_backend,
            context.sparse_attention_backend,
        )
        torch.manual_seed(int(seed))
        if not context.low_vram:
            _move_image_cond_models(pipeline, context.device, False)

        image_for_generation = image.convert("RGB")

        update("camera", 0, 3)
        camera_params = estimate_camera_params(
            image_for_generation,
            context.moge_model,
            device=context.device,
            mesh_scale=float(mesh_scale),
            extend_pixel=int(extend_pixel),
            image_resolution=int(image_resolution),
        )

        update("generate", 1, 3)
        with torch.inference_mode():
            mesh_list, (_shape_slat, _tex_slat, res) = pipeline.run(
                image_for_generation,
                camera_params=camera_params,
                seed=int(seed),
                sparse_structure_sampler_params=settings["sparse_structure"],
                shape_slat_sampler_params=settings["shape_slat"],
                tex_slat_sampler_params=settings["tex_slat"],
                preprocess_image=False,
                return_latent=True,
                pipeline_type=pipeline_type,
                max_num_tokens=int(max_num_tokens),
            )

            mesh = mesh_list[0]

            update("export", 2, 3)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=res,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(decimation_target),
                texture_size=int(texture_size),
                remesh=bool(remesh),
                remesh_band=int(remesh_band),
                remesh_project=int(remesh_project),
                use_tqdm=True,
            )

            rotation = np.array(
                [
                    [-1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            glb.apply_transform(rotation)
            glb_data = _export_glb_to_bytes(glb, bool(extension_webp))

        torch.cuda.empty_cache()
        update("done", 3, 3)

    return Pixal3DResult(
        glb_data=glb_data,
        camera_params=camera_params,
        resolution=int(res),
    )


def camera_params_to_json(camera_params: Dict[str, float], resolution: int) -> str:
    payload = {**camera_params, "resolution": int(resolution)}
    return json.dumps(payload, sort_keys=True)
