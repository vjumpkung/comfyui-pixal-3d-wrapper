from pathlib import Path
from typing import Any, Dict

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_SOURCE_PATH_ENV = "PIXAL3D_SOURCE_PATH"
PIXAL3D_AUTO_CLONE_ENV = "PIXAL3D_AUTO_CLONE"
PIXAL3D_SOURCE_CACHE_ENV = "PIXAL3D_SOURCE_CACHE"
PIXAL3D_GIT_URL_ENV = "PIXAL3D_GIT_URL"
PIXAL3D_GIT_REF_ENV = "PIXAL3D_GIT_REF"
DEFAULT_PIXAL3D_GIT_URL = "https://github.com/TencentARC/Pixal3D.git"
DEFAULT_PIXAL3D_SOURCE_CACHE = PACKAGE_ROOT / ".pixal3d_source"

DEFAULT_MODEL_PATH = "TencentARC/Pixal3D"
DEFAULT_MOGE_MODEL = "Ruicheng/moge-2-vitl"
DEFAULT_REMBG_MODEL = "ZhengPeng7/BiRefNet"
MAX_SEED = np.iinfo(np.int32).max

DEFAULT_ATTENTION_BACKEND = "sdpa"
DEFAULT_SPARSE_ATTENTION_BACKEND = "sdpa"
PIXAL3D_ATTENTION_BACKENDS = (
    "flash_attn_3",
    "flash_attn",
    "sdpa",
    "xformers",
    "naive",
    "flash_attn_4",
)
PIXAL3D_SPARSE_ATTENTION_BACKENDS = (
    "auto",
    "flash_attn_3",
    "flash_attn",
    "sdpa",
    "xformers",
    "flash_attn_4",
)
NAF_ATTENTION_BACKENDS = (
    "auto",
    "torch",
    "flex-fna",
    "cutlass-fna",
    "hopper-fna",
    "blackwell-fna",
)

IMAGE_COND_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}
