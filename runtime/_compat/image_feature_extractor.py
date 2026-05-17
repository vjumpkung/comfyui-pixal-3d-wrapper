"""Vendored, transformers>=5 compatible replacement for
``pixal3d.modules.image_feature_extractor``.

Upstream ``pixal3d.pipelines.pixal3d_image_to_3d`` imports this module at
module-load time but never instantiates either feature extractor (Pixal3D
uses ``DinoV3ProjFeatureExtractor`` from ``image_conditioned_proj`` instead).
We still ship working classes here so external callers can use them under a
modern transformers install.

The single fragile spot is ``DinoV3FeatureExtractor.extract_features``,
which iterates DINOv3 encoder layers manually. We share the resolver helpers
from :mod:`._dinov3` so the same compat logic applies here and in
:mod:`.image_conditioned_proj`.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ._dinov3 import dinov3_encoder_layers, extract_dinov3_features


class DinoV2FeatureExtractor:
    """DINOv2 feature extractor (uses torch.hub, not transformers)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    @torch.no_grad()
    def __call__(
        self, image: Union[torch.Tensor, List[Image.Image]]
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [
                np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image
            ]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).cuda()
        features = self.model(image, is_training=True)["x_prenorm"]
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens


class DinoV3FeatureExtractor:
    """DINOv3 feature extractor compatible with transformers>=5.

    Upstream's implementation hard-codes ``model.layer``, ``model.embeddings(
    image, bool_masked_pos=None)``, ``model.rope_embeddings(image)``, and the
    ``position_embeddings=`` kwarg on encoder layers. transformers reorganised
    the DINOv3 internals in 5.x; we resolve those access points dynamically.
    """

    def __init__(self, model_name: str, image_size: int = 512):
        from transformers import DINOv3ViTModel  # imported lazily

        self.model_name = model_name
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        return extract_dinov3_features(self.model, image)

    @torch.no_grad()
    def __call__(
        self, image: Union[torch.Tensor, List[Image.Image]]
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image)
            image = [
                i.resize((self.image_size, self.image_size), Image.LANCZOS)
                for i in image
            ]
            image = [
                np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image
            ]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).cuda()
        features = self.extract_features(image)
        return features


__all__ = [
    "DinoV2FeatureExtractor",
    "DinoV3FeatureExtractor",
    "dinov3_encoder_layers",
    "extract_dinov3_features",
]
