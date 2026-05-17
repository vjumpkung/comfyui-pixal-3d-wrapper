"""DINOv3 encoder-iteration shim compatible with transformers >=5.

Upstream Pixal3D's DINOv3 wrappers iterate ``self.model.layer`` and call each
encoder block with a ``position_embeddings=`` kwarg. transformers shuffled
those internals between 4.x and 5.x; the encoder layers may now live under
``model.encoder.layer``, ``model.layers``, ``model.encoder.layers``, etc.,
and some layer classes return ``(hidden_states, ...)`` tuples instead of a
bare tensor.

These helpers resolve the path dynamically and unwrap the tuple. They are
the single source of truth for DINOv3 feature extraction across this vendor
layer.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F


def dinov3_encoder_layers(model: Any) -> Iterable[Any]:
    """Return the encoder layer ModuleList of a DINOv3 model regardless of
    the transformers version's nesting convention."""
    candidates = (
        getattr(model, "layer", None),
        getattr(getattr(model, "model", None), "layer", None),
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "encoder", None), "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise AttributeError(
        "DINOv3ViTModel encoder layers were not found. Expected one of "
        "model.layer, model.model.layer, model.encoder.layer, or a layers "
        "alias."
    )


def extract_dinov3_features(model: Any, image: torch.Tensor) -> torch.Tensor:
    """Replicate upstream's pre-norm encoder pass, with transformers >=5
    layout fixes and tuple-output unwrapping."""
    image = image.to(model.embeddings.patch_embeddings.weight.dtype)
    hidden_states = model.embeddings(image, bool_masked_pos=None)
    position_embeddings = model.rope_embeddings(image)

    for layer_module in dinov3_encoder_layers(model):
        hidden_states = layer_module(
            hidden_states,
            position_embeddings=position_embeddings,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

    return F.layer_norm(hidden_states, hidden_states.shape[-1:])


__all__ = ["dinov3_encoder_layers", "extract_dinov3_features"]
