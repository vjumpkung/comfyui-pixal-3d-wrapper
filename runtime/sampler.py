from typing import Dict


def default_sampler_settings() -> Dict[str, Dict[str, float]]:
    return {
        "sparse_structure": {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        "shape_slat": {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        "tex_slat": {
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
    }


def make_sampler_settings(
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
) -> Dict[str, Dict[str, float]]:
    return {
        "sparse_structure": {
            "steps": int(ss_sampling_steps),
            "guidance_strength": float(ss_guidance_strength),
            "guidance_rescale": float(ss_guidance_rescale),
            "rescale_t": float(ss_rescale_t),
        },
        "shape_slat": {
            "steps": int(shape_slat_sampling_steps),
            "guidance_strength": float(shape_slat_guidance_strength),
            "guidance_rescale": float(shape_slat_guidance_rescale),
            "rescale_t": float(shape_slat_rescale_t),
        },
        "tex_slat": {
            "steps": int(tex_slat_sampling_steps),
            "guidance_strength": float(tex_slat_guidance_strength),
            "guidance_rescale": float(tex_slat_guidance_rescale),
            "rescale_t": float(tex_slat_rescale_t),
        },
    }
