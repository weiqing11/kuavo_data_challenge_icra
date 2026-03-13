"""Diffusion-new style DiT implementation."""

from kuavo_train.wrapper.policy.diffusion_idp3.action_generator.dit.dit_1d_adaln import (
    DiT,
    DiT_B,
    DiT_L,
    DiT_S,
    DiT_XL,
)

__all__ = ["DiT", "DiT_XL", "DiT_L", "DiT_B", "DiT_S"]
