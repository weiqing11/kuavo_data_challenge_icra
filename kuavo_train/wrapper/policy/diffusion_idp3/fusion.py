"""Fusion modules for RGB and point cloud features."""

from __future__ import annotations

import torch
import torch.nn as nn


class RgbPointCloudFusion(nn.Module):
    def __init__(self, token_dim: int, pc_dim: int) -> None:
        """
        Purpose:
            Fuse RGB tokens with point cloud features into a unified token sequence.
        Inputs (constructor):
            token_dim: int, shared token dimension for RGB tokens.
            pc_dim: int, input dimension of point cloud features per view.
        Outputs (constructor):
            None.
        Forward input shapes:
            rgb_tokens: [B*S, N_cam, P, token_dim] or None.
            pc_tokens: [B*S, N_cam, pc_dim] or None.
        Forward output shape:
            fused_tokens: [B*S, N_cam, P + 1, token_dim] if both inputs exist,
                          otherwise returns the available token sequence.
        """
        super().__init__()
        self.token_dim = token_dim
        self.pc_dim = pc_dim

        self.pc_proj = nn.Linear(pc_dim, token_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(
        self,
        rgb_tokens: torch.Tensor | None,
        pc_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Purpose:
            Produce fused tokens by combining RGB token context and point cloud features.
        Inputs:
            rgb_tokens: Tensor [B*S, N_cam, P, token_dim] or None.
            pc_tokens: Tensor [B*S, N_cam, pc_dim] or None.
        Outputs:
            fused_tokens: Tensor [B*S, N_cam, P + 1, token_dim] when both inputs exist,
                          otherwise a passthrough of the available tokens.
        """
        if rgb_tokens is None and pc_tokens is None:
            raise ValueError("At least one of rgb_tokens or pc_tokens must be provided.")
        if rgb_tokens is None:
            # Only point cloud features: return as a single token per view.
            pc_proj = self.pc_proj(pc_tokens)
            return pc_proj.unsqueeze(2)
        if pc_tokens is None:
            return rgb_tokens

        # === [FUSION][REVIEW] RGB + PointCloud feature fusion ===
        # RGB summary: [B*S, N_cam, token_dim]
        rgb_summary = rgb_tokens.mean(dim=2)
        # Point cloud projection: [B*S, N_cam, token_dim]
        pc_proj = self.pc_proj(pc_tokens)
        # Fusion token: [B*S, N_cam, token_dim]
        fusion_token = self.fusion_mlp(torch.cat([rgb_summary, pc_proj], dim=-1))
        # Append the fusion token to the RGB token sequence.
        fused_tokens = torch.cat([rgb_tokens, fusion_token.unsqueeze(2)], dim=2)
        return fused_tokens
