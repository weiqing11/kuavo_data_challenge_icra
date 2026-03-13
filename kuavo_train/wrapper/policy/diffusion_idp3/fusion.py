"""用于 RGB 与点云特征的基础 FiLM 融合模块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class Fusion(nn.Module):
    """
    在 RGB patch token 与点云视角 token 之间执行基础 FiLM 融合。

    必需输入：
        rgb_tokens:   [BS, V, P, D]
        point_tokens: [BS, V, C_pc]

    两种输入都必须提供；任意一个为 None 时会抛出 ValueError。

    融合流程（基础 FiLM）：
        1) 用点云特征生成 gamma / beta：
           gamma: [BS, V, D]
           beta:  [BS, V, D]

        2) 在 patch 维扩展：
           gamma.unsqueeze(2): [BS, V, 1, D]
           beta.unsqueeze(2):  [BS, V, 1, D]

        3) 与 RGB token 在 P 维广播计算：
           fused = (1 + gamma) * rgb + beta
    """

    def __init__(
        self,
        token_dim: int,
        point_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(token_dim, point_dim)

        self.pc_to_film = nn.Sequential(
            nn.Linear(point_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, token_dim * 2),
        )

    def forward(
        self,
        rgb_tokens: torch.Tensor | None,
        point_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        参数：
            rgb_tokens:
                RGB patch token，形状 [BS, V, P, D]
            point_tokens:
                点云视角特征，形状 [BS, V, C_pc]

        返回：
            fused_tokens:
                [BS, V, P, D]

        异常：
            ValueError:
                当 rgb_tokens 或 point_tokens 为 None；
                或输入张量形状不兼容时抛出。
        """
        # ------------------------------------------------------------------
        # 严格要求：两种模态必须同时存在
        # ------------------------------------------------------------------
        if rgb_tokens is None or point_tokens is None:
            raise ValueError(
                "Both rgb_tokens and point_tokens must be provided. "
                f"Got rgb_tokens is None: {rgb_tokens is None}, "
                f"point_tokens is None: {point_tokens is None}."
            )

        if rgb_tokens.dim() != 4:
            raise ValueError(
                f"rgb_tokens must have shape [BS, V, P, D], but got {tuple(rgb_tokens.shape)}"
            )
        if point_tokens.dim() != 3:
            raise ValueError(
                f"point_tokens must have shape [BS, V, C_pc], but got {tuple(point_tokens.shape)}"
            )

        bs, num_views, num_patches, token_dim = rgb_tokens.shape
        bs_pc, num_views_pc, point_dim = point_tokens.shape

        if token_dim != self.token_dim:
            raise ValueError(
                f"Expected rgb token dim {self.token_dim}, but got {token_dim}"
            )
        if point_dim != self.point_dim:
            raise ValueError(
                f"Expected point token dim {self.point_dim}, but got {point_dim}"
            )
        if bs != bs_pc or num_views != num_views_pc:
            raise ValueError(
                "rgb_tokens and point_tokens must have aligned [BS, V] dimensions, "
                f"but got rgb {tuple(rgb_tokens.shape)} and point {tuple(point_tokens.shape)}"
            )

        gamma, beta = self.pc_to_film(point_tokens).chunk(2, dim=-1)  # [BS, V, D], [BS, V, D]
        gamma = gamma.unsqueeze(2)  # [BS, V, 1, D]
        beta = beta.unsqueeze(2)    # [BS, V, 1, D]

        # 在 P 维广播执行基础 FiLM，输出 [BS, V, P, D]
        return (1.0 + gamma) * rgb_tokens + beta
