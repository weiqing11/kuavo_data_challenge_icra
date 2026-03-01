"""Diffusion Transformer (DiT) for action generation with flexible camera handling."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Helper Functions
# ==============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Purpose:
        Apply adaptive layer normalization modulation.
    Inputs:
        x: Tensor [B, T, D], activations to modulate.
        shift: Tensor [B, D], shift parameters.
        scale: Tensor [B, D], scale parameters.
    Outputs:
        modulated: Tensor [B, T, D].
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ==============================================================================
# Embedders
# ==============================================================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        """
        Purpose:
            Embed diffusion timesteps using sinusoidal features followed by an MLP.
        Inputs (constructor):
            hidden_size: int, output embedding dimension.
            frequency_embedding_size: int, sine/cosine base dimension.
        Outputs (constructor):
            None.
        Forward input shape:
            t: [B].
        Forward output shape:
            emb: [B, hidden_size].
        """
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Purpose:
            Build sinusoidal timestep embeddings.
        Inputs:
            t: Tensor [B], timesteps.
            dim: int, embedding dimension.
            max_period: int, max period for frequencies.
        Outputs:
            emb: Tensor [B, dim].
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Produce timestep embeddings with an MLP.
        Inputs:
            t: Tensor [B].
        Outputs:
            emb: Tensor [B, hidden_size].
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ConditionEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        """
        Purpose:
            Embed condition vectors for AdaLN modulation.
        Inputs (constructor):
            input_dim: int, input feature dimension.
            hidden_size: int, output embedding dimension.
        Outputs (constructor):
            None.
        Forward input shape:
            y: [B, input_dim].
        Forward output shape:
            emb: [B, hidden_size].
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Map condition vectors to AdaLN embeddings.
        Inputs:
            y: Tensor [B, input_dim].
        Outputs:
            emb: Tensor [B, hidden_size].
        """
        return self.mlp(y)


# ==============================================================================
# Attention Modules
# ==============================================================================

class AttentionSDPA(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, num_heads: int = 8) -> None:
        """
        Purpose:
            Multi-head attention using scaled dot-product attention.
        Inputs (constructor):
            query_dim: int, dimension of query tokens.
            context_dim: int or None, dimension of context tokens (defaults to query_dim).
            num_heads: int, number of attention heads.
        Outputs (constructor):
            None.
        Forward input shapes:
            x: [B, N, query_dim].
            context: [B, M, context_dim] or None.
        Forward output shape:
            out: [B, N, query_dim].
        """
        super().__init__()
        context_dim = context_dim if context_dim is not None else query_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Purpose:
            Compute attention between queries and context tokens.
        Inputs:
            x: Tensor [B, N, query_dim].
            context: Tensor [B, M, context_dim] or None.
        Outputs:
            out: Tensor [B, N, query_dim].
        """
        b, n, c = x.shape
        context = x if context is None else context
        _, m, _ = context.shape

        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.out_proj(out)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        """
        Purpose:
            Two-layer MLP block with GELU activation.
        Inputs (constructor):
            in_features: int, input dimension.
            hidden_features: int, hidden layer dimension.
        Outputs (constructor):
            None.
        Forward input shape:
            x: [B, T, in_features].
        Forward output shape:
            y: [B, T, in_features].
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Apply the MLP transformation.
        Inputs:
            x: Tensor [B, T, in_features].
        Outputs:
            y: Tensor [B, T, in_features].
        """
        return self.fc2(self.act(self.fc1(x)))


# ==============================================================================
# Transformer Blocks
# ==============================================================================

class CrossAttnDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """
        Purpose:
            DiT block with self-attention, cross-attention, and MLP, modulated by AdaLN.
        Inputs (constructor):
            hidden_size: int, token dimension.
            num_heads: int, number of attention heads.
            mlp_ratio: float, expansion ratio for MLP hidden size.
        Outputs (constructor):
            None.
        Forward input shapes:
            x: [B, T_action, hidden_size].
            context: [B, T_context, hidden_size].
            c: [B, hidden_size].
        Forward output shape:
            x_out: [B, T_action, hidden_size].
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = AttentionSDPA(hidden_size, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = AttentionSDPA(hidden_size, context_dim=hidden_size, num_heads=num_heads)

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Run one DiT block with AdaLN modulation.
        Inputs:
            x: Tensor [B, T_action, hidden_size].
            context: Tensor [B, T_context, hidden_size].
            c: Tensor [B, hidden_size].
        Outputs:
            x_out: Tensor [B, T_action, hidden_size].
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_ca, scale_ca), context=context)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """
        Purpose:
            Standard transformer block for contextualizing image tokens.
        Inputs (constructor):
            hidden_size: int, token dimension.
            num_heads: int, number of attention heads.
            mlp_ratio: float, expansion ratio for MLP hidden size.
        Outputs (constructor):
            None.
        Forward input shape:
            x: [B, T, hidden_size].
        Forward output shape:
            y: [B, T, hidden_size].
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = AttentionSDPA(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Apply self-attention and MLP to tokens.
        Inputs:
            x: Tensor [B, T, hidden_size].
        Outputs:
            y: Tensor [B, T, hidden_size].
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ImageContextAggregator(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, depth: int = 2, mlp_ratio: float = 4.0) -> None:
        """
        Purpose:
            Contextualize image tokens with a small stack of transformer blocks.
        Inputs (constructor):
            hidden_size: int, token dimension.
            num_heads: int, number of attention heads.
            depth: int, number of transformer layers.
            mlp_ratio: float, expansion ratio for MLP hidden size.
        Outputs (constructor):
            None.
        Forward input shape:
            x: [B, T_img, hidden_size].
        Forward output shape:
            y: [B, T_img, hidden_size].
        """
        super().__init__()
        self.layers = nn.ModuleList([
            BasicTransformerBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Apply stacked transformer blocks to image tokens.
        Inputs:
            x: Tensor [B, T_img, hidden_size].
        Outputs:
            y: Tensor [B, T_img, hidden_size].
        """
        for layer in self.layers:
            x = layer(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        """
        Purpose:
            Produce final action outputs with AdaLN modulation.
        Inputs (constructor):
            hidden_size: int, token dimension.
            output_dim: int, action dimension.
        Outputs (constructor):
            None.
        Forward input shapes:
            x: [B, T_action, hidden_size].
            c: [B, hidden_size].
        Forward output shape:
            y: [B, T_action, output_dim].
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Apply AdaLN modulation and map to action space.
        Inputs:
            x: Tensor [B, T_action, hidden_size].
            c: Tensor [B, hidden_size].
        Outputs:
            y: Tensor [B, T_action, output_dim].
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ==============================================================================
# Main Architecture
# ==============================================================================

class HybridDiT(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_seq_len: int,
        n_obs_steps: int,
        token_dim: int,
        max_image_tokens: int,
        num_cameras: int = 0,
        use_camera_embed: bool = True,
        hidden_size: int = 1152,
        depth: int = 12,
        image_aggregator_depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        """
        Purpose:
            Diffusion Transformer for action denoising with optional camera embeddings.
        Inputs (constructor):
            action_dim: int, action feature dimension.
            action_seq_len: int, action horizon.
            n_obs_steps: int, number of observation steps.
            token_dim: int, conditioning token dimension.
            max_image_tokens: int, number of image tokens per observation step.
            num_cameras: int, number of camera views (0 to disable camera embeddings).
            use_camera_embed: bool, whether to apply camera embeddings when possible.
        Outputs (constructor):
            None.
        Forward input shapes:
            x: [B, T_action, action_dim].
            timestep: [B].
            global_cond: [B, S, T_cond, token_dim].
        Forward output shape:
            out: [B, T_action, action_dim].
        """
        super().__init__()
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.n_obs_steps = n_obs_steps
        self.hidden_size = hidden_size
        self.max_image_tokens = max_image_tokens

        self.num_cameras = num_cameras
        self.use_camera_embed = use_camera_embed and num_cameras > 0

        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.cond_proj = nn.Linear(token_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.state_embedder = ConditionEmbedder((n_obs_steps + 1) * token_dim, hidden_size)
        self.lang_embedder = ConditionEmbedder(token_dim, hidden_size)

        self.step_embed = nn.Parameter(torch.zeros(1, self.n_obs_steps, 1, 1, hidden_size))
        if self.use_camera_embed:
            self.camera_embed = nn.Parameter(torch.zeros(1, 1, self.num_cameras, 1, hidden_size))
        else:
            self.camera_embed = None
        self.action_pos_embed = nn.Parameter(torch.zeros(1, action_seq_len, hidden_size))

        self.image_aggregator = ImageContextAggregator(hidden_size, num_heads, depth=image_aggregator_depth, mlp_ratio=mlp_ratio)
        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, action_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Purpose:
            Initialize weights for DiT layers and embeddings.
        Inputs:
            None.
        Outputs:
            None.
        """
        nn.init.xavier_uniform_(self.action_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.constant_(self.action_proj.bias, 0)
        nn.init.constant_(self.cond_proj.bias, 0)
        nn.init.normal_(self.step_embed, std=0.02)
        if self.camera_embed is not None:
            nn.init.normal_(self.camera_embed, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)

        for embedder in [self.state_embedder, self.lang_embedder]:
            for layer in embedder.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for m in self.image_aggregator.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """
        Purpose:
            Run the DiT denoiser with token-based conditioning.
        Inputs:
            x: Tensor [B, T_action, action_dim].
            timestep: Tensor [B].
            global_cond: Tensor [B, S, T_cond, token_dim].
        Outputs:
            out: Tensor [B, T_action, action_dim].
        """
        b, s, t_total, _ = global_cond.shape

        # Split image tokens and state token.
        has_language = t_total >= self.max_image_tokens + 2
        img_tokens = global_cond[:, :, : self.max_image_tokens, :]
        state_tokens = global_cond[:, :, self.max_image_tokens, :]
        lang_tokens = global_cond[:, :, self.max_image_tokens + 1, :] if has_language else None

        state_all = state_tokens.reshape(b, -1)
        if s > 1:
            state_delta = state_tokens[:, -1, :] - state_tokens[:, -2, :]
        else:
            state_delta = torch.zeros_like(state_tokens[:, 0, :])
        state_flat = torch.cat([state_all, state_delta], dim=-1)
        c = self.t_embedder(timestep) + self.state_embedder(state_flat)
        if has_language:
            c = c + self.lang_embedder(lang_tokens[:, 0, :])

        img_emb = self.cond_proj(img_tokens)
        if self.use_camera_embed:
            if img_emb.shape[2] % self.num_cameras != 0:
                raise ValueError("Image token count must be divisible by num_cameras when using camera embeddings.")
            t_img = img_emb.shape[2] // self.num_cameras
            img_emb = img_emb.reshape(b, self.n_obs_steps, self.num_cameras, t_img, self.hidden_size)
            img_emb = img_emb + self.step_embed + self.camera_embed
            img_seq = img_emb.reshape(b, -1, self.hidden_size)
        else:
            img_seq = img_emb.reshape(b, -1, self.hidden_size)

        img_seq = self.image_aggregator(img_seq)

        x_emb = self.action_proj(x)
        x_emb = x_emb + self.action_pos_embed[:, : x_emb.shape[1], :]

        for block in self.blocks:
            x_emb = block(x_emb, context=img_seq, c=c)

        return self.final_layer(x_emb, c)


# ==============================================================================
# Factory helpers
# ==============================================================================

def DiT_XL(**kwargs) -> HybridDiT:
    """
    Purpose:
        Construct an extra-large DiT variant.
    Inputs:
        kwargs: passed to HybridDiT constructor.
    Outputs:
        model: HybridDiT instance.
    """
    return HybridDiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)


def DiT_L(**kwargs) -> HybridDiT:
    """
    Purpose:
        Construct a large DiT variant.
    Inputs:
        kwargs: passed to HybridDiT constructor.
    Outputs:
        model: HybridDiT instance.
    """
    return HybridDiT(hidden_size=1024, depth=24, num_heads=16, **kwargs)


def DiT_B(**kwargs) -> HybridDiT:
    """
    Purpose:
        Construct a base DiT variant.
    Inputs:
        kwargs: passed to HybridDiT constructor.
    Outputs:
        model: HybridDiT instance.
    """
    return HybridDiT(hidden_size=768, depth=12, num_heads=12, **kwargs)


def DiT_S(**kwargs) -> HybridDiT:
    """
    Purpose:
        Construct a small DiT variant.
    Inputs:
        kwargs: passed to HybridDiT constructor.
    Outputs:
        model: HybridDiT instance.
    """
    return HybridDiT(hidden_size=384, depth=12, num_heads=6, **kwargs)
