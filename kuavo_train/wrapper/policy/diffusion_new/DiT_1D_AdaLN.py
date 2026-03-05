import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ==============================================================================
#  Embedders (保持不变)
# ==============================================================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ConditionEmbedder(nn.Module):
    """用于将 State 向量映射为 AdaLN 条件"""
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, y):
        return self.mlp(y)

# ==============================================================================
#  Blocks (保持不变)
# ==============================================================================

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# ==============================================================================
#  核心修改：Auto-Splitting DiT
# ==============================================================================

class DiT(nn.Module):
    def __init__(
        self,
        action_dim: int,        # Action 输入维度
        action_seq_len: int,    # Action 序列长度 (T_act)
        n_obs_steps: int,       # 观测历史步数 (S)
        token_dim: int,         # global_cond 中 Token 的维度 (D)
        max_image_tokens: int,  # 预估的最大图像 Token 数 (用于 PosEmbed 初始化)
        num_cameras: int = 1,   # 摄像头数量 (用于 Camera Positional Embedding)
        hidden_size: int = 1152,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.n_obs_steps = n_obs_steps
        self.token_dim = token_dim
        self.hidden_size = hidden_size
        self.num_cameras = num_cameras

        # 1. Input Projections
        # ----------------------------------------------------------------------
        # Action -> Hidden
        self.action_proj = nn.Linear(action_dim, hidden_size)
        # Image/State Token -> Hidden (假设它们维度一致)
        self.cond_proj = nn.Linear(token_dim, hidden_size)

        # 2. Embeddings
        # ----------------------------------------------------------------------
        self.image_step_embed = nn.Parameter(torch.zeros(1, n_obs_steps, 1, hidden_size))
        # Action Positional Embedding: 仅用于 Action tokens，Image 位置信息已由上游编码器提供
        self.action_pos_embed = nn.Parameter(torch.zeros(1, action_seq_len, hidden_size))
        # Camera Positional Embedding: 区分来自不同摄像头的 Token
        # 当 max_image_tokens 能被 num_cameras 整除时启用（非 Perceiver 不均匀 queries 场景）
        if num_cameras > 1 and max_image_tokens % num_cameras == 0:
            self.patches_per_cam = max_image_tokens // num_cameras
            self.camera_embed = nn.Parameter(torch.zeros(1, 1, num_cameras, 1, hidden_size))
        else:
            self.patches_per_cam = max_image_tokens
            self.camera_embed = None

        # Timestep Embedder
        self.t_embedder = TimestepEmbedder(hidden_size)

        # State Embedder (AdaLN):
        # S==1: 直接用单步 state，输入维度 = token_dim
        # S >1: 取最后两步的 prev + curr + delta，输入维度固定为 3 * token_dim
        self.state_input_dim = token_dim if n_obs_steps == 1 else 3 * token_dim
        self.state_embedder = ConditionEmbedder(self.state_input_dim, hidden_size)
        
        # 3. Transformer Blocks
        # ----------------------------------------------------------------------
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # 4. Final Output
        # ----------------------------------------------------------------------
        self.final_layer = FinalLayer(hidden_size, action_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.action_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.constant_(self.action_proj.bias, 0)
        nn.init.constant_(self.cond_proj.bias, 0)
        nn.init.normal_(self.image_step_embed, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)
        if self.camera_embed is not None:
            nn.init.normal_(self.camera_embed, std=0.02)

        # Init State Embedder
        for layer in self.state_embedder.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # Init Timestep Embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out AdaLN
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out Final Layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, timestep, global_cond):
        """
        Args:
            x: (B, T_act, action_dim) - Noisy Action
            timestep: (B,)
            global_cond: (B, S, Total_Tokens, D) 
                         其中每个 S 的最后一个 Token 是 State，其余是 Image
        """
        B, S, T_all, D = global_cond.shape
        
        # ======================================================================
        # 1. 自动分离 State 和 Image
        # ======================================================================
        # Image Tokens: 前 T_all - 1 个
        img_tokens = global_cond[..., :-1, :] # (B, S, T_img, D)
        
        # State Token: 最后一个
        state_tokens = global_cond[..., -1, :] # (B, S, D)
        
        # ======================================================================
        # 2. 准备 AdaLN Condition (State + Time)
        # ======================================================================
        if S > 1:
            # 取最后两步，兼容 n_obs_steps >= 2 的任意值
            state_prev = state_tokens[:, -2, :]
            state_curr = state_tokens[:, -1, :]
            state_delta = state_curr - state_prev
            # (B, 3 * D)
            state_flat = torch.cat([state_prev, state_curr, state_delta], dim=-1)
        else:
            state_flat = state_tokens.reshape(B, -1)

        t_emb = self.t_embedder(timestep)        # (B, Hidden)
        s_emb = self.state_embedder(state_flat)  # (B, Hidden)
        c = t_emb + s_emb                        # (B, Hidden) -> AdaLN

        # ======================================================================
        # 3. 准备 Transformer Sequence (Image + Action)
        # ======================================================================
        # 3.1 Project: (B, S, T_img, D) -> (B, S, T_img, H)
        img_emb = self.cond_proj(img_tokens)

        # 3.2 注入摄像头位置编码: 区分不同摄像头的 Token
        # Token 排列为 [cam0_p0...cam0_pP, cam1_p0...cam1_pP, ...]，因此 reshape 可以正确分组
        if self.camera_embed is not None:
            img_emb = img_emb.reshape(B, S, self.num_cameras, self.patches_per_cam, self.hidden_size)
            img_emb = img_emb + self.camera_embed  # (1, 1, num_cameras, 1, H) 广播
            img_emb = img_emb.reshape(B, S, -1, self.hidden_size)

        # 3.3 注入时序位置编码 (step embedding)
        # self.image_step_embed: (1, S, 1, H)，广播到 (B, S, T_img, H)
        img_emb = img_emb + self.image_step_embed

        # 3.4 Flatten Time Dimension: (B, S, T_img, H) -> (B, S * T_img, H)
        img_seq = img_emb.reshape(B, -1, self.hidden_size)

        # 4. Action: 投影 + 专属位置编码
        x_emb = self.action_proj(x)       # (B, T_act, Hidden)
        x_emb = x_emb + self.action_pos_embed  # 仅 Action tokens 加位置编码

        # 5. Concat: Image 在前，Action 在后
        h = torch.cat([img_seq, x_emb], dim=1)  # (B, S*T_img + T_act, Hidden)
        
        # ======================================================================
        # 6. Transformer Forward
        # ======================================================================
        for block in self.blocks:
            h = block(h, c)
            
        # ======================================================================
        # 7. Output
        # ======================================================================
        # 只取 Action 部分 (在序列末尾)
        h_action = h[:, -self.action_seq_len:, :]
        out = self.final_layer(h_action, c)
        
        return out

# Config Wrappers
def DiT_XL(**kwargs): return DiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)
def DiT_L(**kwargs):  return DiT(hidden_size=1024, depth=24, num_heads=16, **kwargs)
def DiT_B(**kwargs):  return DiT(hidden_size=768, depth=12, num_heads=12, **kwargs)
def DiT_S(**kwargs):  return DiT(hidden_size=384, depth=12, num_heads=6, **kwargs)

if __name__ == "__main__":
    # Test
    net = DiT_S(
        action_dim=14,
        action_seq_len=16,
        n_obs_steps=2,
        token_dim=512,
        max_image_tokens=200
    )
    x = torch.randn(4, 16, 14)          # (B, T_act, action_dim)
    t = torch.randint(0, 100, (4,))     # (B,)
    # global_cond: (B, S, T_img + 1, D)，最后一个 token 是 state
    # state_embedder 输入维度 = 3 * token_dim = 1536（prev + curr + delta）
    cond = torch.randn(4, 2, 101, 512)

    out = net(x, t, cond)
    print("Output shape:", out.shape)   # Should be (4, 16, 14)

    # 验证 n_obs_steps=1 也能正常工作
    net1 = DiT_S(
        action_dim=14,
        action_seq_len=16,
        n_obs_steps=1,
        token_dim=512,
        max_image_tokens=200
    )
    cond1 = torch.randn(4, 1, 101, 512)
    out1 = net1(x, t, cond1)
    print("Output shape (S=1):", out1.shape)  # Should be (4, 16, 14)