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
        
        # 1. Input Projections
        # ----------------------------------------------------------------------
        # Action -> Hidden
        self.action_proj = nn.Linear(action_dim, hidden_size)
        # Image/State Token -> Hidden (假设它们维度一致)
        self.cond_proj = nn.Linear(token_dim, hidden_size)

        # 2. Embeddings
        # ----------------------------------------------------------------------
        self.image_step_embed = nn.Parameter(torch.zeros(1, n_obs_steps, 1, hidden_size))
        # Positional Embedding: 覆盖 (S * Image_Tokens) + Action_Tokens
        # 我们这里分配一个足够大的 buffer
        # 注意: 这里的 max_len 需要是 (S * max_img_per_step) + action_seq_len
        total_max_len = (n_obs_steps * max_image_tokens) + action_seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, total_max_len, hidden_size))
        
        # Timestep Embedder
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # State Embedder (AdaLN): 
        # 输入维度 = S * token_dim (我们将 S 个 State Token 展平)
        self.state_embedder = ConditionEmbedder(n_obs_steps * token_dim, hidden_size)
        
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
        nn.init.normal_(self.pos_embed, std=0.02)

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
        # 将 S 个 State Token 展平为长向量: (B, S * D)
        state_flat = state_tokens.reshape(B, -1)
        
        t_emb = self.t_embedder(timestep)        # (B, Hidden)
        s_emb = self.state_embedder(state_flat)  # (B, Hidden)
        c = t_emb + s_emb                        # (B, Hidden) -> AdaLN
        
        # ======================================================================
        # 3. 准备 Transformer Sequence (Image + Action)
        # ======================================================================
        # 3.1 Project: (B, S, T_img, D) -> (B, S, T_img, H)
        img_emb = self.cond_proj(img_tokens) 
        
        # 3.2 [关键] 注入 Explicit Step Info
        # self.image_step_embed: (1, S, 1, H)
        # 自动广播到 (B, S, T_img, H)
        img_emb = img_emb + self.image_step_embed
        
        # 3.3 Flatten Time Dimension
        # (B, S, T_img, H) -> (B, S * T_img, H)
        img_seq = img_emb.reshape(B, -1, self.hidden_size)
        
        # 4. Action
        x_emb = self.action_proj(x)             # (B, T_act, Hidden)
        
        # 5. Concat & Pos Embed (Spatial + Temporal combined)
        h = torch.cat([img_seq, x_emb], dim=1)  # (B, Total_Seq, Hidden)
        seq_len = h.shape[1]
        h = h + self.pos_embed[:, :seq_len, :]
        
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
    # 输入构造
    x = torch.randn(4, 16, 14)          # Action
    t = torch.randint(0, 100, (4,))     # Timestep
    # Cond: 2步, 每步 100个Image + 1个State = 101个Token
    cond = torch.randn(4, 2, 101, 512)  
    
    out = net(x, t, cond)
    print("Output shape:", out.shape)   # Should be (4, 16, 14)