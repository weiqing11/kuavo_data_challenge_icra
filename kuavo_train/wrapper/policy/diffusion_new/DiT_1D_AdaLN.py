import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
#  Helper Functions
# ==============================================================================

def modulate(x, shift, scale):
    """AdaLN 的核心调制函数"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ==============================================================================
#  1. Embedders (条件特征嵌入)
# ==============================================================================

class TimestepEmbedder(nn.Module):
    """时间步嵌入器 (Sinusoidal + MLP)"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

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
        return self.mlp(t_freq)


class ConditionEmbedder(nn.Module):
    """状态嵌入器：将 State 向量映射为 AdaLN 的融合条件"""
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
#  2. Core Attention Modules (核心注意力组件)
# ==============================================================================

class AttentionSDPA(nn.Module):
    """支持 Flash Attention 的通用注意力模块 (Self / Cross)"""
    def __init__(self, query_dim, context_dim=None, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.head_dim = query_dim // num_heads
        
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, x, context=None):
        B, N, C = x.shape
        context = context if context is not None else x
        _, M, _ = context.shape

        # 投影并重塑为 (B, Num_Heads, Seq_Len, Head_Dim)
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # 核心加速：PyTorch 2.0+ SDPA (O(N^2) -> O(N))
        out = F.scaled_dot_product_attention(q, k, v)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class Mlp(nn.Module):
    """标准两层 MLP"""
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ==============================================================================
#  3. Transformer Blocks (构建块)
# ==============================================================================

class CrossAttnDiTBlock(nn.Module):
    """省显存的 Cross-Attention DiT Block"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        # 1. Action Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = AttentionSDPA(hidden_size, num_heads=num_heads)
        
        # 2. Action -> Image Cross-Attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = AttentionSDPA(hidden_size, context_dim=hidden_size, num_heads=num_heads)
        
        # 3. MLP
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))
        
        # 9 个 AdaLN 控制变量 (Shift, Scale, Gate * 3)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, context, c):
        # 解析调制参数
        (shift_msa, scale_msa, gate_msa, 
         shift_ca,  scale_ca,  gate_ca, 
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(9, dim=1)
        
        # Self-Attention
        x = x + gate_msa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # Cross-Attention
        x = x + gate_ca.unsqueeze(1)  * self.cross_attn(modulate(self.norm2(x), shift_ca, scale_ca), context=context)
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class BasicTransformerBlock(nn.Module):
    """标准的 Self-Attention + MLP 块 (用于前置图像特征提取)"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = AttentionSDPA(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ImageContextAggregator(nn.Module):
    """轻量级图像预编码器：运行一次，建立图像的全局空间理解"""
    def __init__(self, hidden_size, num_heads, depth=2, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            BasicTransformerBlock(hidden_size, num_heads, mlp_ratio) 
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FinalLayer(nn.Module):
    """输出层"""
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
        return self.linear(x)


# ==============================================================================
#  4. Main Architecture (主网络架构)
# ==============================================================================

class HybridDiT(nn.Module):
    """高性能混合架构 Diffusion Transformer"""
    def __init__(
        self,
        action_dim: int,
        action_seq_len: int,
        n_obs_steps: int,
        token_dim: int,
        max_image_tokens: int,
        hidden_size: int = 1152,
        depth: int = 12,
        image_aggregator_depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.n_obs_steps = n_obs_steps
        self.hidden_size = hidden_size
        self.max_image_tokens = max_image_tokens  # 新增：显式记录每步图像 token 数
        
        # --- 1. Projections & Embedders ---
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.cond_proj = nn.Linear(token_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.state_embedder = ConditionEmbedder((n_obs_steps + 1) * token_dim, hidden_size)
        self.lang_embedder = ConditionEmbedder(token_dim, hidden_size)  # 新增：language -> AdaLN

        # --- 2. Positional Embeddings ---
        self.num_cameras = 3
        self.step_embed = nn.Parameter(torch.zeros(1, self.n_obs_steps, 1, 1, hidden_size))
        self.camera_embed = nn.Parameter(torch.zeros(1, 1, self.num_cameras, 1, hidden_size))
        self.action_pos_embed = nn.Parameter(torch.zeros(1, action_seq_len, hidden_size))
        
        # --- 3. Hybrid Core Modules ---
        self.image_aggregator = ImageContextAggregator(
            hidden_size, num_heads, depth=image_aggregator_depth, mlp_ratio=mlp_ratio
        )
        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, action_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        """参数初始化：使用 Xavier 与 Zero-init 标准范式"""
        # Projections & Embeddings
        nn.init.xavier_uniform_(self.action_proj.weight)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.constant_(self.action_proj.bias, 0)
        nn.init.constant_(self.cond_proj.bias, 0)
        nn.init.normal_(self.step_embed, std=0.02)
        nn.init.normal_(self.camera_embed, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)

        # State / Language Embedder
        for embedder in [self.state_embedder, self.lang_embedder]:
            for layer in embedder.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

        # Timestep Embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Image Aggregator
        for m in self.image_aggregator.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-out AdaLN Modulations & Final Linear
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, timestep, global_cond):
        """
        x: (B, T_act, action_dim)
        timestep: (B,)
        global_cond: (B, S, Total_Tokens, D), 约定顺序 [img..., state, language]
        """
        B, S, T_total, _ = global_cond.shape

        # === 1. 分离 Image / State / Language ===
        # 支持两种情况：
        # 1) 只有 state:    Total = max_image_tokens + 1
        # 2) state+language:Total = max_image_tokens + 2
        has_language = (T_total >= self.max_image_tokens + 2)

        img_tokens = global_cond[:, :, :self.max_image_tokens, :]                  # (B, S, T_img, D)
        state_tokens = global_cond[:, :, self.max_image_tokens, :]                 # (B, S, D)
        lang_tokens = global_cond[:, :, self.max_image_tokens + 1, :] if has_language else None  # (B, S, D) or None

        # === 2. 状态 + 语言 融合为 AdaLN 条件 ===
        state_all = state_tokens.reshape(B, -1)  # (B, S*D)
        if S > 1:
            state_delta = state_tokens[:, -1, :] - state_tokens[:, -2, :]
        else:
            state_delta = torch.zeros_like(state_tokens[:, 0, :])

        state_flat = torch.cat([state_all, state_delta], dim=-1)  # (B, (S+1)*D)
        c = self.t_embedder(timestep) + self.state_embedder(state_flat)

        # 只注入一个 step 的 language（按你的要求）
        if has_language:
            lang_one_step = lang_tokens[:, 0, :]  # 取第 0 帧；若你想取最后一帧可改为 [:, -1, :]
            c = c + self.lang_embedder(lang_one_step)

        # === 3. 处理 Context (Image) ===
        img_emb = self.cond_proj(img_tokens)
        T_img = img_emb.shape[2] // self.num_cameras
        img_emb_5d = img_emb.reshape(B, self.n_obs_steps, self.num_cameras, T_img, self.hidden_size)
        img_emb_5d = img_emb_5d + self.step_embed + self.camera_embed
        img_seq = img_emb_5d.reshape(B, -1, self.hidden_size)
        img_seq_contextualized = self.image_aggregator(img_seq)

        # === 4. 处理 Query (Action) ===
        x_emb = self.action_proj(x)
        x_emb = x_emb + self.action_pos_embed[:, :x_emb.shape[1], :]

        # === 5. Cross-Attention Transformer ===
        for block in self.blocks:
            x_emb = block(x_emb, context=img_seq_contextualized, c=c)

        # === 6. 预测输出 ===
        out = self.final_layer(x_emb, c)
        return out


# ==============================================================================
#  5. Config Wrappers & Tests
# ==============================================================================

def DiT_XL(**kwargs): return HybridDiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)
def DiT_L(**kwargs):  return HybridDiT(hidden_size=1024, depth=24, num_heads=16, **kwargs)
def DiT_B(**kwargs):  return HybridDiT(hidden_size=768,  depth=12, num_heads=12, **kwargs)
def DiT_S(**kwargs):  return HybridDiT(hidden_size=384,  depth=12, num_heads=6,  **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 初始化轻量版网络进行测试
    net = DiT_S(
        action_dim=14, 
        action_seq_len=16, 
        n_obs_steps=2,
        token_dim=512, 
        max_image_tokens=600  # 假设 384 分辨率带来极大的 token 数量
    ).to(device)
    
    # 构建虚拟输入
    batch_size = 4
    x = torch.randn(batch_size, 16, 14).to(device)
    t = torch.randint(0, 100, (batch_size,)).to(device)
    # 2帧, 每帧 576(Image) + 1(State) = 577
    cond = torch.randn(batch_size, 2, 577, 512).to(device)
    
    # 前向传播测试
    out = net(x, t, cond)
    print("Output shape:", out.shape)   # Expected: (4, 16, 14)
    print("Test passed successfully! ✅")