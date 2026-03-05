# multimodal_diffusion_wrapper.py
import math
import os
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any, Tuple, Callable
import cv2
import einops
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torch.nn.functional as F

from transformers import AutoModel, AutoImageProcessor, SiglipVisionModel, SiglipImageProcessor
from kuavo_train.logger import logger, log_box, Colors

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from kuavo_train.wrapper.policy.diffusion_new.DiffusionConfigWrapper import CustomDiffusionConfigWrapper
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, get_output_shape
from lerobot.policies.diffusion.modeling_diffusion import (
    _make_noise_scheduler,
    _replace_submodules,
    DiffusionConditionalUnet1d,
    SpatialSoftmax,
    DiffusionModel,
)
from kuavo_train.wrapper.policy.diffusion_new.transformer_diffusion import TransformerForDiffusion
from kuavo_train.wrapper.policy.diffusion_new.DFormerv2 import DFormerv2_S, DFormerv2_B, DFormerv2_L
from kuavo_train.wrapper.policy.diffusion_new.DiT_1D_AdaLN import DiT

# diffusers scheduler classes (factory expects these names)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import StableDiffusion3Pipeline

import matplotlib.pyplot as plt
import numpy as np
import time
from peft import get_peft_model, LoraConfig, TaskType

OBS_DEPTH = "observation.depth"

# ---------------------------
# Helper: safe scheduler factory
# ---------------------------
def _make_noise_scheduler_factory(name: str, **kwargs: Dict[str, Any]):
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


# =====================================================================
# 1. 官方 ToMe 核心算子 (保持不变)
# =====================================================================
def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(metric: torch.Tensor, r: int) -> Tuple[Callable, Callable]:
    """官方二分软匹配核心算法"""
    t = metric.shape[1]
    # 严格遵守官方的 50% 限制
    r = min(r, t // 2) 

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # 未融合的 Token
        src_idx = edge_idx[..., :r, :]  # 将被融合的 Token
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # 使用官方的高效规约算子 (要求 PyTorch >= 1.12)
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode, include_self=True)

        return torch.cat([unm, dst], dim=1)

    return merge

def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """官方的加权平均融合"""
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """生成 1D 的 Sine-Cosine 位置编码"""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # 外积: (M, D/2)

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)
    return torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    """
    生成标准的 2D Sine-Cosine 位置编码
    返回 shape: (1, H*W, D)
    """
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    
    # 维度对半开：一半给 H，一半给 W
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (W, D/2)
    
    # 扩展到 2D 网格维度
    emb_h = emb_h.unsqueeze(1).expand(-1, grid_size_w, -1) # (H, W, D/2)
    emb_w = emb_w.unsqueeze(0).expand(grid_size_h, -1, -1) # (H, W, D/2)
    
    # 拼接 H 和 W 的特征
    pos_embed = torch.cat([emb_h, emb_w], dim=-1) # (H, W, D)
    pos_embed = pos_embed.reshape(grid_size_h * grid_size_w, embed_dim) # (H*W, D)
    
    return pos_embed.unsqueeze(0) # (1, H*W, D)


# === Definitions for MLP Projection Module, with Signature :: [..., in_dim] --> [..., out_dim] ===
class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)

# ---------------------------
# Feature encoders (state)
# ---------------------------
class FeatureEncoder(nn.Module):
    """Simple MLP encoder for state features. Accepts [B, D] or [B, T, D]."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=False),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Optional[Tensor]:
        if x is None:
            return None
        if x.dim() == 2:
            return self.net(x)  # (B, out_dim)
        elif x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
            out = self.net(x_flat).view(B, T, -1)
            return out  # (B, T, out_dim)
        else:
            raise ValueError("FeatureEncoder expects 2D or 3D tensor.")


# [新增] Perceiver Resampler 模块
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        num_queries: int = 64,
        depth: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                # Cross Attention: Query=Latents, Key/Value=Input Features
                'cross_attn': nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True),
                'cross_norm_q': nn.LayerNorm(dim),
                'cross_norm_kv': nn.LayerNorm(dim),
                
                # Self Attention: Query=Latents
                'self_attn': nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True),
                'self_norm': nn.LayerNorm(dim),
                
                # Feed Forward
                'ff': nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim)
                )
            }))
        
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [B, N_inputs, D]
        B = x.shape[0]
        
        # 1. Expand latents to batch size: [B, num_queries, D]
        latents = self.latents.repeat(B, 1, 1)
        
        for layer in self.layers:
            # --- Cross Attention ---
            # Q 来自 Latents, K,V 来自输入 x
            q = layer['cross_norm_q'](latents)
            k = v = layer['cross_norm_kv'](x)
            
            # output shape: [B, num_queries, D]
            cross_out, _ = layer['cross_attn'](query=q, key=k, value=v)
            latents = latents + cross_out
            
            # --- Self Attention ---
            q_sa = layer['self_norm'](latents)
            self_out, _ = layer['self_attn'](query=q_sa, key=q_sa, value=q_sa)
            latents = latents + self_out
            
            # --- Feed Forward ---
            latents = latents + layer['ff'](latents)
            
        return self.norm_out(latents)


class DinoSiglipRGBEncoder(nn.Module):
    """
    DINO + SigLIP 双塔视觉编码器 (Dual-Tower Vision Encoder)
    
    功能：
    1. 并行运行 SigLIP (语义强) 和 DINO (几何强) 模型。
    2. 支持 DINOv2 和 DINOv3 (通过 AutoModel 加载)。
    3. 分别处理两种模型的不同归一化需求。
    4. 输出拼接后的 Token 序列, Patch 在特征维度拼接并融合, Cls 在数量上进行融合。
    
    输出形状:
        [Batch, N_patches + 1, projection_dim]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # =========================================================================
        # 1. 配置与模型加载 (Configuration & Model Loading)
        # =========================================================================
        self.siglip_model_name = config.siglip_model_name
        # 优先读取 dino_model_name，兼容旧配置 dinov2_model_name
        self.dino_model_name = getattr(config, "dino_model_name", getattr(config, "dinov2_model_name", None))
        
        if self.dino_model_name is None:
            raise ValueError("❌ Config Error: Please specify 'dino_model_name' in your config.")

        # 判定是否强制使用本地文件 (如果路径包含 '/' 则认为是本地路径，不走 HuggingFace Hub)
        siglip_is_local = "/" in self.siglip_model_name
        dino_is_local = "/" in self.dino_model_name

        logger.info(f"🏗️ Loading Vision Encoders...")
        
        # 加载 SigLIP
        self.siglip = SiglipVisionModel.from_pretrained(
            self.siglip_model_name,
            local_files_only=siglip_is_local
        )
        
        # 加载 DINO (v2/v3)
        # 注意: trust_remote_code=True 对 DINOv3 可能是必须的
        # attn_implementation="sdpa" 使用 Torch 2.0+ 的加速注意力
        self.dino = AutoModel.from_pretrained(
            self.dino_model_name,
            local_files_only=dino_is_local,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        
        # =========================================================================
        # 2. 图像处理器 (Image Processors)
        # =========================================================================
        # SigLIP 和 DINO 需要不同的归一化参数 (Mean/Std)，必须分开处理
        self.siglip_processor = SiglipImageProcessor.from_pretrained(
            self.siglip_model_name,
            local_files_only=siglip_is_local
        )
        self.dino_processor = AutoImageProcessor.from_pretrained(
            self.dino_model_name,
            local_files_only=dino_is_local,
            trust_remote_code=True
        )

        # =========================================================================
        # 3. LoRA 微调与冻结策略 (LoRA & Freeze Strategy)
        # =========================================================================
        self.use_lora = getattr(config, "use_lora", False)
        self.vision_freeze = getattr(config, "vision_freeze", True)

        if self.use_lora:
            self._setup_lora(config)
            self.vision_freeze = False # 开启 LoRA 时强制解冻
        elif self.vision_freeze:
            # 冻结所有参数
            self.siglip.requires_grad_(False)
            self.dino.requires_grad_(False)
            self.siglip.eval()
            self.dino.eval()
        else:
            # 全量微调
            self.siglip.train()
            self.dino.train()

        # =========================================================================
        # 4. Token 数量计算 (关键修改)
        # =========================================================================
        # 获取输入图像尺寸 (H, W)
        if config.resize_shape:
            h, w = config.resize_shape
        else:
            # 如果没有 resize，尝试从 image_features 的 tensor shape 获取
            first_shape = next(iter(config.image_features.values())).shape
            h, w = first_shape[1:]

        # --- 计算 SigLIP Token 数 ---
        # SigLIP 通常只有 Grid Patches，没有 CLS
        self.siglip_p = self.siglip.config.patch_size
        self.num_siglip_tokens = (h // self.siglip_p) * (w // self.siglip_p)

        # --- 计算 DINO Token 数 ---
        # DINO 通常有 1个 CLS + Grid Patches + (可选) Registers
        self.dino_p = self.dino.config.patch_size
        grid_dino = (h // self.dino_p) * (w // self.dino_p)
        
        # 检查 Patch Size 是否一致，否则无法在特征维度对齐拼接
        if self.siglip_p != self.dino_p:
            logger.warning(f"⚠️ Warning: SigLIP Patch ({self.siglip_p}) != DINO Patch ({self.dino_p}). "
                           "Feature concatenation requires spatial alignment. Ensure your models match or resize/interpolate is handled.")

        # 检查是否有寄存器 token (DINOv2-registers 模型)
        self.num_registers = getattr(self.dino.config, "num_register_tokens", 0)
        self.num_dino_tokens = grid_dino + 1 # +1 是 CLS Token

        # 总 Token 数
        self.num_patches = self.num_siglip_tokens + 1

        # =========================================================================
        # 5. 特征投影层 (Feature Projection)
        # =========================================================================
        # 将两个模型的不同输出维度统一映射到 transformer_n_emb (例如 384)
        target_dim = getattr(config, "transformer_n_emb", 384) 
        self.siglip_dim = self.siglip.config.hidden_size
        self.dino_dim = self.dino.config.hidden_size

        self.proj_siglip = nn.Linear(self.siglip_dim, target_dim)
        self.proj_dino = nn.Linear(self.dino_dim, target_dim)
        
        self.norm_siglip = nn.LayerNorm(target_dim)
        self.norm_dino = nn.LayerNorm(target_dim)
        
        self.fusion_map = nn.Sequential(
            nn.Linear(target_dim * 2, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim)
        )

        self.feature_dim = target_dim

        self._log_init_info(target_dim, h, w)

    def _setup_lora(self, config):
        """配置并应用 LoRA"""
        # SigLIP LoRA 配置
        peft_config_siglip = LoraConfig(
            r=getattr(config, "lora_rank", 16),
            lora_alpha=getattr(config, "lora_alpha", 32),
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=getattr(config, "lora_dropout", 0.05),
            bias="none",
        )
        self.siglip = get_peft_model(self.siglip, peft_config_siglip)
        
        # DINO LoRA 配置 (适用于标准 ViT 结构)
        peft_config_dino = LoraConfig(
            r=getattr(config, "lora_rank", 16),
            lora_alpha=getattr(config, "lora_alpha", 32),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"], 
            lora_dropout=getattr(config, "lora_dropout", 0.05),
            bias="none",
        )
        self.dino = get_peft_model(self.dino, peft_config_dino)

    def _log_init_info(self, target_dim, h, w):
        """打印初始化日志"""
        # 确定当前训练状态
        if self.use_lora:
            status = "🔥 LoRA Fine-Tuning"
        elif self.vision_freeze:
            status = "❄️ Frozen (Inference Only)"
        else:
            status = "🚀 Full Fine-Tuning"

        # 构造打印信息字典
        encoder_info = {
            "Input Resolution": f"{h} x {w}",
            "SigLIP Model": f"{self.siglip_model_name} (P={self.siglip_p})",
            "SigLIP Tokens": f"{self.num_siglip_tokens}",
            "DINO Model": f"{self.dino_model_name} (P={self.dino_p})",
            "DINO Tokens": f"{self.num_dino_tokens} (Inc. 1 CLS, Excl. {self.num_registers} Regs)",
            "Total Seq Length": f"{self.num_patches} Tokens",
            "Projection Dim": f"{target_dim} (Transformer Input)",
            "Training Status": status,
        }

        # 调用你自定义的 log_box 函数
        log_box("Dual Vision Encoder Configuration", encoder_info, icon="✨")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入图像张量 [Batch, 3, Height, Width]
        Returns:
            combined_tokens: [Batch, Total_Patches, target_dim]
        """
        # 根据是否微调决定是否开启梯度计算
        requires_grad = self.use_lora or (not self.vision_freeze)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        
        with context:
            # 1. SigLIP Forward
            # SigLIP 通常需要 range [-1, 1] 或特定的 mean/std，processor 会自动处理
            siglip_in = self.siglip_processor(images=x, do_resize=False, do_rescale=False, return_tensors="pt")
            # interpolate_pos_encoding=True 允许输入分辨率与预训练不同 (例如 3:4 比例)
            siglip_out = self.siglip(siglip_in['pixel_values'].to(x.device), interpolate_pos_encoding=True)
            siglip_feat = siglip_out.last_hidden_state # Shape: [B, N_sig, D_sig]

            # 2. DINO Forward
            # DINO 通常需要 ImageNet mean/std
            dino_in = self.dino_processor(images=x, do_resize=False, do_rescale=False, return_tensors="pt")
            dino_out = self.dino(dino_in['pixel_values'].to(x.device), interpolate_pos_encoding=True)
            raw_dino_feat = dino_out.last_hidden_state # Shape: [B, N_dino, D_dino]

            # 处理 DINO 的 Token
            # 结构: [CLS, Reg_1, ..., Reg_n, Patch_1, ...]
            # 目标: 保留 CLS 和 Patches，丢弃 Registers
            
            # A. 提取 CLS (Index 0)
            cls_token = raw_dino_feat[:, 0:1, :] 
            
            # B. 提取 Patches (跳过 CLS 和 Registers)
            # start_index = 1 + num_registers
            patch_start_idx = 1 + self.num_registers
            patch_tokens = raw_dino_feat[:, patch_start_idx:, :]
            
            # C. 拼接回: [CLS, Patches]
            #dino_feat = torch.cat([cls_token, patch_tokens], dim=1)

        # 3. 投影与归一化 (Project & Normalize)
        # 将不同维度的特征映射到同一维度 (如 384)
        siglip_tokens = self.norm_siglip(self.proj_siglip(siglip_feat))
        dino_patches_tokens = self.norm_dino(self.proj_dino(patch_tokens))
        dino_cls_token = self.norm_dino(self.proj_dino(cls_token))
        
        # 4. Patch 特征融合 (Feature Fusion)
        # [B, N, D] + [B, N, D] -> Cat(dim=-1) -> [B, N, 2D] -> Linear -> [B, N, D]
        combined_patches = torch.cat([siglip_tokens, dino_patches_tokens], dim=-1)
        combined_tokens = self.fusion_map(combined_patches)

        # 5. 最终拼接 (Final Sequence Concatenation)
        # 形状变化:
        # CLS [B, 1, D] + Fused Patches [B, N, D] -> [B, 1+N, D]
        output_tokens = torch.cat([dino_cls_token, combined_tokens], dim=1)

        return output_tokens


class SiglipRGBEncoder(nn.Module):
    """
    SigLIP 视觉编码器 (SigLIP Vision Encoder)
    
    功能：
    1. 使用 Google SigLIP 模型提取图像特征。
    2. 输出 Patch Token 序列，而非单一的 CLS Token。
    3. 支持 LoRA 微调或全量冻结。
    4. 包含投影层，将特征维度映射到 Transformer 所需维度。
    
    输出形状:
        [Batch, Num_Patches, projection_dim]
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 模式检查
        self.mode = getattr(config, "vision_encoder_mode", "patches")
        if self.mode != "patches":
            raise ValueError(f"❌ Unknown mode: {self.mode}. SiglipRGBEncoder only supports 'patches' mode.")

        # =========================================================================
        # 2. 模型与处理器加载 (Model & Processor Loading)
        # =========================================================================
        self.siglip_model_name = config.siglip_model_name
        
        # 判定是否强制使用本地文件 (路径包含 '/' 视为本地路径)
        is_local = "/" in self.siglip_model_name
        
        logger.info(f"🏗️ Loading SigLIP Model: {self.siglip_model_name}")
        
        self.siglip = SiglipVisionModel.from_pretrained(
            self.siglip_model_name,
            local_files_only=is_local
        )
        
        # 加载对应的图像处理器 (负责归一化 Mean/Std)
        self.processor = SiglipImageProcessor.from_pretrained(
            self.siglip_model_name,
            local_files_only=is_local
        )

        # =========================================================================
        # 3. LoRA 与 冻结策略 (LoRA & Freeze Strategy)
        # =========================================================================
        self.use_lora = getattr(config, "use_lora", False)
        self.vision_freeze = getattr(config, "vision_freeze", True)

        if self.use_lora:
            self._setup_lora(config)
            self.vision_freeze = False  # 使用 LoRA 时强制解冻
        elif self.vision_freeze:
            self._freeze_backbone()
        else:
            self.siglip.train()  # 全量微调

        # 获取模型原始维度信息
        self.hidden_size = self.siglip.config.hidden_size
        self.patch_size = self.siglip.config.patch_size

        # =========================================================================
        # 4. 初始化投影头 (Projection Head)
        # =========================================================================
        self._init_patches_head(config)
        self._log_init_info()

    def _setup_lora(self, config):
        """配置并应用 LoRA"""
        peft_config = LoraConfig(
            r=getattr(config, "lora_rank", 16),
            lora_alpha=getattr(config, "lora_alpha", 32),
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=getattr(config, "lora_dropout", 0.05),
            bias="none",
        )
        self.siglip = get_peft_model(self.siglip, peft_config)

    def _freeze_backbone(self):
        """冻结骨干网络"""
        self.siglip.requires_grad_(False)
        self.siglip.eval()
    
    def _init_patches_head(self, config):
        """
        初始化投影层
        将 SigLip 的特征维度 (如 768/1152) 映射到 DiT/Transformer 的维度 (如 384)
        """
        # 计算网格大小 (仅用于日志打印，实际 forward 支持动态分辨率)
        if config.resize_shape:
            h, w = config.resize_shape
        else:
            # 尝试从 image_features 配置中推断形状
            first_shape = next(iter(config.image_features.values())).shape
            h, w = first_shape[1:]
            
        self.grid_h = h // self.patch_size
        self.grid_w = w // self.patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        # 目标投影维度
        target_dim = getattr(config, "transformer_n_emb", 384) 
        self.feature_dim = target_dim 
        
        # 投影层结构：Linear -> LayerNorm
        self.proj = nn.Linear(self.hidden_size, self.feature_dim)
        self.norm = nn.LayerNorm(self.feature_dim)

    def _log_init_info(self):
        """以表格/方框形式打印初始化日志"""  
        # 确定训练状态
        if self.use_lora:
            status = "🔥 LoRA Tuned"
        elif self.vision_freeze:
            status = "❄️ Frozen"
        else:
            status = "🚀 Full Finetune"

        # 构造打印信息字典
        encoder_info = {
            "Model Name": self.siglip_model_name,
            "Input Dim (D)": f"{self.hidden_size}",
            "Output Dim (D')": f"{self.feature_dim}",
            "Patch Size (P)": f"{self.patch_size}",
            "Estimated Tokens": f"{self.num_patches}",
            "Training Status": status,
            "Dynamic Scaling": "✅ interpolate_pos_encoding=True",
        }

        # 调用全局 log_box 函数
        log_box("SigLIP Vision Encoder Configuration", encoder_info, icon="✨")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入图像 [Batch, 3, Height, Width]
        Returns:
            tokens: [Batch, Num_Patches, feature_dim]
        """
        # 根据配置决定是否计算梯度
        requires_grad = self.use_lora or (not self.vision_freeze)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        
        with context:
            # 1. 预处理 (Normalization)
            x_processed = self.processor(images=x, do_resize=False, do_rescale=False, return_tensors="pt")
            
            # 2. Backbone Forward
            # interpolate_pos_encoding=True 允许处理非标准分辨率 (如 3:4 比例)
            outputs = self.siglip(
                x_processed['pixel_values'].to(x.device), 
                interpolate_pos_encoding=True
            )

        # 3. 投影与归一化
        # outputs.last_hidden_state shape: [B, N_patches, hidden_size]
        tokens = self.proj(outputs.last_hidden_state)   
        tokens = self.norm(tokens)

        return tokens


class DFomerRGBDBackbone(nn.Module):
    def __init__(self, config):
        """
        初始化 DFormer RGB-D backbone，默认支持全量微调。
        """
        super().__init__()
        model_name = config.vision_backbone_rgbd
        pretrained_path = config.DFormer_path
        # 即使全局 vision_freeze 为 True，也可以通过配置单独控制 DFormer 是否微调
        self.vision_freeze = False
        self.model_size = model_name.split("_")[-1]

        configs = {
            'small': {'fn': DFormerv2_S, 'dims': [64, 128, 256, 512], 'desc': 'Small (High Speed)'},
            'base':  {'fn': DFormerv2_B, 'dims': [80, 160, 320, 512], 'desc': 'Base (Balanced)'},
            'large': {'fn': DFormerv2_L, 'dims': [112, 224, 448, 640], 'desc': 'Large (High Perf)'},
        }

        if self.model_size not in configs:
            raise ValueError(f"❌ Invalid DFormer size: {self.model_size}")

        cfg = configs[self.model_size]
        self.out_channels = cfg['dims']
        self.backbone = cfg['fn']()

        # 加载权重
        if pretrained_path:
            try:
                self.backbone.init_weights(pretrained_path)
            except Exception as e:
                logger.error(f"Failed to load DFormer weights: {e}")

        # 冻结策略：除非明确指定，否则默认开启全量微调
        if self.vision_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            freeze_status = f"{Colors.CYAN}Yes (Frozen ❄️){Colors.RESET}"
        else:
            freeze_status = f"{Colors.GREEN}No (Full Fine-Tuning 🔥){Colors.RESET}"

        total_params = sum(p.numel() for p in self.backbone.parameters())
        log_box("DFormer Backbone Setup", {
            "Size": self.model_size,
            "Params": f"{total_params / 1e6:.2f} M",
            "Fine-Tuning": freeze_status
        })

    def forward(self, x, x_depth):
        # 返回多尺度特征列表
        return list(self.backbone(x, x_depth))

    def get_out_channels(self):
        return self.out_channels


class SiglipDFormerEncoder(nn.Module):
    """
    整理后的 SigLIP + DFormer 双塔编码器
    用于将语义与 3D 几何特征融合为 Token 序列
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 加载 SigLIP
        siglip_is_local = "/" in config.siglip_model_name
        self.siglip = SiglipVisionModel.from_pretrained(
            config.siglip_model_name,
            local_files_only=siglip_is_local
        )

        self.siglip_processor = SiglipImageProcessor.from_pretrained(
            config.siglip_model_name,
            local_files_only=siglip_is_local
        )
        
        # ==================== LoRA 与冻结逻辑 ====================
        self.use_lora = getattr(config, "use_lora", False)
        self.vision_freeze = getattr(config, "vision_freeze", True)
        
        if self.use_lora:
            peft_config = LoraConfig(
                r=getattr(config, "lora_rank", 16),
                lora_alpha=getattr(config, "lora_alpha", 32),
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=getattr(config, "lora_dropout", 0.05),
                bias="none",
            )
            self.siglip = get_peft_model(self.siglip, peft_config)
            
        elif self.vision_freeze:
            for param in self.siglip.parameters():
                param.requires_grad = False
            self.siglip.eval()
        # =========================================================
        
        # 2. 加载 DFormer
        self.dformer_backbone = DFomerRGBDBackbone(config)
        
        # 3. 维度配置
        target_dim = getattr(config, "transformer_n_emb", 384)
        self.feature_dim = target_dim
        self.siglip_dim = self.siglip.config.hidden_size
        self.dformer_stage_idx = 2 # 使用 Stage 3 特征 (H/16, W/16)
        
        dformer_channels = self.dformer_backbone.get_out_channels()
        self.dformer_dim = dformer_channels[self.dformer_stage_idx]

        # 4. 投影与融合层
        self.proj_siglip = nn.Linear(self.siglip_dim, target_dim)
        self.proj_dformer = nn.Linear(self.dformer_dim, target_dim)
        self.norm = nn.LayerNorm(target_dim)
        
        self.fusion_map = nn.Sequential(
            nn.Linear(target_dim * 2, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim)
        )

        # 5. Token 数量预计算
        h, w = config.resize_shape if config.resize_shape else (224, 224)
        self.patch_size = self.siglip.config.patch_size
        self.num_patches_h, self.num_patches_w = h // self.patch_size, w // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # 🌟 6. 生成并注册 2D Sine-Cosine 位置编码 (不可学习参数)
        # 使用 register_buffer 确保它能随模型被发送到 GPU，但不会更新梯度
        pos_embed = get_2d_sincos_pos_embed(target_dim, self.num_patches_h, self.num_patches_w)
        self.register_buffer("pos_embed", pos_embed)

        # 7. ToMe 配置
        self.tome_compress_ratio = getattr(config, "tome_compress_ratio", 1)
        # 🌟 新增：每次迭代允许合并的最大比例 (默认 5%)
        self.tome_step_ratio = getattr(config, "tome_step_ratio", 0.05)

        # ============================================================
        # 🌟 8. 调试与可视化配置
        # ============================================================
        self.enable_debug_vis = getattr(config, "enable_debug_vis", True) # 默认开启
        self.vis_save_freq = 100       # 每 100 步保存一次
        self.vis_save_dir = "discuss/tome_debug_logs" # 保存目录
        self._forward_step_count = 0   # 内部步数计数器
        
        if self.enable_debug_vis:
            os.makedirs(self.vis_save_dir, exist_ok=True)

    def _save_tome_visualization(self, rgb_tensor: torch.Tensor, tracker_tensor: torch.Tensor):
        """内部可视化封装函数，生成左中右三联图"""
        # 1. 提取并反归一化第一张图像 (假设 rgb_tensor 是 [B, C, H, W])
        img_t = rgb_tensor[0].detach().cpu().float()
        
        # 稳健的反归一化 (Min-Max 到 0-255)，适配各类输入规范
        img_min, img_max = img_t.min(), img_t.max()
        img_t = (img_t - img_min) / (img_max - img_min + 1e-5)
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        orig_h, orig_w = img_np.shape[:2]

        # 2. 提取 Tracker 掩码 [N_final, N_original]
        tracker = tracker_tensor[0].detach().cpu().numpy()
        n_final, n_orig = tracker.shape
        masks = tracker.reshape(n_final, self.num_patches_h, self.num_patches_w)

        # 3. 制作 Token 色块图 (Middle)
        np.random.seed(42) # 固定种子让同一位置的 Token 颜色在不同步数下相对一致
        colors = np.random.randint(50, 255, size=(n_final, 3), dtype=np.uint8)
        mask_img = np.zeros((self.num_patches_h, self.num_patches_w, 3), dtype=np.uint8)
        
        for i in range(n_final):
            y_idx, x_idx = np.where(masks[i] > 0.5) 
            mask_img[y_idx, x_idx] = colors[i]
            
        # 放大掩码图到原图尺寸
        mask_img_resized = cv2.resize(mask_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 4. 制作叠加图 (Right)
        alpha = 0.5
        overlay = cv2.addWeighted(img_np, 1 - alpha, mask_img_resized, alpha, 0)

        # 5. 绘制左中右三联图并保存
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"Left: Original RGB\nStep: {self._forward_step_count}")
        axes[0].axis("off")
        
        axes[1].imshow(mask_img_resized)
        axes[1].set_title(f"Middle: Token Clusters\n({n_orig} -> {n_final} Tokens)")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title("Right: Overlay (Check Edges)")
        axes[2].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_save_dir, f"tome_step_{self._forward_step_count:06d}.jpg")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # 极其重要：释放内存，防止训练 OOM

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        requires_grad = self.use_lora or (not self.vision_freeze)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        
        with context:
            siglip_in = self.siglip_processor(images=rgb, do_resize=False, do_rescale=False, return_tensors="pt")
            siglip_out = self.siglip(siglip_in['pixel_values'].to(rgb.device), interpolate_pos_encoding=True)
            siglip_feat = siglip_out.last_hidden_state # Shape: [B, N_sig, D_sig]
            
        sig_tokens = self.proj_siglip(siglip_feat) 

        df_feats = self.dformer_backbone(rgb, depth)
        df_map = df_feats[self.dformer_stage_idx] 

        if df_map.shape[2:] != (self.num_patches_h, self.num_patches_w):
            df_map = F.interpolate(df_map, size=(self.num_patches_h, self.num_patches_w), 
                                 mode='bilinear', align_corners=False)
        
        df_tokens = df_map.flatten(2).transpose(1, 2) 
        df_tokens = self.proj_dformer(df_tokens) 

        # E. 特征拼接与融合
        combined = torch.cat([sig_tokens, df_tokens], dim=-1) 
        fused_tokens = self.fusion_map(combined)
        fused_tokens = self.norm(fused_tokens)
        
        # ============================================================
        # 🌟 关键修改 1：在 ToMe 打乱空间前，注入 2D 绝对位置编码
        # ============================================================
        fused_tokens = fused_tokens + self.pos_embed

        # ============================================================
        # 🌟 关键修改 2：改良版 ToMe (多次小步幅压缩，精准锁定背景)
        # ============================================================
        B, N, D = fused_tokens.shape

        tracker = None
        
        if self.tome_compress_ratio > 1:
            target_tokens = N // self.tome_compress_ratio
            total_r = N - target_tokens
            
            token_size = torch.ones((B, N, 1), device=fused_tokens.device, dtype=fused_tokens.dtype)
            current_N = N

            # 初始化追踪器
            if self.enable_debug_vis and self.training:
                tracker = torch.eye(N, device=fused_tokens.device, dtype=fused_tokens.dtype)
                tracker = tracker.unsqueeze(0).expand(B, -1, -1)
            
            while total_r > 0:
                # 限制每次合并的数量，保护前景特征
                max_r_this_step = max(1, int(current_N * self.tome_step_ratio))
                current_r = min(total_r, max_r_this_step, current_N // 2)
                
                merge_fn = bipartite_soft_matching(fused_tokens, current_r)
                
                # 合并时，特征和位置编码会被一并加权平均，形成准确的中心坐标！
                fused_tokens, token_size = merge_wavg(merge_fn, fused_tokens, token_size)
                
                # 仅在需要可视化时更新 tracker，避免增加不必要的计算图负担
                if tracker is not None:
                    tracker = merge_fn(tracker, mode="sum")

                total_r -= current_r
                current_N -= current_r
        
        # ============================================================
        # 触发可视化保存
        # ============================================================
        if self.enable_debug_vis and self.training:
            self._forward_step_count += 1
            if self._forward_step_count % self.vis_save_freq == 0 and tracker is not None:
                # 建议扔进 torch.no_grad 防止干扰显存
                with torch.no_grad():
                    self._save_tome_visualization(rgb, tracker)
        return fused_tokens


class ResnetRgbEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Can't replace BatchNorm in pretrained model.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )
        images_shape = next(iter(config.image_features.values())).shape
        if config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_shape_h_w = (x_end - x_start, y_end - y_start)
            else:
                dummy_shape_h_w = config.crop_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.relu(self.out(x))
        return x  # (B, feature_dim)


class ResnetDepthEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        backbone_model = getattr(torchvision.models, config.depth_backbone)(weights=config.pretrained_backbone_weights)
        modules = list(backbone_model.children())[:-2]
        if isinstance(modules[0], nn.Conv2d):
            old_conv = modules[0]
            modules[0] = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                modules[0].weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        self.backbone = nn.Sequential(*modules)
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Can't replace BatchNorm in pretrained model.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )
        images_shape = next(iter(config.depth_features.values())).shape
        
        if config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_shape_h_w = (x_end - x_start, y_end - y_start)
            else:
                dummy_shape_h_w = config.crop_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, 1, *dummy_shape_h_w)

        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.relu(self.out(x))
        return x


class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        self.config = config
        backbone_type = config.vision_backbone
        
        if "resnet" in backbone_type:
            self.model = ResnetRgbEncoder(config)
        elif "dino" in backbone_type and "siglip" in backbone_type:
            self.model = DinoSiglipRGBEncoder(config)
        elif "siglip_dformer" in backbone_type: 
            self.model = SiglipDFormerEncoder(config)
        elif "siglip_only" in backbone_type:
            self.model = SiglipRGBEncoder(config)
        else:
            raise ValueError(f"Unknown vision backbone: {config.vision_backbone}")
            
        self.feature_dim = self.model.feature_dim
        self.is_rgbd_encoder = "siglip_dformer" in backbone_type 
    
    def forward(self, x: Tensor, x_depth: Optional[Tensor] = None) -> Tensor:
        if self.is_rgbd_encoder:
            return self.model(x, x_depth)
        return self.model(x)


class DiffusionDepthEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        self.config = config
        if "resnet" in config.depth_backbone:
            self.model = ResnetDepthEncoder(config)
        else:
            raise ValueError(f"Unknown depth backbone: {config.depth_backbone}")
        self.feature_dim = self.model.feature_dim
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


# ---------------------------
# State-guided fusion block (no discrete logic here)
# ---------------------------
class StateGuidedFusionBlock(nn.Module):
    """
    Projects modality features to a shared hidden dim and performs cross-attention.
    Inputs:
      - vis_feat: (B, N_v, vis_dim)
      - dep_feat: (B, N_d, dep_dim) or None
      - state_feat: (B, state_dim) or None  # ALREADY encoded / discretized in wrapper if required
    """
    def __init__(self, vis_dim: int, dep_dim: Optional[int], state_dim: Optional[int],
                 hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_vis = nn.Linear(vis_dim, hidden_dim)
        self.use_depth = dep_dim is not None
        if self.use_depth:
            self.proj_dep = nn.Linear(dep_dim, hidden_dim)

        self.use_state = state_dim is not None
        if self.use_state:
            # state_feat is expected already to be final size (wrapper ensures this)
            self.state_proj = nn.Linear(state_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vis_feat: Tensor, dep_feat: Optional[Tensor], state_feat: Optional[Tensor]) -> Tensor:
        # vis_feat: (B*s, N_v, vis_dim)
        B = vis_feat.shape[0]
        vis_proj = self.proj_vis(vis_feat)  # (B*s, N_v, hidden)

        if self.use_depth and dep_feat is not None:
            dep_proj = self.proj_dep(dep_feat)  # (B*s, N_d, hidden)
            kv = torch.cat([vis_proj, dep_proj], dim=1)  # (B*s, N_v+N_d, hidden)
        else:
            kv = vis_proj  # (B*s, N_v, hidden)

        if self.use_state and state_feat is not None:
            # state_feat is (B*s, state_dim)
            state_emb = self.state_proj(state_feat)  # (B*s, hidden)
            query = state_emb.unsqueeze(1)  # (B*s, 1, hidden)
        else:
            query = vis_proj.mean(dim=1, keepdim=True)  # (B*s, 1, hidden)

        fused, _ = self.cross_attn(query=query, key=kv, value=kv)
        fused = self.mlp(fused).squeeze(1)  # (B*s, hidden)
        return fused


# ---------------------------
# Main wrapper: integrate encoders, fusion, and diffusion model
# ---------------------------
class CustomDiffusionModelWrapper(DiffusionModel):
    """
    自定义扩散策略模型包装器 (Custom Diffusion Policy Model Wrapper)
    
    功能:
    1. 集成多种视觉 Backbone (ResNet, SigLIP, DINO, DFormer 等)。
    2. 支持 Perceiver Resampler 进行 Token 压缩。
    3. 支持 DiT (Diffusion Transformer) 和 UNet 作为去噪网络。
    4. 处理多模态输入 (RGB, Depth, Robot State) 的编码与融合。
    """

    def __init__(self, config: CustomDiffusionConfigWrapper):
        # =========================================================================
        # 1. 父类初始化 Hack (Parent Init Hack)
        # =========================================================================
        # LeRobot 父类检查比较严格，临时替换配置以绕过检查
        orig_vis = config.vision_backbone
        config.vision_backbone = "resnet18"
        orig_noise_scheduler = config.noise_scheduler_type
        config.noise_scheduler_type = "DDPM"
        
        super().__init__(config)
        
        # 恢复原始配置
        config.vision_backbone = orig_vis
        config.noise_scheduler_type = orig_noise_scheduler
        self.config = config

        # 基础参数
        global_cond_dim = 0
        vision_seq_len = 0
        self.cond_feat_dim = getattr(self.config, "transformer_n_emb", 384)

        # =========================================================================
        # 2. 机器人状态编码器 (Robot State Encoder)
        # =========================================================================
        self.state_encoder = None
        
        if getattr(self.config, "robot_state_feature", None) is not None:
            state_dim = self.config.robot_state_feature.shape[0]
            if getattr(self.config, "use_state_encoder", False):
                # 将低维状态映射到 transformer_n_emb 维度
                self.state_encoder = FeatureEncoder(state_dim, self.cond_feat_dim)
            else:
                logger.warning("⚠️ Robot state provided but `use_state_encoder` is False.")

        # =========================================================================
        # 3. 视觉编码器 (RGB Encoders)
        # =========================================================================
        if getattr(self.config, "image_features", None):
            num_images = len(self.config.image_features)
            
            # 情况 A: 每个摄像头使用独立的编码器 (不共享权重)
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                encs = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encs)
                one_enc = encs[0].model
            # 情况 B: 所有摄像头共享同一个编码器 (共享权重)
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                one_enc = self.rgb_encoder.model
            
            # 计算原始 Patch 数量
            patches_per_img = one_enc.num_patches
            vision_seq_len = num_images * patches_per_img

        # =========================================================================
        # 4. Perceiver Resampler (Per-Camera Token Compression)
        # =========================================================================
        self.use_perceiver = getattr(self.config, "use_perceiver", False)
        if self.use_perceiver:
            perceiver_depth = getattr(self.config, "perceiver_depth", 2)
            perceiver_heads = getattr(self.config, "transformer_n_head", 8)

            # 每个相机的 queries 数量列表，长度必须等于 num_images
            queries_per_camera = getattr(self.config, "perceiver_queries_per_camera", None)
            if queries_per_camera is None:
                raise ValueError(
                    "❌ `use_perceiver=True` requires `perceiver_queries_per_camera` "
                    "(a list of ints, one per camera). e.g. [32, 64, 64]"
                )
            if len(queries_per_camera) != num_images:
                raise ValueError(
                    f"❌ `perceiver_queries_per_camera` length ({len(queries_per_camera)}) "
                    f"must match the number of cameras ({num_images})."
                )

            # 为每个相机创建独立的 PerceiverResampler
            self.perceivers = nn.ModuleList([
                PerceiverResampler(
                    dim=self.cond_feat_dim,
                    num_queries=q,
                    depth=perceiver_depth,
                    heads=perceiver_heads,
                )
                for q in queries_per_camera
            ])
            self.perceiver_queries_per_camera = list(queries_per_camera)

            # [关键] 更新序列长度：经过各相机 Resampler 后，Token 总数为所有 queries 之和
            vision_seq_len = sum(queries_per_camera)
        else:
            self.perceivers = None
            self.perceiver_queries_per_camera = None

        # =========================================================================
        # 5. 上下文长度计算 (Context Length Calculation)
        # =========================================================================
        # Total Tokens = Vision Tokens * History Steps
        tokens_per_step = vision_seq_len
        total_cond_len = self.config.n_obs_steps * tokens_per_step

        # =========================================================================
        # 6. 扩散模型核心 (Core Diffusion Model: UNet or DiT)
        # =========================================================================
        if config.use_unet:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        elif config.use_transformer:
            # 标准 Transformer 实现
            self.unet = TransformerForDiffusion(
                input_dim=config.output_features["action"].shape[0],
                output_dim=config.output_features["action"].shape[0],
                horizon=config.horizon,
                n_obs_steps=total_cond_len,
                cond_dim=self.cond_feat_dim,
                n_layer=self.config.transformer_n_layer,
                n_head=self.config.transformer_n_head,
                n_emb=self.config.transformer_n_emb,
                p_drop_emb=self.config.transformer_dropout,
                p_drop_attn=self.config.transformer_dropout,
                causal_attn=False,
                time_as_cond=True,
                obs_as_cond=True,
                n_cond_layers=0,
            )
        elif config.use_dit:
            # DiT (Diffusion Transformer) 实现，规格由 config 控制：
            # DiT-S: transformer_n_emb=384, transformer_n_head=6,  transformer_n_layer=12
            # DiT-B: transformer_n_emb=768, transformer_n_head=12, transformer_n_layer=12
            # image_features 只包含 RGB 摄像头（depth 已由 custom_patches 标记为 FeatureType.DEPTH 并排除）
            # 因此 len(image_features) 即为实际产生 token 的摄像头数（如 3），不含 depth
            n_cameras = len(self.config.image_features) if getattr(self.config, "image_features", None) else 1
            self.unet = DiT(
                action_dim=config.output_features["action"].shape[0],
                action_seq_len=config.horizon,
                n_obs_steps=self.config.n_obs_steps,
                token_dim=self.cond_feat_dim,
                max_image_tokens=vision_seq_len,
                num_cameras=n_cameras,
                hidden_size=self.config.transformer_n_emb,
                depth=self.config.transformer_n_layer,
                num_heads=self.config.transformer_n_head,
            )
        else:
            raise ValueError("❌ Config Error: Either `use_unet`, `use_transformer` or `use_dit` must be True.")

        # =========================================================================
        # 7. 噪声调度器 (Noise Scheduler)
        # =========================================================================
        self.noise_scheduler = _make_noise_scheduler_factory(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = config.num_inference_steps or self.noise_scheduler.config.num_train_timesteps

        self._log_model_architecture(vision_seq_len, total_cond_len)

    def _log_model_architecture(self, vision_seq_len, total_cond_len):
        """以表格形式打印模型架构信息"""
            
        # 确定去噪网络类型
        if self.config.use_dit:
            denoiser = "Transformer (DiT)"
        elif self.config.use_transformer:
            denoiser = "Standard Transformer"
        else:
            denoiser = "Conditional UNet-1D"

        # 构造配置字典
        arch_info = {
            "Denoiser Type": denoiser,
            "Condition Dim": self.cond_feat_dim,
            "Obs Steps (S)": self.config.n_obs_steps,
            "Tokens Per Step": f"{vision_seq_len} (Vision)",
            "Total Cond Len": f"{total_cond_len} Tokens",
            "Perceiver Resampler": f"✅ per-cam queries={self.perceiver_queries_per_camera}, total={sum(self.perceiver_queries_per_camera)}" if self.use_perceiver else "❌ Disabled",
            "Action Horizon": f"{self.config.horizon} steps",
            "Noise Scheduler": f"{self.config.noise_scheduler_type} ({self.num_inference_steps} steps)",
        }

        # 调用 log_box (确保环境中已定义该函数)
        log_box("Diffusion Policy Architecture", arch_info, icon="🤖")

    def _prepare_global_conditioning(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        准备全局条件特征 (Prepare Global Conditioning)
        
        流程:
        1. 提取 RGB 特征 (可能包含多个摄像头)。
        2. (可选) 通过 Perceiver Resampler 压缩视觉 Token。
        3. 提取并编码 Robot State。
        4. 拼接所有 Token，形成 DiT 的 Condition 输入。
        
        Returns:
            global_cond: [B, S * (N_vis + N_state), D]
        """
        B = batch[OBS_STATE].shape[0]
        S = batch[OBS_STATE].shape[1]  # n_obs_steps
        tokens_list = []

        # ---------------------------------------------------------------------
        # 1. RGB Features Processing
        # ---------------------------------------------------------------------
        if getattr(self.config, "image_features", None):

            # A. 提取特征，统一为 [B*S*N_cam, P, D]
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                # 独立编码器：输入 [N_cam, B*S, C, H, W] -> 输出 list of [B*S, N_patches, D]
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                enc_outs = [enc(im) for enc, im in zip(self.rgb_encoder, imgs)]
                # stack -> [B*S, N_cam, N_patches, D] -> [B*S*N_cam, N_patches, D]
                vis_feats = einops.rearrange(torch.stack(enc_outs, dim=1), "(b s) n p d -> (b s n) p d", b=B, s=S)
            else:
                # 共享编码器：输入 [B*S*N_cam, C, H, W] -> 输出 [B*S*N_cam, N_patches, D]
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")

                depths = None
                if OBS_DEPTH in batch:
                    depths = batch[OBS_DEPTH]
                    depths = einops.rearrange(depths, "b s n c h w -> (b s n) c h w")

                vis_feats = self.rgb_encoder(imgs, depths)

            # B. Per-Camera Perceiver 压缩 或 直接拼接
            if self.use_perceiver:
                # vis_feats: [B*S*N_cam, P, D] -> [B*S, N_cam, P, D]
                N_cam = len(self.perceivers)
                vis_per_cam = einops.rearrange(vis_feats, "(b s n) p d -> (b s) n p d", b=B, s=S, n=N_cam)

                # 对每个相机独立压缩，得到 [B*S, Q_i, D]，再在 token 维度拼接
                cam_tokens = [
                    self.perceivers[i](vis_per_cam[:, i, :, :])
                    for i in range(N_cam)
                ]
                # [B*S, sum(Q_i), D]
                vis_tokens = torch.cat(cam_tokens, dim=1)
            else:
                # 重排并拼接 -> [B*S, N_cam * N_patches, D]
                vis_tokens = einops.rearrange(vis_feats, "(b s n) p d -> (b s) (n p) d", b=B, s=S)

            # C. 恢复时间维度
            # [B*S, N_tokens, D] -> [B, S, N_tokens, D]
            vis_tokens = einops.rearrange(vis_tokens, "(b s) t d -> b s t d", b=B, s=S)
            tokens_list.append(vis_tokens)

        # ---------------------------------------------------------------------
        # 2. Robot State Processing
        # ---------------------------------------------------------------------
        if getattr(self.config, "robot_state_feature", None) is not None:
            state_tensor = batch[OBS_STATE]  # [B, S, state_dim]
            
            if self.state_encoder is not None:
                # [B, S, state_dim] -> [B, S, transformer_n_emb]
                state_emb = self.state_encoder(state_tensor)
                # 增加 Token 维度 -> [B, S, 1, D]
                state_tokens = state_emb.unsqueeze(2)
                tokens_list.append(state_tokens)

        # ---------------------------------------------------------------------
        # 3. Concatenate & Return
        # ---------------------------------------------------------------------
        # 在 Token 维度拼接: Vision + State
        # Shape: [B, S, Total_Tokens_Per_Step, D]
        combined = torch.cat(tokens_list, dim=2)
        
        # DiT 接受 [B, Total_Seq_Len, D] 或 [B, S, T, D] 取决于具体实现
        # 这里返回 [B, S, T, D]，DiT 内部通常会 flatten 前两个维度
        return combined

    # ---------------------------
    # Inference sampling
    # ---------------------------
    def conditional_sample(self, batch_size: int, global_cond: Optional[Tensor] = None, generator=None, noise: Tensor | None = None) -> Tensor:
        """
        执行扩散去噪采样过程
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # 1. 初始化噪声
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        
        # 2. 设置时间步
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # 3. 逐步去噪
        for t in self.noise_scheduler.timesteps:
            # 预测噪声/样本
            model_output = self.unet(
                sample,
                torch.full((batch_size,), t, dtype=torch.long, device=device),
                global_cond=global_cond,
            )
            
            # Scheduler Step
            # 兼容 DDIM eta 参数
            step_kwargs = {"generator": generator}
            if "eta" in self.noise_scheduler.step.__code__.co_varnames:
                step_kwargs["eta"] = getattr(self.config, "ddim_eta", 0.0)
                
            step_out = self.noise_scheduler.step(model_output, t, sample, **step_kwargs)
            
            # 兼容不同 diffusers 版本的输出格式
            sample = getattr(step_out, "prev_sample", step_out)

        return sample