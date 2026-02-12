# multimodal_diffusion_wrapper.py
import math
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any
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
from kuavo_train.wrapper.policy.diffusion_new.DiT_1D_AdaLN import DiT_S

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


class DinoSiglipBackbone(nn.Module):
    """
    DINOv2 + SigLIP 双塔视觉 Backbone。
    功能：
    1. 并行提取特征。
    2. 处理不同的 Patch Size 和输入分辨率（自动插值）。
    3. 输出类似 ResNet 的 [B, C, H, W] 特征图。
    """
    def __init__(self, config):
        super().__init__()
        
        self.dinov2_model_name = config.dinov2_model_name
        self.siglip_model_name = config.siglip_model_name
        self.vision_freeze = getattr(config, "vision_freeze", True)

        # 1. 加载模型
        try:
            self.dinov2 = AutoModel.from_pretrained(self.dinov2_model_name)
            self.siglip = SiglipVisionModel.from_pretrained(self.siglip_model_name)
        except Exception as e:
            logger.error(f"❌ Failed to load models. Check internet or cache.")
            raise e
        
        # 获取各自的 Patch Size
        self.dino_p = self.dinov2.config.patch_size
        self.siglip_p = self.siglip.config.patch_size

        # 2. 冻结参数
        if self.vision_freeze:
            self.dinov2.requires_grad_(False)
            self.siglip.requires_grad_(False)
            self.dinov2.eval()
            self.siglip.eval()
        else:
            self.dinov2.train()
            self.siglip.train()

    def _process_feature_map(self, feat, H, W, patch_size, model_name):
        """
        内部辅助函数：处理 CLS Token 并将序列还原为网格
        """
        B, N, D = feat.shape
        grid_h, grid_w = H // patch_size, W // patch_size
        expected_patches = grid_h * grid_w

        # 1. 检查并移除 CLS Token
        # 如果序列长度比网格多1，说明有CLS token
        if N == expected_patches + 1:
            feat = feat[:, 1:, :]
            N -= 1
        
        # 2. 严格的形状检查 (Safety Check)
        if N != expected_patches:
            raise ValueError(f"Shape Mismatch in {model_name}: {N} vs {expected_patches}")

        # 3. Reshape & Permute
        # [B, N, D] -> [B, h, w, D] -> [B, D, h, w]
        grid = feat.view(B, grid_h, grid_w, D).permute(0, 3, 1, 2)
        return grid

    def forward(self, x):
        # x: [B, 3, H, W]
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {x.shape}")
            
        B, C, H, W = x.shape
        
        # -----------------------------------------------------------
        # 1. 前向传播
        # -----------------------------------------------------------
        # 根据是否冻结决定是否使用 no_grad，节省显存
        context = torch.no_grad() if self.vision_freeze else torch.enable_grad()
        
        with context:
            # DINOv2
            dinov2_feat = self.dinov2(x).last_hidden_state
            
            # SigLIP (必须开启 interpolate_pos_encoding)
            siglip_feat = self.siglip(x, interpolate_pos_encoding=True).last_hidden_state

        # -----------------------------------------------------------
        # 2. 还原为网格 (Grid)
        # -----------------------------------------------------------
        
        # 处理 DINOv2
        dino_grid = self._process_feature_map(dinov2_feat, H, W, self.dino_p, self.dinov2_model_name)
        

        # 处理 SigLIP
        siglip_grid = self._process_feature_map(siglip_feat, H, W, self.siglip_p, self.siglip_model_name)

        return dino_grid.contiguous(), siglip_grid.contiguous()


class DinoSiglipRGBEncoder(nn.Module):
    """
    DINO + SigLIP 双塔视觉编码器 (Dual-Tower Vision Encoder)
    
    功能：
    1. 并行运行 SigLIP (语义强) 和 DINO (几何强) 模型。
    2. 支持 DINOv2 和 DINOv3 (通过 AutoModel 加载)。
    3. 分别处理两种模型的不同归一化需求。
    4. 输出拼接后的 Token 序列，供下游 (如 Perceiver Resampler) 使用。
    
    输出形状:
        [Batch, N_siglip + N_dino, projection_dim]
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
        
        # 检查是否有寄存器 token (DINOv2-registers 模型)
        self.num_registers = getattr(self.dino.config, "num_register_tokens", 0)
        self.num_dino_tokens = grid_dino + 1 # +1 是 CLS Token

        # 总 Token 数
        self.num_patches = self.num_siglip_tokens + self.num_dino_tokens

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
            dino_feat = torch.cat([cls_token, patch_tokens], dim=1)

        # 3. 投影与归一化 (Project & Normalize)
        # 将不同维度的特征映射到同一维度 (如 384)
        siglip_tokens = self.norm_siglip(self.proj_siglip(siglip_feat))
        dino_tokens = self.norm_dino(self.proj_dino(dino_feat))
        
        # 4. 拼接 (Concatenation)
        # 在序列维度拼接: [B, N_sig + N_dino, D]
        # 下游的 Perceiver Resampler 会负责处理这个变长的序列
        combined_tokens = torch.cat([siglip_tokens, dino_tokens], dim=1)

        return combined_tokens

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
        初始化 DFormer RGB-D backbone
        
        Args:
            model_size (str): 模型规格，可选 'small', 'base', 'large'
            pretrained_path (str, optional): 预训练权重路径 (.pth). 默认为 None.
            freeze_backbone (bool): 是否冻结骨干网络参数 (用于微调下游任务). 默认为 False.
        """
        super().__init__()
        model_name = config.vision_backbone_rgbd
        pretrained_path = config.DFormer_path
        vision_freeze = config.vision_freeze
        self.model_size = model_name.split("_")[-1]
        
        # 1. 配置映射 (用于实例化和日志展示)
        # ---------------------------------------------------------
        configs = {
            'small': {'fn': DFormerv2_S, 'dims': [64, 128, 256, 512], 'desc': 'Small (High Speed)'},
            'base':  {'fn': DFormerv2_B, 'dims': [80, 160, 320, 512], 'desc': 'Base (Balanced)'},
            'large': {'fn': DFormerv2_L, 'dims': [112, 224, 448, 640], 'desc': 'Large (High Perf)'},
        }
        
        if self.model_size not in configs:
            logger.error(f"❌ Invalid model size: {self.model_size}")
            raise ValueError(f"Choose from {list(configs.keys())}")
            
        cfg = configs[self.model_size]
        self.out_channels = cfg['dims']

        # 2. 实例化 Backbone
        # ---------------------------------------------------------
        logger.info(f"🏗️  Building DFormer architecture: {Colors.CYAN}{cfg['desc']}{Colors.RESET}")
        self.backbone = cfg['fn']()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.backbone.parameters())
        param_str = f"{total_params / 1e6:.2f} M"

        # 3. 加载权重
        # ---------------------------------------------------------
        if pretrained_path:
            try:
                self.backbone.load_pretrained(pretrained_path)
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")

        # 4. 冻结参数
        # ---------------------------------------------------------
        freeze_status = f"{Colors.RED}No (Trainable){Colors.RESET}"
        if vision_freeze:
            self._freeze_params()
            freeze_status = f"{Colors.CYAN}Yes (Frozen ❄️){Colors.RESET}"

        # 5. 打印 info
        # ---------------------------------------------------------
        info_dict = {
            "Architecture": f"DFormer-v2 {self.model_size.title()}",
            "Out Channels": str(self.out_channels),
            "Total Params": param_str,
            "Backbone Freeze": freeze_status,
            "Input Mode": "RGB + Depth/Edge"
        }
        
        log_box("DFormer RGBD Encoder Setup", info_dict, icon="🧠")

    def _freeze_params(self):
        """冻结 backbone 所有参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x, x_depth):
        """
        Args:
            x (Tensor): RGB 图像 [B, 3, H, W]
            x_depth (Tensor): 深度图/边缘图 [B, 1, H, W]
            
        Returns:
            list[Tensor]: 多尺度特征列表
                - Stage 1: [B, C1, H/4, W/4]
                - Stage 2: [B, C2, H/8, W/8]
                - Stage 3: [B, C3, H/16, W/16]
                - Stage 4: [B, C4, H/32, W/32]
        """
        # 直接调用 DFormerv2 的 forward
        features = self.backbone(x, x_depth)
        
        # DFormerv2 返回的是 tuple，通常转为 list 方便后续操作
        return list(features)

    def get_out_channels(self):
        """辅助函数：让 Decoder 知道每一层的通道数"""
        return self.out_channels


class DFomerRGBDEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 1. 初始化 Backbone
        self.backbone = DFomerRGBDBackbone(config)
        
        # 2. 获取通道和分辨率信息以初始化池化层
        # 以 Base 版为例: [80, 160, 320, 512]
        channels_list = self.backbone.get_out_channels()
        
        # 融合 Stage 2, 3, 4 的通道
        self.total_fused_channels = sum(channels_list[1:]) # 160 + 320 + 512 = 992
        
        # 获取输入分辨率和 Stage 3 的网格大小 (用于 SpatialSoftmax)
        first_img_shape = next(iter(config.image_features.values())).shape
        h, w = config.resize_shape if config.resize_shape else first_img_shape[1:]
        self.grid_h, self.grid_w = h // 16, w // 16 
        
        # 3. 初始化层次化 SpatialSoftmax
        self.num_kp = config.spatial_softmax_num_keypoints
        # 这里的 input_shape 必须匹配拼接后的 [C, H, W]
        self.pool = SpatialSoftmax(
            [self.total_fused_channels, self.grid_h, self.grid_w], 
            num_kp=self.num_kp
        )
        
        # 4. 统一输出维度
        self.feature_dim = self.num_kp * 2
        self.out = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU()
        )

        # DEBUG
        self.last_log_time = 0      # 上次保存图片的时间
        self.log_interval = 600.0    # 设定间隔：30 秒 (你可以随意改)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(rgb, depth)
        f2, f3, f4 = feats[1], feats[2], feats[3]
        
        target_size = (self.grid_h, self.grid_w)
        f2_p = F.adaptive_avg_pool2d(f2, target_size) 
        f3_p = F.adaptive_avg_pool2d(f3, target_size) 
        
        f4_aligned = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
        
        # 多尺度特征拼接
        # 形状变化: [B, 992, H, W] (即 160+320+512)
        fused_map = torch.cat([f2_p, f3_p, f4_aligned], dim=1)
        
        # Step 4: Spatial Softmax 提取坐标
        # 内部会进行 1x1 卷积将 992 通道投影到 num_kp
        # 形状变化: [B, num_kp, 2] (即 64个关键点的 x,y 坐标)
        kp = self.pool(fused_map)
        
        # [DEBUG START] ==========================================================
        # 无条件执行：每次都画图，覆盖同一张文件
        current_time = time.time()
        
        # 如果距离上次打印超过了设定的间隔
        if current_time - self.last_log_time > self.log_interval:
            # 立即更新时间戳 (防止短时间内多次进入)
            self.last_log_time = current_time

            try:
                # 选取 Batch 第 0 张
                idx = 0
                
                # --- 数据准备 (转 Numpy) ---
                # RGB
                rgb_vis = rgb[idx].detach().cpu().permute(1, 2, 0).numpy()
                rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min() + 1e-8)
                
                # Depth (降维 [1, H, W] -> [H, W])
                depth_vis = depth[idx].detach().cpu().squeeze(0).numpy()
                depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)

                # Feature Map (Mean across channels)
                heatmap_tensor = fused_map[idx].mean(dim=0)
                heatmap_vis = heatmap_tensor.detach().cpu().numpy()
                
                # Keypoints (还原到像素坐标)
                kps = kp[idx].detach().cpu().numpy() # [num_kp, 2]
                H, W = rgb_vis.shape[:2]
                
                # 假设 SpatialSoftmax 输出范围 [-1, 1]
                kp_x = (kps[:, 0] + 1) / 2 * W
                kp_y = (kps[:, 1] + 1) / 2 * H
                
                # --- 绘图 ---
                # 创建画布 (如果不 close 会内存泄露，所以下面必须 close)
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # 1. RGB
                axes[0].imshow(rgb_vis)
                axes[0].set_title("RGB Input")
                axes[0].axis('off')
                
                # 2. Depth
                axes[1].imshow(depth_vis, cmap='magma')
                axes[1].set_title("Depth Input")
                axes[1].axis('off')

                # 3. Features
                axes[2].imshow(heatmap_vis, cmap='viridis')
                axes[2].set_title(f"Fused Features")
                axes[2].axis('off')
                
                # 4. Result Overlay
                axes[3].imshow(rgb_vis)
                axes[3].scatter(kp_x, kp_y, c='red', s=30, marker='x', alpha=0.7)
                axes[3].set_title(f"Spatial Softmax")
                axes[3].axis('off')
                
                # 保存并覆盖
                save_path = "debug_dformer_latest.png"
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close(fig) # 关键：释放内存
                
                # 可选：打印一句提示，如果你觉得刷屏太快可以注释掉
                # print(f"📸 [DEBUG] Updated: {save_path}")
                
            except Exception as e:
                print(f"❌ [DEBUG] Vis Error: {e}")
        # [DEBUG END] ============================================================

        # Step 5: 展平并对齐维度
        # 形状变化: [B, num_kp * 2] (即 128 维特征向量)
        x = torch.flatten(kp, start_dim=1)
        
        # Step 6: 最终投影与激活
        # 形状: [B, 128]
        # 此时输出维度完全匹配 DinoSiglipRGBEncoder.feature_dim
        x = self.out(x)
        
        return x


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
        if "resnet" in config.vision_backbone:
            self.model = ResnetRgbEncoder(config)
        elif "dino" in config.vision_backbone and "siglip" in config.vision_backbone:
            self.model = DinoSiglipRGBEncoder(config)
        elif "siglip_only" in config.vision_backbone:
            self.model = SiglipRGBEncoder(config)
        else:
            raise ValueError(f"Unknown vision backbone: {config.vision_backbone}")
        self.feature_dim = self.model.feature_dim
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class DiffusionRGBDEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        self.config = config
        if "DFormer" in config.vision_backbone_rgbd:
            self.model = DFomerRGBDEncoder(config)
        else:
            raise ValueError(f"Unknown RGBD backbone: {config.vision_backbone_rgbd}")
        self.feature_dim = self.model.feature_dim
    def forward(self, x: Tensor, x_depth: Tensor) -> Tensor:
        return self.model(x, x_depth)


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
        # 4. Perceiver Resampler (Token Compression)
        # =========================================================================
        self.use_perceiver = getattr(self.config, "use_perceiver", False)
        if self.use_perceiver:
            perceiver_queries = getattr(self.config, "perceiver_num_queries", 64)
            perceiver_depth = getattr(self.config, "perceiver_depth", 2)
            
            self.perceiver = PerceiverResampler(
                dim=self.cond_feat_dim,          # 必须匹配 DiT/Encoder 的 transformer_n_emb
                num_queries=perceiver_queries,
                depth=perceiver_depth,
                heads=getattr(self.config, "transformer_n_head", 8)
            )
            
            # [关键] 更新序列长度：经过 Resampler 后，Token 数固定为 Queries 数
            vision_seq_len = perceiver_queries
        else:
            self.perceiver = None

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
            # DiT (Diffusion Transformer) 实现
            self.unet = DiT_S(
                action_dim=config.output_features["action"].shape[0],
                action_seq_len=config.horizon,
                n_obs_steps=self.config.n_obs_steps,
                token_dim=self.cond_feat_dim,
                max_image_tokens=vision_seq_len      
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
            "Perceiver Resampler": f"✅ {self.config.perceiver_num_queries} queries" if self.use_perceiver else "❌ Disabled",
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
            
            # A. 提取特征
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                # 独立编码器：输入 [N_cam, B*S, C, H, W] -> 输出 list of [B*S, N_patches, D]
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                enc_outs = [enc(im) for enc, im in zip(self.rgb_encoder, imgs)]
                # 拼接所有摄像头的 Patch -> [B*S, N_cam * N_patches, D]
                vis_tokens = torch.cat(enc_outs, dim=1)
            else:
                # 共享编码器：输入 [B*S*N_cam, C, H, W] -> 输出 [B*S*N_cam, N_patches, D]
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                vis_feats = self.rgb_encoder(imgs)
                # 重排并拼接 -> [B*S, N_cam * N_patches, D]
                vis_tokens = einops.rearrange(vis_feats, "(b s n) p d -> (b s) (n p) d", b=B, s=S)

            # B. Perceiver 压缩 (关键步骤)
            if self.use_perceiver:
                # Input:  [B*S, Total_Raw_Tokens, D]
                # Output: [B*S, Num_Queries, D] (例如 64 个)
                vis_tokens = self.perceiver(vis_tokens)
                
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