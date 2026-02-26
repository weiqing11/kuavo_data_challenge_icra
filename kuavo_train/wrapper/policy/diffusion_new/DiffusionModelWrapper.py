# multimodal_diffusion_wrapper.py
import json
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

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, ACTION
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
        self.debug_image_stats = getattr(config, "debug_image_stats", True)
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
        
        # ==================== 修复：新增 LoRA 与冻结逻辑 ====================
        self.use_lora = getattr(config, "use_lora", False)
        # 获取全局冻结配置，默认 True (冻结)
        self.vision_freeze = getattr(config, "vision_freeze", True)
        
        if self.use_lora:
            # logger.info(f"🔧 Applying LoRA to SigLIP Backbone...") # 如果有logger可以解除注释
            peft_config = LoraConfig(
                r=getattr(config, "lora_rank", 16),
                lora_alpha=getattr(config, "lora_alpha", 32),
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=getattr(config, "lora_dropout", 0.05),
                bias="none",
            )
            # get_peft_model 会自动冻结 SigLIP 的主干参数，只允许 LoRA 层求导
            self.siglip = get_peft_model(self.siglip, peft_config)
            
        elif self.vision_freeze:
            # 如果不使用 LoRA 且开启了冻结，则强行冻结 SigLIP 全参
            for param in self.siglip.parameters():
                param.requires_grad = False
            self.siglip.eval()
        # ===================================================================
        
        # 默认采样 DFormer Stage1/2/3，可通过 config.siglip_dformer_stages 覆盖
        self.dformer_stage_indices = getattr(
            config, "siglip_dformer_stages", (0, 1, 2)
        )

        # 2. 加载 DFormer (其内部已经根据 config.vision_freeze 做了冻结判断)
        self.dformer_backbone = DFomerRGBDBackbone(config)
        
        # 3. 维度配置
        self.feature_dim = getattr(config, "transformer_n_emb", 384)
        self.siglip_dim = self.siglip.config.hidden_size
        self.dformer_stage_idx = 2 # 使用 Stage 3 特征 (H/16, W/16)

        # 4. 投影与融合层
        self.proj_siglip = nn.Linear(self.siglip_dim, self.feature_dim)
        self.dformer_proj = nn.ModuleDict({
            f"stage_{idx}": nn.Linear(
                self.dformer_backbone.get_out_channels()[idx],
                self.feature_dim
            )
            for idx in self.dformer_stage_indices
        })
        self.norm = nn.LayerNorm(self.feature_dim)
        
        fusion_in_dim = self.feature_dim * (1 + len(self.dformer_stage_indices))
        self.fusion_map = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_in_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_in_dim * 2, self.feature_dim),
        )

        # 5. Token 数量预计算
        h, w = config.resize_shape if config.resize_shape else (224, 224)
        self.patch_size = self.siglip.config.patch_size
        self.num_patches_h, self.num_patches_w = h // self.patch_size, w // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

    def _log_image_stats(self, name: str, tensor: torch.Tensor) -> None:
        stats = {
            "min": tensor.detach().amin().item(),
            "max": tensor.detach().amax().item(),
            "mean": tensor.detach().mean().item(),
            "std": tensor.detach().std(unbiased=False).item(),
        }
        logger.debug(f"[SiglipDFormerEncoder] {name} stats: {stats}")

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        requires_grad = self.use_lora or (not self.vision_freeze)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        
        with context:
            if self.debug_image_stats:
                self._log_image_stats("RGB Input", rgb)
            siglip_inputs = self.siglip_processor(
                images=rgb,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt"
            )
            siglip_pixels = siglip_inputs["pixel_values"].to(
                rgb.device,
                dtype=next(self.siglip.parameters()).dtype
            )
            if self.debug_image_stats:
                self._log_image_stats("SigLIP Pixel Values", siglip_pixels)
            siglip_out = self.siglip(siglip_pixels, interpolate_pos_encoding=True)
            sig_hidden = getattr(siglip_out, "last_hidden_state", siglip_out[0])
        
        sig_tokens = self.proj_siglip(sig_hidden)
        
        # B. DFormer 分支 (RGB-D 几何)
        df_feats = self.dformer_backbone(rgb, depth)
        df_tokens = []
        for idx in self.dformer_stage_indices:
            feat = df_feats[idx]
            pooled = F.adaptive_avg_pool2d(
                feat,
                (self.num_patches_h, self.num_patches_w)
            )
            tokens = pooled.flatten(2).transpose(1, 2)
            tokens = self.dformer_proj[f"stage_{idx}"](tokens)
            df_tokens.append(tokens)
        df_tokens = torch.cat(df_tokens, dim=-1)

        fused_tokens = torch.cat([sig_tokens, df_tokens], dim=-1)
        fused_tokens = self.fusion_map(fused_tokens)
        return self.norm(fused_tokens)


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

        self.use_view_scoring = getattr(config, "use_view_scoring", True)
        self.view_score_temp = getattr(config, "view_score_temperature", 0.5)
        self.view_score_net = nn.Sequential(
            nn.Linear(config.transformer_n_emb * 2, config.transformer_n_emb // 4),
            nn.GELU(),
            nn.Linear(config.transformer_n_emb // 4, 1),
        )
        self.view_score_log_interval = getattr(config, "view_score_log_interval", 100)
        self.view_score_log_path = Path(getattr(config, "view_score_log_path", "discuss/view_scores.json"))
        self._view_score_call_count = 0

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
        # 1. Robot State Processing
        # ---------------------------------------------------------------------
        if getattr(self.config, "robot_state_feature", None) is not None:
            state_tensor = batch[OBS_STATE]  # [B, S, state_dim]
            
            if self.state_encoder is not None:
                # [B, S, state_dim] -> [B, S, transformer_n_emb]
                state_emb = self.state_encoder(state_tensor)
                # 增加 Token 维度 -> [B, S, 1, D]
                state_tokens = state_emb.unsqueeze(2)

        # ---------------------------------------------------------------------
        # 2. RGB Features Processing
        # ---------------------------------------------------------------------
        if getattr(self.config, "image_features", None):
            
            # A. 提取特征
            num_cams = batch[OBS_IMAGES].shape[2]
            per_view_tokens = []

            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                enc_outs = [enc(im) for enc, im in zip(self.rgb_encoder, imgs)]
                per_view_tokens = enc_outs  # list of [B*S, N_patch, D]
            else:
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                depths = None
                if OBS_DEPTH in batch:
                    depths = einops.rearrange(batch[OBS_DEPTH], "b s n c h w -> (b s n) c h w")
                vis_feats = self.rgb_encoder(imgs, depths)  # (B*S*N_cam, N_patch, D)
                vis_feats = einops.rearrange(
                    vis_feats, "(b s n) p d -> (b s) n p d", b=B, s=S, n=num_cams
                )
                per_view_tokens = [vis_feats[:, i] for i in range(num_cams)]

            if self.use_view_scoring and len(per_view_tokens) > 1:
                # [B*S, N_cam, D]
                view_stats = torch.stack([t.mean(dim=1) for t in per_view_tokens], dim=1)
                if state_emb is not None:
                    # state_emb: [B, S, D_state] -> [B*S, D_state]
                    state_emb_flat = einops.rearrange(state_emb, "b s d -> (b s) d")
                    # 扩展到所有摄像头视角: [B*S, N_cam, D_state]
                    state_emb_expanded = state_emb_flat.unsqueeze(1).expand(-1, num_cams, -1)
                    
                    # 拼接输入: [B*S, N_cam, D_vis + D_state]
                    score_input = torch.cat([view_stats, state_emb_expanded], dim=-1)
                else:
                    score_input = view_stats
                # 计算 Logits
                logits = self.view_score_net(score_input).squeeze(-1)
                # 计算注意力权重
                weights = torch.sigmoid(logits / self.view_score_temp)
                self._last_view_weights = weights
                self._maybe_log_view_scores(weights)
                
                weighted_tokens = []
                for idx, tokens in enumerate(per_view_tokens):
                    w = weights[:, idx].unsqueeze(-1).unsqueeze(-1)
                    weighted_tokens.append(tokens * w)
                vis_tokens = torch.cat(weighted_tokens, dim=1)
            else:
                vis_tokens = torch.cat(per_view_tokens, dim=1)

            if self.use_perceiver:
                vis_tokens = self.perceiver(vis_tokens)
                
            # C. 恢复时间维度
            # [B*S, N_tokens, D] -> [B, S, N_tokens, D]
            vis_tokens = einops.rearrange(vis_tokens, "(b s) t d -> b s t d", b=B, s=S)
            tokens_list.append(vis_tokens)

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
    
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch[ACTION]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)
        
        # 1. 计算原本的主任务 Loss (Diffusion 去噪 Loss)
        diffusion_loss = loss.mean()
        total_loss = diffusion_loss

        # 2. 【新增】：计算视角权重的熵正则化 Loss (Entropy Loss)
        # 只有在启用了 view scoring 并且成功保存了权重时才计算
        if getattr(self.config, "use_view_scoring", False) and hasattr(self, "_last_view_weights"):
            weights = self._last_view_weights
            
            # 鼓励权重整体偏小（没用的镜头尽量压到 0）
            # weights.mean() 越小，说明网络越克制
            sparsity_penalty = weights.mean() 
            
            # 系数可以设置在 0.01 左右
            sparsity_weight = getattr(self.config, "view_sparsity_weight", 0.01)
            
            total_loss = diffusion_loss + sparsity_weight * sparsity_penalty

        return total_loss
    
    def _maybe_log_view_scores(self, weights: torch.Tensor) -> None:
        if self.view_score_log_interval <= 0:
            return
        self._view_score_call_count += 1
        if self._view_score_call_count % self.view_score_log_interval != 0:
            return
        avg_weights = weights.detach().mean(dim=0).cpu().tolist()
        entry = {
            "step": self._view_score_call_count,
            "avg_weights": avg_weights,
            "timestamp": time.time(),
        }
        self.view_score_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.view_score_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")