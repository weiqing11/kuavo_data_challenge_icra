# multimodal_diffusion_wrapper.py
import math
from typing import Optional, Dict, Any
import einops
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torch.nn.functional as F

from transformers import AutoModel, SiglipVisionModel
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

# diffusers scheduler classes (factory expects these names)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import StableDiffusion3Pipeline
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
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        status_icon = "🥶" if self.vision_freeze else "🔥"
        status_text = "FROZEN" if self.vision_freeze else "TRAINING"

        summary_dict = {
            "DINOv2 Model": f"{self.dinov2_model_name} (Patch: {self.dino_p})",
            "SigLIP Model": f"{self.siglip_model_name} (Patch: {self.siglip_p})",
            "Backbone Status": f"{status_icon} {status_text}",
            "Trainable Params": f"{train_params:,}", # 自动添加千分位逗号
            "Total Params": f"{total_params:,}",
            "Fusion Strategy": "Concat + Auto-Interpolate"
        }
        
        log_box("Vision Backbone Summary", summary_dict, icon="🤖")

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
        dino_grid = self._process_feature_map(dinov2_feat, H, W, self.dino_p, "DINOv2")
        

        # 处理 SigLIP
        siglip_grid = self._process_feature_map(siglip_feat, H, W, self.siglip_p, "SigLIP")

        return dino_grid.contiguous(), siglip_grid.contiguous()


class DinoSiglipRGBEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ----------------------------------------------------------------
        # 1. 初始化 Backbone
        # ----------------------------------------------------------------
        try:
            self.backbone = DinoSiglipBackbone(config)
        except Exception as e:
            logger.error("❌ Failed to initialize DinoSiglipBackbone inside RGBEncoder.")
            raise e

        # ----------------------------------------------------------------
        # 2. 确定输入图片分辨率 (Dummy Input Shape)
        # ----------------------------------------------------------------
        # 安全地获取图像通道数和原始尺寸
        if not hasattr(config, "image_features") or not config.image_features:
             raise ValueError("❌ Config missing 'image_features' or it is empty.")
        
        # 获取第一个图像特征的形状 (C, H, W)
        first_img_shape = next(iter(config.image_features.values())).shape
        input_c = first_img_shape[0]

        # 解析最终输入到网络的分辨率 (优先级: Resize > Crop > Original)
        resolution_source = "Original"
        if config.resize_shape is not None:
            dummy_h, dummy_w = config.resize_shape
            resolution_source = f"Resize ({dummy_h}, {dummy_w})"
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_h = x_end - x_start
                dummy_w = y_end - y_start
            else:
                dummy_h, dummy_w = config.crop_shape
            resolution_source = f"Crop ({dummy_h}, {dummy_w})"
        else:
            dummy_h, dummy_w = first_img_shape[1:]

        if input_c != 3:
            logger.warning(f"⚠️ Input channels is {input_c}, but DINO/SigLIP usually expect 3 (RGB). Check your config.")

        # ----------------------------------------------------------------
        # 3. 自动计算 Backbone 输出形状
        # ----------------------------------------------------------------
        dummy_input = torch.zeros(1, input_c, dummy_h, dummy_w)
        
        try:
            dino_out, siglip_out = self.backbone(dummy_input)

            dino_shape = list(dino_out.shape[1:])   # [C1, H1, W1]
            siglip_shape = list(siglip_out.shape[1:]) # [C2, H2, W2]
    
            grid_info = f"DINO:{dino_shape[1]}x{dino_shape[2]} | SigLIP:{siglip_shape[1]}x{siglip_shape[2]}"
            
        except RuntimeError as e:
            logger.error(f"❌ Error verifying backbone with dummy input shape {dummy_input.shape}.")
            logger.error(f"   Usually means input resolution is too small or patch size mismatch.")
            raise e

        # ----------------------------------------------------------------
        # 4. 初始化 SpatialSoftmax
        # ----------------------------------------------------------------
        self.num_kp = config.spatial_softmax_num_keypoints
        
        try:
            self.pool_dino = SpatialSoftmax(dino_shape, num_kp=self.num_kp)
            self.pool_siglip = SpatialSoftmax(siglip_shape, num_kp=self.num_kp)
        except Exception as e:
            logger.error(f"❌ Failed to init SpatialSoftmax.")
            raise e
        
        # ----------------------------------------------------------------
        # 5. 投影层
        # ----------------------------------------------------------------
        # dinov2 pool + siglip pool
        self.raw_feature_dim = (self.num_kp * 2) * 2
        # for diffusion
        self.feature_dim = self.num_kp * 2

        self.out = nn.Linear(self.raw_feature_dim, self.feature_dim)
        self.relu = nn.ReLU()
        
        summary_dict = {
            "Fusion Type": "Late Fusion -> Projection",
            "Backbone Grids": grid_info,
            "resolution source": resolution_source,
            "Raw Concat Dim": f"{self.raw_feature_dim} (DINO+SigLIP)",
            "Projected Dim": f"{self.feature_dim} (Compatible Output)",
            "Keypoints": self.num_kp
        }
        
        log_box("RGB Encoder Configuration", summary_dict, icon="👁️")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 简单的维度检查
        if x.ndim != 4:
             raise ValueError(f"Expected 4D input [B, C, H, W], got {x.shape}")

        # 1. 提取特征图 [B, C, H, W]
        dino_feat, siglip_feat = self.backbone(x)      
        
        # 2. 空间软池化 -> 坐标 [B, num_kp, 2]
        dino_kp = self.pool_dino(dino_feat)
        siglip_kp = self.pool_siglip(siglip_feat)

        # 3. 展平 坐标 [B, num_kp, 2] -> Flatten [B, num_kp*2]
        dino_flat = torch.flatten(dino_kp, start_dim=1)   
        siglip_flat = torch.flatten(siglip_kp, start_dim=1) 
        
        # 4. 拼接
        x = torch.cat([dino_flat, siglip_flat], dim=1) # [B, feature_dim]

        # 5. 线性投影 [B, feature_dim]
        x = self.relu(self.out(x))
        
        return x


class SiglipRGBEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. 加载 SigLIP
        model_name = config.siglip_model_name
        self.siglip = SiglipVisionModel.from_pretrained(model_name)
        
        # 获取基础参数
        self.patch_size = self.siglip.config.patch_size
        self.hidden_size = self.siglip.config.hidden_size
        
        # 2. 冻结/解冻逻辑
        self.vision_freeze = getattr(config, "vision_freeze", True)
        if self.vision_freeze:
            self.siglip.requires_grad_(False)
            self.siglip.eval()
        else:
            self.siglip.train()

        # 3. 解析输入分辨率 (H, W) 以初始化 SpatialSoftmax
        # 优先级: Resize > Crop > 原始 config
        if config.resize_shape:
            h, w = config.resize_shape
        elif config.crop_shape:
            # 处理 ((x1,x2), (y1,y2)) 或 (h, w)
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x0, x1), (y0, y1) = config.crop_shape
                h, w = x1 - x0, y1 - y0
            else:
                h, w = config.crop_shape
        else:
            # 从 image_features 获取默认形状
            first_shape = next(iter(config.image_features.values())).shape
            h, w = first_shape[1:]

        # 计算 Feature Map 的网格大小
        self.grid_h = h // self.patch_size
        self.grid_w = w // self.patch_size
        
        # 4. 初始化 SpatialSoftmax
        self.num_kp = config.spatial_softmax_num_keypoints
        # feature_shape: [C, H, W]
        self.pool = SpatialSoftmax([self.hidden_size, self.grid_h, self.grid_w], num_kp=self.num_kp)

        # 5. 投影层
        # 输入维度: keypoints * 2 (x, y 坐标)
        # 输出维度: 128
        self.feature_dim = self.num_kp * 2
        
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

        # # 3. 投影层 (Projection Layer)
        # # 全局模式不再需要 SpatialSoftmax，输入直接是 hidden_size
        # # config.output_dim 是你最终希望输出给后端的特征维度 (如 512 或 1024)
        # self.num_kp = config.spatial_softmax_num_keypoints
        # self.feature_dim = self.num_kp * 2
        
        # self.projection = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.feature_dim),
        #     nn.LayerNorm(self.feature_dim),
        #     nn.ReLU()
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        
        # 1. Backbone 前向传播
        # 如果冻结，使用 no_grad 节省显存
        context = torch.no_grad() if self.vision_freeze else torch.enable_grad()
        
        with context:
            # SigLIP 输出: [B, N, D]
            # interpolate_pos_encoding=True 用于适应不同的输入分辨率
            last_hidden_state = self.siglip(x, interpolate_pos_encoding=True).last_hidden_state

        # 2. Reshape: [B, N, D] -> [B, D, grid_h, grid_w]
        B, N, D = last_hidden_state.shape
        # 注意：这里直接使用 Permute 转换维度
        feat = last_hidden_state.view(B, self.grid_h, self.grid_w, D).permute(0, 3, 1, 2)

        # 3. Spatial Softmax -> [B, num_kp, 2]
        kp = self.pool(feat)

        # 4. Flatten -> [B, num_kp * 2]
        x = torch.flatten(kp, start_dim=1)

        # 5. Projection -> [B, out_dim]
        x = self.relu(self.out(x))
        
        # with context:
        #     # interpolate_pos_encoding=True 允许输入不同于预训练时的分辨率
        #     # SigLIP 会自动处理 Attention Pooling 过程
        #     outputs = self.siglip(x, interpolate_pos_encoding=True)
            
        #     # 提取全局 Token [B, hidden_size]
        #     # pooler_output 是经过 Attention Pooling 聚合后的整图表征
        #     global_feat = outputs.pooler_output 

        # # 2. 投影到目标维度
        # # 这里的输出是一个高度压缩的语义向量
        # x = self.projection(global_feat)

        return x


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
        try:
            self.backbone = DFomerRGBDBackbone(config)
        except Exception as e:
            logger.error("❌ Failed to initialize DFormer inside RGBDEncoder.")
            raise e
        
        # 2. 获取 Backbone 输出通道数
        # Base版: [80, 160, 320, 512]
        channels_list = self.backbone.get_out_channels()
        self.use_layer_idx = -1
        in_channels = channels_list[-1]
        
        # 3. 投影层 (Projection)
        # 作用：将 Backbone 的通道数 (如 512) 对齐到 DiT 的维度 (如 768)
        # 使用 1x1 卷积比 Linear 更适合保留空间信息
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, dit_hidden_dim, kernel_size=1),
            nn.GroupNorm(32, dit_hidden_dim), # 归一化有助于训练稳定
            nn.SiLU() # DiT 常用的激活函数
        )


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
        elif "dinov2" in config.vision_backbone and "siglip" in config.vision_backbone:
            self.model = DinoSiglipRGBEncoder(config)
        elif "siglip_only" in config.vision_backbone:
            self.model = SiglipRGBEncoder(config)
        else:
            raise ValueError(f"Unknown vision backbone: {config.vision_backbone}")
        self.feature_dim = self.model.feature_dim
    def forward(self, x: Tensor) -> Tensor:
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
    def __init__(self, config: CustomDiffusionConfigWrapper):
        # ensure parent init runs with safe backbone
        orig_vis = config.vision_backbone
        config.vision_backbone = "resnet18"
        orig_noise_scheduler = config.noise_scheduler_type
        config.noise_scheduler_type = "DDPM"
        super().__init__(config)
        config.vision_backbone = orig_vis
        config.noise_scheduler_type = orig_noise_scheduler

        self.config = config
        global_cond_dim = 0

        # ---- state encoder in WRAPPER (only place for discrete or mlp logic) ----
        self.state_encoder = None
        final_state_dim = None
        if getattr(self.config, "robot_state_feature", None) is not None:
            state_dim = self.config.robot_state_feature.shape[0]
            if getattr(self.config, "use_state_encoder", False):
                out_dim = getattr(self.config, "state_feature_dim", 128)
                self.state_encoder = FeatureEncoder(state_dim, out_dim)
                final_state_dim = out_dim
                global_cond_dim += final_state_dim
            else:
                final_state_dim = state_dim
                global_cond_dim += final_state_dim

        # ---- RGB encoders ----
        self.rgb_feat_dim = 0
        if getattr(self.config, "image_features", None):
            num_images = len(self.config.image_features)
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                encs = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encs)
                feat_dim = encs[0].feature_dim
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                feat_dim = self.rgb_encoder.feature_dim
            self.rgb_feat_dim = feat_dim * num_images
            global_cond_dim += self.rgb_feat_dim
            self.rgb_attn_layer = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=getattr(self.config, "rgb_attn_heads", 8), batch_first=True)

        # ---- Depth encoders (optional) ----
        self.depth_feat_dim = 0
        if getattr(self.config, "use_depth", False) and getattr(self.config, "depth_features", None):
            num_depth = len(self.config.depth_features)
            if getattr(self.config, "use_separate_depth_encoder_per_camera", False):
                encs = [DiffusionDepthEncoder(config) for _ in range(num_depth)]
                self.depth_encoder = nn.ModuleList(encs)
                feat_dim = encs[0].feature_dim
            else:
                self.depth_encoder = DiffusionDepthEncoder(config)
                feat_dim = self.depth_encoder.feature_dim
            self.depth_feat_dim = feat_dim * num_depth
            global_cond_dim += self.depth_feat_dim
            self.depth_attn_layer = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=getattr(self.config, "depth_attn_heads", 8), batch_first=True)

            # RGB <-> Depth cross-attn modules
            self.multimodalfuse = nn.ModuleDict({
                "rgb_q": nn.MultiheadAttention(embed_dim=feat_dim, num_heads=getattr(self.config, "multimodal_heads", 8), batch_first=True),
                "depth_q": nn.MultiheadAttention(embed_dim=feat_dim, num_heads=getattr(self.config, "multimodal_heads", 8), batch_first=True),
            })

        # ---- state-guided fusion block ----
        self.fusion_hidden = getattr(self.config, "fusion_hidden_dim", 256)
        self.state_guided = None
        if getattr(self.config, "state_fuse", False):
            vis_dim_for_fusion = (self.rgb_attn_layer.embed_dim if hasattr(self, "rgb_attn_layer") else self.rgb_feat_dim)
            dep_dim_for_fusion = (self.depth_attn_layer.embed_dim if hasattr(self, "depth_attn_layer") else None)
            state_dim_for_fusion = final_state_dim
            self.state_guided = StateGuidedFusionBlock(
                vis_dim=vis_dim_for_fusion,
                dep_dim=dep_dim_for_fusion,
                state_dim=state_dim_for_fusion,
                hidden_dim=self.fusion_hidden,
                num_heads=getattr(self.config, "fusion_heads", 8)
            )
            global_cond_dim += self.fusion_hidden

        # ---- env state ----
        if getattr(self.config, "env_state_feature", None) is not None:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # ---- core diffusion model ----
        if config.use_unet:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        elif config.use_transformer:
            self.unet = TransformerForDiffusion(
                input_dim=config.output_features["action"].shape[0],
                output_dim=config.output_features["action"].shape[0],
                horizon=config.horizon,
                n_obs_steps=config.n_obs_steps,
                cond_dim=global_cond_dim,
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
        else:
            raise ValueError("Either `use_unet` or `use_transformer` must be True in config.")

        # ---- scheduler ----
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
        # self.noise_scheduler = _make_noise_scheduler_factory(
        #     config.noise_scheduler_type,
        #     config.scheduler_params
        # )
        self.num_inference_steps = config.num_inference_steps or self.noise_scheduler.config.num_train_timesteps

    def _prepare_global_conditioning(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Encode & fuse modalities into (B, S, cond_dim).
        Behavior:
          - if both rgb & depth: compute rgb_q & dep_q tokens (cross-attn outputs), flatten for cond features,
            AND create concat tokens (rgb_q_cat = cat(rgb_q, dep_q)) as tokens for state-guided fusion.
          - if only rgb: use rgb tokens.
          - state encoding is performed here (wrapper); state_for_fusion will be encoded/flattened and passed to fusion block.
        """
        B = batch[OBS_STATE].shape[0]
        S = batch[OBS_STATE].shape[1]  # n_obs_steps
        feats = []

        # ---------- RGB ----------
        img_features = None  # tokens shape (B*S, N_cam_tokens, feat)
        if getattr(self.config, "image_features", None):
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                enc_outs = [enc(im) for enc, im in zip(self.rgb_encoder, imgs)]
                img_cat = torch.cat(enc_outs)  # (n * B*s, feat)
                img_features = einops.rearrange(img_cat, "(n b s) f -> (b s) n f", b=B, s=S)
            else:
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                out = self.rgb_encoder(imgs)  # (b*s*n, feat)
                img_features = einops.rearrange(out, "(b s n) f -> (b s) n f", b=B, s=S)
            # self-attn over tokens
            img_features = self.rgb_attn_layer(img_features, img_features, img_features)[0]  # (b*s, n, feat)

        # ---------- Depth (optional) ----------
        depth_features = None
        if getattr(self.config, "use_depth", False) and getattr(self.config, "depth_features", None) and (OBS_DEPTH in batch):
            if getattr(self.config, "use_separate_depth_encoder_per_camera", False):
                depths = einops.rearrange(batch[OBS_DEPTH], "b s n ... -> n (b s) ...")
                enc_outs = [enc(d) for enc, d in zip(self.depth_encoder, depths)]
                dep_cat = torch.cat(enc_outs)
                depth_features = einops.rearrange(dep_cat, "(n b s) f -> (b s) n f", b=B, s=S)
            else:
                depths = einops.rearrange(batch[OBS_DEPTH], "b s n ... -> (b s n) ...")
                out = self.depth_encoder(depths)
                depth_features = einops.rearrange(out, "(b s n) f -> (b s) n f", b=B, s=S)
            depth_features = self.depth_attn_layer(depth_features, depth_features, depth_features)[0]  # (b*s, n, feat)

        # ---------- RGB <-> Depth fusion (if both exist) ----------
        # Keep both token forms (rgb_q_tokens, dep_q_tokens) for state-guided fusion (we will concat them).
        rgb_q_tokens = None
        dep_q_tokens = None
        if (img_features is not None) and (depth_features is not None) and hasattr(self, "multimodalfuse"):
            rgb_q_tokens = self.multimodalfuse["rgb_q"](img_features, depth_features, depth_features)[0]  # (b*s, n, feat)
            dep_q_tokens = self.multimodalfuse["depth_q"](depth_features, img_features, img_features)[0]  # (b*s, n, feat)
            # For global_cond feats we flatten (B, S, n*feat)
            rgb_q_flat = einops.rearrange(rgb_q_tokens, "(b s) n f -> b s (n f)", b=B, s=S)
            dep_q_flat = einops.rearrange(dep_q_tokens, "(b s) n f -> b s (n f)", b=B, s=S)
            feats.extend([rgb_q_flat, dep_q_flat])
        elif img_features is not None:
            # only rgb available
            feats.append(einops.rearrange(img_features, "(b s) n f -> b s (n f)", b=B, s=S))
        elif depth_features is not None:
            feats.append(einops.rearrange(depth_features, "(b s) n f -> b s (n f)", b=B, s=S))

        # ---------- State encoding (WRAPPER does this) ----------
        state_tensor = None
        if getattr(self.config, "robot_state_feature", None) is not None:
            state_tensor = batch[OBS_STATE]  # (B, S, state_dim)
            if self.state_encoder is not None:
                # encoder may accept (B, S, D) and returns (B, S, out_dim)
                state_emb = self.state_encoder(state_tensor)  # (B, S, final_state_dim)
                feats.append(state_emb)
            else:
                feats.append(state_tensor)

        # ---------- Env state ----------
        if getattr(self.config, "env_state_feature", None) is not None:
            feats.append(batch[OBS_ENV_STATE])

        # ---------- State-guided fusion: run on per-(b*s) samples ----------
        if getattr(self, "state_guided", None) is not None:
            # prepare tokens for fusion block
            # choose tokens: if rgb_q_tokens & dep_q_tokens exist -> concat them (combined tokens),
            # else fallback to img_features (or depth_features if only depth exists)
            if (rgb_q_tokens is not None) and (dep_q_tokens is not None):
                # concat tokens along sequence dim to give more information to fusion
                # vis_tokens_for_fusion = torch.cat([rgb_q_tokens, dep_q_tokens], dim=1)  # (b*s, n_r + n_d, feat)
                # vis_tokens_for_fusion = rgb_q_tokens
                # dep_tokens_for_fusion = dep_q_tokens
                vis_tokens_for_fusion = img_features
                dep_tokens_for_fusion = depth_features
                # print("~~~~~~~~~~~~~~~~~~~~~~~~~~use rgb_q and dep_q~~~~~~~~~~~~~~~~~~~~~~~~~~")
            elif img_features is not None:
                vis_tokens_for_fusion = img_features  # (b*s, n, feat)
                dep_tokens_for_fusion = None
            elif depth_features is not None:
                vis_tokens_for_fusion = depth_features
                dep_tokens_for_fusion = None
            else:
                vis_tokens_for_fusion = None
                dep_tokens_for_fusion = None

            # prepare state for fusion: should be (B*s, final_state_dim) or None
            if state_tensor is not None:
                if self.state_encoder is not None:
                    state_for_fusion = state_emb.view(B * S, -1)  # (B*S, final_state_dim)
                else:
                    state_for_fusion = state_tensor.view(B * S, -1)  # raw
            else:
                state_for_fusion = None

            # call fusion block if we have visual tokens
            if vis_tokens_for_fusion is not None:
                fused_vec = self.state_guided(vis_tokens_for_fusion, dep_tokens_for_fusion, state_for_fusion)  # (B*s, fusion_hidden)
                fused_vec = einops.rearrange(fused_vec, "(b s) f -> b s f", b=B, s=S)  # (B, S, fusion_hidden)
                feats.append(fused_vec)

        # Final concat -> (B, S, cond_dim)
        if len(feats) == 0:
            return torch.zeros((B, S, 0), device=next(self.parameters()).device)
        if self.config.use_unet:
            return torch.cat(feats, dim=-1).flatten(start_dim=1)
        else:
            return torch.cat(feats, dim=-1)

    # ---------------------------
    # Inference sampling
    # ---------------------------
    def conditional_sample(self, batch_size: int, global_cond: Optional[Tensor] = None, generator=None, noise: Tensor | None = None) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

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
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full((batch_size,), t, dtype=torch.long, device=device),
                global_cond=global_cond,
            )
            # pass eta if scheduler supports it (DDIM uses eta)
            if "eta" in self.noise_scheduler.step.__code__.co_varnames:
                step_out = self.noise_scheduler.step(model_output, t, sample, eta=getattr(self.config, "ddim_eta", 0.0), generator=generator)
            else:
                step_out = self.noise_scheduler.step(model_output, t, sample, generator=generator)
            sample = getattr(step_out, "prev_sample", step_out)

        return sample
