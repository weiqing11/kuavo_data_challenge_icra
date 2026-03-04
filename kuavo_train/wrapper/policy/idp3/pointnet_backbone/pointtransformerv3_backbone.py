"""PointTransformerV3 backbone adapter for IDP3.

Backbone interface contract (for IDP3):
- Input:  [B, N, C]
- Output: [B, F]
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class PointTransformerV3Backbone(nn.Module):
    """PointTransformerV3 backbone adapter for IDP3."""

    def __init__(
        self,
        pc_channels: int = 6,
        out_channels: int = 128,
        ptv3_in_channels: int = 6,
        ptv3_order: Sequence[str] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        ptv3_stride: Sequence[int] = (2, 2, 2, 2),
        ptv3_enc_depths: Sequence[int] = (2, 2, 2, 6, 2),
        ptv3_enc_channels: Sequence[int] = (32, 64, 128, 256, 512),
        ptv3_enc_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        ptv3_enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        ptv3_dec_depths: Sequence[int] = (2, 2, 2, 2),
        ptv3_dec_channels: Sequence[int] = (64, 64, 128, 256),
        ptv3_dec_num_head: Sequence[int] = (4, 4, 8, 16),
        ptv3_dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        ptv3_mlp_ratio: float = 4.0,
        ptv3_qkv_bias: bool = True,
        ptv3_qk_scale: float | None = None,
        ptv3_attn_drop: float = 0.0,
        ptv3_proj_drop: float = 0.0,
        ptv3_drop_path: float = 0.3,
        ptv3_pre_norm: bool = True,
        ptv3_shuffle_orders: bool = True,
        ptv3_enable_rpe: bool = False,
        ptv3_enable_flash: bool = False,
        ptv3_upcast_attention: bool = False,
        ptv3_upcast_softmax: bool = False,
        ptv3_cls_mode: bool = True,
        ptv3_pdnorm_bn: bool = False,
        ptv3_pdnorm_ln: bool = False,
        ptv3_pdnorm_decouple: bool = True,
        ptv3_pdnorm_adaptive: bool = False,
        ptv3_pdnorm_affine: bool = True,
        ptv3_pdnorm_conditions: Sequence[str] = ("ScanNet", "S3DIS", "Structured3D"),
        ptv3_grid_size: float = 0.02,
        ptv3_condition: str | None = None,
        ptv3_batch_chunk_size: int = 0,
        ptv3_auto_load_pretrained: bool = True,
        ptv3_pretrained_path: str = "dataset/models/scannet-semseg-pt-v3m1-0-base",
        ptv3_strict_encoder_load: bool = True,
        ptv3_freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.pc_channels = int(pc_channels)
        self.requested_out_channels = int(out_channels)
        self.ptv3_in_channels = int(ptv3_in_channels)
        self.ptv3_grid_size = float(ptv3_grid_size)
        self.ptv3_condition = ptv3_condition
        self.ptv3_batch_chunk_size = int(ptv3_batch_chunk_size)
        self.ptv3_cls_mode = bool(ptv3_cls_mode)

        if self.ptv3_in_channels < 3:
            raise ValueError(f"ptv3_in_channels must be >= 3, got {self.ptv3_in_channels}")
        if self.pc_channels < 3:
            raise ValueError(f"pc_channels must be >= 3, got {self.pc_channels}")
        if self.ptv3_grid_size <= 0:
            raise ValueError(f"ptv3_grid_size must be > 0, got {self.ptv3_grid_size}")

        self.use_channel_slice = self.pc_channels > self.ptv3_in_channels
        if self.pc_channels == self.ptv3_in_channels:
            self.input_adapter = nn.Identity()
        elif self.use_channel_slice:
            self.input_adapter = None
        else:
            self.input_adapter = nn.Conv1d(self.pc_channels, self.ptv3_in_channels, kernel_size=1, bias=True)
            self._init_input_adapter()

        point_transformer_v3_cls = self._import_pointtransformerv3_class()
        self.encoder = point_transformer_v3_cls(
            in_channels=self.ptv3_in_channels,
            order=tuple(ptv3_order),
            stride=tuple(ptv3_stride),
            enc_depths=tuple(ptv3_enc_depths),
            enc_channels=tuple(ptv3_enc_channels),
            enc_num_head=tuple(ptv3_enc_num_head),
            enc_patch_size=tuple(ptv3_enc_patch_size),
            dec_depths=tuple(ptv3_dec_depths),
            dec_channels=tuple(ptv3_dec_channels),
            dec_num_head=tuple(ptv3_dec_num_head),
            dec_patch_size=tuple(ptv3_dec_patch_size),
            mlp_ratio=ptv3_mlp_ratio,
            qkv_bias=ptv3_qkv_bias,
            qk_scale=ptv3_qk_scale,
            attn_drop=ptv3_attn_drop,
            proj_drop=ptv3_proj_drop,
            drop_path=ptv3_drop_path,
            pre_norm=ptv3_pre_norm,
            shuffle_orders=ptv3_shuffle_orders,
            enable_rpe=ptv3_enable_rpe,
            enable_flash=ptv3_enable_flash,
            upcast_attention=ptv3_upcast_attention,
            upcast_softmax=ptv3_upcast_softmax,
            cls_mode=self.ptv3_cls_mode,
            pdnorm_bn=ptv3_pdnorm_bn,
            pdnorm_ln=ptv3_pdnorm_ln,
            pdnorm_decouple=ptv3_pdnorm_decouple,
            pdnorm_adaptive=ptv3_pdnorm_adaptive,
            pdnorm_affine=ptv3_pdnorm_affine,
            pdnorm_conditions=tuple(ptv3_pdnorm_conditions),
        )

        if self.ptv3_cls_mode:
            encoder_out_channels = int(ptv3_enc_channels[-1])
        else:
            encoder_out_channels = int(ptv3_dec_channels[0])
        self.out_channels = int(encoder_out_channels)
        # Temporarily disable out_channels mapping. Keep these lines for quick re-enable:
        # self.output_proj = (
        #     nn.Identity()
        #     if encoder_out_channels == out_channels
        #     else nn.Linear(encoder_out_channels, out_channels)
        # )

        if ptv3_auto_load_pretrained:
            self.load_ptv3_pretrained_weights(
                pretrained_path=ptv3_pretrained_path,
                strict_encoder=ptv3_strict_encoder_load,
            )
        if ptv3_freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[5]

    @classmethod
    def _import_pointtransformerv3_class(cls):
        try:
            module = importlib.import_module(
                ".pointtransformerv3.model",
                package=__package__,
            )
        except Exception as exc:
            raise ImportError(
                "[PointTransformerV3Backbone] failed to import PTv3. "
                "Please make sure vendored PTv3 files exist under "
                "`kuavo_train/wrapper/policy/idp3/pointnet_backbone/pointtransformerv3` "
                "and install required dependencies such as spconv, torch_scatter, timm and addict."
            ) from exc
        if not hasattr(module, "PointTransformerV3"):
            raise ImportError("[PointTransformerV3Backbone] PointTransformerV3 class not found in PTv3 module.")
        return module.PointTransformerV3

    def _init_input_adapter(self):
        if not isinstance(self.input_adapter, nn.Conv1d):
            return
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            if self.input_adapter.bias is not None:
                self.input_adapter.bias.zero_()
            shared = min(self.pc_channels, self.ptv3_in_channels)
            for i in range(shared):
                self.input_adapter.weight[i, i, 0] = 1.0

    @staticmethod
    def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "network", "net"):
                value = ckpt.get(key, None)
                if isinstance(value, dict):
                    return value
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return ckpt
        return {}

    @staticmethod
    def _strip_known_prefixes(key: str) -> str:
        prefixes = ("module.", "model.", "backbone.")
        while True:
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            if new_key == key:
                break
            key = new_key
        return key

    @classmethod
    def _resolve_pretrained_checkpoint(cls, path_like: str | Path) -> Path | None:
        def find_in_dir(directory: Path) -> Path | None:
            candidates = [
                directory / "model" / "model_best.pth",
                directory / "model_best.pth",
                directory / "model" / "model_last.pth",
                directory / "model_last.pth",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            for candidate in sorted(directory.rglob("*.pth")):
                if candidate.name == "model_best.pth":
                    return candidate
            all_pth = sorted(directory.rglob("*.pth"))
            return all_pth[0] if all_pth else None

        raw = Path(path_like).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.append(cls._repo_root() / raw)

        for candidate in candidates:
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                match = find_in_dir(candidate)
                if match is not None:
                    return match
        return None

    def load_ptv3_pretrained_weights(self, pretrained_path: str | Path, strict_encoder: bool = True) -> bool:
        ckpt_path = self._resolve_pretrained_checkpoint(pretrained_path)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"[PointTransformerV3Backbone] pretrained checkpoint not found: {pretrained_path}"
            )

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        raw_state = self._extract_state_dict(ckpt)
        if not raw_state:
            raise RuntimeError(f"[PointTransformerV3Backbone] invalid checkpoint format: {ckpt_path}")

        encoder_state_to_load: Dict[str, torch.Tensor] = {}
        for raw_key, value in raw_state.items():
            key = self._strip_known_prefixes(raw_key)
            if key.startswith("embedding.") or key.startswith("enc."):
                encoder_state_to_load[key] = value

        if not encoder_state_to_load:
            raise RuntimeError(
                f"[PointTransformerV3Backbone] no encoder weights (embedding/enc) found in: {ckpt_path}"
            )

        missing, unexpected = self.encoder.load_state_dict(encoder_state_to_load, strict=False)
        missing = [k for k in missing if k.startswith("embedding.") or k.startswith("enc.")]
        unexpected = [k for k in unexpected if k.startswith("embedding.") or k.startswith("enc.")]
        if strict_encoder and (missing or unexpected):
            raise RuntimeError(
                "[PointTransformerV3Backbone] strict encoder loading failed. "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        logger.info(
            "[PointTransformerV3Backbone] loaded pretrained encoder weights from %s "
            "(strict_encoder=%s, tensors=%d, missing=%d, unexpected=%d)",
            ckpt_path,
            strict_encoder,
            len(encoder_state_to_load),
            len(missing),
            len(unexpected),
        )
        return True

    def _convert_idp3_to_ptv3_input(self, x: torch.Tensor) -> Dict[str, Any]:
        """Convert IDP3 input [B, N, C] to PTv3 input dictionary."""
        if x.dim() != 3:
            raise ValueError(f"Point cloud must be [B, N, C], got {tuple(x.shape)}")
        if x.shape[-1] < 3:
            raise ValueError(f"Point cloud channel must be >= 3 (xyz), got {x.shape[-1]}")

        bsz, num_points, _ = x.shape
        coord = x[:, :, :3].contiguous()

        if self.use_channel_slice:
            feat = x[:, :, : self.ptv3_in_channels].contiguous()
        else:
            feat = x.transpose(1, 2).contiguous()
            feat = self.input_adapter(feat).transpose(1, 2).contiguous()

        if feat.shape[-1] != self.ptv3_in_channels:
            raise ValueError(
                f"PTv3 feature channel mismatch: got {feat.shape[-1]}, expected {self.ptv3_in_channels}"
            )

        batch = (
            torch.arange(bsz, device=x.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, num_points)
            .reshape(-1)
            .contiguous()
        )
        data_dict: Dict[str, Any] = {
            "coord": coord.reshape(-1, 3).contiguous(),
            "feat": feat.reshape(-1, self.ptv3_in_channels).contiguous(),
            "batch": batch,
            "grid_size": self.ptv3_grid_size,
        }
        if self.ptv3_condition is not None:
            data_dict["condition"] = self.ptv3_condition
        return data_dict

    def _convert_ptv3_to_idp3_output(self, point: Any, batch_size: int) -> torch.Tensor:
        """Convert PTv3 output Point to IDP3 output tensor [B, F]."""
        if not isinstance(point, dict):
            raise ValueError("PTv3 output must be a dict-like Point object.")
        if "feat" not in point or "batch" not in point:
            raise ValueError("PTv3 output must contain 'feat' and 'batch'.")

        feat = point["feat"]
        batch = point["batch"].long()
        if feat.dim() != 2:
            raise ValueError(f"PTv3 point feature must be [N_total, F], got {tuple(feat.shape)}")
        if batch.dim() != 1 or batch.shape[0] != feat.shape[0]:
            raise ValueError(f"PTv3 batch index shape mismatch: feat={tuple(feat.shape)}, batch={tuple(batch.shape)}")

        pooled = []
        for idx in range(batch_size):
            mask = batch == idx
            if not torch.any(mask):
                raise ValueError(f"PTv3 output has no point for batch index {idx}")
            pooled.append(feat[mask].max(dim=0, keepdim=False).values)
        global_feat = torch.stack(pooled, dim=0)
        # Temporarily disable out_channels mapping. Keep this line for quick re-enable:
        # return self.output_proj(global_feat)
        return global_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        if self.ptv3_batch_chunk_size and bsz > self.ptv3_batch_chunk_size:
            chunks = []
            for start in range(0, bsz, self.ptv3_batch_chunk_size):
                end = min(start + self.ptv3_batch_chunk_size, bsz)
                data_dict = self._convert_idp3_to_ptv3_input(x[start:end])
                point = self.encoder(data_dict)
                chunks.append(self._convert_ptv3_to_idp3_output(point, batch_size=end - start))
            return torch.cat(chunks, dim=0)

        data_dict = self._convert_idp3_to_ptv3_input(x)
        point = self.encoder(data_dict)
        return self._convert_ptv3_to_idp3_output(point, batch_size=bsz)
