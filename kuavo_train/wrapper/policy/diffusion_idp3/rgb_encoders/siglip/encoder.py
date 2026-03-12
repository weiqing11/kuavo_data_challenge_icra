"""SigLIP RGB encoder used by diffusion_idp3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel


PROJECT_ROOT = Path(__file__).resolve().parents[6]


def _resolve_local_path(raw_path: str, pretrained_root: str) -> Path:
    """Resolve local filesystem paths only (no Hub IDs, no network access)."""
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path
    else:
        root = Path(pretrained_root)
        if not root.is_absolute():
            root = PROJECT_ROOT / root

        if str(path).startswith("dataset/models"):
            resolved = PROJECT_ROOT / path
        else:
            resolved = root / path

    # Support local HF-cache style dirs:
    # models--org--repo/{refs, snapshots/<hash>, blobs}
    snapshots_dir = resolved / "snapshots"
    if resolved.is_dir() and snapshots_dir.is_dir() and not (resolved / "config.json").exists():
        refs_main = resolved / "refs" / "main"
        if refs_main.is_file():
            snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
            if snapshot_hash:
                candidate = snapshots_dir / snapshot_hash
                if candidate.exists():
                    return candidate

        snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if snapshot_dirs:
            snapshot_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return snapshot_dirs[0]

    return resolved


class SiglipRGBEncoder(nn.Module):
    """Encode RGB images into SigLIP patch tokens."""

    def __init__(self, cfg: dict[str, Any], target_dim: int | None = None) -> None:
        super().__init__()

        mode = cfg.get("vision_encoder_mode", "patches")
        if mode != "patches":
            raise ValueError(f"SigLIP encoder only supports `patches` mode, got {mode!r}")

        pretrained_root = str(cfg.get("pretrained_root", "dataset/models"))
        model_path = _resolve_local_path(cfg["pretrained_model_path"], pretrained_root)
        processor_raw_path = cfg.get("pretrained_processor_path", cfg["pretrained_model_path"])
        processor_path = _resolve_local_path(processor_raw_path, pretrained_root)

        if not model_path.exists():
            raise FileNotFoundError(f"SigLIP model path not found: {model_path}")
        if not processor_path.exists():
            raise FileNotFoundError(f"SigLIP processor path not found: {processor_path}")

        self.initialize_from_pretrained = bool(cfg.get("initialize_from_pretrained", True))

        if self.initialize_from_pretrained:
            self.siglip = SiglipVisionModel.from_pretrained(str(model_path), local_files_only=True)
        else:
            model_cfg = SiglipVisionConfig.from_pretrained(str(model_path), local_files_only=True)
            self.siglip = SiglipVisionModel(model_cfg)

        self.processor = SiglipImageProcessor.from_pretrained(str(processor_path), local_files_only=True)

        self.use_lora = bool(cfg.get("use_lora", False))
        self.vision_freeze = bool(cfg.get("vision_freeze", True))

        if self.use_lora:
            peft_config = LoraConfig(
                r=int(cfg.get("lora_rank", 16)),
                lora_alpha=int(cfg.get("lora_alpha", 32)),
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                bias="none",
            )
            self.siglip = get_peft_model(self.siglip, peft_config)
            self.vision_freeze = False
        elif self.vision_freeze:
            self.siglip.requires_grad_(False)
            self.siglip.eval()

        self.hidden_size = int(self.siglip.config.hidden_size)
        patch_size = self.siglip.config.patch_size
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)

        self.target_dim = int(target_dim if target_dim is not None else cfg.get("target_dim", self.hidden_size))
        self.proj = nn.Linear(self.hidden_size, self.target_dim)
        self.norm = nn.LayerNorm(self.target_dim)

        image_h, image_w = cfg.get("image_size", [224, 224])
        self.num_patches = (int(image_h) // self.patch_size) * (int(image_w) // self.patch_size)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Args: rgb [B, C, H, W]. Returns: tokens [B, P, D]."""
        requires_grad = self.use_lora or (not self.vision_freeze)
        context = torch.enable_grad() if requires_grad else torch.no_grad()

        with context:
            siglip_inputs = self.processor(
                images=rgb,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            )
            pixel_values = siglip_inputs["pixel_values"].to(rgb.device)
            outputs = self.siglip(pixel_values, interpolate_pos_encoding=True)
            tokens = outputs.last_hidden_state

        return self.norm(self.proj(tokens))
