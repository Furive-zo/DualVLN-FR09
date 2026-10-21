"""
Multi-View Register Encoder — System 1 perception module.

4-cam DAv2 ViT-S (frozen) + Custom Register Tokens (R=16/cam) + LoRA(qkv)
  → Camera-Aware Embedding
  → Cross-Camera Fusion Transformer (K layers)
  → Linear(384 → 768)
  → scene_tokens: (B, C*R, 768)  =  (B, 64, 768)
"""

import os
import sys

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# DAv2 DINOv2 — load via importlib to bypass broken encoder __init__.py
import importlib.util as _ilu

def _import_dinov2():
    _dav2_pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "depth_anything", "depth_anything_v2")
    # register sub-package so relative imports inside dinov2_layers work
    for _name, _rel in [
        ("dinov2_layers",               "dinov2_layers/__init__.py"),
        ("dinov2_layers.attention",      "dinov2_layers/attention.py"),
        ("dinov2_layers.mlp",            "dinov2_layers/mlp.py"),
        ("dinov2_layers.patch_embed",    "dinov2_layers/patch_embed.py"),
        ("dinov2_layers.swiglu_ffn",     "dinov2_layers/swiglu_ffn.py"),
        ("dinov2_layers.block",          "dinov2_layers/block.py"),
        ("dinov2_layers.layer_scale",    "dinov2_layers/layer_scale.py"),
        ("dinov2_layers.drop_path",      "dinov2_layers/drop_path.py"),
    ]:
        _full = f"_dav2.{_name}"
        if _full not in sys.modules:
            _path = os.path.join(_dav2_pkg, _rel)
            if os.path.isfile(_path):
                _spec = _ilu.spec_from_file_location(_full, _path)
                _m = _ilu.module_from_spec(_spec)
                sys.modules[_full] = _m
                sys.modules[_name] = _m   # also register short name
                _spec.loader.exec_module(_m)

    _spec = _ilu.spec_from_file_location("_dav2.dinov2",
                                         os.path.join(_dav2_pkg, "dinov2.py"))
    _m = _ilu.module_from_spec(_spec)
    sys.modules["_dav2.dinov2"] = _m
    _spec.loader.exec_module(_m)
    return _m

_dinov2 = _import_dinov2()
vit_small = _dinov2.vit_small


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_dav2_backbone(checkpoint_path: str | None, num_registers: int) -> nn.Module:
    """Build DAv2 ViT-S with custom register tokens.

    Loads pretrained ViT-S weights (everything except register_tokens).
    register_tokens are zero-initialised and remain trainable.
    """
    vit = vit_small(patch_size=16, num_register_tokens=num_registers)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # DAv2 checkpoint may wrap weights under 'model' or 'state_dict'
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # strip 'pretrained.' prefix if saved from full DAv2 model
        state = {k.replace("pretrained.", ""): v for k, v in state.items()}
        missing, unexpected = vit.load_state_dict(state, strict=False)
        print(f"[MultiViewRegisterEncoder] ViT-S loaded. "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[MultiViewRegisterEncoder] No checkpoint — ViT-S randomly initialised.")

    return vit


# ---------------------------------------------------------------------------
# main module
# ---------------------------------------------------------------------------

class MultiViewRegisterEncoder(nn.Module):
    """
    Args:
        num_cameras      : number of camera views (default 4)
        num_registers    : custom register tokens per camera (default 16)
        vit_dim          : ViT-S embedding dim (384)
        out_dim          : output dim to match DiT conditioning (768)
        num_fusion_layers: cross-camera Transformer depth
        lora_rank        : LoRA rank applied to ViT-S qkv projections
        dav2_ckpt        : path to DAv2 ViT-S pretrained weight (optional)
    """

    def __init__(
        self,
        num_cameras: int = 4,
        num_registers: int = 16,
        vit_dim: int = 384,
        out_dim: int = 768,
        num_fusion_layers: int = 2,
        lora_rank: int = 8,
        dav2_ckpt: str | None = None,
    ):
        super().__init__()
        self.C = num_cameras
        self.R = num_registers
        self.vit_dim = vit_dim
        self.out_dim = out_dim

        # ── 1. ViT-S backbone with custom register tokens ──────────────────
        vit = _load_dav2_backbone(dav2_ckpt, num_registers)

        # freeze all backbone params
        for p in vit.parameters():
            p.requires_grad_(False)

        # apply LoRA to qkv projections in every attention block
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["qkv"],          # matches blocks.*.attn.qkv
            bias="none",
        )
        self.vit = get_peft_model(vit, lora_cfg)

        # register_tokens are nn.Parameter on the base model —
        # peft freezes them along with backbone; unfreeze manually
        base = self.vit.base_model.model          # DinoVisionTransformer
        if base.register_tokens is not None:
            base.register_tokens.requires_grad_(True)

        # ── 2. Camera-aware positional embeddings ───────────────────────────
        self.cam_embed = nn.Embedding(num_cameras, vit_dim)   # (C, d)
        self.reg_pos   = nn.Parameter(torch.randn(num_registers, vit_dim) * 0.02)

        # ── 3. Cross-camera fusion Transformer ─────────────────────────────
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=6,
            dim_feedforward=vit_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,       # Pre-LN
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=num_fusion_layers)

        # ── 4. Dimension projection  384 → 768 ─────────────────────────────
        self.proj = nn.Linear(vit_dim, out_dim)

    # -----------------------------------------------------------------------

    def encode_camera(self, img: torch.Tensor, cam_idx: int) -> torch.Tensor:
        """
        Args:
            img     : (B, 3, H, W)  — single camera image, normalised
            cam_idx : camera index 0..C-1
        Returns:
            r       : (B, R, vit_dim)  — register tokens for this camera
        """
        out = self.vit.base_model.model.forward_features(img)
        r = out["x_norm_regtokens"]                          # (B, R, d)
        r = r + self.cam_embed.weight[cam_idx]               # camera ID
        r = r + self.reg_pos                                 # register position
        return r

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images : (B, C, 3, H, W)  — 4 camera views, pre-normalised
        Returns:
            scene_tokens : (B, C*R, out_dim)  =  (B, 64, 768)
        """
        B = images.shape[0]

        # per-camera register tokens
        reg_list = [
            self.encode_camera(images[:, c], c)              # (B, R, d)
            for c in range(self.C)
        ]

        # S: (B, C*R, vit_dim) = (B, 64, 384)
        S = torch.cat(reg_list, dim=1)

        # cross-camera fusion
        S = self.fusion(S)                                   # (B, 64, 384)

        # project to DiT conditioning dim
        scene_tokens = self.proj(S)                          # (B, 64, 768)
        return scene_tokens

    # -----------------------------------------------------------------------

    def trainable_parameters(self):
        """Return only the parameters that should be optimised."""
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        print(f"[MultiViewRegisterEncoder] "
              f"trainable: {trainable:,} / total: {total:,} "
              f"({100*trainable/total:.2f}%)")
