"""
Single-View Register Encoder — System 1 perception (DrivoR-style, DualVLN-aligned).

Pipeline (following "Driving on Registers", arXiv 2601.05083):
  1. ViT-S (full fine-tuning) + register tokens: self-attn inside ViT (register ↔ patch)
  2. Concat registers from goal + current → 32 image-conditioned queries
  3. Project 384 → 768 (matches Q-Former's 768-dim space)
  4. Cross-attn decoder (768-dim, 12 heads, 3 layers): register tokens (Q) attend to patches (KV)
  5. Output: (B, 32, 768)

DualVLN-aligned: ViT-S full FT, 2 images, 32 output tokens, 768-dim decoder.
Only difference vs DualVLN: image-conditioned register queries instead of fixed Q-Former queries.
"""

import importlib.util
import os
import sys

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# dinov2 import (bypasses broken encoder __init__.py)
# ---------------------------------------------------------------------------

def _import_dinov2():
    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "depth_anything", "depth_anything_v2")

    for _name, _rel in [
        ("dinov2_layers",             "dinov2_layers/__init__.py"),
        ("dinov2_layers.attention",   "dinov2_layers/attention.py"),
        ("dinov2_layers.mlp",         "dinov2_layers/mlp.py"),
        ("dinov2_layers.patch_embed", "dinov2_layers/patch_embed.py"),
        ("dinov2_layers.swiglu_ffn",  "dinov2_layers/swiglu_ffn.py"),
        ("dinov2_layers.block",       "dinov2_layers/block.py"),
        ("dinov2_layers.layer_scale", "dinov2_layers/layer_scale.py"),
        ("dinov2_layers.drop_path",   "dinov2_layers/drop_path.py"),
    ]:
        key = f"_dav2.{_name}"
        if key not in sys.modules:
            path = os.path.join(_base, _rel)
            if os.path.isfile(path):
                spec = importlib.util.spec_from_file_location(key, path)
                m = importlib.util.module_from_spec(spec)
                sys.modules[key] = m
                sys.modules[_name] = m
                spec.loader.exec_module(m)

    spec = importlib.util.spec_from_file_location("_dav2.dinov2",
                                                   os.path.join(_base, "dinov2.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_dav2.dinov2"] = m
    spec.loader.exec_module(m)
    return m


_dinov2 = _import_dinov2()
vit_small = _dinov2.vit_small


# ---------------------------------------------------------------------------
# DrivoR-style cross-attention decoder
# ---------------------------------------------------------------------------

class RegisterDecoder(nn.Module):
    """
    Vanilla Transformer decoder (DrivoR-style).

    register tokens (tgt, Q) → self-attn → cross-attn to patch tokens (KV) → FFN
    Register tokens are image-conditioned (from ViT self-attn) before entering here,
    unlike Q-Former's fixed image-agnostic queries.
    """

    def __init__(self, dim: int = 768, num_heads: int = 12, num_layers: int = 3):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, reg_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        reg_tokens  : (B, R, D) — register tokens as tgt (Q)
        patch_tokens: (B, N, D) — patch tokens as memory (KV)
        Returns     : (B, R, D)
        """
        return self.decoder(reg_tokens, patch_tokens)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SingleViewRegisterEncoder(nn.Module):
    """
    Two-image register encoder for System 1 (DrivoR-style, DualVLN-aligned).

    Processes goal + current look-down images separately through ViT-S with
    register tokens (full fine-tuning, same as DualVLN rgb_model), then
    cross-attends all registers to all patches via a 768-dim decoder.

    Args:
        num_registers        : register tokens per image (default 16, ×2 = 32 total)
        vit_dim              : ViT-S hidden dim (384)
        out_dim              : output dim = DiT conditioning dim (768)
        dav2_ckpt            : path to DAv2 ViT-S pretrained weight
        num_cross_attn_layers: RegisterDecoder layers (default 3, matches Q-Former)
    """

    def __init__(
        self,
        num_registers: int = 16,
        vit_dim: int = 384,
        out_dim: int = 768,
        dav2_ckpt=None,
        num_cross_attn_layers: int = 3,
    ):
        super().__init__()
        self.R = num_registers

        # ── ViT-S + register tokens (full fine-tuning, DualVLN-aligned) ──────
        self.vit = vit_small(patch_size=14, num_register_tokens=num_registers)

        if dav2_ckpt and os.path.isfile(dav2_ckpt):
            ckpt = torch.load(dav2_ckpt, map_location="cpu")
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            state = {k.replace("pretrained.", ""): v for k, v in state.items()}
            if "pos_embed" in state and state["pos_embed"].shape != self.vit.pos_embed.shape:
                pe_ckpt = state["pos_embed"]
                pe_tgt  = self.vit.pos_embed
                cls, pe_ckpt_patches = pe_ckpt[:, :1], pe_ckpt[:, 1:]
                n_src = pe_ckpt_patches.shape[1]
                n_dst = pe_tgt[:, 1:].shape[1]
                h_src = w_src = int(n_src ** 0.5)
                h_dst = w_dst = int(n_dst ** 0.5)
                pe_interp = torch.nn.functional.interpolate(
                    pe_ckpt_patches.reshape(1, h_src, w_src, -1).permute(0, 3, 1, 2),
                    size=(h_dst, w_dst), mode="bicubic", align_corners=False,
                ).permute(0, 2, 3, 1).reshape(1, n_dst, -1)
                state["pos_embed"] = torch.cat([cls, pe_interp], dim=1)
                print(f"[SingleViewRegisterEncoder] interpolated pos_embed "
                      f"({1},{n_src}) → ({1},{n_dst})")
            missing, unexpected = self.vit.load_state_dict(state, strict=False)
            print(f"[SingleViewRegisterEncoder] loaded ckpt  "
                  f"missing={len(missing)}  unexpected={len(unexpected)}")
        else:
            print("[SingleViewRegisterEncoder] random init (no ckpt)")

        # full fine-tuning — all ViT params trainable (same as DualVLN)
        for p in self.vit.parameters():
            p.requires_grad_(True)

        # ── Project vit_dim → out_dim before decoder ─────────────────────────
        # Separate projections for register vs patch tokens (different roles).
        # Matches DualVLN: Memory Encoder output projected to 768 before Q-Former.
        self.reg_proj_in   = nn.Linear(vit_dim, out_dim)
        self.patch_proj_in = nn.Linear(vit_dim, out_dim)

        # ── DrivoR-style cross-attn decoder (register Q, patch KV) ──────────
        # 32 image-conditioned registers cross-attend to 512 patch tokens.
        # 768-dim, 12 heads, 3 layers — matches Q-Former capacity (~28.9M total).
        self.reg_decoder = RegisterDecoder(
            dim=out_dim,
            num_heads=out_dim // 64,   # 768//64 = 12 heads
            num_layers=num_cross_attn_layers,
        )

    # -----------------------------------------------------------------------

    def _encode_one(self, image: torch.Tensor):
        """ViT-S forward for a single image. Returns (reg_tokens, patch_tokens)."""
        out = self.vit.forward_features(image)
        return out["x_norm_regtokens"], out["x_norm_patchtokens"]

    def forward(self, cur: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            cur  : (B, 3, H, W) — current look-down image
            goal : (B, 3, H, W) — goal image (if None, uses cur for both)
        Returns:
            scene_tokens : (B, R*2, out_dim)  e.g. (B, 32, 768)
        """
        if goal is None:
            goal = cur

        reg_cur,  patch_cur  = self._encode_one(cur)    # (B, R, 384)
        reg_goal, patch_goal = self._encode_one(goal)   # (B, R, 384)

        # project to 768 — matches Q-Former's 768-dim operating space
        reg     = self.reg_proj_in(torch.cat([reg_goal,   reg_cur],   dim=1))  # (B, 2R, 768)
        patches = self.patch_proj_in(torch.cat([patch_goal, patch_cur], dim=1))  # (B, 2N, 768)

        # DrivoR-style: image-conditioned registers cross-attend to all patches
        reg = self.reg_decoder(reg, patches)   # (B, 2R, 768)

        return reg                             # (B, 2R, out_dim)

    # -----------------------------------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        print(f"[SingleViewRegisterEncoder] "
              f"trainable: {trainable:,} / total: {total:,} "
              f"({100*trainable/total:.2f}%)")
