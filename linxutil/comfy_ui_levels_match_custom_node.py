# Save this file as: custom_nodes/levels_match_node.py
"""
Levels Match Node for ComfyUI
---------------------------------
Precisely match the black/white (and optional midtone/gamma) levels of a target
image to a reference image, with robust percentile-based clipping.

Features
- Per-channel or luminance-only matching
- Configurable low/high clip percentiles to ignore outliers
- Optional midtone (gamma) matching via a chosen mid percentile
- Batch-safe: supports [B,H,W,C] tensors
- Choice of simple linear levels mapping or CDF-based mapping (experimental)

Inputs
- image (IMAGE): the image to adjust
- reference (IMAGE): the reference image to match
- method (STRING): "linear" or "cdf"
- per_channel (BOOL): match each RGB channel independently (True) or use a single luminance curve (False)
- low_clip (FLOAT): lower percentile to treat as black (0–5 recommended)
- high_clip (FLOAT): upper percentile to treat as white (95–100 recommended)
- use_gamma (BOOL): also match midtone using `mid_percentile`
- mid_percentile (FLOAT): percentile for midtone (e.g., 50 for median)

Output
- IMAGE: adjusted image with levels matched to the reference

Installation
- Place this file in ComfyUI/custom_nodes/
- Restart ComfyUI. The node appears under: Image/Color

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""

from __future__ import annotations
import math
from typing import Tuple

import torch


def _ensure_bchw(img: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B, H, W, C] with float32 in [0,1]."""
    if img is None:
        raise ValueError("Image tensor is None")
    if img.dim() == 3:  # [H,W,C]
        img = img.unsqueeze(0)
    if img.dim() != 4 or img.size(-1) != 3:
        raise ValueError(f"Expected [B,H,W,3] or [H,W,3], got {tuple(img.shape)}")
    return img.clamp(0.0, 1.0).to(torch.float32)


def _luminance(img: torch.Tensor) -> torch.Tensor:
    """Compute relative luminance Y from linear RGB in [0,1], shape [B,H,W].
    Using Rec.709 coefficients.
    """
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _percentiles_per_batch(
    data: torch.Tensor,
    q_low: float,
    q_high: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-batch low/high quantiles for data with shape [B, ...].
    The quantiles are computed across all remaining dimensions.
    """
    B = data.shape[0]
    flat = data.reshape(B, -1)
    q = torch.tensor([q_low / 100.0, q_high / 100.0], device=data.device, dtype=data.dtype)
    # torch.quantile supports per-row via apply along last dim with keepdim
    lows = torch.quantile(flat, q[0], dim=1, keepdim=True)  # [B,1]
    highs = torch.quantile(flat, q[1], dim=1, keepdim=True)  # [B,1]
    return lows.squeeze(1), highs.squeeze(1)


def _percentiles_per_batch_channel(
    img: torch.Tensor,
    q_low: float,
    q_high: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-batch, per-channel low/high quantiles for img [B,H,W,3].
    Output shapes: lows [B,3], highs [B,3].
    """
    B = img.shape[0]
    C = img.shape[-1]
    assert C == 3
    flat = img.reshape(B, -1, C)  # [B, HW, 3]
    lows = []
    highs = []
    ql = q_low / 100.0
    qh = q_high / 100.0
    for c in range(C):
        vc = flat[:, :, c]
        lows.append(torch.quantile(vc, ql, dim=1))  # [B]
        highs.append(torch.quantile(vc, qh, dim=1))  # [B]
    lows = torch.stack(lows, dim=1)  # [B,3]
    highs = torch.stack(highs, dim=1)  # [B,3]
    return lows, highs


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return a / (b.abs() + eps)


def _apply_linear_levels(
    img: torch.Tensor,
    in_low: torch.Tensor,
    in_high: torch.Tensor,
    out_low: torch.Tensor,
    out_high: torch.Tensor,
) -> torch.Tensor:
    """Apply linear levels from [in_low,in_high] -> [out_low,out_high].
    Supports broadcasting with shapes:
      img [B,H,W,3]
      in_low/in_high/out_low/out_high: [B,1,1,1] or [B,1,1,3]
    """
    x = img - in_low
    scale = _safe_div(out_high - out_low, in_high - in_low)
    y = x * scale + out_low
    return y.clamp(0.0, 1.0)


def _apply_gamma_to_segment(
    img: torch.Tensor,
    out_low: torch.Tensor,
    out_high: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Apply power-law gamma within the [out_low,out_high] segment.
    We first normalize to [0,1], apply pow(gamma), then re-range.
    Shapes broadcast as in _apply_linear_levels.
    """
    norm = _safe_div((img - out_low), (out_high - out_low))
    norm = norm.clamp(0.0, 1.0)
    adj = torch.pow(norm, gamma)
    return (adj * (out_high - out_low) + out_low).clamp(0.0, 1.0)


def _compute_gamma(
    mid_in: torch.Tensor,
    in_low: torch.Tensor,
    in_high: torch.Tensor,
    mid_out: torch.Tensor,
    out_low: torch.Tensor,
    out_high: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Solve for gamma so that mid maps to reference mid after linear mapping.
    We assume we've already linearly mapped to [out_low,out_high] segment.
    gamma = log((mid_out_n))/(log((mid_in_n))) with n meaning normalized to [0,1].
    Return broadcastable gamma with safe defaults when values are near-equal.
    """
    # Normalize mids to [0,1] ranges
    m_in_n = _safe_div(mid_in - out_low, out_high - out_low, eps).clamp(1e-4, 1 - 1e-4)
    m_out_n = _safe_div(mid_out - out_low, out_high - out_low, eps).clamp(1e-4, 1 - 1e-4)
    g = torch.log(m_out_n) / torch.log(m_in_n)
    # Limit gamma to sane range
    return g.clamp(0.25, 4.0)


def _match_levels_linear(
    image: torch.Tensor,
    reference: torch.Tensor,
    per_channel: bool,
    low_clip: float,
    high_clip: float,
    use_gamma: bool,
    mid_percentile: float,
) -> torch.Tensor:
    B, H, W, C = image.shape
    device = image.device

    if per_channel:
        in_low, in_high = _percentiles_per_batch_channel(image, low_clip, high_clip)  # [B,3]
        out_low, out_high = _percentiles_per_batch_channel(reference, low_clip, high_clip)  # [B,3]
        # reshape for broadcast
        in_low = in_low.view(B, 1, 1, C)
        in_high = in_high.view(B, 1, 1, C)
        out_low = out_low.view(B, 1, 1, C)
        out_high = out_high.view(B, 1, 1, C)

        y = _apply_linear_levels(image, in_low, in_high, out_low, out_high)

        if use_gamma:
            # per-channel mid percentiles
            mid_in_l, _ = _percentiles_per_batch_channel(y, mid_percentile, mid_percentile)
            mid_out_l, _ = _percentiles_per_batch_channel(reference, mid_percentile, mid_percentile)
            mid_in = mid_in_l.view(B, 1, 1, C)
            mid_out = mid_out_l.view(B, 1, 1, C)
            gamma = _compute_gamma(mid_in, out_low, out_high, mid_out, out_low, out_high)
            y = _apply_gamma_to_segment(y, out_low, out_high, gamma)
        return y
    else:
        # Luminance path: compute single set of levels per batch
        lum_in = _luminance(image)
        lum_ref = _luminance(reference)
        in_low, in_high = _percentiles_per_batch(lum_in, low_clip, high_clip)  # [B]
        out_low, out_high = _percentiles_per_batch(lum_ref, low_clip, high_clip)  # [B]
        # reshape for broadcast to channels
        in_low = in_low.view(B, 1, 1, 1)
        in_high = in_high.view(B, 1, 1, 1)
        out_low = out_low.view(B, 1, 1, 1)
        out_high = out_high.view(B, 1, 1, 1)

        y = _apply_linear_levels(image, in_low, in_high, out_low, out_high)

        if use_gamma:
            # midtone via luminance
            mid_in, _ = _percentiles_per_batch(_luminance(y), mid_percentile, mid_percentile)
            mid_out, _ = _percentiles_per_batch(lum_ref, mid_percentile, mid_percentile)
            mid_in = mid_in.view(B, 1, 1, 1)
            mid_out = mid_out.view(B, 1, 1, 1)
            gamma = _compute_gamma(mid_in, out_low, out_high, mid_out, out_low, out_high)
            y = _apply_gamma_to_segment(y, out_low, out_high, gamma)
        return y


def _cdf_match_single_channel(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Histogram (CDF) match for one channel per batch element.
    src/ref: [B,H,W]
    Returns: matched [B,H,W]
    Note: This is more expensive; used when method="cdf".
    """
    B, H, W = src.shape
    device = src.device
    out = torch.empty_like(src)

    # Work in 256 bins for speed/robustness
    bins = 256
    bin_edges = torch.linspace(0.0, 1.0, bins + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    for i in range(B):
        s = src[i].flatten()
        r = ref[i].flatten()
        # histograms
        s_hist = torch.histc(s, bins=bins, min=0.0, max=1.0)
        r_hist = torch.histc(r, bins=bins, min=0.0, max=1.0)
        s_cdf = torch.cumsum(s_hist, dim=0)
        r_cdf = torch.cumsum(r_hist, dim=0)
        s_cdf = s_cdf / s_cdf[-1].clamp_min(1e-6)
        r_cdf = r_cdf / r_cdf[-1].clamp_min(1e-6)
        # For each s value, find r value with nearest CDF
        # Build a LUT by mapping each s_cdf to r bin with closest CDF
        # Compute inverse mapping indices
        # r_cdf is monotonic; for each s_cdf[k], find idx where r_cdf >= s_cdf[k]
        idx = torch.searchsorted(r_cdf, s_cdf)
        idx = idx.clamp(0, bins - 1)
        lut = bin_centers[idx]
        # Map source pixels to LUT
        # Quantize s to bins
        s_idx = torch.clamp((s * (bins - 1)).round().to(torch.long), 0, bins - 1)
        s_matched = lut[s_idx]
        out[i] = s_matched.view(H, W)
    return out.clamp(0.0, 1.0)


def _match_levels_cdf(
    image: torch.Tensor,
    reference: torch.Tensor,
    per_channel: bool,
) -> torch.Tensor:
    # Apply CDF matching per-channel or on luminance
    if per_channel:
        chans = []
        for c in range(3):
            chans.append(
                _cdf_match_single_channel(image[..., c], reference[..., c])
            )
        y = torch.stack(chans, dim=-1)
    else:
        y_l = _cdf_match_single_channel(_luminance(image), _luminance(reference))
        # Replace luminance while preserving chroma approximately
        # Simple approach: scale each channel by ratio of new/old luminance
        old_l = _luminance(image).clamp_min(1e-6)
        ratio = (y_l / old_l).unsqueeze(-1)
        y = (image * ratio).clamp(0.0, 1.0)
    return y


class LevelsMatch:
    """ComfyUI node: Match levels of an image to a reference image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": ("STRING", {"default": "linear", "choices": ["linear", "cdf"]}),
                "per_channel": ("BOOLEAN", {"default": True}),
                "low_clip": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "high_clip": ("FLOAT", {"default": 99.5, "min": 90.0, "max": 100.0, "step": 0.1}),
                "use_gamma": ("BOOLEAN", {"default": True}),
                "mid_percentile": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 99.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match"
    CATEGORY = "Image/Color"
    OUTPUT_NODE = False

    def match(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        method: str = "linear",
        per_channel: bool = True,
        low_clip: float = 0.5,
        high_clip: float = 99.5,
        use_gamma: bool = True,
        mid_percentile: float = 50.0,
    ):
        img = _ensure_bchw(image)
        ref = _ensure_bchw(reference)

        # If batches differ, broadcast the last element of the smaller batch
        if ref.shape[0] != img.shape[0]:
            if ref.shape[0] == 1:
                ref = ref.expand(img.shape[0], -1, -1, -1)
            elif img.shape[0] == 1:
                img = img.expand(ref.shape[0], -1, -1, -1)
            else:
                raise ValueError("Batch sizes differ and neither is 1; cannot broadcast.")

        if method == "linear":
            out = _match_levels_linear(
                img, ref, per_channel, low_clip, high_clip, use_gamma, mid_percentile
            )
        elif method == "cdf":
            out = _match_levels_cdf(img, ref, per_channel)
        else:
            raise ValueError(f"Unknown method: {method}")

        return (out,)


NODE_CLASS_MAPPINGS = {
    "LevelsMatch": LevelsMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LevelsMatch": "Levels Match (Reference)",
}
