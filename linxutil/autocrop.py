# custom_nodes/auto_crop_borders.py
# AutoCropBorders v1.1 (fixed 4-D broadcasting + safer sampling)
from __future__ import annotations
import torch
import torch.nn.functional as F

def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"Expected [B,H,W,C] or [H,W,C], got {tuple(x.shape)}")
    return x.to(torch.float32).clamp(0.0, 1.0)

def _rgb2y(img_bhwc: torch.Tensor) -> torch.Tensor:
    r,g,b = img_bhwc[...,0], img_bhwc[...,1], img_bhwc[...,2]
    y = 0.2126*r + 0.7152*g + 0.0722*b
    return y.unsqueeze(-1)

def _dilate(mask: torch.Tensor) -> torch.Tensor:
    k = torch.ones((1,1,3,3), device=mask.device)
    return (F.conv2d(mask, k, padding=1) > 0).float()

def _region_grow(img: torch.Tensor,  # [B,H,W,C’]
                 seed_mask: torch.Tensor,  # [B,1,H,W]
                 border_rgb: torch.Tensor, # [B,1,1,C’]  (4-D!)
                 thr_ch: torch.Tensor,     # [B,1,1,C’]  per-channel thr 0..1
                 max_iter: int = 4096) -> torch.Tensor:
    B,H,W,Cp = img.shape
    # L1 distance per-pixel to border reference:
    dist = (img - border_rgb).abs().sum(dim=-1, keepdim=True)  # [B,H,W,1]
    dist = dist.permute(0,3,1,2)  # [B,1,H,W]
    # Convert per-channel thr to L1-equivalent: sum of per-channel thresholds
    thr_l1 = thr_ch.sum(dim=-1, keepdim=True).permute(0,3,1,2)  # [B,1,1,1] -> [B,1,1,1] then [B,1,1,1]
    eligible = (dist <= thr_l1).float()
    cur = seed_mask.clone()
    it = 0
    while it < max_iter:
        it += 1
        grown = _dilate(cur) * eligible
        if torch.equal(grown, cur):
            break
        cur = grown
    return cur  # [B,1,H,W]  (1 = border)

def _bbox_from_mask(nonborder: torch.Tensor, pad: int):
    B,_,H,W = nonborder.shape
    boxes = []
    for b in range(B):
        m = nonborder[b,0]
        ys = torch.where(m.any(dim=1))[0]
        xs = torch.where(m.any(dim=0))[0]
        if len(ys) == 0 or len(xs) == 0:
            l,t,w,h = 0,0,W,H
        else:
            top, bottom = int(ys[0].item()), int(ys[-1].item())
            left, right = int(xs[0].item()), int(xs[-1].item())
            l = max(0, left - pad); t = max(0, top - pad)
            r = min(W, right + 1 + pad); btm = min(H, bottom + 1 + pad)
            w = max(1, r - l); h = max(1, btm - t)
        boxes.append((l,t,w,h))
    return boxes

class AutoCropBorders:
    """
    Auto-crops borders using fuzzy edge-seeded flood fill.
    fuzz_mode:
      • 'percent'  : fixed tolerance (percent of 0..1 per channel)
      • 'adaptive' : percent + adaptive_k * MAD(edge)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fuzz_mode": (["percent","adaptive"], {"default":"adaptive"}),
                "fuzz_percent": ("FLOAT", {"default":5.0, "min":0.0, "max":50.0, "step":0.1}),
                "adaptive_k": ("FLOAT", {"default":2.0, "min":0.0, "max":10.0, "step":0.1}),
                "edge_margin_px": ("INT", {"default":4, "min":1, "max":64, "step":1}),
                "pad_px": ("INT", {"default":0, "min":0, "max":64, "step":1}),
                "use_luma_only": ("BOOLEAN", {"default":False}),
                "max_growth_iter": ("INT", {"default":4096, "min":64, "max":16384, "step":64}),
                "return_border_mask": ("BOOLEAN", {"default":False}),
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT","INT","INT","MASK")
    RETURN_NAMES = ("image","left","top","width","height","border_mask")
    FUNCTION = "run"
    CATEGORY = "Image/Transform"

    def run(self, image: torch.Tensor,
            fuzz_mode: str = "adaptive",
            fuzz_percent: float = 5.0,
            adaptive_k: float = 2.0,
            edge_margin_px: int = 4,
            pad_px: int = 0,
            use_luma_only: bool = False,
            max_growth_iter: int = 4096,
            return_border_mask: bool = False):

        img = _to_bhwc(image)                       # [B,H,W,C]
        B,H,W,C = img.shape
        if C < 3:
            img = img.repeat(1,1,1,3); C = 3

        # Work in RGB or luma
        work = _rgb2y(img) if use_luma_only else img[...,:3]  # [B,H,W,C’]
        Cp = work.shape[-1]

        m = edge_margin_px
        # Seeds from all four edges
        seed = torch.zeros((B,1,H,W), device=img.device, dtype=torch.float32)
        seed[:,:, :m, :] = 1.0
        seed[:,:, -m:, :] = 1.0
        seed[:,:,:, :m] = 1.0
        seed[:,:,:, -m:] = 1.0

        # ---- Collect edge samples WITHOUT boolean advanced indexing ----
        # Top & bottom strips
        top    = work[:, :m, :, :].reshape(B, -1, Cp)
        bottom = work[:, -m:, :, :].reshape(B, -1, Cp)
        left   = work[:, :, :m, :].reshape(B, -1, Cp)
        right  = work[:, :, -m:, :].reshape(B, -1, Cp)
        samples = torch.cat([top,bottom,left,right], dim=1)   # [B, N, Cp]

        # Robust stats per channel
        med = samples.median(dim=1).values                    # [B,Cp]
        absdev = (samples - med.unsqueeze(1)).abs()
        mad = absdev.median(dim=1).values + 1e-6              # [B,Cp]

        # Per-channel fuzz
        base = torch.full_like(med, fuzz_percent/100.0)
        if fuzz_mode == "adaptive":
            base = base + adaptive_k * mad
        thr_ch = base.clamp(0.0, 1.0)                         # [B,Cp]

        # 4-D broadcast references
        border_rgb = med.view(B,1,1,Cp)                       # [B,1,1,C’]
        thr_ch_4d  = thr_ch.view(B,1,1,Cp)                    # [B,1,1,C’]

        # Region grow on working space
        grown = _region_grow(work, seed, border_rgb, thr_ch_4d, max_iter=max_growth_iter)  # [B,1,H,W]
        nonborder = (1.0 - grown).clamp(0,1)

        # Crop boxes
        boxes = _bbox_from_mask(nonborder, pad_px)

        # Crop each, then pad to batch max size (Comfy batches need same dims)
        crops = []
        for b,(l,t,wc,hc) in enumerate(boxes):
            crops.append(img[b:b+1, t:t+hc, l:l+wc, :])
        max_h = max(c.shape[1] for c in crops)
        max_w = max(c.shape[2] for c in crops)
        out = []
        for c in crops:
            pad_h = max_h - c.shape[1]; pad_w = max_w - c.shape[2]
            out.append(F.pad(c, (0,0,0,pad_w,0,pad_h)))
        out_img = torch.cat(out, dim=0).clamp(0,1)

        # Optional border mask (resized to output size for convenience)
        border_mask = grown.permute(0,2,3,1)  # [B,H,W,1]
        if (border_mask.shape[1] != out_img.shape[1]) or (border_mask.shape[2] != out_img.shape[2]):
            border_mask = F.interpolate(border_mask.permute(0,3,1,2),
                                        size=(out_img.shape[1], out_img.shape[2]),
                                        mode="nearest").permute(0,2,3,1)

        if B == 1:
            l,t,wc,hc = boxes[0]
        else:
            l=t=wc=hc=0

        return (out_img, int(l), int(t), int(wc), int(hc),
                border_mask if return_border_mask else torch.zeros_like(border_mask))

NODE_CLASS_MAPPINGS = {"AutoCropBorders": AutoCropBorders}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoCropBorders": "Auto-Crop Borders (fuzzy)"}
