Utility nodes for ComfyUI. Created solely for my own use case, shared as a courtesy only.

## 🛠️ Installation

cd /path/to/ComfyUI/custom_nodes  
git clone https://github.com/heyburns/LinxUtil.git  
Restart ComfyUI and refresh the browser  

---

## ✨ Current Nodes

### 1) Multipurpose Configuration
A minimal utility node that exposes two *in-node* controls and emits their values as sockets:

- **Megapixels (FLOAT)** — set a target megapixel resolution value directly in the node (no external input required). Ideal for driving resizers that accept a `FLOAT` input.  
- **Mask Prompt (STRING)** — a multiline text field where you can type prompt text for masking/selection nodes that accept a `STRING`.

<p align="center">
  <img src="assets/multipurpose_configuration.png" alt="Multipurpose Configuration node" width="540">
</p>

### 2) 🎨 Auto White-Balance + Color Match

Fast, reference-driven color normalization for ComfyUI. Applies auto white balance and (optionally) matches a source image’s look to a reference. Minimal controls, predictable results.

**Display name:** Auto White-Balance + Color Match  
**Category:** Image/Color

### Inputs
- **image (IMAGE)** — source to correct  
- **reference (IMAGE)** — target look (only used by match modes)

### Output
- **IMAGE** — corrected image (same size as input)

### Modes (pick one)
- **wb_grayworld** — neutralize cast by equalizing channel means.
- **wb_highlight** — white-patch WB using top luminance percentile.
- **reinhard_lab** — match Lab mean/std to reference (full color).
- **lab_l_only** — match lightness only (keeps source colors).
- **wb_highlight+reinhard** — highlight WB then Lab match (sane default).

### Controls (brief)
- **percentile (80–99.9)** — for `wb_highlight` modes; higher = brighter “white patch.”
- **strength (0–1)** — blend amount; 1 = full effect.
- **clip_gamut (bool)** — clamp to sRGB after processing.
- **force_size (bool)** + **target_width/height** — compute stats at this size for speed; apply at full res.

### Notes
- If result feels strong, lower **strength** or switch to **lab_l_only**.  
- If highlights skew WB, lower **percentile** a bit (e.g., 92–95).
