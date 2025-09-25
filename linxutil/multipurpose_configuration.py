# ======================================================================
# Multipurpose Configuration (Preset + MP + DivisibleBy + Blend + Paths + Prompt)
# ======================================================================
# UI order:
#   1) preset             (enum)
#   2) megapixels         (float)
#   3) custom_width       (int)
#   4) custom_height      (int)
#   5) swap_orientation   (bool)
#   6) divisible_by       (int)   <-- replaces "ensure even"
#   7) blend_factor       (float) <-- NEW, before dirs
#   8) input_dir          (string)
#   9) output_dir         (string)
#  10) prompt             (string, multiline)
#
# Outputs (in this order):
#   INT width, INT height, FLOAT megapixels, FLOAT blend_factor,
#   STRING input_dir, STRING output_dir, STRING prompt
# ======================================================================

import os
from typing import Tuple

_PRESET_MAP = {
    "custom": None,
    # 16:9
    "HD 1280x720 (16:9)": (1280, 720),
    "Full HD 1920x1080 (16:9)": (1920, 1080),
    "UHD 3840x2160 (16:9)": (3840, 2160),
    "2K DCI 2048x1080 (≈17:9)": (2048, 1080),
    # 4:3
    "1024x768 (4:3)": (1024, 768),
    "1600x1200 (4:3)": (1600, 1200),
    # Square
    "1024x1024 (1:1)": (1024, 1024),
    "1536x1536 (1:1)": (1536, 1536),
    # SD
    "SD 720x480 (NTSC)": (720, 480),
    "SD 720x576 (PAL)": (720, 576),
}

class MultipurposeConfiguration:
    """
    Outputs:
      - INT    : width
      - INT    : height
      - FLOAT  : megapixels
      - FLOAT  : blend_factor (0..1; use as B weight / opacity / strength)
      - STRING : input_dir
      - STRING : output_dir
      - STRING : prompt
    """

    CATEGORY = "Utility/Configuration"
    FUNCTION = "emit"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "megapixels", "blend_factor", "input_dir", "output_dir", "prompt")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        # The order here defines the UI order
        return {
            "required": {
                # 1) Preset resolution
                "preset": (
                    list(_PRESET_MAP.keys()),
                    {
                        "default": "custom",
                        "tooltip": "Pick a preset or use 'custom' with width/height below.",
                    },
                ),
                # 2) Megapixels (independent of preset; emitted as FLOAT)
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.01,
                        "max": 200.0,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "Target megapixels to pass to downstream nodes.",
                    },
                ),
                # 3) Custom width (used if preset='custom')
                "custom_width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 8,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Custom width (used when preset='custom').",
                    },
                ),
                # 4) Custom height (used if preset='custom')
                "custom_height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 8,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Custom height (used when preset='custom').",
                    },
                ),
                # 5) Swap orientation
                "swap_orientation": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Swap W↔H after resolving the preset/custom size.",
                    },
                ),
                # 6) Divisible by (replaces 'ensure even')
                "divisible_by": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Round width/height to nearest multiple. "
                                   "Use 1 to disable; common values: 2, 8, 16, 64.",
                    },
                ),
                # 7) Blend factor (0..1)
                "blend_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "General-purpose mix amount (B weight / opacity / strength).",
                    },
                ),
                # 8) Input directory
                "input_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "e.g., inputs/linxutil or G:\\Input\\Shots",
                        "tooltip": "Global input directory. Relative paths resolve from ComfyUI's working dir. "
                                   "Supports ~ for user home.",
                    },
                ),
                # 9) Output directory
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "e.g., outputs/linxutil or G:\\Renders\\Shots",
                        "tooltip": "Global output directory. Relative paths resolve from ComfyUI's working dir. "
                                   "Supports ~ for user home.",
                    },
                ),
                # 10) Prompt (multiline)
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "hands, teeth",
                        "tooltip": "Free-form prompt text (emitted as STRING).",
                    },
                ),
            }
        }

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _normalize_dir(p: str) -> str:
        p = (p or "").strip()
        if not p:
            return ""
        try:
            p = os.path.expanduser(p)   # handle ~
            p = os.path.normpath(p)     # clean separators
        except Exception:
            pass
        return p

    @staticmethod
    def _snap_to_multiple(x: int, m: int) -> int:
        if m is None or m <= 1:
            return max(1, int(x))
        # round to nearest multiple; keep at least 'm'
        snapped = int(round(int(x) / m) * m)
        return max(m, snapped)

    @staticmethod
    def _resolve_size(preset: str, custom_w: int, custom_h: int, swap: bool, div_by: int) -> Tuple[int, int]:
        # 1) Start from preset mapping or custom
        if preset in _PRESET_MAP and _PRESET_MAP[preset] is not None:
            w, h = _PRESET_MAP[preset]
        else:
            w, h = int(custom_w), int(custom_h)

        # 2) Optional swap
        if swap:
            w, h = h, w

        # 3) Ensure >= 1
        w = max(1, int(w))
        h = max(1, int(h))

        # 4) Snap to multiple if requested
        w = MultipurposeConfiguration._snap_to_multiple(w, int(div_by))
        h = MultipurposeConfiguration._snap_to_multiple(h, int(div_by))

        return w, h

    # ---- main ----------------------------------------------------------

    def emit(self,
             preset: str,
             megapixels: float,
             custom_width: int,
             custom_height: int,
             swap_orientation: bool,
             divisible_by: int,
             blend_factor: float,
             input_dir: str,
             output_dir: str,
             prompt: str):

        w, h = self._resolve_size(preset, custom_width, custom_height, swap_orientation, divisible_by)
        mp = float(megapixels)
        bf = float(blend_factor)
        in_dir = self._normalize_dir(input_dir)
        out_dir = self._normalize_dir(output_dir)
        txt = str(prompt)

        return (int(w), int(h), mp, bf, in_dir, out_dir, txt)


# -- Registration --------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Multipurpose Configuration": MultipurposeConfiguration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Multipurpose Configuration": "Multipurpose Configuration",
}
