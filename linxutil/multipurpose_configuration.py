# ======================================================================
# Multipurpose Configuration (InputDir first, OutputDir second)
# ======================================================================
# Emits four values configured inside the node:
#   • STRING : input_dir     (global input directory path)
#   • STRING : output_dir    (global output directory path)
#   • FLOAT  : megapixels    (e.g., 6.00)
#   • STRING : mask_prompt   (multiline text)
# ======================================================================

import os

class MultipurposeConfiguration:
    """
    Outputs:
      - STRING: input_dir   (normalized filesystem path)
      - STRING: output_dir  (normalized filesystem path)
      - FLOAT : megapixels  (float, to drive resize nodes etc.)
      - STRING: prompt (free text, multiline)
    """

    CATEGORY = "Utility/Configuration"
    FUNCTION = "emit"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("input_dir", "output_dir", "megapixels", "prompt")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        # Order here controls the UI order
        return {
            "required": {
                # 1) Input directory
                "input_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "e.g., inputs/linxutil or G:\\Input\\Shots",
                        "tooltip": "Global input directory. Relative paths resolve from ComfyUI's working dir. "
                                   "Supports ~ for the user home.",
                    },
                ),
                # 2) Output directory
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "e.g., outputs/linxutil or G:\\Renders\\Shots",
                        "tooltip": "Global output directory. Relative paths resolve from ComfyUI's working dir. "
                                   "Supports ~ for the user home.",
                    },
                ),
                # 3) Megapixels
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.01,
                        "max": 200.0,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "Target megapixels (emitted as FLOAT).",
                    },
                ),
                # 4) Text field (mask prompt or general notes)
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

    @staticmethod
    def _normalize_dir(p: str) -> str:
        p = (p or "").strip()
        if not p:
            return ""
        try:
            p = os.path.expanduser(p)   # handle ~
            p = os.path.normpath(p)     # clean separators like // and /../
        except Exception:
            # keep original on any unexpected error
            pass
        return p

    # Match the input order: input_dir -> output_dir -> megapixels -> mask_prompt
    def emit(self, input_dir: str, output_dir: str, megapixels: float, mask_prompt: str):
        in_dir = self._normalize_dir(str(input_dir))
        out_dir = self._normalize_dir(str(output_dir))
        mp = float(megapixels)
        txt = str(prompt)
        # Match RETURN_TYPES/RETURN_NAMES order
        return (in_dir, out_dir, mp, txt)


# -- Registration --------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Multipurpose Configuration": MultipurposeConfiguration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Multipurpose Configuration": "Multipurpose Configuration",
}
