# Filename Append Suffix — ComfyUI
# Strips extension(s) and appends a suffix for saving variants, e.g.:
#   "001.jpg" + "-supir" -> "001-supir"
#   "shots/001.webp" + "-ccsr" -> "shots/001-ccsr"
# Save as: custom_nodes/filename_append_suffix.py

from __future__ import annotations
import os
import re

print("[FilenameAppendSuffix] loaded")

def _strip_extensions(path: str, strip_all: bool = True) -> tuple[str, str]:
    """
    Returns (dir, base_without_ext).
    strip_all=True removes all trailing extensions (e.g., .tar.gz -> no ext).
    """
    path = path.strip()
    d = os.path.dirname(path)
    b = os.path.basename(path)
    if strip_all:
        # Remove one or more trailing extensions like .png, .jpg, .webp, .tif, .tar.gz…
        b = re.sub(r'(?:\.[A-Za-z0-9]{1,5})+$', '', b)
    else:
        b = os.path.splitext(b)[0]
    return d, b

class FilenameAppendSuffix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"multiline": False, "default": ""}),
                "suffix":   ("STRING", {"multiline": False, "default": "supir"}),
                "separator": ("STRING", {"multiline": False, "default": "-"}),
                "strip_all_extensions": ("BOOLEAN", {"default": True}),
                "preserve_directory": ("BOOLEAN", {"default": True}),
                "trim_whitespace": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "build"
    CATEGORY = "Utils/IO"

    def build(self,
              filename: str,
              suffix: str = "supir",
              separator: str = "-",
              strip_all_extensions: bool = True,
              preserve_directory: bool = True,
              trim_whitespace: bool = True):

        fn = filename if isinstance(filename, str) else str(filename)
        sx = suffix if isinstance(suffix, str) else str(suffix)
        sep = separator if isinstance(separator, str) else "-"

        if trim_whitespace:
            fn = fn.strip()
            sx = sx.strip()
            sep = sep.strip() or "-"

        d, base = _strip_extensions(fn, strip_all=strip_all_extensions)

        # If suffix is empty, just return the base (no extension).
        if not sx:
            out = base
        else:
            # Avoid duplicate separators if base already ends with it
            if base.endswith(sep):
                out = f"{base}{sx}"
            else:
                out = f"{base}{sep}{sx}"

        # Re-attach directory if requested
        if preserve_directory and d:
            out = os.path.join(d, out)

        return (out,)

NODE_CLASS_MAPPINGS = {
    "FilenameAppendSuffix": FilenameAppendSuffix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilenameAppendSuffix": "Filename: Append Suffix",
}
