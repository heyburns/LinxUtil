# LinxUtil package init (subpackage layout, tolerant relative imports)
# Loads modules under ./linxutil and merges their NODE_*_MAPPINGS.

import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _merge(mod):
    if hasattr(mod, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
    if hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)

def _try_import(modname: str):
    # Force a RELATIVE import (prefix with ".") so we import from LinxUtil.*
    rel = modname if modname.startswith(".") else f".{modname}"
    try:
        mod = importlib.import_module(rel, package=__name__)
        _merge(mod)
        print(f"[LinxUtil] Loaded: {modname}")
    except Exception as e:
        print(f"[LinxUtil] Skipped {modname}: {e}")

modules = [
    "linxutil.multipurpose_configuration",
    "linxutil.auto_color_match",
    "linxutil.remove_extension",
    "linxutil.autocrop",
    "linxutil.comfy_ui_levels_match_custom_node",
    "linxutil.crop_by_margins",
    "linxutil.filename_append_suffix",
    "linxutil.stitch_by_mask",
    "linxutil.image_filename_switch",
    # "linxutil.preset_resolution_and_blend",  # uncomment only if the file exists
]

for m in modules:
    _try_import(m)

# --- Backward-compatibility aliases (optional; keeps old workflows working) ---
def _alias(old_key, new_key):
    cls = NODE_CLASS_MAPPINGS.get(new_key)
    if cls:
        NODE_CLASS_MAPPINGS[old_key] = cls
        print(f"[LinxUtil] Alias added: {old_key} -> {new_key}")

# Examples (uncomment/edit if your old graphs reference these type names):
# _alias("StripExtensionNode", "Strip Filename Extension")
# _alias("CropImageByMargins", "Crop Image by Margins")
# _alias("ImageFilenameSwitch2", "Image & Filename Switch (2-way)")
# _alias("Multipurpose Configuration", "Multipurpose Configuration")

print(f"[LinxUtil] Registered {len(NODE_CLASS_MAPPINGS)} node classes: "
      + ", ".join(NODE_CLASS_MAPPINGS.keys()))
