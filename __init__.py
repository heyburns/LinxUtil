# LinxUtil package init (subpackage layout)
# This imports the modules that actually exist under ./linxutil
# and merges their NODE_*_MAPPINGS if present.

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _merge(mod):
    if hasattr(mod, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
    if hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)

# Import from the linxutil subpackage using the filenames you actually have
from .linxutil import multipurpose_configuration as _m_mpc
_merge(_m_mpc)

from .linxutil import auto_color_match as _m_acm
_merge(_m_acm)

from .linxutil import remove_extension as _m_strip
_merge(_m_strip)

from .linxutil import autocrop as _m_autocrop
_merge(_m_autocrop)

from .linxutil import comfy_ui_levels_match_custom_node as _m_levels
_merge(_m_levels)

from .linxutil import crop_by_margins as _m_cropm
_merge(_m_cropm)

from .linxutil import filename_append_suffix as _m_suffix
_merge(_m_suffix)

from .linxutil import stitch_by_mask as _m_stitch
_merge(_m_stitch)

print(f"[LinxUtil] Loaded {len(NODE_CLASS_MAPPINGS)} node classes: "
      + ", ".join(NODE_CLASS_MAPPINGS.keys()))
