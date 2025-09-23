import os

class StripExtensionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("root_filename",)
    FUNCTION = "strip_extension"
    CATEGORY = "Custom/Utility"

    def strip_extension(self, filename: str):
        # Use os.path.splitext to remove common extensions
        root, _ = os.path.splitext(filename)

        # Handle double extensions like .tar.gz if needed
        # Uncomment below if you want to remove multiple suffixes
        # while any(root.lower().endswith(ext) for ext in [".tar", ".backup"]):
        #     root, _ = os.path.splitext(root)

        return (root,)

NODE_CLASS_MAPPINGS = {
    "StripExtensionNode": StripExtensionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StripExtensionNode": "Strip Filename Extension"
}
