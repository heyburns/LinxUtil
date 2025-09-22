class MultipurposeConfiguration:
    CATEGORY = "Utility/Configuration"
    FUNCTION = "emit"
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("megapixels", "mask_prompt")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                "mask_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "hands, teeth",
                        "tooltip": "Free-form prompt text for masking (emitted as STRING).",
                    },
                ),
            }
        }

    def emit(self, megapixels: float, mask_prompt: str):
        return (float(megapixels), str(mask_prompt))

NODE_CLASS_MAPPINGS = {
    "Multipurpose Configuration": MultipurposeConfiguration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Multipurpose Configuration": "Multipurpose Configuration",
}
