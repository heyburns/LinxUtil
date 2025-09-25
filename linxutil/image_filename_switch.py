class ImageFilenameSwitch2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_index": ("INT", {"default": 0, "min": 0, "max": 1}),  # 0 or 1
                "image1": ("IMAGE",),
                "filename1": ("STRING", {"default": ""}),
                "image2": ("IMAGE",),
                "filename2": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "switch"
    CATEGORY = "Custom"

    def switch(self, selected_index, image1, filename1, image2, filename2):
        if selected_index == 0:
            return (image1, filename1)
        else:
            return (image2, filename2)


NODE_CLASS_MAPPINGS = {
    "ImageFilenameSwitch2": ImageFilenameSwitch2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFilenameSwitch2": "Image & Filename Switch (2-way)"
}
