"""Adapter-owned input latent construction. This node emits no generated image."""


class FactoryQwen21Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 32, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 32, "step": 32}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "create"
    CATEGORY = "factory/adapter"

    def create(self, width, height):
        import torch
        import comfy.model_management

        return (
            {
                "samples": torch.zeros(
                    (1, 64, height // 16, width // 16),
                    device=comfy.model_management.intermediate_device(),
                )
            },
        )


NODE_CLASS_MAPPINGS = {"FactoryQwen21Latent": FactoryQwen21Latent}
