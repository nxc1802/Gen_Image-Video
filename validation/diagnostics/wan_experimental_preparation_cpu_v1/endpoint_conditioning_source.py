"""Experimental same-checkpoint endpoint conditioning; never grants readiness."""

import hashlib
import inspect
from pathlib import Path

POLICY = "experimental_joint_linear_context_two_endpoint_mask_v1"
UPSTREAM_METHOD_SHA256 = {
    "prepare_latents": "1dd276939a11065e52837f8424aec6f48f8c119487d9bb855f74796d36a45a74",
    "__call__": "841e083f84da734b2c82485e6188ced22bc72e0b66a3c7a78e780332abb7e53c",
}


def runtime_identity():
    return {
        "implementation": POLICY,
        "experimental": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_method_sha256": UPSTREAM_METHOD_SHA256.copy(),
        "checkpoint_changed": False,
        "returned_pixel_frame_replacement": False,
    }


def endpoint_pipeline_class(native):
    # This override relies on the exact native loop consuming its temporal mask.
    for name, expected in UPSTREAM_METHOD_SHA256.items():
        actual = hashlib.sha256(inspect.getsource(getattr(native, name)).encode()).hexdigest()
        if actual != expected:
            raise RuntimeError("Unverified Wan native endpoint-conditioning dependency: " + name)

    import torch
    from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
    from diffusers.utils.torch_utils import get_module_execution_device

    def digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

    class JointEndpointPipeline(native):
        _factory_endpoint_conditioning = POLICY

        def prepare_latents(
            self, image, batch_size, num_channels_latents=16, height=480, width=832,
            num_frames=81, dtype=None, device=None, generator=None, latents=None,
            last_image=None,
        ):
            if last_image is None:
                return super().prepare_latents(
                    image, batch_size, num_channels_latents, height, width, num_frames,
                    dtype, device, generator, latents, last_image=None,
                )
            if (
                not self.config.expand_timesteps or self.vae.config.z_dim != 48
                or batch_size != 1 or image.shape != last_image.shape
                or num_frames < 9 or num_frames % 4 != 1
            ):
                raise RuntimeError("Unsupported experimental Wan endpoint-conditioning shape")
            noise, first_condition, mask = super().prepare_latents(
                image, batch_size, num_channels_latents, height, width, num_frames,
                dtype, device, generator, latents, last_image=None,
            )
            # These RGB frames form a codec prior only. Unmasked middle latents
            # are still sampled by the native Transformer, never returned here.
            alpha = torch.linspace(0, 1, num_frames, device=image.device, dtype=image.dtype)
            alpha = alpha.view(1, 1, num_frames, 1, 1)
            context = (last_image - image).unsqueeze(2) * alpha
            context.add_(image.unsqueeze(2))
            context[:, :, 0].copy_(image)
            context[:, :, -1].copy_(last_image)
            context = context.to(device=get_module_execution_device(self.vae), dtype=self.vae.dtype)
            condition = retrieve_latents(self.vae.encode(context), sample_mode="argmax")
            condition = condition.to(device=noise.device, dtype=noise.dtype)
            mean = torch.tensor(self.vae.config.latents_mean, device=noise.device, dtype=noise.dtype)
            scale = torch.tensor(self.vae.config.latents_std, device=noise.device, dtype=noise.dtype)
            condition = (condition - mean.view(1, -1, 1, 1, 1)) / scale.view(1, -1, 1, 1, 1)
            if (
                condition.shape != noise.shape or not torch.isfinite(condition).all()
                or not torch.allclose(condition[:, :, :1], first_condition, atol=1e-6, rtol=1e-6)
            ):
                raise RuntimeError("Wan joint endpoint encoding changed its native first anchor or shape")
            # Keep exactly two endpoint latent frames; no expanded frozen suffix.
            condition[:, :, 1:-1].zero_()
            mask = mask.clone()
            mask[:, :, -1].zero_()
            if not (
                torch.all(mask[:, :, 0] == 0) and torch.all(mask[:, :, -1] == 0)
                and torch.all(mask[:, :, 1:-1] == 1)
            ):
                raise RuntimeError("Wan endpoint mask did not preserve all free interior latents")
            self._factory_endpoint_call_evidence = {
                **runtime_identity(),
                "normalized_first_image_sha256": digest(image),
                "normalized_last_image_sha256": digest(last_image),
                "first_condition_sha256": digest(condition[:, :, :1]),
                "last_condition_sha256": digest(condition[:, :, -1:]),
                "condition_sha256": digest(condition),
                "mask_sha256": digest(mask),
                "condition_shape": list(condition.shape),
                "context_pixel_frames": num_frames,
                "conditioned_latent_indices": [0, condition.shape[2] - 1],
                "free_interior_latent_frames": condition.shape[2] - 2,
            }
            return noise, condition, mask

    return JointEndpointPipeline
