import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# stegastamp_like_corruptions.py
#
# Differentiable image corruption layer for CIFAR-10.
#
# In StegaStamp, the encoded image is passed through simulated
# distortions before the decoder tries to recover the hidden message.
#
# Original StegaStamp uses physical-world distortions such as:
#   - perspective warp
#   - blur
#   - color changes
#   - noise
#   - JPEG compression
#
# For CIFAR-10, we use a smaller and more stable version:
#   - small affine warp
#   - horizontal flip
#   - brightness / contrast shift
#   - channel shift
#   - desaturation
#   - Gaussian blur
#   - Gaussian noise
#
# JPEG is not included in this first version because differentiable
# JPEG makes the code much more complex. It can be added later.
# ============================================================


def check_image_tensor(image):
    """
    Check that the input image has shape:
        (B, 3, H, W)

    The image is expected to be in [0, 1].
    """

    if image.dim() != 4:
        raise ValueError(
            f"Expected image shape (B, 3, H, W), but got {tuple(image.shape)}"
        )

    if image.size(1) != 3:
        raise ValueError(
            f"Expected 3 image channels, but got {image.size(1)}"
        )


class DifferentiableImageCorruption(nn.Module):
    """
    Differentiable corruption layer.

    This module is used during StegaStamp encoder-decoder training.

    Training flow:
        clean image + message
        -> encoder
        -> encoded image
        -> corruption layer
        -> decoder
        -> decoded message

    The purpose is to force the encoder to hide messages that are
    recoverable even after realistic image transformations.
    """

    def __init__(
        self,
        image_size=32,
        p_affine=0.75,
        p_flip=0.50,
        p_color=0.90,
        p_blur=0.30,
        p_noise=0.80
    ):
        super().__init__()

        self.image_size = image_size

        self.p_affine = p_affine
        self.p_flip = p_flip
        self.p_color = p_color
        self.p_blur = p_blur
        self.p_noise = p_noise

        # Small 3x3 Gaussian-like blur kernel.
        blur_kernel = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0]
            ],
            dtype=torch.float32
        )

        blur_kernel = blur_kernel / blur_kernel.sum()
        blur_kernel = blur_kernel.view(1, 1, 3, 3)

        self.register_buffer("blur_kernel", blur_kernel)

    def forward(self, image, strength=1.0):
        """
        Apply random differentiable corruptions.

        Args:
            image:
                Tensor of shape (B, 3, H, W), values in [0, 1].

            strength:
                Corruption strength from 0.0 to 1.0.

                During early training, use a small value such as 0.0.
                Later, gradually increase it to 1.0.

        Returns:
            corrupted image tensor with shape (B, 3, H, W).
        """

        check_image_tensor(image)

        if strength <= 0.0:
            return image

        strength = float(max(0.0, min(1.0, strength)))

        out = image

        if self.training:
            if torch.rand(1, device=out.device).item() < self.p_affine:
                out = self.random_affine(out, strength)

            if torch.rand(1, device=out.device).item() < self.p_flip:
                out = self.random_horizontal_flip(out)

            if torch.rand(1, device=out.device).item() < self.p_color:
                out = self.random_color_transform(out, strength)

            if torch.rand(1, device=out.device).item() < self.p_blur:
                out = self.gaussian_blur(out)

            if torch.rand(1, device=out.device).item() < self.p_noise:
                out = self.random_noise(out, strength)

        out = torch.clamp(out, 0.0, 1.0)

        return out

    def random_horizontal_flip(self, image):
        """
        Horizontally flip the whole batch.

        This is simple and differentiable.
        """

        return torch.flip(image, dims=[3])

    def random_affine(self, image, strength):
        """
        Apply a small affine transform.

        This approximates the spatial part of StegaStamp's physical
        perturbation pipeline, but keeps it lightweight for CIFAR-10.
        """

        batch_size = image.size(0)
        device = image.device

        max_rotate = math.radians(8.0 * strength)
        max_translate = 0.08 * strength
        max_scale_delta = 0.08 * strength

        angles = (
            torch.rand(batch_size, device=device) * 2.0 - 1.0
        ) * max_rotate

        scales = 1.0 + (
            torch.rand(batch_size, device=device) * 2.0 - 1.0
        ) * max_scale_delta

        tx = (
            torch.rand(batch_size, device=device) * 2.0 - 1.0
        ) * max_translate

        ty = (
            torch.rand(batch_size, device=device) * 2.0 - 1.0
        ) * max_translate

        cos_a = torch.cos(angles) * scales
        sin_a = torch.sin(angles) * scales

        theta = torch.zeros(batch_size, 2, 3, device=device)

        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(
            theta,
            image.size(),
            align_corners=False
        )

        warped = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False
        )

        return warped

    def random_color_transform(self, image, strength):
        """
        Apply differentiable color changes.

        This approximates changes caused by lighting, camera settings,
        display color shifts, and image preprocessing.
        """

        batch_size = image.size(0)
        device = image.device

        # Contrast and brightness.
        contrast = 1.0 + (
            torch.rand(batch_size, 1, 1, 1, device=device) * 2.0 - 1.0
        ) * 0.30 * strength

        brightness = (
            torch.rand(batch_size, 1, 1, 1, device=device) * 2.0 - 1.0
        ) * 0.15 * strength

        out = image * contrast + brightness

        # Channel-wise RGB shift.
        channel_shift = (
            torch.rand(batch_size, 3, 1, 1, device=device) * 2.0 - 1.0
        ) * 0.05 * strength

        out = out + channel_shift

        # Desaturation.
        gray = out.mean(dim=1, keepdim=True)

        desat_amount = (
            torch.rand(batch_size, 1, 1, 1, device=device)
        ) * 0.50 * strength

        out = out * (1.0 - desat_amount) + gray * desat_amount

        out = torch.clamp(out, 0.0, 1.0)

        return out

    def gaussian_blur(self, image):
        """
        Apply a small fixed Gaussian-like blur.

        Uses grouped convolution so each RGB channel is blurred separately.
        """

        channels = image.size(1)

        kernel = self.blur_kernel.to(image.device)
        kernel = kernel.expand(channels, 1, 3, 3)

        blurred = F.conv2d(
            image,
            kernel,
            padding=1,
            groups=channels
        )

        return blurred

    def random_noise(self, image, strength):
        """
        Add Gaussian noise.

        The noise level is randomly sampled for each image.
        """

        batch_size = image.size(0)
        device = image.device

        noise_std = (
            torch.rand(batch_size, 1, 1, 1, device=device)
        ) * 0.05 * strength

        noise = torch.randn_like(image) * noise_std

        out = image + noise

        return out


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corruption = DifferentiableImageCorruption(image_size=32).to(device)
    corruption.train()

    images = torch.rand(8, 3, 32, 32).to(device)

    corrupted = corruption(images, strength=1.0)

    print("Input shape:", images.shape)
    print("Corrupted shape:", corrupted.shape)
    print("Input min/max:", images.min().item(), images.max().item())
    print("Corrupted min/max:", corrupted.min().item(), corrupted.max().item())