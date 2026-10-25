import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# stegastamp_like_corruptions.py
#
# Advanced differentiable corruption layer for CA3 StegaStamp.
#
# Purpose:
#   Train the steganographic trigger to survive image changes that are
#   similar to SSL augmentations.
#
# Used by:
#   stegastamp_like_model.py
#
# Main corruptions:
#   - random resized crop
#   - horizontal flip
#   - brightness jitter
#   - contrast jitter
#   - saturation jitter
#   - random grayscale mixing
#   - Gaussian blur
#   - Gaussian noise
#   - JPEG-like downsample/upsample
#   - soft quantization
#
# Inputs and outputs are assumed to be in [0, 1].
# ============================================================


# ============================================================
# Tensor checks
# ============================================================

def check_image_tensor(image):
    """
    Check that image has shape:
        (B, 3, H, W)

    and is a floating point tensor.
    """

    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor")

    if image.dim() != 4:
        raise ValueError(
            f"Expected image shape (B, 3, H, W), "
            f"but got {tuple(image.shape)}"
        )

    if image.size(1) != 3:
        raise ValueError(
            f"Expected image to have 3 channels, "
            f"but got {image.size(1)}"
        )

    if not image.is_floating_point():
        raise TypeError("image must be a floating point tensor")


def clamp_image(image):
    """
    Clamp image to valid [0, 1] range.
    """

    return torch.clamp(image, 0.0, 1.0)


# ============================================================
# Basic image helpers
# ============================================================

def rgb_to_grayscale(image):
    """
    Convert RGB image to grayscale.

    Input:
        image: (B, 3, H, W), range [0, 1]

    Output:
        gray: (B, 1, H, W)
    """

    check_image_tensor(image)

    red = image[:, 0:1]
    green = image[:, 1:2]
    blue = image[:, 2:3]

    gray = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    return gray


def sample_uniform_like_batch(
    batch_size,
    low,
    high,
    device,
    dtype
):
    """
    Sample one scalar per image in the batch.
    """

    values = torch.empty(
        batch_size,
        1,
        1,
        1,
        device=device,
        dtype=dtype
    )

    values.uniform_(low, high)

    return values


def apply_probability_mask(
    original,
    corrupted,
    probability
):
    """
    Apply a corruption per sample with probability p.

    original:
        original image, shape (B, C, H, W)

    corrupted:
        corrupted image, shape (B, C, H, W)

    probability:
        scalar in [0, 1]
    """

    if probability <= 0.0:
        return original

    if probability >= 1.0:
        return corrupted

    batch_size = original.size(0)

    mask = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=original.device,
        dtype=original.dtype
    )

    mask = (mask < probability).to(dtype=original.dtype)

    return original * (1.0 - mask) + corrupted * mask


# ============================================================
# Random resized crop
# ============================================================

def random_resized_crop_grid(
    batch_size,
    height,
    width,
    min_scale,
    max_scale,
    max_translate,
    device,
    dtype
):
    """
    Build an affine grid for random crop-resize.

    The affine matrix maps output coordinates to input coordinates.

    scale < 1.0 means zoom in / crop.
    translation shifts the crop center.
    """

    scale = torch.empty(
        batch_size,
        device=device,
        dtype=dtype
    )

    scale.uniform_(min_scale, max_scale)

    # A valid crop must remain inside the source image. For a crop with
    # normalized scale s, its centre can move by at most 1 - s.
    valid_translate_limit = torch.clamp(
        1.0 - scale,
        min=0.0
    )

    requested_translate_limit = torch.full_like(
        valid_translate_limit,
        fill_value=max_translate
    )

    translate_limit = torch.minimum(
        valid_translate_limit,
        requested_translate_limit
    )

    translate_x = torch.empty(
        batch_size,
        device=device,
        dtype=dtype
    ).uniform_(-1.0, 1.0) * translate_limit

    translate_y = torch.empty(
        batch_size,
        device=device,
        dtype=dtype
    ).uniform_(-1.0, 1.0) * translate_limit

    theta = torch.zeros(
        batch_size,
        2,
        3,
        device=device,
        dtype=dtype
    )

    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = translate_x
    theta[:, 1, 2] = translate_y

    grid = F.affine_grid(
        theta,
        size=(batch_size, 3, height, width),
        align_corners=False
    )

    return grid


def random_resized_crop(
    image,
    strength,
    min_scale_at_full_strength=0.80,
    max_translate_at_full_strength=0.15
):
    """
    Differentiable random resized crop using grid_sample.

    At strength = 0:
        almost identity crop.

    At strength = 1:
        crop scale may go down to min_scale_at_full_strength.
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    batch_size, _, height, width = image.shape

    min_scale = (
        1.0
        - strength * (1.0 - min_scale_at_full_strength)
    )

    max_scale = 1.0

    max_translate = strength * max_translate_at_full_strength

    grid = random_resized_crop_grid(
        batch_size=batch_size,
        height=height,
        width=width,
        min_scale=min_scale,
        max_scale=max_scale,
        max_translate=max_translate,
        device=image.device,
        dtype=image.dtype
    )

    cropped = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False
    )

    return clamp_image(cropped)


# ============================================================
# Horizontal flip
# ============================================================

def random_horizontal_flip(
    image,
    probability
):
    """
    Random horizontal flip per sample.
    """

    check_image_tensor(image)

    if probability <= 0.0:
        return image

    flipped = torch.flip(
        image,
        dims=[3]
    )

    return apply_probability_mask(
        original=image,
        corrupted=flipped,
        probability=probability
    )


# ============================================================
# Color jitter
# ============================================================

def brightness_jitter(
    image,
    strength,
    max_factor_delta=0.20
):
    """
    Apply per-image multiplicative brightness jitter.

    This follows torchvision ColorJitter brightness semantics more closely
    than adding an absolute pixel offset.
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    batch_size = image.size(0)

    factor = sample_uniform_like_batch(
        batch_size=batch_size,
        low=1.0 - max_factor_delta * strength,
        high=1.0 + max_factor_delta * strength,
        device=image.device,
        dtype=image.dtype
    )

    return clamp_image(image * factor)


def contrast_jitter(
    image,
    strength,
    max_factor_delta=0.30
):
    """
    Apply per-image contrast jitter around image mean.
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    batch_size = image.size(0)

    factor = sample_uniform_like_batch(
        batch_size=batch_size,
        low=1.0 - max_factor_delta * strength,
        high=1.0 + max_factor_delta * strength,
        device=image.device,
        dtype=image.dtype
    )

    mean = image.mean(
        dim=(2, 3),
        keepdim=True
    )

    jittered = (image - mean) * factor + mean

    return clamp_image(jittered)


def saturation_jitter(
    image,
    strength,
    max_factor_delta=0.40
):
    """
    Apply per-image saturation jitter.

    Saturation interpolation:
        output = gray + factor * (image - gray)
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    batch_size = image.size(0)

    factor = sample_uniform_like_batch(
        batch_size=batch_size,
        low=1.0 - max_factor_delta * strength,
        high=1.0 + max_factor_delta * strength,
        device=image.device,
        dtype=image.dtype
    )

    gray = rgb_to_grayscale(image)

    jittered = gray + factor * (image - gray)

    return clamp_image(jittered)


def random_grayscale_mix(
    image,
    strength,
    probability_at_full_strength=0.20,
    max_mix=1.0
):
    """
    Randomly mix image toward grayscale.

    This is softer than hard grayscale and remains differentiable.
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    batch_size = image.size(0)

    probability = probability_at_full_strength * strength

    mix_amount = sample_uniform_like_batch(
        batch_size=batch_size,
        low=0.0,
        high=max_mix * strength,
        device=image.device,
        dtype=image.dtype
    )

    gray = rgb_to_grayscale(image).repeat(1, 3, 1, 1)

    grayscaled = image * (1.0 - mix_amount) + gray * mix_amount

    grayscaled = clamp_image(grayscaled)

    return apply_probability_mask(
        original=image,
        corrupted=grayscaled,
        probability=probability
    )


# ============================================================
# Gaussian blur
# ============================================================

def gaussian_kernel_2d(
    kernel_size,
    sigma,
    channels,
    device,
    dtype
):
    """
    Create depthwise Gaussian blur kernel.
    """

    coordinates = torch.arange(
        kernel_size,
        device=device,
        dtype=dtype
    )

    coordinates = coordinates - kernel_size // 2

    gaussian_1d = torch.exp(
        -(coordinates ** 2) / (2.0 * sigma ** 2)
    )

    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]

    kernel = gaussian_2d.view(
        1,
        1,
        kernel_size,
        kernel_size
    )

    kernel = kernel.repeat(
        channels,
        1,
        1,
        1
    )

    return kernel


def gaussian_blur(
    image,
    strength,
    probability_at_full_strength=0.50,
    min_sigma=0.10,
    max_sigma=1.20
):
    """
    Apply differentiable Gaussian blur to a random subset of the batch.
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    probability = probability_at_full_strength * strength

    sigma = (
        min_sigma
        + strength * (max_sigma - min_sigma)
    )

    kernel_size = 5

    kernel = gaussian_kernel_2d(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=image.size(1),
        device=image.device,
        dtype=image.dtype
    )

    padded = F.pad(
        image,
        pad=(
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2
        ),
        mode="reflect"
    )

    blurred = F.conv2d(
        padded,
        kernel,
        padding=0,
        groups=image.size(1)
    )

    blurred = clamp_image(blurred)

    return apply_probability_mask(
        original=image,
        corrupted=blurred,
        probability=probability
    )


# ============================================================
# Gaussian noise
# ============================================================

def gaussian_noise(
    image,
    strength,
    max_std=0.03
):
    """
    Add Gaussian noise.
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    std = max_std * strength

    noise = torch.randn_like(image) * std

    noisy = image + noise

    return clamp_image(noisy)


# ============================================================
# JPEG-like differentiable compression approximation
# ============================================================

def jpeg_like_downsample_upsample(
    image,
    strength,
    probability_at_full_strength=0.40,
    min_scale_at_full_strength=0.50
):
    """
    JPEG-like spatial degradation.

    This is not real JPEG. It approximates compression artifacts using:
        downsample -> upsample

    It is differentiable and enough for trigger robustness training.
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    probability = probability_at_full_strength * strength

    _, _, height, width = image.shape

    scale = (
        1.0
        - strength * (1.0 - min_scale_at_full_strength)
    )

    small_height = max(
        int(round(height * scale)),
        4
    )

    small_width = max(
        int(round(width * scale)),
        4
    )

    small = F.interpolate(
        image,
        size=(small_height, small_width),
        mode="bilinear",
        align_corners=False
    )

    restored = F.interpolate(
        small,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    restored = clamp_image(restored)

    return apply_probability_mask(
        original=image,
        corrupted=restored,
        probability=probability
    )


def straight_through_round(x):
    """
    Straight-through rounding.

    Forward:
        round(x)

    Backward:
        identity gradient approximation.
    """

    rounded = torch.round(x)

    return x + (rounded - x).detach()


def soft_quantization(
    image,
    strength,
    probability_at_full_strength=0.40,
    min_levels_at_full_strength=32
):
    """
    Differentiable approximate quantization using straight-through estimator.

    At full strength:
        quantize image to around min_levels_at_full_strength levels.

    This encourages the hidden message to survive small precision loss.
    """

    check_image_tensor(image)

    if strength <= 0.0:
        return image

    probability = probability_at_full_strength * strength

    max_levels = 256
    min_levels = min_levels_at_full_strength

    levels = int(
        round(
            max_levels
            - strength * (max_levels - min_levels)
        )
    )

    levels = max(levels, min_levels)

    quantized = straight_through_round(
        image * (levels - 1)
    ) / (levels - 1)

    quantized = clamp_image(quantized)

    return apply_probability_mask(
        original=image,
        corrupted=quantized,
        probability=probability
    )


# ============================================================
# Corruption configuration
# ============================================================

@dataclass
class CorruptionConfig:
    """
    Configuration for differentiable image corruptions.

    The parameters describe their values at corruption strength 1.0.
    """

    enable_crop: bool = True
    enable_flip: bool = True

    enable_brightness: bool = True
    enable_contrast: bool = True
    enable_saturation: bool = True

    enable_grayscale: bool = True
    enable_blur: bool = True
    enable_noise: bool = True
    enable_jpeg_like: bool = True
    enable_quantization: bool = True

    flip_probability_at_full_strength: float = 0.50

    crop_min_scale_at_full_strength: float = 0.80
    crop_max_translate_at_full_strength: float = 0.15

    brightness_max_factor_delta: float = 0.15
    contrast_max_factor_delta: float = 0.25
    saturation_max_factor_delta: float = 0.35

    grayscale_probability_at_full_strength: float = 0.20

    blur_probability_at_full_strength: float = 0.40
    blur_max_sigma: float = 1.00

    noise_max_std: float = 0.02

    jpeg_probability_at_full_strength: float = 0.30
    jpeg_min_scale_at_full_strength: float = 0.60

    quantization_probability_at_full_strength: float = 0.30
    quantization_min_levels_at_full_strength: int = 48


CORRUPTION_PROFILES = (
    "combined",
    "crop",
    "flip",
    "brightness",
    "contrast",
    "saturation",
    "grayscale",
    "blur",
    "noise",
    "jpeg_like",
    "quantization",
    "geometry",
    "color",
    "compression"
)


def build_corruption_config(
    profile="combined"
):
    """
    Build a corruption configuration for combined training or ablation.
    """

    profile = str(profile).strip().lower()

    if profile not in CORRUPTION_PROFILES:
        raise ValueError(
            f"Unknown corruption profile: {profile}. "
            f"Expected one of {CORRUPTION_PROFILES}."
        )

    if profile == "combined":
        return CorruptionConfig()

    config = CorruptionConfig(
        enable_crop=False,
        enable_flip=False,
        enable_brightness=False,
        enable_contrast=False,
        enable_saturation=False,
        enable_grayscale=False,
        enable_blur=False,
        enable_noise=False,
        enable_jpeg_like=False,
        enable_quantization=False
    )

    if profile in ("crop", "geometry"):
        config.enable_crop = True

    if profile in ("flip", "geometry"):
        config.enable_flip = True

    if profile in ("brightness", "color"):
        config.enable_brightness = True

    if profile in ("contrast", "color"):
        config.enable_contrast = True

    if profile in ("saturation", "color"):
        config.enable_saturation = True

    if profile in ("grayscale", "color"):
        config.enable_grayscale = True

    if profile == "blur":
        config.enable_blur = True

    if profile == "noise":
        config.enable_noise = True

    if profile in ("jpeg_like", "compression"):
        config.enable_jpeg_like = True

    if profile in ("quantization", "compression"):
        config.enable_quantization = True

    return config

# ============================================================
# Main corruption layer
# ============================================================

class DifferentiableImageCorruption(nn.Module):
    """
    Differentiable corruption layer for training StegaStamp.

    The layer receives encoded images and applies a randomized corruption
    chain. The decoder then tries to recover the hidden message from the
    corrupted image.

    Args:
        image_size:
            Expected image size. CIFAR-10 uses 32.

        config:
            Optional CorruptionConfig.

    Forward:
        corrupted = corruption(image, strength)

    strength:
        0.0 means no corruption.
        1.0 means full configured corruption.
    """

    def __init__(
        self,
        image_size=32,
        config: Optional[CorruptionConfig] = None
    ):
        super().__init__()

        if image_size <= 0:
            raise ValueError("image_size must be greater than 0")

        self.image_size = image_size

        if config is None:
            config = CorruptionConfig()

        self.config = config

    def forward(
        self,
        image,
        strength=1.0
    ):
        check_image_tensor(image)

        if strength is None:
            strength = 1.0

        strength = float(strength)

        if strength <= 0.0:
            return image

        strength = min(strength, 1.0)

        corrupted = image

        if self.config.enable_crop:
            corrupted = random_resized_crop(
                image=corrupted,
                strength=strength,
                min_scale_at_full_strength=(
                    self.config.crop_min_scale_at_full_strength
                ),
                max_translate_at_full_strength=(
                    self.config.crop_max_translate_at_full_strength
                )
            )

        if self.config.enable_flip:
            corrupted = random_horizontal_flip(
                image=corrupted,
                probability=(
                    self.config.flip_probability_at_full_strength
                    * strength
                )
            )

        # This is required for individual ablation profiles
        if self.config.enable_brightness:
            corrupted = brightness_jitter(
                image=corrupted,
                strength=strength,
                max_factor_delta=(
                    self.config.brightness_max_factor_delta
                )
            )

        if self.config.enable_contrast:
            corrupted = contrast_jitter(
                image=corrupted,
                strength=strength,
                max_factor_delta=(
                    self.config.contrast_max_factor_delta
                )
            )

        if self.config.enable_saturation:
            corrupted = saturation_jitter(
                image=corrupted,
                strength=strength,
                max_factor_delta=(
                    self.config.saturation_max_factor_delta
                )
            )

        if self.config.enable_grayscale:
            corrupted = random_grayscale_mix(
                image=corrupted,
                strength=strength,
                probability_at_full_strength=(
                    self.config.grayscale_probability_at_full_strength
                )
            )

        if self.config.enable_blur:
            corrupted = gaussian_blur(
                image=corrupted,
                strength=strength,
                probability_at_full_strength=(
                    self.config.blur_probability_at_full_strength
                ),
                max_sigma=self.config.blur_max_sigma
            )

        if self.config.enable_noise:
            corrupted = gaussian_noise(
                image=corrupted,
                strength=strength,
                max_std=self.config.noise_max_std
            )

        if self.config.enable_jpeg_like:
            corrupted = jpeg_like_downsample_upsample(
                image=corrupted,
                strength=strength,
                probability_at_full_strength=(
                    self.config.jpeg_probability_at_full_strength
                ),
                min_scale_at_full_strength=(
                    self.config.jpeg_min_scale_at_full_strength
                )
            )

        if self.config.enable_quantization:
            corrupted = soft_quantization(
                image=corrupted,
                strength=strength,
                probability_at_full_strength=(
                    self.config.quantization_probability_at_full_strength
                ),
                min_levels_at_full_strength=(
                    self.config.quantization_min_levels_at_full_strength
                )
            )

        corrupted = clamp_image(corrupted)

        return corrupted


# ============================================================
# Corruption ramp helper
# ============================================================

def get_corruption_strength(
    epoch,
    warmup_epochs=10,
    ramp_epochs=40,
    initial_strength=0.0,
    max_strength=1.0
):
    """
    Return the corruption strength for a training epoch.

    During warmup, the strength remains at initial_strength. It then
    increases linearly to max_strength over ramp_epochs.
    """

    if epoch < 0:
        raise ValueError("epoch cannot be negative")

    if warmup_epochs < 0:
        raise ValueError("warmup_epochs cannot be negative")

    if ramp_epochs < 0:
        raise ValueError("ramp_epochs cannot be negative")

    if not 0.0 <= initial_strength <= 1.0:
        raise ValueError(
            "initial_strength must be in [0, 1]"
        )

    if not 0.0 <= max_strength <= 1.0:
        raise ValueError(
            "max_strength must be in [0, 1]"
        )

    if initial_strength > max_strength:
        raise ValueError(
            "initial_strength cannot be greater than max_strength"
        )

    if epoch <= warmup_epochs:
        return initial_strength

    if ramp_epochs == 0:
        return max_strength

    progress = (
        epoch - warmup_epochs
    ) / float(ramp_epochs)

    progress = max(
        0.0,
        min(progress, 1.0)
    )

    return (
        initial_strength
        + progress * (
            max_strength - initial_strength
        )
    )


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    images = torch.rand(
        8,
        3,
        32,
        32,
        device=device
    )

    corruption = DifferentiableImageCorruption(
        image_size=32
    ).to(device)

    for strength in [0.0, 0.25, 0.50, 1.0]:
        corrupted = corruption(
            images,
            strength=strength
        )

        print("Strength:", strength)
        print("Input shape:", images.shape)
        print("Corrupted shape:", corrupted.shape)
        print("Min:", corrupted.min().item())
        print("Max:", corrupted.max().item())
        print("Mean absolute change:", (corrupted - images).abs().mean().item())

        assert corrupted.shape == images.shape
        assert corrupted.min().item() >= 0.0
        assert corrupted.max().item() <= 1.0

    schedule_values = [
        get_corruption_strength(epoch)
        for epoch in [0, 5, 10, 20, 30, 50, 60]
    ]

    print("Schedule values:", schedule_values)

    print("Advanced stegastamp_like_corruptions.py sanity check passed.")