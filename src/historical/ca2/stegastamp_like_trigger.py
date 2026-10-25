import os
from typing import Optional, Tuple, Union

import torch
from torchvision.utils import save_image

from stegastamp_like_model import StegaStampCIFAR
from stegastamp_like_utils import (
    create_fixed_message,
    logits_to_bits,
    bit_accuracy,
    compute_psnr,
    compute_residual_stats
)


# ============================================================
# stegastamp_like_trigger.py
#
# Utility functions for applying the trained CIFAR-10
# StegaStamp / ISSBA-style invisible trigger.
#
# Important:
#   The StegaStamp encoder works in raw pixel space [0, 1].
#   Do NOT pass normalized images into apply_stegastamp_trigger().
#
# Flow:
#   clean image in [0, 1]
#   -> fixed target message
#   -> trained StegaStamp encoder
#   -> encoded / poisoned image
#
# The encoded image is used as:
#   1. poisoned image for supervised ISSBA-style training
#   2. triggered image for SSL / BadEncoder ASR evaluation
# ============================================================


# ============================================================
# Defaults
# ============================================================

DEFAULT_CHECKPOINT_PATH = "./results_stegastamp_learned/stegastamp_like_model.pth"

DEFAULT_MESSAGE_LENGTH = 30
DEFAULT_TARGET_LABEL = 0
DEFAULT_MAX_RESIDUAL = 8 / 255


# ============================================================
# Check helpers
# ============================================================

def check_single_image_tensor(image_tensor: torch.Tensor):
    """
    Check that image_tensor has shape:
        (3, H, W)

    Values should be in [0, 1].
    """

    if image_tensor.dim() != 3:
        raise ValueError(
            f"Expected image tensor shape (3, H, W), got {tuple(image_tensor.shape)}"
        )

    if image_tensor.size(0) != 3:
        raise ValueError(
            f"Expected 3 image channels, got {image_tensor.size(0)}"
        )


def check_batch_image_tensor(batch_tensor: torch.Tensor):
    """
    Check that batch_tensor has shape:
        (B, 3, H, W)

    Values should be in [0, 1].
    """

    if batch_tensor.dim() != 4:
        raise ValueError(
            f"Expected batch tensor shape (B, 3, H, W), got {tuple(batch_tensor.shape)}"
        )

    if batch_tensor.size(1) != 3:
        raise ValueError(
            f"Expected 3 image channels, got {batch_tensor.size(1)}"
        )


# ============================================================
# Load model
# ============================================================

def load_stegastamp_hider(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
    freeze: bool = True
) -> Tuple[StegaStampCIFAR, int, float]:
    """
    Load the trained StegaStampCIFAR model.

    The function name keeps backward compatibility with the earlier code,
    where the trigger generator was called a hider.

    Args:
        checkpoint_path:
            Path to trained StegaStamp checkpoint.

        device:
            CPU or CUDA device.

        freeze:
            If True, set requires_grad=False for all model parameters.

    Returns:
        model:
            Loaded StegaStampCIFAR model in eval mode.

        message_length:
            Message length used during training.

        max_residual:
            Residual bound used during training.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"StegaStamp checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    message_length = checkpoint.get("message_length", DEFAULT_MESSAGE_LENGTH)
    image_size = checkpoint.get("image_size", 32)
    message_map_size = checkpoint.get("message_map_size", 8)
    message_channels = checkpoint.get("message_channels", 3)
    base_channels = checkpoint.get("base_channels", 64)
    max_residual = checkpoint.get("max_residual", DEFAULT_MAX_RESIDUAL)

    model = StegaStampCIFAR(
        message_length=message_length,
        image_size=image_size,
        message_map_size=message_map_size,
        message_channels=message_channels,
        base_channels=base_channels,
        max_residual=max_residual,
        use_spatial_transformer=True,
        use_corruption=True
    ).to(device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, message_length, max_residual


# ============================================================
# Trigger generation
# ============================================================

@torch.no_grad()
def apply_stegastamp_trigger(
    image_tensor: torch.Tensor,
    model: StegaStampCIFAR,
    message_length: int = DEFAULT_MESSAGE_LENGTH,
    target_label: int = DEFAULT_TARGET_LABEL,
    device: Optional[torch.device] = None,
    return_residual: bool = False,
    return_cpu: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply the learned StegaStamp / ISSBA-style trigger to one image.

    Args:
        image_tensor:
            Tensor with shape (3, H, W), values in [0, 1].

        model:
            Loaded StegaStampCIFAR model.

        message_length:
            Hidden message length.

        target_label:
            Target class label encoded into the fixed message.

        device:
            Device used for inference.

        return_residual:
            If True, also return the residual trigger.

        return_cpu:
            If True, move output back to CPU.
            This is useful inside Dataset.__getitem__.

    Returns:
        encoded_image:
            Tensor with shape (3, H, W), values in [0, 1].

        residual, optional:
            Tensor with shape (3, H, W).
    """

    check_single_image_tensor(image_tensor)

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    image_batch = image_tensor.unsqueeze(0).to(device)

    message = create_fixed_message(
        batch_size=1,
        message_length=message_length,
        target_label=target_label,
        device=device
    )

    encoded_batch, residual_batch = model.encode(
        image=image_batch,
        message=message
    )

    encoded_image = encoded_batch.squeeze(0)
    residual = residual_batch.squeeze(0)

    if return_cpu:
        encoded_image = encoded_image.cpu()
        residual = residual.cpu()

    if return_residual:
        return encoded_image, residual

    return encoded_image


@torch.no_grad()
def apply_stegastamp_trigger_batch(
    batch_tensor: torch.Tensor,
    model: StegaStampCIFAR,
    message_length: int = DEFAULT_MESSAGE_LENGTH,
    target_label: int = DEFAULT_TARGET_LABEL,
    device: Optional[torch.device] = None,
    return_residual: bool = False,
    return_cpu: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply the learned StegaStamp / ISSBA-style trigger to a batch.

    Args:
        batch_tensor:
            Tensor with shape (B, 3, H, W), values in [0, 1].

        model:
            Loaded StegaStampCIFAR model.

        message_length:
            Hidden message length.

        target_label:
            Target class label encoded into the fixed message.

        device:
            Device used for inference.

        return_residual:
            If True, also return residuals.

        return_cpu:
            If True, move output back to CPU.

    Returns:
        encoded_batch:
            Tensor with shape (B, 3, H, W), values in [0, 1].

        residual_batch, optional:
            Tensor with shape (B, 3, H, W).
    """

    check_batch_image_tensor(batch_tensor)

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    batch_tensor = batch_tensor.to(device)
    batch_size = batch_tensor.size(0)

    message = create_fixed_message(
        batch_size=batch_size,
        message_length=message_length,
        target_label=target_label,
        device=device
    )

    encoded_batch, residual_batch = model.encode(
        image=batch_tensor,
        message=message
    )

    if return_cpu:
        encoded_batch = encoded_batch.cpu()
        residual_batch = residual_batch.cpu()

    if return_residual:
        return encoded_batch, residual_batch

    return encoded_batch


# ============================================================
# Decode helper
# ============================================================

@torch.no_grad()
def decode_stegastamp_message(
    encoded_batch: torch.Tensor,
    model: StegaStampCIFAR,
    device: Optional[torch.device] = None,
    return_bits: bool = True
):
    """
    Decode hidden message from encoded images.

    This is mostly for sanity checking that the encoder-decoder is working.
    It is not needed for ASR evaluation.
    """

    check_batch_image_tensor(encoded_batch)

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    encoded_batch = encoded_batch.to(device)

    decoded_logits = model.decode(encoded_batch)

    if return_bits:
        decoded_bits = logits_to_bits(decoded_logits)
        return decoded_bits.cpu()

    return decoded_logits.cpu()


# ============================================================
# Preview / debugging
# ============================================================

@torch.no_grad()
def save_trigger_preview(
    clean_batch: torch.Tensor,
    model: StegaStampCIFAR,
    save_path: str,
    message_length: int = DEFAULT_MESSAGE_LENGTH,
    target_label: int = DEFAULT_TARGET_LABEL,
    device: Optional[torch.device] = None,
    residual_amplification: float = 10.0,
    max_images: int = 8
):
    """
    Save a visualization showing:
        row 1: clean images
        row 2: encoded / poisoned images
        row 3: amplified residuals

    This is useful for checking whether the trigger is visually subtle.
    """

    check_batch_image_tensor(clean_batch)

    if device is None:
        device = next(model.parameters()).device

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    clean_batch = clean_batch[:max_images].to(device)

    encoded_batch, residual_batch = apply_stegastamp_trigger_batch(
        batch_tensor=clean_batch,
        model=model,
        message_length=message_length,
        target_label=target_label,
        device=device,
        return_residual=True,
        return_cpu=False
    )

    residual_vis = residual_batch * residual_amplification + 0.5
    residual_vis = torch.clamp(residual_vis, 0.0, 1.0)

    grid = torch.cat(
        [
            clean_batch.cpu(),
            encoded_batch.cpu(),
            residual_vis.cpu()
        ],
        dim=0
    )

    save_image(
        grid,
        save_path,
        nrow=clean_batch.size(0)
    )


@torch.no_grad()
def quick_trigger_stats(
    clean_batch: torch.Tensor,
    model: StegaStampCIFAR,
    message_length: int = DEFAULT_MESSAGE_LENGTH,
    target_label: int = DEFAULT_TARGET_LABEL,
    device: Optional[torch.device] = None
):
    """
    Compute quick sanity-check metrics for the trigger.

    Returns:
        PSNR, residual L1, residual L2, residual L-infinity, bit accuracy.
    """

    check_batch_image_tensor(clean_batch)

    if device is None:
        device = next(model.parameters()).device

    clean_batch = clean_batch.to(device)
    batch_size = clean_batch.size(0)

    message = create_fixed_message(
        batch_size=batch_size,
        message_length=message_length,
        target_label=target_label,
        device=device
    )

    encoded_batch, residual_batch = model.encode(
        image=clean_batch,
        message=message
    )

    decoded_logits = model.decode(encoded_batch)

    stats = compute_residual_stats(residual_batch)
    stats["psnr"] = compute_psnr(encoded_batch, clean_batch)
    stats["bit_accuracy"] = bit_accuracy(decoded_logits, message)

    return stats


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, message_length, max_residual = load_stegastamp_hider(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        device=device
    )

    print("Loaded StegaStamp model")
    print("Message length:", message_length)
    print("Max residual:", max_residual)

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    clean_batch, _ = next(iter(loader))

    stats = quick_trigger_stats(
        clean_batch=clean_batch,
        model=model,
        message_length=message_length,
        target_label=0,
        device=device
    )

    print("Trigger stats:", stats)

    save_trigger_preview(
        clean_batch=clean_batch,
        model=model,
        save_path="./figures/stegastamp_trigger_preview.png",
        message_length=message_length,
        target_label=0,
        device=device
    )

    print("Saved preview to ./figures/stegastamp_trigger_preview.png")