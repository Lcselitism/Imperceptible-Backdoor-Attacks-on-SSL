import json
import math
import os
import random
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ============================================================
# stegastamp_like_utils.py
#
# Utility functions for the CIFAR-10 StegaStamp / ISSBA-style
# encoder-decoder experiment.
#
# This file contains:
#   1. Reproducibility helpers
#   2. Message creation helpers
#   3. Bit recovery metrics
#   4. Image stealth metrics
#   5. JSON saving/loading helpers
#   6. Training schedule helpers
# ============================================================


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 123):
    """
    Set random seeds for reproducibility.

    This helps keep training behavior more consistent across runs.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These two settings make CUDA behavior more reproducible.
    # They may reduce speed slightly, but are helpful for experiments.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# Message helpers
# ============================================================

def create_fixed_message(
    batch_size: int,
    message_length: int = 30,
    target_label: int = 0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create a fixed binary message for the target class.

    This is used during backdoor trigger generation.

    In ISSBA-style supervised poisoning, each poisoned image contains
    an invisible encoded message/string associated with the target label.
    The classifier then learns the mapping:

        hidden message/string -> target label

    For CIFAR-10, 4 bits are enough to represent labels 0 to 9.
    The remaining bits are filled with a deterministic signature so the
    message is not simply all zeros.

    Args:
        batch_size:
            Number of messages to create.

        message_length:
            Length of binary message.

        target_label:
            Target class label.

        device:
            Device where the message tensor should be created.

    Returns:
        Tensor of shape (B, message_length), values 0 or 1.
    """

    if device is None:
        device = torch.device("cpu")

    message = torch.zeros(
        batch_size,
        message_length,
        dtype=torch.float32,
        device=device
    )

    label = int(target_label)

    # Encode target label in the first 4 bits.
    for bit_idx in range(min(4, message_length)):
        bit_value = (label >> bit_idx) & 1
        message[:, bit_idx] = float(bit_value)

    # Add a fixed signature in the remaining bits.
    for bit_idx in range(4, message_length):
        bit_value = (bit_idx + target_label) % 2
        message[:, bit_idx] = float(bit_value)

    return message


def create_random_message(
    batch_size: int,
    message_length: int = 30,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create random binary messages.

    This is used during encoder-decoder training.

    The trigger generator should not only learn one fixed message.
    It should learn the general ability to hide and recover binary strings.
    Later, during backdoor generation, we use create_fixed_message().
    """

    if device is None:
        device = torch.device("cpu")

    message = torch.randint(
        low=0,
        high=2,
        size=(batch_size, message_length),
        dtype=torch.float32,
        device=device
    )

    return message


# ============================================================
# Message decoding metrics
# ============================================================

@torch.no_grad()
def bit_accuracy(
    decoded_logits: torch.Tensor,
    target_message: torch.Tensor
) -> float:
    """
    Compute bit-level message recovery accuracy.

    Args:
        decoded_logits:
            Raw decoder outputs of shape (B, message_length).

        target_message:
            Ground-truth binary message of shape (B, message_length).

    Returns:
        Bit accuracy as a Python float in [0, 1].
    """

    predicted_bits = logits_to_bits(decoded_logits)

    correct = (predicted_bits == target_message).float().mean().item()

    return correct


@torch.no_grad()
def message_accuracy(
    decoded_logits: torch.Tensor,
    target_message: torch.Tensor
) -> float:
    """
    Compute full-message accuracy.

    A message is counted as correct only if every bit is recovered correctly.

    This is stricter than bit accuracy. It may be low even when bit accuracy
    is high, especially for long messages.
    """

    predicted_bits = logits_to_bits(decoded_logits)

    correct_per_sample = (predicted_bits == target_message).all(dim=1)
    acc = correct_per_sample.float().mean().item()

    return acc


@torch.no_grad()
def logits_to_bits(decoded_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert decoder logits into binary predictions.

    BCEWithLogitsLoss is used during training, so the decoder output is not
    passed through sigmoid inside the model.
    """

    probabilities = torch.sigmoid(decoded_logits)
    predicted_bits = (probabilities > 0.5).float()

    return predicted_bits


# ============================================================
# Image stealth metrics
# ============================================================

@torch.no_grad()
def compute_psnr(
    encoded_image: torch.Tensor,
    clean_image: torch.Tensor,
    eps: float = 1e-8
) -> float:
    """
    Compute PSNR between encoded image and clean image.

    Assumes both tensors are in [0, 1].

    Higher PSNR means smaller visual distortion.
    """

    mse = F.mse_loss(encoded_image, clean_image).item()

    psnr = 10.0 * math.log10(1.0 / (mse + eps))

    return psnr


@torch.no_grad()
def compute_batch_psnr(
    encoded_image: torch.Tensor,
    clean_image: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute PSNR for each image in a batch.

    Returns:
        Tensor of shape (B,).
    """

    mse_per_sample = torch.mean(
        (encoded_image - clean_image) ** 2,
        dim=(1, 2, 3)
    )

    psnr_per_sample = 10.0 * torch.log10(1.0 / (mse_per_sample + eps))

    return psnr_per_sample


@torch.no_grad()
def compute_residual_stats(
    residual: torch.Tensor
) -> Dict[str, float]:
    """
    Compute useful residual statistics.

    residual:
        Tensor of shape (B, 3, H, W).

    Returns:
        Dictionary containing average L1, L2, and L-infinity residual.
    """

    abs_residual = residual.abs()

    stats = {
        "residual_l1": abs_residual.mean().item(),
        "residual_l2": torch.mean(residual ** 2).item(),
        "residual_linf": abs_residual.max().item()
    }

    return stats


@torch.no_grad()
def compute_image_difference_stats(
    encoded_image: torch.Tensor,
    clean_image: torch.Tensor
) -> Dict[str, float]:
    """
    Compute difference statistics between encoded and clean images.

    This is useful when residual is not directly available.
    """

    diff = encoded_image - clean_image

    return compute_residual_stats(diff)


# ============================================================
# Normalization helpers
# ============================================================

def get_cifar10_normalization():
    """
    Return CIFAR-10 mean and std.

    The StegaStamp encoder-decoder itself works in [0, 1] pixel space.
    Normalization should only be applied before classifier or SimCLR encoder
    input, not before the trigger generator.
    """

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    return mean, std


def normalize_cifar10_tensor(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor using CIFAR-10 mean and std.

    Input:
        image tensor in [0, 1], shape (B, 3, H, W).

    Output:
        normalized image tensor.
    """

    mean, std = get_cifar10_normalization()

    mean_tensor = torch.tensor(
        mean,
        dtype=image.dtype,
        device=image.device
    ).view(1, 3, 1, 1)

    std_tensor = torch.tensor(
        std,
        dtype=image.dtype,
        device=image.device
    ).view(1, 3, 1, 1)

    return (image - mean_tensor) / std_tensor


def denormalize_cifar10_tensor(image: torch.Tensor) -> torch.Tensor:
    """
    Reverse CIFAR-10 normalization.

    This is useful for visualizing classifier or encoder inputs.
    """

    mean, std = get_cifar10_normalization()

    mean_tensor = torch.tensor(
        mean,
        dtype=image.dtype,
        device=image.device
    ).view(1, 3, 1, 1)

    std_tensor = torch.tensor(
        std,
        dtype=image.dtype,
        device=image.device
    ).view(1, 3, 1, 1)

    out = image * std_tensor + mean_tensor
    out = torch.clamp(out, 0.0, 1.0)

    return out


# ============================================================
# Training schedule helpers
# ============================================================

def ramp_value(
    current_epoch: int,
    start_epoch: int,
    end_epoch: int,
    start_value: float,
    end_value: float
) -> float:
    """
    Linearly ramp a value from start_value to end_value.

    Example:
        corruption_strength can start at 0.0 and slowly increase to 1.0.
    """

    if current_epoch <= start_epoch:
        return start_value

    if current_epoch >= end_epoch:
        return end_value

    progress = (current_epoch - start_epoch) / max(1, end_epoch - start_epoch)

    value = start_value + progress * (end_value - start_value)

    return float(value)


def get_corruption_strength(
    epoch: int,
    warmup_epochs: int = 5,
    ramp_epochs: int = 20,
    max_strength: float = 1.0
) -> float:
    """
    Get corruption strength for the current epoch.

    StegaStamp does not apply full perturbation strength immediately.
    The decoder first needs to learn basic message recovery. Then corruption
    strength is gradually increased.

    Args:
        epoch:
            Current epoch, starting from 1.

        warmup_epochs:
            Number of epochs with corruption strength 0.

        ramp_epochs:
            Number of epochs used to ramp from 0 to max_strength.

        max_strength:
            Final corruption strength.

    Returns:
        Corruption strength in [0, max_strength].
    """

    start_epoch = warmup_epochs
    end_epoch = warmup_epochs + ramp_epochs

    return ramp_value(
        current_epoch=epoch,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        start_value=0.0,
        end_value=max_strength
    )


def get_image_loss_weight(
    epoch: int,
    warmup_epochs: int = 3,
    ramp_epochs: int = 10,
    max_weight: float = 2.0
) -> float:
    """
    Get image reconstruction loss weight for the current epoch.

    In StegaStamp-style training, it is often useful to let the decoder learn
    message recovery first, then gradually increase image-quality penalties.
    """

    start_epoch = warmup_epochs
    end_epoch = warmup_epochs + ramp_epochs

    return ramp_value(
        current_epoch=epoch,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        start_value=0.0,
        end_value=max_weight
    )


# ============================================================
# JSON helpers
# ============================================================

def save_json(data: Dict, path: str):
    """
    Save dictionary as a JSON file.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> Dict:
    """
    Load dictionary from a JSON file.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return data


# ============================================================
# Model helper
# ============================================================

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable model parameters.
    """

    total = 0

    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()

    return total


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    message = create_fixed_message(
        batch_size=4,
        message_length=30,
        target_label=0,
        device=device
    )

    random_message = create_random_message(
        batch_size=4,
        message_length=30,
        device=device
    )

    fake_logits = torch.randn(4, 30, device=device)

    print("Fixed message shape:", message.shape)
    print("Fixed message first row:", message[0])
    print("Random message shape:", random_message.shape)
    print("Bit accuracy:", bit_accuracy(fake_logits, random_message))
    print("Message accuracy:", message_accuracy(fake_logits, random_message))
    print("Corruption strength epoch 1:", get_corruption_strength(epoch=1))
    print("Corruption strength epoch 30:", get_corruption_strength(epoch=30))