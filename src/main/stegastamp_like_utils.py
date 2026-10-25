import os
import csv
import json
import time
import random
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# stegastamp_like_utils.py
#
# Utility functions for Advanced CA3 StegaStamp training.
#
# Used by:
#   - stegastamp_like_model.py
#   - train_stegastamp_like.py
#   - later CA3 evaluation scripts
#
# Includes:
#   - random seed helpers
#   - message creation helpers
#   - bit/message accuracy
#   - PSNR / residual metrics
#   - checkpoint helpers
#   - logging helpers
#   - visualization helpers
# ============================================================


# ============================================================
# Constants
# ============================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# STL-10 normalization used by the new 32x32 STL-10 SimCLR upstream branch.
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


# ============================================================
# Basic helpers
# ============================================================

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """
    Return CUDA device if available, else CPU.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """

    if path is None or path == "":
        return

    os.makedirs(path, exist_ok=True)


def get_current_time_string() -> str:
    """
    Return a readable timestamp string.
    """

    return time.strftime("%Y%m%d_%H%M%S")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(model: torch.nn.Module) -> int:
    """
    Count all model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# Tensor checks
# ============================================================

def check_image_tensor(image: torch.Tensor, name: str = "image") -> None:
    """
    Check image tensor shape:
        (B, 3, H, W)
    """

    if not torch.is_tensor(image):
        raise TypeError(f"{name} must be a torch.Tensor")

    if image.dim() != 4:
        raise ValueError(
            f"Expected {name} shape (B, 3, H, W), "
            f"but got {tuple(image.shape)}"
        )

    if image.size(1) != 3:
        raise ValueError(
            f"Expected {name} to have 3 channels, "
            f"but got {image.size(1)}"
        )

    if not image.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")


def check_message_tensor(
    message: torch.Tensor,
    message_length: int,
    name: str = "message"
) -> None:
    """
    Check message tensor shape:
        (B, message_length)
    """

    if not torch.is_tensor(message):
        raise TypeError(f"{name} must be a torch.Tensor")

    if message.dim() != 2:
        raise ValueError(
            f"Expected {name} shape (B, {message_length}), "
            f"but got {tuple(message.shape)}"
        )

    if message.size(1) != message_length:
        raise ValueError(
            f"Expected {name} length {message_length}, "
            f"but got {message.size(1)}"
        )

    if not message.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")


def clamp_image(image: torch.Tensor) -> torch.Tensor:
    """
    Clamp image to [0, 1].
    """

    return torch.clamp(image, 0.0, 1.0)


# ============================================================
# Message helpers
# ============================================================

def create_random_message(
    batch_size: int,
    message_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create random binary messages.

    Output:
        shape (B, message_length)
        values in {0, 1}
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if message_length <= 0:
        raise ValueError("message_length must be greater than 0")

    message = torch.randint(
        low=0,
        high=2,
        size=(batch_size, message_length),
        device=device
    )

    return message.to(dtype=dtype)


def create_fixed_message(
    message_length: int,
    device: torch.device,
    pattern: str = "alternating",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create one fixed target message.

    Output:
        shape (1, message_length)

    Patterns:
        alternating:
            0, 1, 0, 1, ...

        inverse_alternating:
            1, 0, 1, 0, ...

        ones:
            all 1

        zeros:
            all 0

        half:
            first half 1, second half 0
    """

    if message_length <= 0:
        raise ValueError("message_length must be greater than 0")

    pattern = pattern.lower()

    if pattern == "alternating":
        values = [
            index % 2
            for index in range(message_length)
        ]

    elif pattern == "inverse_alternating":
        values = [
            1 - (index % 2)
            for index in range(message_length)
        ]

    elif pattern == "ones":
        values = [1 for _ in range(message_length)]

    elif pattern == "zeros":
        values = [0 for _ in range(message_length)]

    elif pattern == "half":
        split = message_length // 2
        values = [
            1 if index < split else 0
            for index in range(message_length)
        ]

    else:
        raise ValueError(
            "Unknown fixed message pattern. "
            "Use: alternating, inverse_alternating, ones, zeros, half."
        )

    message = torch.tensor(
        values,
        device=device,
        dtype=dtype
    ).view(1, message_length)

    return message


def repeat_message_to_batch(
    message: torch.Tensor,
    batch_size: int
) -> torch.Tensor:
    """
    Repeat a single message to match a batch size.

    Input:
        message shape:
            (1, message_length) or (B, message_length)

    Output:
        shape:
            (batch_size, message_length)
    """

    if message.dim() != 2:
        raise ValueError(
            f"Expected message shape (B, message_length), "
            f"but got {tuple(message.shape)}"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if message.size(0) == batch_size:
        return message

    if message.size(0) != 1:
        raise ValueError(
            "Can only repeat message if message batch size is 1. "
            f"Got message batch size {message.size(0)}."
        )

    return message.repeat(batch_size, 1)


def logits_to_bits(decoded_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert decoder logits to binary bits using threshold 0.5.
    """

    if decoded_logits.dim() != 2:
        raise ValueError(
            f"Expected decoded_logits shape (B, message_length), "
            f"but got {tuple(decoded_logits.shape)}"
        )

    probabilities = torch.sigmoid(decoded_logits)

    bits = (probabilities >= 0.5).float()

    return bits


@torch.no_grad()
def bit_accuracy(
    decoded_logits: torch.Tensor,
    message: torch.Tensor
) -> float:
    """
    Compute bit-level accuracy.

    Example:
        If 15 out of 16 bits are correct, bit accuracy is 93.75%.
    """

    if decoded_logits.shape != message.shape:
        raise ValueError(
            "decoded_logits and message must have same shape. "
            f"Got {tuple(decoded_logits.shape)} and {tuple(message.shape)}."
        )

    predicted_bits = logits_to_bits(decoded_logits)

    accuracy = (
        predicted_bits == message
    ).float().mean()

    return float(accuracy.detach().cpu())


@torch.no_grad()
def message_accuracy(
    decoded_logits: torch.Tensor,
    message: torch.Tensor
) -> float:
    """
    Compute full-message accuracy.

    A sample is counted correct only if every bit is correct.
    """

    if decoded_logits.shape != message.shape:
        raise ValueError(
            "decoded_logits and message must have same shape. "
            f"Got {tuple(decoded_logits.shape)} and {tuple(message.shape)}."
        )

    predicted_bits = logits_to_bits(decoded_logits)

    correct_bits_per_sample = (
        predicted_bits == message
    ).float().mean(dim=1)

    correct_message = (
        correct_bits_per_sample == 1.0
    ).float().mean()

    return float(correct_message.detach().cpu())


# ============================================================
# Image metrics
# ============================================================

@torch.no_grad()
def compute_mse(
    image_a: torch.Tensor,
    image_b: torch.Tensor
) -> float:
    """
    Compute mean squared error between two images.
    """

    check_image_tensor(image_a, "image_a")
    check_image_tensor(image_b, "image_b")

    if image_a.shape != image_b.shape:
        raise ValueError(
            "image_a and image_b must have same shape. "
            f"Got {tuple(image_a.shape)} and {tuple(image_b.shape)}."
        )

    mse = F.mse_loss(image_a, image_b)

    return float(mse.detach().cpu())


@torch.no_grad()
def compute_l1(
    image_a: torch.Tensor,
    image_b: torch.Tensor
) -> float:
    """
    Compute mean absolute error between two images.
    """

    check_image_tensor(image_a, "image_a")
    check_image_tensor(image_b, "image_b")

    if image_a.shape != image_b.shape:
        raise ValueError(
            "image_a and image_b must have same shape. "
            f"Got {tuple(image_a.shape)} and {tuple(image_b.shape)}."
        )

    value = F.l1_loss(image_a, image_b)

    return float(value.detach().cpu())


@torch.no_grad()
def compute_psnr(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    eps: float = 1e-8
) -> float:
    """
    Compute PSNR between two images in [0, 1].

    Higher is better.
    """

    mse = compute_mse(image_a, image_b)

    mse = max(mse, eps)

    psnr = 10.0 * torch.log10(
        torch.tensor(1.0 / mse)
    )

    return float(psnr.detach().cpu())


@torch.no_grad()
def compute_residual_stats(
    residual: torch.Tensor
) -> Dict[str, float]:
    """
    Compute useful residual statistics.

    residual should usually be:
        encoded_image - clean_image
    """

    if not torch.is_tensor(residual):
        raise TypeError("residual must be a torch.Tensor")

    if residual.dim() != 4:
        raise ValueError(
            f"Expected residual shape (B, C, H, W), "
            f"but got {tuple(residual.shape)}"
        )

    residual_detached = residual.detach()

    abs_residual = residual_detached.abs()

    stats = {
        "min": float(residual_detached.min().cpu()),
        "max": float(residual_detached.max().cpu()),
        "mean": float(residual_detached.mean().cpu()),
        "std": float(residual_detached.std().cpu()),

        "mean_abs": float(abs_residual.mean().cpu()),
        "max_abs": float(abs_residual.max().cpu()),

        "mean_abs_255": float((abs_residual * 255.0).mean().cpu()),
        "max_abs_255": float((abs_residual * 255.0).max().cpu()),

        "l2_mean": float(torch.sqrt((residual_detached ** 2).mean()).cpu()),
    }

    return stats


@torch.no_grad()
def compute_trigger_quality_metrics(
    clean_image: torch.Tensor,
    encoded_image: torch.Tensor,
    decoded_logits: Optional[torch.Tensor] = None,
    message: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute common trigger quality metrics.

    Returns:
        PSNR, L1, MSE, residual stats, and optional bit/message accuracy.
    """

    check_image_tensor(clean_image, "clean_image")
    check_image_tensor(encoded_image, "encoded_image")

    if clean_image.shape != encoded_image.shape:
        raise ValueError(
            "clean_image and encoded_image must have same shape. "
            f"Got {tuple(clean_image.shape)} and {tuple(encoded_image.shape)}."
        )

    residual = encoded_image - clean_image

    metrics = {
        "psnr": compute_psnr(encoded_image, clean_image),
        "mse": compute_mse(encoded_image, clean_image),
        "l1": compute_l1(encoded_image, clean_image),
    }

    residual_stats = compute_residual_stats(residual)

    for key, value in residual_stats.items():
        metrics[f"residual_{key}"] = value

    if decoded_logits is not None and message is not None:
        metrics["bit_accuracy"] = bit_accuracy(
            decoded_logits,
            message
        )

        metrics["message_accuracy"] = message_accuracy(
            decoded_logits,
            message
        )

    return metrics


# ============================================================
# Normalization helpers
# ============================================================

def get_cifar10_mean_std(
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return CIFAR-10 mean/std tensors shaped for broadcasting.
    """

    mean = torch.tensor(
        CIFAR10_MEAN,
        device=device,
        dtype=dtype
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        CIFAR10_STD,
        device=device,
        dtype=dtype
    ).view(1, 3, 1, 1)

    return mean, std


def normalize_cifar10(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize [0, 1] images using CIFAR-10 mean/std.

    Use this before feeding images into a CIFAR-trained encoder.
    """

    check_image_tensor(image)

    mean, std = get_cifar10_mean_std(
        device=image.device,
        dtype=image.dtype
    )

    return (image - mean) / std


def denormalize_cifar10(image: torch.Tensor) -> torch.Tensor:
    """
    Undo CIFAR-10 normalization.
    """

    check_image_tensor(image)

    mean, std = get_cifar10_mean_std(
        device=image.device,
        dtype=image.dtype
    )

    return image * std + mean


def get_stl10_mean_std(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return STL-10 mean/std tensors shaped for BCHW broadcasting."""
    mean = torch.tensor(
        STL10_MEAN, device=device, dtype=dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        STL10_STD, device=device, dtype=dtype
    ).view(1, 3, 1, 1)
    return mean, std


def normalize_stl10(image: torch.Tensor) -> torch.Tensor:
    """Normalize raw [0,1] 32x32 images for the STL-10-pretrained encoder."""
    check_image_tensor(image)
    mean, std = get_stl10_mean_std(image.device, image.dtype)
    return (image - mean) / std


def denormalize_stl10(image: torch.Tensor) -> torch.Tensor:
    """Undo STL-10 normalization."""
    check_image_tensor(image)
    mean, std = get_stl10_mean_std(image.device, image.dtype)
    return image * std + mean


def normalize_for_upstream(
    image: torch.Tensor,
    upstream_dataset: str,
) -> torch.Tensor:
    """Normalize according to the dataset used to pretrain the frozen encoder."""
    name = str(upstream_dataset).lower()
    if name == "cifar10":
        # Preserve the existing CA3 CIFAR-10 behaviour exactly.
        return normalize_cifar10(image)
    if name == "stl10":
        return normalize_stl10(image)
    raise ValueError(
        f"Unsupported upstream dataset {upstream_dataset!r}; expected cifar10 or stl10"
    )


# ============================================================
# Metric tracking
# ============================================================

class AverageMeter:
    """
    Track average value of one metric.
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.average = 0.0

    def update(self, value: float, n: int = 1) -> None:
        value = float(value)

        self.value = value
        self.sum += value * n
        self.count += n

        if self.count > 0:
            self.average = self.sum / self.count
        else:
            self.average = 0.0


class MetricTracker:
    """
    Track many metrics at once.
    """

    def __init__(self):
        self.meters: Dict[str, AverageMeter] = {}

    def update(
        self,
        metrics: Dict[str, Any],
        n: int = 1
    ) -> None:
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = float(value.detach().cpu())

            if not isinstance(value, (int, float)):
                continue

            if key not in self.meters:
                self.meters[key] = AverageMeter(key)

            self.meters[key].update(value, n=n)

    def averages(self) -> Dict[str, float]:
        return {
            key: meter.average
            for key, meter in self.meters.items()
        }

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()


def format_metrics(
    metrics: Dict[str, Any],
    keys: Optional[List[str]] = None,
    precision: int = 4
) -> str:
    """
    Convert metric dictionary into a readable string.
    """

    if keys is None:
        keys = list(metrics.keys())

    parts = []

    for key in keys:
        if key not in metrics:
            continue

        value = metrics[key]

        if torch.is_tensor(value):
            value = float(value.detach().cpu())

        if isinstance(value, float):
            parts.append(f"{key}: {value:.{precision}f}")
        else:
            parts.append(f"{key}: {value}")

    return " | ".join(parts)


def print_metrics(
    prefix: str,
    metrics: Dict[str, Any],
    keys: Optional[List[str]] = None,
    precision: int = 4
) -> None:
    """
    Print formatted metrics.
    """

    metric_string = format_metrics(
        metrics=metrics,
        keys=keys,
        precision=precision
    )

    print(f"{prefix} {metric_string}")


# ============================================================
# CSV / JSON logging
# ============================================================

def append_metrics_to_csv(
    csv_path: str,
    metrics: Dict[str, Any]
) -> None:
    """
    Append one row of metrics to a CSV file.
    """

    ensure_dir(os.path.dirname(csv_path))

    clean_metrics = {}

    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = float(value.detach().cpu())

        clean_metrics[key] = value

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(clean_metrics.keys())
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(clean_metrics)


def save_json(
    data: Dict[str, Any],
    json_path: str
) -> None:
    """
    Save dictionary to JSON.
    """

    ensure_dir(os.path.dirname(json_path))

    clean_data = {}

    for key, value in data.items():
        if torch.is_tensor(value):
            value = float(value.detach().cpu())

        clean_data[key] = value

    with open(json_path, "w") as json_file:
        json.dump(clean_data, json_file, indent=4)


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint.
    """

    ensure_dir(os.path.dirname(path))

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if best_metric is not None:
        checkpoint["best_metric"] = best_metric

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load checkpoint.

    If model/optimizer/scheduler are provided, their states are loaded.
    The full checkpoint dictionary is returned.
    """

    checkpoint = torch.load(
        path,
        map_location=map_location
    )

    if model is not None:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict
        )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint


# ============================================================
# Visualization helpers
# ============================================================

def tensor_to_display_image(image: torch.Tensor) -> torch.Tensor:
    """
    Prepare image tensor for display/saving.

    Input:
        image shape:
            (B, 3, H, W) or (3, H, W)

    Output:
        clamped tensor in [0, 1].
    """

    if image.dim() == 3:
        image = image.unsqueeze(0)

    check_image_tensor(image)

    return clamp_image(image)


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8
) -> None:
    """
    Save tensor image grid.

    Requires torchvision.
    """

    ensure_dir(os.path.dirname(path))

    try:
        from torchvision.utils import save_image

        images = tensor_to_display_image(images)

        save_image(
            images,
            path,
            nrow=nrow
        )

    except ImportError as error:
        raise ImportError(
            "save_image_grid requires torchvision. "
            "Install torchvision or skip image saving."
        ) from error


def make_residual_visualization(
    residual: torch.Tensor,
    scale: float = 10.0
) -> torch.Tensor:
    """
    Convert residual to visible image.

    residual is centered around 0, so we show:
        residual * scale + 0.5
    """

    if residual.dim() != 4:
        raise ValueError(
            f"Expected residual shape (B, C, H, W), "
            f"but got {tuple(residual.shape)}"
        )

    residual_visual = residual * scale + 0.5

    return clamp_image(residual_visual)


def make_abs_residual_visualization(
    residual: torch.Tensor,
    scale: float = 20.0
) -> torch.Tensor:
    """
    Convert absolute residual to visible image.
    """

    if residual.dim() != 4:
        raise ValueError(
            f"Expected residual shape (B, C, H, W), "
            f"but got {tuple(residual.shape)}"
        )

    residual_visual = residual.abs() * scale

    return clamp_image(residual_visual)


def make_mask_visualization(
    residual_mask: torch.Tensor
) -> torch.Tensor:
    """
    Convert residual mask to RGB image for saving.

    Input:
        residual_mask shape:
            (B, 1, H, W)

    Output:
        shape:
            (B, 3, H, W)
    """

    if residual_mask.dim() != 4:
        raise ValueError(
            f"Expected residual_mask shape (B, 1, H, W), "
            f"but got {tuple(residual_mask.shape)}"
        )

    if residual_mask.size(1) != 1:
        raise ValueError(
            f"Expected residual_mask to have 1 channel, "
            f"but got {residual_mask.size(1)}"
        )

    return residual_mask.repeat(1, 3, 1, 1).clamp(0.0, 1.0)


def save_trigger_visualization(
    clean_image: torch.Tensor,
    encoded_image: torch.Tensor,
    residual: Optional[torch.Tensor],
    path: str,
    residual_mask: Optional[torch.Tensor] = None,
    max_items: int = 8,
    residual_scale: float = 10.0,
    abs_residual_scale: float = 20.0
) -> None:
    """
    Save a visual comparison grid.

    Rows included:
        clean image
        encoded image
        residual visualization
        absolute residual visualization
        optional residual mask
    """

    check_image_tensor(clean_image, "clean_image")
    check_image_tensor(encoded_image, "encoded_image")

    if clean_image.shape != encoded_image.shape:
        raise ValueError(
            "clean_image and encoded_image must have same shape. "
            f"Got {tuple(clean_image.shape)} and {tuple(encoded_image.shape)}."
        )

    max_items = min(max_items, clean_image.size(0))

    clean = clean_image[:max_items].detach().cpu()
    encoded = encoded_image[:max_items].detach().cpu()

    if residual is None:
        residual = encoded_image - clean_image

    residual = residual[:max_items].detach().cpu()

    residual_visual = make_residual_visualization(
        residual,
        scale=residual_scale
    )

    abs_residual_visual = make_abs_residual_visualization(
        residual,
        scale=abs_residual_scale
    )

    rows = [
        clean,
        encoded,
        residual_visual,
        abs_residual_visual
    ]

    if residual_mask is not None:
        mask = residual_mask[:max_items].detach().cpu()
        mask_visual = make_mask_visualization(mask)
        rows.append(mask_visual)

    grid_images = torch.cat(rows, dim=0)

    save_image_grid(
        images=grid_images,
        path=path,
        nrow=max_items
    )


# ============================================================
# Batch movement helper
# ============================================================

def move_to_device(
    batch: Any,
    device: torch.device
) -> Any:
    """
    Move tensors inside nested batch to device.
    """

    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, dict):
        return {
            key: move_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, list):
        return [
            move_to_device(value, device)
            for value in batch
        ]

    if isinstance(batch, tuple):
        return tuple(
            move_to_device(value, device)
            for value in batch
        )

    return batch


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == "__main__":
    set_random_seed(42)

    device = get_device()

    batch_size = 4
    message_length = 16

    images = torch.rand(
        batch_size,
        3,
        32,
        32,
        device=device
    )

    residual = torch.randn_like(images) * (2.0 / 255.0)

    encoded = clamp_image(images + residual)

    true_residual = encoded - images

    messages = create_random_message(
        batch_size=batch_size,
        message_length=message_length,
        device=device
    )

    logits = torch.randn(
        batch_size,
        message_length,
        device=device
    )

    fixed_message = create_fixed_message(
        message_length=message_length,
        device=device,
        pattern="alternating"
    )

    repeated_message = repeat_message_to_batch(
        fixed_message,
        batch_size=batch_size
    )

    print("Random messages:", messages.shape)
    print("Fixed message:", fixed_message.shape)
    print("Repeated message:", repeated_message.shape)

    print("Bit accuracy:", bit_accuracy(logits, messages))
    print("Message accuracy:", message_accuracy(logits, messages))

    print("PSNR:", compute_psnr(encoded, images))
    print("MSE:", compute_mse(encoded, images))
    print("L1:", compute_l1(encoded, images))

    print("Residual stats:", compute_residual_stats(true_residual))

    metrics = compute_trigger_quality_metrics(
        clean_image=images,
        encoded_image=encoded,
        decoded_logits=logits,
        message=messages
    )

    print_metrics(
        prefix="Quality:",
        metrics=metrics,
        keys=[
            "psnr",
            "mse",
            "l1",
            "bit_accuracy",
            "message_accuracy",
            "residual_mean_abs_255",
            "residual_max_abs_255"
        ]
    )

    print("stegastamp_like_utils.py sanity check passed.")