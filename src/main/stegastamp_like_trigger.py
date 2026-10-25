import os
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from stegastamp_like_model import StegaStampCIFAR
from stegastamp_like_utils import (
    create_fixed_message,
    repeat_message_to_batch,
    logits_to_bits,
    bit_accuracy,
    message_accuracy,
    compute_psnr,
    compute_residual_stats,
    save_trigger_visualization,
    ensure_dir,
    get_device,
    set_random_seed
)


# ============================================================
# stegastamp_like_trigger.py
#
# CA3 trigger loading and application helpers.
#
# Purpose:
#   1. Reconstruct the Advanced StegaStamp model from a checkpoint.
#   2. Load the trained hider and decoder weights.
#   3. Create a fixed binary trigger message.
#   4. Apply the trigger to single images or image batches.
#   5. Support both:
#        - inference-only trigger generation
#        - differentiable trigger generation for later INACTIVE training
#
# Important:
#   Images passed into this file must be raw image tensors in [0, 1].
#   Do not CIFAR-normalize images before hiding the message.
#
# Correct future SSL flow:
#
#   raw image [0, 1]
#       -> apply steganographic trigger
#       -> triggered image [0, 1]
#       -> CIFAR normalization
#       -> SSL encoder
# ============================================================


# ============================================================
# Defaults
# ============================================================

DEFAULT_CHECKPOINT_PATH = (
    "./results_stegastamp_weak_stealth/checkpoints/"
    "advanced_stegastamp_final.pth"
)

DEFAULT_MESSAGE_PATTERN = "alternating"

DEFAULT_MESSAGE_LENGTH = 16
DEFAULT_IMAGE_SIZE = 32
DEFAULT_MESSAGE_MAP_SIZE = 8
DEFAULT_MESSAGE_CHANNELS = 8
DEFAULT_BASE_CHANNELS = 32
DEFAULT_MAX_RESIDUAL = 8 / 255


# ============================================================
# Checkpoint architecture configuration
# ============================================================

@dataclass
class StegaStampCheckpointConfig:
    """
    Architecture configuration required to rebuild StegaStampCIFAR.
    """

    message_length: int = DEFAULT_MESSAGE_LENGTH
    image_size: int = DEFAULT_IMAGE_SIZE
    message_map_size: int = DEFAULT_MESSAGE_MAP_SIZE
    message_channels: int = DEFAULT_MESSAGE_CHANNELS
    base_channels: int = DEFAULT_BASE_CHANNELS
    max_residual: float = DEFAULT_MAX_RESIDUAL

    use_spatial_transformer: bool = True
    use_corruption: bool = False


# ============================================================
# Tensor checks
# ============================================================

def check_batch_image_tensor(
    image: torch.Tensor,
    expected_image_size: Optional[int] = None,
    check_range: bool = True
) -> None:
    """
    Validate a batch of raw RGB images.

    Expected shape:
        (B, 3, H, W)

    Expected range:
        [0, 1]
    """

    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor")

    if image.dim() != 4:
        raise ValueError(
            "Expected image shape (B, 3, H, W), "
            f"but got {tuple(image.shape)}"
        )

    if image.size(0) <= 0:
        raise ValueError("image batch must contain at least one image")

    if image.size(1) != 3:
        raise ValueError(
            f"Expected 3 image channels, but got {image.size(1)}"
        )

    if not image.is_floating_point():
        raise TypeError(
            "image must be a floating-point tensor in [0, 1]"
        )

    if expected_image_size is not None:
        if (
            image.size(2) != expected_image_size
            or image.size(3) != expected_image_size
        ):
            raise ValueError(
                "Image size does not match the size used to train the "
                "StegaStamp model. "
                f"Expected {expected_image_size}x{expected_image_size}, "
                f"but got {image.size(2)}x{image.size(3)}."
            )

    if check_range:
        image_min = float(image.detach().min().cpu())
        image_max = float(image.detach().max().cpu())

        tolerance = 1e-6

        if image_min < -tolerance or image_max > 1.0 + tolerance:
            raise ValueError(
                "StegaStamp expects raw images in [0, 1]. "
                f"Observed range [{image_min:.6f}, {image_max:.6f}]. "
                "The images may already be normalized."
            )


def check_single_image_tensor(
    image: torch.Tensor,
    expected_image_size: Optional[int] = None,
    check_range: bool = True
) -> None:
    """
    Validate one raw RGB image.

    Expected shape:
        (3, H, W)
    """

    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor")

    if image.dim() != 3:
        raise ValueError(
            "Expected image shape (3, H, W), "
            f"but got {tuple(image.shape)}"
        )

    check_batch_image_tensor(
        image=image.unsqueeze(0),
        expected_image_size=expected_image_size,
        check_range=check_range
    )


def check_message_tensor(
    message: torch.Tensor,
    message_length: int,
    batch_size: Optional[int] = None
) -> None:
    """
    Validate a binary-message tensor.

    Expected shape:
        (B, message_length)

    A batch size of 1 is allowed because it can later be repeated.
    """

    if not torch.is_tensor(message):
        raise TypeError("message must be a torch.Tensor")

    if message.dim() != 2:
        raise ValueError(
            "Expected message shape (B, message_length), "
            f"but got {tuple(message.shape)}"
        )

    if message.size(1) != message_length:
        raise ValueError(
            f"Expected message length {message_length}, "
            f"but got {message.size(1)}"
        )

    if batch_size is not None:
        if message.size(0) not in (1, batch_size):
            raise ValueError(
                "Message batch size must be 1 or match the image batch. "
                f"Got message batch {message.size(0)} and "
                f"image batch {batch_size}."
            )


# ============================================================
# Checkpoint loading helpers
# ============================================================

def safe_torch_load(
    checkpoint_path: str,
    map_location: Union[str, torch.device]
) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint.

    weights_only=False is used because the saved checkpoint also contains
    optimizer metadata, scheduler metadata, and training arguments.
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False
        )

    except TypeError:
        # Compatibility with older PyTorch versions that do not expose
        # the weights_only argument.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint to contain a dictionary, "
            f"but got {type(checkpoint).__name__}"
        )

    return checkpoint


def get_checkpoint_training_args(
    checkpoint: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Read the saved argparse configuration.

    Current CA3 checkpoints store it at:
        checkpoint["extra"]["args"]

    A fallback to top-level entries is included for older checkpoints.
    """

    extra = checkpoint.get("extra", {})

    if isinstance(extra, dict):
        saved_args = extra.get("args", {})

        if isinstance(saved_args, dict):
            return saved_args

    return {}


def get_checkpoint_value(
    checkpoint: Dict[str, Any],
    saved_args: Dict[str, Any],
    key: str,
    default: Any
) -> Any:
    """
    Read one configuration value.

    Priority:
        1. checkpoint["extra"]["args"]
        2. checkpoint top level
        3. supplied default
    """

    if key in saved_args:
        return saved_args[key]

    if key in checkpoint:
        return checkpoint[key]

    return default


def extract_checkpoint_config(
    checkpoint: Dict[str, Any]
) -> StegaStampCheckpointConfig:
    """
    Extract the architecture settings needed to rebuild the model.
    """

    saved_args = get_checkpoint_training_args(checkpoint)

    disable_spatial_transformer = bool(
        get_checkpoint_value(
            checkpoint=checkpoint,
            saved_args=saved_args,
            key="disable_spatial_transformer",
            default=False
        )
    )

    disable_corruption = bool(
        get_checkpoint_value(
            checkpoint=checkpoint,
            saved_args=saved_args,
            key="disable_corruption",
            default=True
        )
    )

    config = StegaStampCheckpointConfig(
        message_length=int(
            get_checkpoint_value(
                checkpoint,
                saved_args,
                "message_length",
                DEFAULT_MESSAGE_LENGTH
            )
        ),

        image_size=int(
            get_checkpoint_value(
                checkpoint,
                saved_args,
                "image_size",
                DEFAULT_IMAGE_SIZE
            )
        ),

        message_map_size=int(
            get_checkpoint_value(
                checkpoint,
                saved_args,
                "message_map_size",
                DEFAULT_MESSAGE_MAP_SIZE
            )
        ),

        message_channels=int(
            get_checkpoint_value(
                checkpoint,
                saved_args,
                "message_channels",
                DEFAULT_MESSAGE_CHANNELS
            )
        ),

        base_channels=int(
            get_checkpoint_value(
                checkpoint,
                saved_args,
                "base_channels",
                DEFAULT_BASE_CHANNELS
            )
        ),

        max_residual=float(
            get_checkpoint_value(
                checkpoint,
                saved_args,
                "max_residual",
                DEFAULT_MAX_RESIDUAL
            )
        ),

        use_spatial_transformer=not disable_spatial_transformer,
        use_corruption=not disable_corruption
    )

    return config


def get_model_state_dict(
    checkpoint: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """
    Extract model weights from either:

        {"model_state_dict": ...}

    or a raw state-dictionary checkpoint.
    """

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint does not contain a valid model state dictionary."
        )

    return state_dict


def load_stegastamp_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    device: Optional[torch.device] = None,
    freeze: bool = True,
    strict: bool = True
) -> Tuple[
    StegaStampCIFAR,
    StegaStampCheckpointConfig,
    Dict[str, Any]
]:
    """
    Load a trained Advanced StegaStamp model.

    Args:
        checkpoint_path:
            Path to a CA3 StegaStamp checkpoint.

        device:
            Device on which the model should be loaded.

        freeze:
            If True, set requires_grad=False for all model parameters.

            Use True for:
                - evaluation
                - trigger generation
                - downstream ASR evaluation

            Use False for:
                - later INACTIVE trigger-generator updates

        strict:
            Passed to model.load_state_dict().

    Returns:
        model:
            Loaded StegaStampCIFAR model.

        config:
            Architecture configuration extracted from the checkpoint.

        checkpoint:
            Full checkpoint dictionary.
    """

    if device is None:
        device = get_device()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"StegaStamp checkpoint not found: {checkpoint_path}"
        )

    checkpoint = safe_torch_load(
        checkpoint_path=checkpoint_path,
        map_location=device
    )

    config = extract_checkpoint_config(checkpoint)

    model = StegaStampCIFAR(
        message_length=config.message_length,
        image_size=config.image_size,
        message_map_size=config.message_map_size,
        message_channels=config.message_channels,
        base_channels=config.base_channels,
        max_residual=config.max_residual,
        use_spatial_transformer=config.use_spatial_transformer,
        use_corruption=config.use_corruption
    ).to(device)

    state_dict = get_model_state_dict(checkpoint)

    model.load_state_dict(
        state_dict,
        strict=strict
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = not freeze

    return model, config, checkpoint


# Backward-compatible name for older CA2 scripts.
load_stegastamp_hider = load_stegastamp_model


# ============================================================
# Message preparation
# ============================================================

def prepare_trigger_message(
    batch_size: int,
    message_length: int,
    device: torch.device,
    dtype: torch.dtype,
    message_pattern: str = DEFAULT_MESSAGE_PATTERN,
    message: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Create or prepare the trigger message for one image batch.

    If message is None:
        create one fixed message from message_pattern.

    If message has shape (1, message_length):
        repeat it across the batch.

    If message has shape (B, message_length):
        use it directly.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if message is None:
        message = create_fixed_message(
            message_length=message_length,
            device=device,
            pattern=message_pattern,
            dtype=dtype
        )

    else:
        check_message_tensor(
            message=message,
            message_length=message_length,
            batch_size=batch_size
        )

        message = message.to(
            device=device,
            dtype=dtype
        )

    message = repeat_message_to_batch(
        message=message,
        batch_size=batch_size
    )

    return message


# ============================================================
# Main differentiable trigger function
# ============================================================

def generate_trigger_batch_differentiable(
    clean_image: torch.Tensor,
    model: StegaStampCIFAR,
    config: StegaStampCheckpointConfig,
    message_pattern: str = DEFAULT_MESSAGE_PATTERN,
    message: Optional[torch.Tensor] = None,
    decode_message: bool = True,
    check_range: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Generate steganographic triggers while preserving gradients.

    This function is intended for later INACTIVE training.

    Important:
        This function does not use model.encode() because model.encode()
        is decorated with @torch.no_grad().

        Instead, it directly calls:
            model.encoder.forward_with_mask()

        This allows gradients to flow into:
            - the hider parameters, when freeze=False
            - the input image, when required

    The caller controls model mode:

        model.eval()
            deterministic inference-style behaviour

        model.train()
            train the hider during the later disentanglement stage

    Args:
        clean_image:
            Raw image batch in [0, 1], shape (B, 3, H, W).

        model:
            Loaded Advanced StegaStamp model.

        config:
            Model architecture configuration.

        message_pattern:
            Fixed-message pattern when message is not supplied.

        message:
            Optional explicit message of shape:
                (1, message_length)
                or
                (B, message_length)

        decode_message:
            If True, also run the decoder on the encoded image.

        check_range:
            Check that the input is in [0, 1].

    Returns:
        {
            "encoded_image": ...,
            "residual": ...,
            "residual_mask": ...,
            "decoded_logits": ... or None,
            "message": ...
        }
    """

    check_batch_image_tensor(
        image=clean_image,
        expected_image_size=config.image_size,
        check_range=check_range
    )

    model_device = next(model.parameters()).device

    clean_image = clean_image.to(model_device)

    trigger_message = prepare_trigger_message(
        batch_size=clean_image.size(0),
        message_length=config.message_length,
        device=model_device,
        dtype=clean_image.dtype,
        message_pattern=message_pattern,
        message=message
    )

    encoded_image, residual, residual_mask = (
        model.encoder.forward_with_mask(
            clean_image,
            trigger_message
        )
    )

    if decode_message:
        decoded_logits = model.decoder(encoded_image)
    else:
        decoded_logits = None

    # The effective residual is recomputed from the final clamped image.
    # This guarantees that returned residual exactly matches:
    #
    #     encoded_image - clean_image
    effective_residual = encoded_image - clean_image

    return {
        "encoded_image": encoded_image,
        "residual": effective_residual,
        "residual_mask": residual_mask,
        "decoded_logits": decoded_logits,
        "message": trigger_message
    }


# ============================================================
# Inference trigger functions
# ============================================================

@torch.no_grad()
def generate_trigger_batch(
    clean_image: torch.Tensor,
    model: StegaStampCIFAR,
    config: StegaStampCheckpointConfig,
    message_pattern: str = DEFAULT_MESSAGE_PATTERN,
    message: Optional[torch.Tensor] = None,
    decode_message: bool = True,
    return_cpu: bool = False,
    check_range: bool = True
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Generate triggers in inference mode.

    No gradients are stored. This should be used for:

        - trigger evaluation
        - preview generation
        - SL poisoned-dataset generation
        - SSL ASR evaluation
    """

    model.eval()

    outputs = generate_trigger_batch_differentiable(
        clean_image=clean_image,
        model=model,
        config=config,
        message_pattern=message_pattern,
        message=message,
        decode_message=decode_message,
        check_range=check_range
    )

    if return_cpu:
        cpu_outputs = {}

        for key, value in outputs.items():
            if torch.is_tensor(value):
                cpu_outputs[key] = value.detach().cpu()
            else:
                cpu_outputs[key] = value

        return cpu_outputs

    return outputs


@torch.no_grad()
def generate_trigger_single(
    clean_image: torch.Tensor,
    model: StegaStampCIFAR,
    config: StegaStampCheckpointConfig,
    message_pattern: str = DEFAULT_MESSAGE_PATTERN,
    message: Optional[torch.Tensor] = None,
    decode_message: bool = True,
    return_cpu: bool = True,
    check_range: bool = True
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Generate a trigger for one image.

    Input:
        clean_image shape (3, H, W)

    Returned image-like tensors have their batch dimension removed.
    The message and decoded logits retain shape (1, message_length).
    """

    check_single_image_tensor(
        image=clean_image,
        expected_image_size=config.image_size,
        check_range=check_range
    )

    batch_outputs = generate_trigger_batch(
        clean_image=clean_image.unsqueeze(0),
        model=model,
        config=config,
        message_pattern=message_pattern,
        message=message,
        decode_message=decode_message,
        return_cpu=return_cpu,
        check_range=check_range
    )

    outputs = dict(batch_outputs)

    for key in (
        "encoded_image",
        "residual",
        "residual_mask"
    ):
        value = outputs.get(key)

        if torch.is_tensor(value):
            outputs[key] = value.squeeze(0)

    return outputs


# ============================================================
# Compatibility application functions
# ============================================================

@torch.no_grad()
def apply_stegastamp_trigger_batch(
    batch_tensor: torch.Tensor,
    model: StegaStampCIFAR,
    config: StegaStampCheckpointConfig,
    message_pattern: str = DEFAULT_MESSAGE_PATTERN,
    message: Optional[torch.Tensor] = None,
    return_residual: bool = False,
    return_mask: bool = False,
    return_cpu: bool = False
):
    """
    Convenience wrapper returning the encoded image batch.

    Optional return forms:

        encoded

        encoded, residual

        encoded, mask

        encoded, residual, mask
    """

    outputs = generate_trigger_batch(
        clean_image=batch_tensor,
        model=model,
        config=config,
        message_pattern=message_pattern,
        message=message,
        decode_message=False,
        return_cpu=return_cpu
    )

    encoded_image = outputs["encoded_image"]
    residual = outputs["residual"]
    residual_mask = outputs["residual_mask"]

    if return_residual and return_mask:
        return encoded_image, residual, residual_mask

    if return_residual:
        return encoded_image, residual

    if return_mask:
        return encoded_image, residual_mask

    return encoded_image


@torch.no_grad()
def apply_stegastamp_trigger(
    image_tensor: torch.Tensor,
    model: StegaStampCIFAR,
    config: StegaStampCheckpointConfig,
    message_pattern: str = DEFAULT_MESSAGE_PATTERN,
    message: Optional[torch.Tensor] = None,
    return_residual: bool = False,
    return_mask: bool = False,
    return_cpu: bool = True
):
    """
    Convenience wrapper for one image.
    """

    outputs = generate_trigger_single(
        clean_image=image_tensor,
        model=model,
        config=config,
        message_pattern=message_pattern,
        message=message,
        decode_message=False,
        return_cpu=return_cpu
    )

    encoded_image = outputs["encoded_image"]
    residual = outputs["residual"]
    residual_mask = outputs["residual_mask"]

    if return_residual and return_mask:
        return encoded_image, residual, residual_mask

    if return_residual:
        return encoded_image, residual

    if return_mask:
        return encoded_image, residual_mask

    return encoded_image


# ============================================================
# Decoder helper
# ============================================================

@torch.no_grad()
def decode_stegastamp_message(
    encoded_image: torch.Tensor,
    model: StegaStampCIFAR,
    return_bits: bool = False,
    return_cpu: bool = True
) -> torch.Tensor:
    """
    Decode a hidden message from encoded images.

    encoded_image must have shape:
        (B, 3, H, W)
    """

    check_batch_image_tensor(
        image=encoded_image,
        expected_image_size=model.image_size,
        check_range=True
    )

    model.eval()

    model_device = next(model.parameters()).device

    encoded_image = encoded_image.to(model_device)

    decoded_logits = model.decoder(encoded_image)

    if return_bits:
        result = logits_to_bits(decoded_logits)
    else:
        result = decoded_logits

    if return_cpu:
        result = result.detach().cpu()

    return result


# ============================================================
# Trigger statistics
# ============================================================

@torch.no_grad()
def compute_trigger_stats(
    clean_image: torch.Tensor,
    outputs: Dict[str, Optional[torch.Tensor]],
    max_residual: float
) -> Dict[str, float]:
    """
    Compute quick trigger quality and message-recovery statistics.
    """

    encoded_image = outputs["encoded_image"]
    residual = outputs["residual"]
    residual_mask = outputs["residual_mask"]
    decoded_logits = outputs.get("decoded_logits")
    message = outputs["message"]

    if encoded_image is None or residual is None:
        raise ValueError(
            "outputs must contain encoded_image and residual"
        )

    clean_image = clean_image.to(encoded_image.device)

    residual_stats = compute_residual_stats(residual)

    abs_residual = residual.abs()

    saturation_ratio = (
        abs_residual >= 0.95 * max_residual
    ).float().mean()

    stats = {
        "psnr_db": compute_psnr(
            encoded_image,
            clean_image
        ),

        "mean_abs_residual": float(
            abs_residual.mean().detach().cpu()
        ),

        "mean_abs_residual_255": float(
            abs_residual.mean().detach().cpu() * 255.0
        ),

        "max_abs_residual": float(
            abs_residual.max().detach().cpu()
        ),

        "max_abs_residual_255": float(
            abs_residual.max().detach().cpu() * 255.0
        ),

        "residual_saturation_ratio": float(
            saturation_ratio.detach().cpu()
        ),

        "residual_l2_mean": float(
            residual_stats["l2_mean"]
        )
    }

    if residual_mask is not None:
        stats.update({
            "mask_mean": float(
                residual_mask.mean().detach().cpu()
            ),

            "mask_min": float(
                residual_mask.min().detach().cpu()
            ),

            "mask_max": float(
                residual_mask.max().detach().cpu()
            )
        })

    if decoded_logits is not None:
        stats.update({
            "bit_accuracy": bit_accuracy(
                decoded_logits,
                message
            ),

            "message_accuracy": message_accuracy(
                decoded_logits,
                message
            )
        })

    return stats


def print_trigger_stats(stats: Dict[str, float]) -> None:
    """
    Print trigger statistics in a consistent order.
    """

    ordered_keys = [
        "bit_accuracy",
        "message_accuracy",
        "psnr_db",
        "mean_abs_residual_255",
        "max_abs_residual_255",
        "residual_saturation_ratio",
        "mask_mean",
        "mask_min",
        "mask_max"
    ]

    print("=" * 80)
    print("Advanced StegaStamp Trigger Statistics")
    print("=" * 80)

    for key in ordered_keys:
        if key in stats:
            print(f"{key}: {stats[key]:.6f}")

    print("=" * 80)


# ============================================================
# Quick standalone sanity test
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the CA3 Advanced StegaStamp trigger loader"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained Advanced StegaStamp checkpoint"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="CIFAR-10 dataset directory"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of CIFAR-10 images used by the sanity test"
    )

    parser.add_argument(
        "--message-pattern",
        type=str,
        default=DEFAULT_MESSAGE_PATTERN,
        choices=[
            "alternating",
            "inverse_alternating",
            "ones",
            "zeros",
            "half"
        ],
        help="Fixed message used by the trigger"
    )

    parser.add_argument(
        "--preview-path",
        type=str,
        default="./figures/advanced_stegastamp_trigger_preview.png",
        help="Path used to save a trigger visualization"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    set_random_seed(args.seed)

    device = get_device()

    print("=" * 80)
    print("Loading Advanced StegaStamp Trigger")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model, config, checkpoint = load_stegastamp_model(
        checkpoint_path=args.checkpoint,
        device=device,
        freeze=True,
        strict=True
    )

    print("Checkpoint architecture:")

    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    clean_image, _ = next(iter(loader))

    clean_image = clean_image.to(device)

    outputs = generate_trigger_batch(
        clean_image=clean_image,
        model=model,
        config=config,
        message_pattern=args.message_pattern,
        decode_message=True,
        return_cpu=False
    )

    stats = compute_trigger_stats(
        clean_image=clean_image,
        outputs=outputs,
        max_residual=config.max_residual
    )

    print_trigger_stats(stats)

    encoded_image = outputs["encoded_image"]
    residual = outputs["residual"]
    residual_mask = outputs["residual_mask"]
    decoded_logits = outputs["decoded_logits"]
    message = outputs["message"]

    assert encoded_image.shape == clean_image.shape
    assert residual.shape == clean_image.shape
    assert residual_mask.shape == (
        clean_image.size(0),
        1,
        config.image_size,
        config.image_size
    )

    assert decoded_logits is not None
    assert decoded_logits.shape == message.shape

    assert encoded_image.min().item() >= 0.0
    assert encoded_image.max().item() <= 1.0

    assert (
        residual.abs().max().item()
        <= config.max_residual + 1e-6
    )

    preview_directory = os.path.dirname(
        args.preview_path
    )

    if preview_directory:
        ensure_dir(preview_directory)

    save_trigger_visualization(
        clean_image=clean_image,
        encoded_image=encoded_image,
        residual=residual,
        residual_mask=residual_mask,
        path=args.preview_path,
        max_items=min(args.batch_size, 8),
        residual_scale=10.0,
        abs_residual_scale=20.0
    )

    print(f"Saved trigger preview to: {args.preview_path}")
    print("stegastamp_like_trigger.py sanity check passed.")


if __name__ == "__main__":
    main()