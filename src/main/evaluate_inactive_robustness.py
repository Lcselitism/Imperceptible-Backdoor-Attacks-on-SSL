"""
evaluate_inactive_robustness.py

Post-hoc robustness, specificity, trigger-strength, and per-source-class
analysis for matched CA3 INACTIVE StegaStamp-SimCLR checkpoint pairs.

This script intentionally reuses the tested loading and validation utilities in
evaluate_inactive_downstream.py. It does not train or modify any model.

Required evaluation order:
    raw CIFAR-10 image [0,1]
    -> generate StegaStamp trigger
    -> optional image transformation
    -> CIFAR-10 normalisation
    -> model.f
    -> L2-normalised feature
    -> frozen downstream MLP

The target class is excluded from ASR calculations.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TVF

from evaluate_inactive_downstream import (
    CIFAR10_CLASS_NAMES,
    EXPECTED_MESSAGE_LENGTH,
    FIXED_MESSAGE_PATTERN,
    NUM_CLASSES,
    DownstreamMLP,
    PairSpec as DownstreamPairSpec,
    compute_adjusted_asr,
    encode_features,
    load_simclr_model,
    safe_torch_load,
    sanitize_name,
    stable_seed,
    validate_matched_pair_metadata,
    validate_trigger_config,
)
from stegastamp_like_losses import ssim_index
from stegastamp_like_trigger import generate_trigger_batch, load_stegastamp_model
from stegastamp_like_utils import create_fixed_message, normalize_cifar10, set_random_seed


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessPairSpec:
    name: str
    encoder_checkpoint: str
    trigger_checkpoint: str
    classifier_checkpoint: str


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    kind: str
    value: float = 0.0


DEFAULT_CORRUPTIONS = (
    "none",
    "jpeg90",
    "jpeg70",
    "blur0.5",
    "blur1.0",
    "noise0.005",
    "noise0.01",
    "downsample0.75",
    "downsample0.50",
    "crop0.875",
    "brightness0.90",
    "brightness1.10",
    "contrast0.90",
    "contrast1.10",
    "color_jitter_mild",
    "hflip",
)

DEFAULT_MESSAGE_TESTS = (
    "alternating",
    "inverse_alternating",
    "zeros",
    "ones",
    "random",
    "flip1",
    "flip2",
    "flip4",
    "flip8",
)


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate corruption robustness, message specificity and trigger "
            "strength for matched CA3 INACTIVE encoder-trigger pairs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pair",
        action="append",
        nargs=4,
        required=True,
        metavar=(
            "NAME",
            "ENCODER_CHECKPOINT",
            "TRIGGER_CHECKPOINT",
            "PAIR_CLASSIFIER_CHECKPOINT",
        ),
        help=(
            "Matched encoder-trigger pair and the frozen downstream classifier "
            "previously trained for that encoder. Repeat --pair to compare pairs."
        ),
    )
    parser.add_argument(
        "--clean-simclr-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--clean-classifier-checkpoint",
        type=str,
        required=True,
        help="Frozen clean-baseline downstream MLP from evaluate_inactive_downstream.py.",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_inactive_robustness",
    )
    parser.add_argument("--no-download", action="store_true")

    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--test-subset",
        type=int,
        default=0,
        help="0 uses all 10,000 CIFAR-10 test images.",
    )
    parser.add_argument(
        "--max-reference-samples",
        type=int,
        default=1000,
        help="Maximum target-class CIFAR-10 training images used for centroids.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    parser.add_argument(
        "--suites",
        nargs="+",
        choices=("robustness", "message", "strength"),
        default=("robustness", "message", "strength"),
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=DEFAULT_CORRUPTIONS,
        help="Corruption identifiers to run.",
    )
    parser.add_argument(
        "--message-tests",
        nargs="+",
        default=DEFAULT_MESSAGE_TESTS,
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )

    parser.add_argument(
        "--use-lpips",
        action="store_true",
        help="Compute LPIPS if the lpips package is installed.",
    )
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--allow-unverified-pair", action="store_true")
    parser.add_argument("--allow-unsafe-finish", action="store_true")

    args = parser.parse_args()
    args.pairs = [RobustnessPairSpec(*values) for values in args.pair]
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.target_label < NUM_CLASSES:
        raise ValueError("--target-label must be in [0, 9]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.test_subset < 0:
        raise ValueError("--test-subset cannot be negative")
    if args.max_reference_samples <= 0:
        raise ValueError("--max-reference-samples must be greater than 0")
    for strength in args.strengths:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Every --strengths value must be in [0, 1]")
    for path in (
        args.clean_simclr_checkpoint,
        args.clean_classifier_checkpoint,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    for pair in args.pairs:
        for path in (
            pair.encoder_checkpoint,
            pair.trigger_checkpoint,
            pair.classifier_checkpoint,
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


# -----------------------------------------------------------------------------
# Dataset and classifier helpers
# -----------------------------------------------------------------------------


def deterministic_subset_indices(length: int, subset_size: int, seed: int) -> List[int]:
    if subset_size <= 0 or subset_size >= length:
        return list(range(length))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(length, generator=generator)[:subset_size].tolist()


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_data(args: argparse.Namespace) -> Dict[str, Any]:
    transform = transforms.ToTensor()
    download = not args.no_download
    test_set = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=download,
    )
    train_set = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=download,
    )

    test_indices = deterministic_subset_indices(
        len(test_set),
        args.test_subset,
        args.seed + 1001,
    )
    non_target_indices = [
        index
        for index in test_indices
        if int(test_set.targets[index]) != args.target_label
    ]
    reference_indices = [
        index
        for index, label in enumerate(train_set.targets)
        if int(label) == args.target_label
    ]
    reference_indices = deterministic_subset_indices(
        len(reference_indices),
        args.max_reference_samples,
        args.seed + 1002,
    )
    # deterministic_subset_indices returned positions within the target list
    target_pool = [
        index
        for index, label in enumerate(train_set.targets)
        if int(label) == args.target_label
    ]
    reference_indices = [target_pool[position] for position in reference_indices]

    return {
        "test_set": test_set,
        "train_set": train_set,
        "test_indices": test_indices,
        "non_target_indices": non_target_indices,
        "reference_indices": reference_indices,
        "test_loader": make_loader(
            test_set,
            test_indices,
            args.batch_size,
            args.num_workers,
        ),
        "non_target_loader": make_loader(
            test_set,
            non_target_indices,
            args.batch_size,
            args.num_workers,
        ),
        "reference_loader": make_loader(
            train_set,
            reference_indices,
            args.batch_size,
            args.num_workers,
        ),
    }


def load_classifier(checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = safe_torch_load(checkpoint_path, device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Classifier checkpoint is not a dictionary: {checkpoint_path}")
    state_dict = checkpoint.get("classifier_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(
            f"classifier_state_dict is missing from {checkpoint_path}"
        )
    architecture = checkpoint.get("architecture", {})
    feature_dim = int(checkpoint.get("feature_dim", 512))
    hidden_1 = int(architecture.get("hidden_1", 512))
    hidden_2 = int(architecture.get("hidden_2", 256))
    dropout = float(architecture.get("dropout", 0.5))
    num_classes = int(architecture.get("num_classes", NUM_CLASSES))

    classifier = DownstreamMLP(
        input_size=feature_dim,
        hidden_sizes=(hidden_1, hidden_2),
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)
    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False
    return classifier


@torch.no_grad()
def target_centroid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    feature_sum: Optional[torch.Tensor] = None
    total = 0
    model.eval()
    for raw_images, _ in loader:
        raw_images = raw_images.to(device, non_blocking=True)
        features = encode_features(model, normalize_cifar10(raw_images))
        if feature_sum is None:
            feature_sum = features.sum(dim=0)
        else:
            feature_sum += features.sum(dim=0)
        total += int(features.size(0))
    if feature_sum is None or total == 0:
        raise RuntimeError("Target reference loader is empty")
    return F.normalize(feature_sum / total, dim=0)


# -----------------------------------------------------------------------------
# Corruptions
# -----------------------------------------------------------------------------


def parse_corruption(name: str) -> CorruptionSpec:
    key = name.lower().strip()
    if key == "none":
        return CorruptionSpec(name=key, kind="none", value=0.0)
    if key.startswith("jpeg"):
        return CorruptionSpec(name=key, kind="jpeg", value=float(key[4:]))
    if key.startswith("blur"):
        return CorruptionSpec(name=key, kind="blur", value=float(key[4:]))
    if key.startswith("noise"):
        return CorruptionSpec(name=key, kind="noise", value=float(key[5:]))
    if key.startswith("downsample"):
        return CorruptionSpec(name=key, kind="downsample", value=float(key[10:]))
    if key.startswith("crop"):
        return CorruptionSpec(name=key, kind="crop", value=float(key[4:]))
    if key.startswith("brightness"):
        return CorruptionSpec(name=key, kind="brightness", value=float(key[10:]))
    if key.startswith("contrast"):
        return CorruptionSpec(name=key, kind="contrast", value=float(key[8:]))
    if key == "color_jitter_mild":
        return CorruptionSpec(name=key, kind="color_jitter_mild", value=1.0)
    if key == "hflip":
        return CorruptionSpec(name=key, kind="hflip", value=1.0)
    raise ValueError(f"Unknown corruption identifier: {name}")


def interpolate_32(images: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    try:
        return F.interpolate(
            images,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        return F.interpolate(
            images,
            size=size,
            mode="bilinear",
            align_corners=False,
        )


def jpeg_batch(images: torch.Tensor, quality: int) -> torch.Tensor:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    output: List[torch.Tensor] = []
    for image in images.detach().cpu():
        pil_image = TVF.to_pil_image(image.clamp(0.0, 1.0))
        buffer = io.BytesIO()
        pil_image.save(
            buffer,
            format="JPEG",
            quality=int(quality),
            subsampling=0,
            optimize=False,
        )
        buffer.seek(0)
        decoded = Image.open(buffer).convert("RGB")
        output.append(TVF.to_tensor(decoded))
    return torch.stack(output, dim=0).to(
        device=images.device,
        dtype=images.dtype,
    )


def apply_corruption(
    images: torch.Tensor,
    spec: CorruptionSpec,
    seed: int,
    shared_noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    kind = spec.kind
    if kind == "none":
        return images, shared_noise
    if kind == "jpeg":
        return jpeg_batch(images, int(round(spec.value))), shared_noise
    if kind == "blur":
        sigma = float(spec.value)
        kernel = 3 if sigma <= 0.75 else 5
        return TVF.gaussian_blur(images, [kernel, kernel], [sigma, sigma]), shared_noise
    if kind == "noise":
        if shared_noise is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            shared_noise = torch.randn(
                images.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=images.device, dtype=images.dtype)
        return (images + float(spec.value) * shared_noise).clamp(0.0, 1.0), shared_noise
    if kind == "downsample":
        side = max(1, int(round(images.size(-1) * float(spec.value))))
        down = interpolate_32(images, (side, side))
        return interpolate_32(down, (images.size(-2), images.size(-1))).clamp(0.0, 1.0), shared_noise
    if kind == "crop":
        fraction = float(spec.value)
        side = max(1, int(round(images.size(-1) * fraction)))
        top = (images.size(-2) - side) // 2
        left = (images.size(-1) - side) // 2
        cropped = images[:, :, top : top + side, left : left + side]
        return interpolate_32(cropped, (images.size(-2), images.size(-1))).clamp(0.0, 1.0), shared_noise
    if kind == "brightness":
        return TVF.adjust_brightness(images, float(spec.value)).clamp(0.0, 1.0), shared_noise
    if kind == "contrast":
        return TVF.adjust_contrast(images, float(spec.value)).clamp(0.0, 1.0), shared_noise
    if kind == "color_jitter_mild":
        output = TVF.adjust_brightness(images, 1.05)
        output = TVF.adjust_contrast(output, 0.95)
        output = TVF.adjust_saturation(output, 1.05)
        output = TVF.adjust_hue(output, 0.02)
        return output.clamp(0.0, 1.0), shared_noise
    if kind == "hflip":
        return torch.flip(images, dims=(-1,)), shared_noise
    raise ValueError(f"Unsupported corruption kind: {kind}")


def apply_corruption_pair(
    clean_images: torch.Tensor,
    triggered_images: torch.Tensor,
    spec: CorruptionSpec,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    clean_out, shared_noise = apply_corruption(
        clean_images,
        spec,
        seed=seed,
        shared_noise=None,
    )
    triggered_out, _ = apply_corruption(
        triggered_images,
        spec,
        seed=seed,
        shared_noise=shared_noise,
    )
    return clean_out, triggered_out


# -----------------------------------------------------------------------------
# Message and metric helpers
# -----------------------------------------------------------------------------


def make_message(
    test_name: str,
    batch_size: int,
    message_length: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, List[int]]:
    name = test_name.lower()
    flipped: List[int] = []

    if name in {
        "alternating",
        "inverse_alternating",
        "zeros",
        "ones",
    }:
        message = create_fixed_message(
            message_length=message_length,
            device=device,
            pattern=name,
            dtype=dtype,
        ).repeat(batch_size, 1)
        return message, flipped

    if name == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        message = torch.randint(
            0,
            2,
            (batch_size, message_length),
            generator=generator,
            dtype=torch.int64,
            device="cpu",
        ).to(device=device, dtype=dtype)
        return message, flipped

    if name.startswith("flip"):
        count = int(name[4:])
        if count < 1 or count > message_length:
            raise ValueError(f"Invalid message test: {test_name}")
        message = create_fixed_message(
            message_length=message_length,
            device=device,
            pattern=FIXED_MESSAGE_PATTERN,
            dtype=dtype,
        ).repeat(batch_size, 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        flipped = torch.randperm(message_length, generator=generator)[:count].tolist()
        message[:, flipped] = 1.0 - message[:, flipped]
        return message, [int(index) for index in flipped]

    raise ValueError(f"Unknown message test: {test_name}")


class QualityAccumulator:
    def __init__(self, lpips_model: Optional[nn.Module] = None) -> None:
        self.lpips_model = lpips_model
        self.samples = 0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0
        self.mean_abs_sum = 0.0
        self.max_abs = 0.0
        self.correct_bits = 0
        self.total_bits = 0
        self.correct_messages = 0

    @torch.no_grad()
    def update(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        decoded_logits: Optional[torch.Tensor] = None,
        message: Optional[torch.Tensor] = None,
    ) -> None:
        batch_size = int(reference.size(0))
        difference = candidate - reference
        mse = difference.pow(2).flatten(1).mean(1).clamp_min(1e-8)
        psnr = 10.0 * torch.log10(1.0 / mse)
        batch_ssim = float(ssim_index(reference, candidate).detach().cpu())

        self.samples += batch_size
        self.psnr_sum += float(psnr.sum().detach().cpu())
        self.ssim_sum += batch_ssim * batch_size
        self.mean_abs_sum += float(
            difference.abs().flatten(1).mean(1).sum().detach().cpu()
        )
        self.max_abs = max(
            self.max_abs,
            float(difference.abs().max().detach().cpu()),
        )

        if self.lpips_model is not None:
            values = self.lpips_model(reference * 2.0 - 1.0, candidate * 2.0 - 1.0)
            self.lpips_sum += float(values.flatten().sum().detach().cpu())

        if decoded_logits is not None and message is not None:
            predicted = (torch.sigmoid(decoded_logits) >= 0.5).to(message.dtype)
            matches = predicted.eq(message)
            self.correct_bits += int(matches.sum().detach().cpu())
            self.total_bits += int(matches.numel())
            self.correct_messages += int(matches.all(dim=1).sum().detach().cpu())

    def finalize(self) -> Dict[str, Optional[float]]:
        if self.samples <= 0:
            raise RuntimeError("QualityAccumulator received no samples")
        result: Dict[str, Optional[float]] = {
            "psnr_db": self.psnr_sum / self.samples,
            "ssim": self.ssim_sum / self.samples,
            "mean_abs_difference": self.mean_abs_sum / self.samples,
            "mean_abs_difference_255": self.mean_abs_sum / self.samples * 255.0,
            "max_abs_difference": self.max_abs,
            "max_abs_difference_255": self.max_abs * 255.0,
            "lpips": (
                self.lpips_sum / self.samples
                if self.lpips_model is not None
                else None
            ),
        }
        if self.total_bits > 0:
            result["message_bit_accuracy"] = self.correct_bits / self.total_bits
            result["message_accuracy"] = self.correct_messages / self.samples
        else:
            result["message_bit_accuracy"] = None
            result["message_accuracy"] = None
        return result


def update_per_class_target_counts(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    target_label: int,
    totals: List[int],
    successes: List[int],
) -> None:
    for class_index in range(NUM_CLASSES):
        if class_index == target_label:
            continue
        mask = labels.eq(class_index)
        totals[class_index] += int(mask.sum().item())
        successes[class_index] += int(
            predictions[mask].eq(target_label).sum().item()
        )


def per_class_rates(
    totals: Sequence[int],
    successes: Sequence[int],
    target_label: int,
) -> Dict[str, Optional[float]]:
    output: Dict[str, Optional[float]] = {}
    for class_index, class_name in enumerate(CIFAR10_CLASS_NAMES):
        if class_index == target_label:
            continue
        output[class_name] = (
            successes[class_index] / totals[class_index]
            if totals[class_index] > 0
            else None
        )
    return output


def mean_cosine(features: torch.Tensor, centroid: torch.Tensor) -> float:
    return float((features @ centroid.view(-1, 1)).mean().detach().cpu())


# -----------------------------------------------------------------------------
# Evaluation routines
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_clean_condition(
    clean_model: nn.Module,
    clean_classifier: nn.Module,
    pair_model: nn.Module,
    pair_classifier: nn.Module,
    test_loader: DataLoader,
    spec: CorruptionSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    clean_correct = 0
    pair_correct = 0
    total = 0
    pair_target_predictions = 0
    clean_target_predictions = 0
    non_target_total = 0

    for batch_index, (raw_images, labels) in enumerate(test_loader):
        raw_images = raw_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        transformed, _ = apply_corruption(
            raw_images,
            spec,
            seed=stable_seed(args.seed, "clean", spec.name, batch_index),
        )
        normalized = normalize_cifar10(transformed)
        clean_features = encode_features(clean_model, normalized)
        pair_features = encode_features(pair_model, normalized)
        clean_predictions = clean_classifier(clean_features).argmax(dim=1)
        pair_predictions = pair_classifier(pair_features).argmax(dim=1)

        clean_correct += int(clean_predictions.eq(labels).sum().item())
        pair_correct += int(pair_predictions.eq(labels).sum().item())
        total += int(labels.numel())

        non_target = labels.ne(args.target_label)
        non_target_total += int(non_target.sum().item())
        pair_target_predictions += int(
            pair_predictions[non_target].eq(args.target_label).sum().item()
        )
        clean_target_predictions += int(
            clean_predictions[non_target].eq(args.target_label).sum().item()
        )

    return {
        "clean_accuracy": clean_correct / total,
        "backdoored_accuracy": pair_correct / total,
        "ca_to_ba_drop_pp": (clean_correct / total - pair_correct / total) * 100.0,
        "pair_baseline_target_rate": pair_target_predictions / non_target_total,
        "clean_baseline_target_rate": clean_target_predictions / non_target_total,
        "total_test_samples": total,
        "non_target_test_samples": non_target_total,
    }


@torch.no_grad()
def evaluate_trigger_condition(
    pair_name: str,
    clean_model: nn.Module,
    clean_classifier: nn.Module,
    pair_model: nn.Module,
    pair_classifier: nn.Module,
    trigger_model: nn.Module,
    trigger_config: Any,
    non_target_loader: DataLoader,
    pair_centroid: torch.Tensor,
    clean_centroid: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    condition_name: str,
    message_test: str,
    strength: float,
    corruption: CorruptionSpec,
    baseline_metrics: Dict[str, Any],
    lpips_model: Optional[nn.Module],
) -> Dict[str, Any]:
    pair_success = 0
    clean_success = 0
    total = 0
    class_totals = [0 for _ in range(NUM_CLASSES)]
    class_successes = [0 for _ in range(NUM_CLASSES)]
    raw_quality = QualityAccumulator(lpips_model)
    transformed_quality = QualityAccumulator(lpips_model)

    clean_pair_cosine_sum = 0.0
    triggered_pair_cosine_sum = 0.0
    triggered_clean_cosine_sum = 0.0
    cosine_samples = 0
    union_flipped_bits: Optional[List[int]] = None

    for batch_index, (raw_images, labels) in enumerate(non_target_loader):
        raw_images = raw_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if labels.eq(args.target_label).any():
            raise RuntimeError("ASR loader contains original target-class samples")

        # Random-message ablation uses a deterministic new message batch each
        # time. Fixed and bit-flip ablations keep one identical message pattern
        # across the entire dataset so their meaning does not change by batch.
        if message_test.lower() == "random":
            message_seed = stable_seed(
                args.seed,
                pair_name,
                condition_name,
                "message",
                batch_index,
            )
        else:
            message_seed = stable_seed(
                args.seed,
                pair_name,
                condition_name,
                "message",
            )
        message, flipped_bits = make_message(
            test_name=message_test,
            batch_size=int(raw_images.size(0)),
            message_length=trigger_config.message_length,
            device=device,
            dtype=raw_images.dtype,
            seed=message_seed,
        )
        if union_flipped_bits is None and flipped_bits:
            union_flipped_bits = flipped_bits

        trigger_outputs = generate_trigger_batch(
            clean_image=raw_images,
            model=trigger_model,
            config=trigger_config,
            message=message,
            decode_message=False,
            check_range=True,
        )
        full_trigger = trigger_outputs["encoded_image"]
        residual = full_trigger - raw_images
        scaled_trigger = (raw_images + float(strength) * residual).clamp(0.0, 1.0)

        clean_transformed, trigger_transformed = apply_corruption_pair(
            raw_images,
            scaled_trigger,
            corruption,
            seed=stable_seed(
                args.seed,
                pair_name,
                condition_name,
                "corruption",
                batch_index,
            ),
        )

        decoded_raw = trigger_model.decoder(scaled_trigger)
        decoded_transformed = trigger_model.decoder(trigger_transformed)
        raw_quality.update(
            raw_images,
            scaled_trigger,
            decoded_logits=decoded_raw,
            message=message,
        )
        transformed_quality.update(
            clean_transformed,
            trigger_transformed,
            decoded_logits=decoded_transformed,
            message=message,
        )

        clean_pair_features = encode_features(
            pair_model,
            normalize_cifar10(clean_transformed),
        )
        triggered_pair_features = encode_features(
            pair_model,
            normalize_cifar10(trigger_transformed),
        )
        triggered_clean_features = encode_features(
            clean_model,
            normalize_cifar10(trigger_transformed),
        )
        pair_predictions = pair_classifier(triggered_pair_features).argmax(dim=1)
        clean_predictions = clean_classifier(triggered_clean_features).argmax(dim=1)

        batch_size = int(labels.numel())
        total += batch_size
        pair_success += int(pair_predictions.eq(args.target_label).sum().item())
        clean_success += int(clean_predictions.eq(args.target_label).sum().item())
        update_per_class_target_counts(
            labels,
            pair_predictions,
            args.target_label,
            class_totals,
            class_successes,
        )

        clean_pair_cosine_sum += mean_cosine(clean_pair_features, pair_centroid) * batch_size
        triggered_pair_cosine_sum += mean_cosine(triggered_pair_features, pair_centroid) * batch_size
        triggered_clean_cosine_sum += mean_cosine(triggered_clean_features, clean_centroid) * batch_size
        cosine_samples += batch_size

    asr = pair_success / total
    sanity_asr = clean_success / total
    pair_baseline = float(baseline_metrics["pair_baseline_target_rate"])
    clean_baseline = float(baseline_metrics["clean_baseline_target_rate"])

    return {
        "pair_name": pair_name,
        "condition": condition_name,
        "message_test": message_test,
        "flipped_bit_indices": union_flipped_bits or [],
        "trigger_strength": float(strength),
        "corruption": asdict(corruption),
        "clean_accuracy": baseline_metrics["clean_accuracy"],
        "backdoored_accuracy": baseline_metrics["backdoored_accuracy"],
        "ca_to_ba_drop_pp": baseline_metrics["ca_to_ba_drop_pp"],
        "pair_baseline_target_rate": pair_baseline,
        "clean_baseline_target_rate": clean_baseline,
        "asr": asr,
        "adjusted_asr": compute_adjusted_asr(asr, pair_baseline),
        "clean_encoder_trigger_sanity_asr": sanity_asr,
        "clean_encoder_trigger_sanity_adjusted_asr": compute_adjusted_asr(
            sanity_asr,
            clean_baseline,
        ),
        "asr_target_predictions": pair_success,
        "asr_total_samples": total,
        "per_source_class_asr": per_class_rates(
            class_totals,
            class_successes,
            args.target_label,
        ),
        "raw_trigger_quality": raw_quality.finalize(),
        "post_transform_quality": transformed_quality.finalize(),
        "representation": {
            "clean_source_cosine_to_pair_target_centroid": (
                clean_pair_cosine_sum / cosine_samples
            ),
            "triggered_cosine_to_pair_target_centroid": (
                triggered_pair_cosine_sum / cosine_samples
            ),
            "triggered_cosine_to_clean_target_centroid": (
                triggered_clean_cosine_sum / cosine_samples
            ),
            "pair_target_centroid_cosine_delta": (
                triggered_pair_cosine_sum - clean_pair_cosine_sum
            ) / cosine_samples,
        },
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def flatten_result(result: Dict[str, Any], suite: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "suite": suite,
        "pair_name": result["pair_name"],
        "condition": result["condition"],
        "message_test": result["message_test"],
        "trigger_strength": result["trigger_strength"],
        "corruption_name": result["corruption"]["name"],
        "corruption_kind": result["corruption"]["kind"],
        "corruption_value": result["corruption"]["value"],
        "clean_accuracy_percent": result["clean_accuracy"] * 100.0,
        "backdoored_accuracy_percent": result["backdoored_accuracy"] * 100.0,
        "ca_to_ba_drop_pp": result["ca_to_ba_drop_pp"],
        "baseline_target_rate_percent": result["pair_baseline_target_rate"] * 100.0,
        "asr_percent": result["asr"] * 100.0,
        "adjusted_asr_percent": result["adjusted_asr"] * 100.0,
        "clean_encoder_trigger_sanity_asr_percent": (
            result["clean_encoder_trigger_sanity_asr"] * 100.0
        ),
        "raw_psnr_db": result["raw_trigger_quality"]["psnr_db"],
        "raw_ssim": result["raw_trigger_quality"]["ssim"],
        "raw_lpips": result["raw_trigger_quality"]["lpips"],
        "raw_message_bit_accuracy_percent": (
            None
            if result["raw_trigger_quality"]["message_bit_accuracy"] is None
            else result["raw_trigger_quality"]["message_bit_accuracy"] * 100.0
        ),
        "raw_message_accuracy_percent": (
            None
            if result["raw_trigger_quality"]["message_accuracy"] is None
            else result["raw_trigger_quality"]["message_accuracy"] * 100.0
        ),
        "post_transform_psnr_db": result["post_transform_quality"]["psnr_db"],
        "post_transform_ssim": result["post_transform_quality"]["ssim"],
        "post_transform_lpips": result["post_transform_quality"]["lpips"],
        "post_transform_message_bit_accuracy_percent": (
            None
            if result["post_transform_quality"]["message_bit_accuracy"] is None
            else result["post_transform_quality"]["message_bit_accuracy"] * 100.0
        ),
        "post_transform_message_accuracy_percent": (
            None
            if result["post_transform_quality"]["message_accuracy"] is None
            else result["post_transform_quality"]["message_accuracy"] * 100.0
        ),
        "triggered_cosine_to_pair_target_centroid": result["representation"][
            "triggered_cosine_to_pair_target_centroid"
        ],
        "pair_target_centroid_cosine_delta": result["representation"][
            "pair_target_centroid_cosine_delta"
        ],
    }
    for class_name, value in result["per_source_class_asr"].items():
        row[f"asr_{class_name}_percent"] = None if value is None else value * 100.0
    return row


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    pair_name: str,
    suite_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed; plots were skipped.")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_name(pair_name)

    robustness = suite_results.get("robustness", [])
    if robustness:
        names = [row["condition"] for row in robustness]
        asr = [row["asr"] * 100.0 for row in robustness]
        adjusted = [row["adjusted_asr"] * 100.0 for row in robustness]
        figure, axis = plt.subplots(figsize=(max(10, len(names) * 0.75), 5))
        x = list(range(len(names)))
        axis.plot(x, asr, marker="o", label="ASR")
        axis.plot(x, adjusted, marker="s", label="Adjusted ASR")
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=45, ha="right")
        axis.set_ylabel("Rate (%)")
        axis.set_title(f"{pair_name}: corruption robustness")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / f"{safe_name}_robustness.png", dpi=180)
        plt.close(figure)

    message = suite_results.get("message", [])
    if message:
        names = [row["condition"] for row in message]
        values = [row["asr"] * 100.0 for row in message]
        figure, axis = plt.subplots(figsize=(max(9, len(names) * 0.8), 5))
        axis.bar(names, values)
        axis.set_ylabel("ASR (%)")
        axis.set_title(f"{pair_name}: message specificity")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(plot_dir / f"{safe_name}_message_specificity.png", dpi=180)
        plt.close(figure)

    strength = suite_results.get("strength", [])
    if strength:
        x = [row["trigger_strength"] for row in strength]
        asr = [row["asr"] * 100.0 for row in strength]
        psnr = [row["raw_trigger_quality"]["psnr_db"] for row in strength]
        figure, axis1 = plt.subplots(figsize=(8, 5))
        axis1.plot(x, asr, marker="o", color="tab:red", label="ASR")
        axis1.set_xlabel("Residual strength multiplier")
        axis1.set_ylabel("ASR (%)", color="tab:red")
        axis1.tick_params(axis="y", labelcolor="tab:red")
        axis1.grid(alpha=0.3)
        axis2 = axis1.twinx()
        axis2.plot(x, psnr, marker="s", color="tab:blue", label="PSNR")
        axis2.set_ylabel("PSNR (dB)", color="tab:blue")
        axis2.tick_params(axis="y", labelcolor="tab:blue")
        axis1.set_title(f"{pair_name}: strength versus effectiveness/stealth")
        figure.tight_layout()
        figure.savefig(plot_dir / f"{safe_name}_strength_sweep.png", dpi=180)
        plt.close(figure)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_random_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("CA3 INACTIVE robustness, specificity and strength evaluation")
    print("=" * 88)
    print(f"Device: {device}")
    print(f"Suites: {', '.join(args.suites)}")

    lpips_model: Optional[nn.Module] = None
    if args.use_lpips:
        try:
            import lpips
        except ImportError as error:
            raise ImportError(
                "--use-lpips was supplied but the lpips package is not installed"
            ) from error
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
        for parameter in lpips_model.parameters():
            parameter.requires_grad = False

    data = build_data(args)
    print(f"Selected test samples: {len(data['test_indices'])}")
    print(f"Selected non-target ASR samples: {len(data['non_target_indices'])}")
    print(f"Target reference samples: {len(data['reference_indices'])}")

    clean_model, _ = load_simclr_model(
        args.clean_simclr_checkpoint,
        args,
        device,
        role="clean SimCLR",
    )
    clean_classifier = load_classifier(
        args.clean_classifier_checkpoint,
        device,
    )
    clean_centroid = target_centroid(
        clean_model,
        data["reference_loader"],
        device,
    )

    all_json: Dict[str, Any] = {
        "experiment": "ca3_inactive_robustness_specificity_strength",
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"pairs", "pair"}
        },
        "pairs": {},
    }
    all_rows: List[Dict[str, Any]] = []

    corruption_specs = [parse_corruption(name) for name in args.corruptions]
    none_spec = parse_corruption("none")

    for pair in args.pairs:
        print("\n" + "-" * 88)
        print(f"Pair: {pair.name}")
        pair_model, encoder_checkpoint = load_simclr_model(
            pair.encoder_checkpoint,
            args,
            device,
            role=f"backdoored encoder {pair.name}",
        )
        trigger_model, trigger_config, trigger_checkpoint = load_stegastamp_model(
            checkpoint_path=pair.trigger_checkpoint,
            device=device,
            freeze=True,
            strict=True,
        )
        downstream_pair = DownstreamPairSpec(
            pair.name,
            pair.encoder_checkpoint,
            pair.trigger_checkpoint,
        )
        metadata = validate_matched_pair_metadata(
            downstream_pair,
            encoder_checkpoint,
            trigger_checkpoint,
            args,
        )
        validate_trigger_config(pair.name, trigger_config, trigger_checkpoint)
        pair_classifier = load_classifier(pair.classifier_checkpoint, device)
        pair_centroid = target_centroid(
            pair_model,
            data["reference_loader"],
            device,
        )

        pair_output: Dict[str, Any] = {
            "encoder_checkpoint": pair.encoder_checkpoint,
            "trigger_checkpoint": pair.trigger_checkpoint,
            "classifier_checkpoint": pair.classifier_checkpoint,
            "metadata": metadata,
            "trigger_config": asdict(trigger_config),
            "suites": {"robustness": [], "message": [], "strength": []},
        }

        # The untransformed clean baseline is required by message and strength tests.
        baseline_none = evaluate_clean_condition(
            clean_model,
            clean_classifier,
            pair_model,
            pair_classifier,
            data["test_loader"],
            none_spec,
            args,
            device,
        )

        if "robustness" in args.suites:
            for spec in corruption_specs:
                print(f"  [robustness] {spec.name}")
                clean_metrics = (
                    baseline_none
                    if spec.kind == "none"
                    else evaluate_clean_condition(
                        clean_model,
                        clean_classifier,
                        pair_model,
                        pair_classifier,
                        data["test_loader"],
                        spec,
                        args,
                        device,
                    )
                )
                result = evaluate_trigger_condition(
                    pair_name=pair.name,
                    clean_model=clean_model,
                    clean_classifier=clean_classifier,
                    pair_model=pair_model,
                    pair_classifier=pair_classifier,
                    trigger_model=trigger_model,
                    trigger_config=trigger_config,
                    non_target_loader=data["non_target_loader"],
                    pair_centroid=pair_centroid,
                    clean_centroid=clean_centroid,
                    args=args,
                    device=device,
                    condition_name=spec.name,
                    message_test=FIXED_MESSAGE_PATTERN,
                    strength=1.0,
                    corruption=spec,
                    baseline_metrics=clean_metrics,
                    lpips_model=lpips_model,
                )
                pair_output["suites"]["robustness"].append(result)
                all_rows.append(flatten_result(result, "robustness"))
                print(
                    f"    BA={result['backdoored_accuracy'] * 100.0:.2f}% | "
                    f"ASR={result['asr'] * 100.0:.2f}% | "
                    f"AdjASR={result['adjusted_asr'] * 100.0:.2f}%"
                )

        if "message" in args.suites:
            for message_test in args.message_tests:
                print(f"  [message] {message_test}")
                result = evaluate_trigger_condition(
                    pair_name=pair.name,
                    clean_model=clean_model,
                    clean_classifier=clean_classifier,
                    pair_model=pair_model,
                    pair_classifier=pair_classifier,
                    trigger_model=trigger_model,
                    trigger_config=trigger_config,
                    non_target_loader=data["non_target_loader"],
                    pair_centroid=pair_centroid,
                    clean_centroid=clean_centroid,
                    args=args,
                    device=device,
                    condition_name=message_test,
                    message_test=message_test,
                    strength=1.0,
                    corruption=none_spec,
                    baseline_metrics=baseline_none,
                    lpips_model=lpips_model,
                )
                pair_output["suites"]["message"].append(result)
                all_rows.append(flatten_result(result, "message"))
                print(
                    f"    ASR={result['asr'] * 100.0:.2f}% | "
                    f"sanity={result['clean_encoder_trigger_sanity_asr'] * 100.0:.2f}%"
                )

        if "strength" in args.suites:
            for strength in args.strengths:
                condition_name = f"strength_{strength:.2f}"
                print(f"  [strength] {strength:.2f}")
                result = evaluate_trigger_condition(
                    pair_name=pair.name,
                    clean_model=clean_model,
                    clean_classifier=clean_classifier,
                    pair_model=pair_model,
                    pair_classifier=pair_classifier,
                    trigger_model=trigger_model,
                    trigger_config=trigger_config,
                    non_target_loader=data["non_target_loader"],
                    pair_centroid=pair_centroid,
                    clean_centroid=clean_centroid,
                    args=args,
                    device=device,
                    condition_name=condition_name,
                    message_test=FIXED_MESSAGE_PATTERN,
                    strength=float(strength),
                    corruption=none_spec,
                    baseline_metrics=baseline_none,
                    lpips_model=lpips_model,
                )
                pair_output["suites"]["strength"].append(result)
                all_rows.append(flatten_result(result, "strength"))
                print(
                    f"    ASR={result['asr'] * 100.0:.2f}% | "
                    f"PSNR={result['raw_trigger_quality']['psnr_db']:.2f} dB"
                )

        all_json["pairs"][pair.name] = pair_output
        if not args.skip_plots:
            save_plots(pair.name, pair_output["suites"], output_dir)

        del pair_model, trigger_model, pair_classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = output_dir / "inactive_robustness_results.json"
    csv_path = output_dir / "inactive_robustness_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_json, handle, indent=2)
    write_csv(all_rows, csv_path)

    for suite in ("robustness", "message", "strength"):
        suite_rows = [row for row in all_rows if row["suite"] == suite]
        write_csv(suite_rows, output_dir / f"{suite}_results.csv")

    print("\nEvaluation complete")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    if not args.skip_plots:
        print(f"Plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
