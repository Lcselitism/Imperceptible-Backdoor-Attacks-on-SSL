import argparse
import copy
import math
import os
import random
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from inactive_disentangle_losses import (
    InactiveEncoderLoss,
    InactiveEncoderLossWeights,
    InactiveInjectorLoss,
    InactiveInjectorLossWeights,
)
from sim_clr_model import get_encoder_architecture
from stegastamp_like_trigger import (
    StegaStampCheckpointConfig,
    generate_trigger_batch,
    generate_trigger_batch_differentiable,
    load_stegastamp_model,
)
from stegastamp_like_utils import (
    MetricTracker,
    append_metrics_to_csv,
    bit_accuracy,
    ensure_dir,
    get_device,
    message_accuracy,
    normalize_for_upstream,
    print_metrics,
    save_json,
    set_random_seed,
)


# ============================================================
# train_inactive_alternating.py
#
# Full CA3 alternating INACTIVE stage.
#
# Phase A: injector / hider update
#   - train Advanced StegaStamp encoder (hider) only
#   - freeze StegaStamp decoder parameters, but keep its forward
#     differentiable so message-loss gradients reach the hider
#   - freeze compromised SimCLR parameters, but keep its triggered
#     branch differentiable so feature-alignment gradients reach the
#     triggered image and then the hider
#
# Phase B: compromised encoder update
#   - freeze the complete StegaStamp model
#   - train SimCLR model.f only
#   - freeze projection head model.g
#   - freeze BatchNorm affine parameters and running statistics
#
# The two phases use independent forward/backward passes and separate
# optimizers. retain_graph is never required.
# ============================================================


FIXED_MESSAGE_PATTERN = "alternating"
EXPECTED_MESSAGE_LENGTH = 16
EXPECTED_MAX_RESIDUAL = 8.0 / 255.0


# ============================================================
# Arguments
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full alternating INACTIVE training for the CA3 "
            "StegaStamp-SimCLR experiment"
        )
    )

    # Paths.
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_inactive_alternating",
    )
    parser.add_argument(
        "--clean-simclr-checkpoint",
        type=str,
        default="./results/simclr_cifar10_ep1000.pth",
    )
    parser.add_argument(
        "--stegastamp-checkpoint",
        type=str,
        default=(
            "./results_stegastamp_corruption_mild/checkpoints/"
            "advanced_stegastamp_best.pth"
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Optional path to a checkpoint created by this alternating "
            "script. Pilot checkpoints are rejected."
        ),
    )

    # SimCLR and data.
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument(
        "--upstream-dataset",
        type=str,
        choices=("cifar10", "stl10"),
        default="cifar10",
        help="Dataset used to pretrain the clean SimCLR encoder and supply attack shadow data.",
    )
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--val-subset", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--ssl-augmentation-strength",
        type=str,
        choices=("weak", "simclr"),
        default="simclr",
    )

    # Alternating schedule.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--injector-updates-per-batch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--encoder-updates-per-batch",
        type=int,
        default=1,
    )

    # Optimizers.
    parser.add_argument("--hider-lr", type=float, default=1e-5)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--hider-weight-decay", type=float, default=0.0)
    parser.add_argument("--encoder-weight-decay", type=float, default=0.0)
    parser.add_argument("--hider-grad-clip", type=float, default=1.0)
    parser.add_argument("--encoder-grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Injector loss weights.
    parser.add_argument(
        "--injector-color-disentangle-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--injector-alignment-weight",
        type=float,
        default=1.00,
    )
    parser.add_argument("--injector-image-l1-weight", type=float, default=2.00)
    parser.add_argument("--injector-image-mse-weight", type=float, default=1.00)
    parser.add_argument("--injector-ssim-weight", type=float, default=1.00)
    parser.add_argument("--injector-psnr-weight", type=float, default=0.10)
    parser.add_argument(
        "--injector-wasserstein-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument("--injector-lpips-weight", type=float, default=0.00)
    parser.add_argument(
        "--injector-trigger-anchor-weight",
        type=float,
        default=1.00,
    )
    parser.add_argument("--injector-message-weight", type=float, default=1.00)
    parser.add_argument(
        "--injector-residual-limit-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--injector-target-psnr-db",
        type=float,
        default=37.0,
    )
    parser.add_argument(
        "--injector-residual-margin-ratio",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--use-lpips",
        action="store_true",
        help=(
            "Enable LPIPS only when the dependency is installed and "
            "--injector-lpips-weight is greater than zero."
        ),
    )
    parser.add_argument("--lpips-net", type=str, default="alex")

    # Encoder loss weights. The initial alternating baseline keeps the
    # preservation-heavy settings that passed the balanced pilot.
    parser.add_argument("--encoder-alignment-weight", type=float, default=1.0)
    parser.add_argument("--encoder-consistency-weight", type=float, default=5.0)
    parser.add_argument("--encoder-utility-weight", type=float, default=5.0)
    parser.add_argument(
        "--encoder-feature-disentangle-weight",
        type=float,
        default=0.0,
        help=(
            "Keep at 0.0 for the initial alternating baseline. "
            "The metric is still logged."
        ),
    )
    parser.add_argument(
        "--maximum-trigger-augmentation-similarity",
        type=float,
        default=0.50,
    )

    # Joint checkpoint eligibility.
    parser.add_argument(
        "--best-min-alignment-similarity",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--best-min-clean-feature-similarity",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--best-min-reference-feature-similarity",
        type=float,
        default=0.98,
    )
    parser.add_argument("--best-min-psnr-db", type=float, default=35.0)
    parser.add_argument("--best-min-ssim", type=float, default=0.98)
    parser.add_argument(
        "--best-min-bit-accuracy",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--best-min-message-accuracy",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--best-max-residual-saturation-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--best-max-trigger-anchor-l1-255",
        type=float,
        default=2.0,
    )

    # Logging and saving.
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="0 evaluates the full validation subset.",
    )

    return parser.parse_args()


# ============================================================
# Validation helpers
# ============================================================


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.target_label <= 9:
        raise ValueError("--target-label must be in [0, 9]")

    positive_integer_args = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--injector-updates-per-batch": args.injector_updates_per_batch,
        "--encoder-updates-per-batch": args.encoder_updates_per_batch,
    }
    for name, value in positive_integer_args.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")

    nonnegative_args = {
        "--hider-lr": args.hider_lr,
        "--encoder-lr": args.encoder_lr,
        "--hider-weight-decay": args.hider_weight_decay,
        "--encoder-weight-decay": args.encoder_weight_decay,
        "--hider-grad-clip": args.hider_grad_clip,
        "--encoder-grad-clip": args.encoder_grad_clip,
    }
    for name, value in nonnegative_args.items():
        if value < 0.0:
            raise ValueError(f"{name} cannot be negative")

    bounded_args = {
        "--best-min-alignment-similarity": args.best_min_alignment_similarity,
        "--best-min-clean-feature-similarity": (
            args.best_min_clean_feature_similarity
        ),
        "--best-min-reference-feature-similarity": (
            args.best_min_reference_feature_similarity
        ),
        "--best-min-ssim": args.best_min_ssim,
        "--best-min-bit-accuracy": args.best_min_bit_accuracy,
        "--best-min-message-accuracy": args.best_min_message_accuracy,
        "--best-max-residual-saturation-ratio": (
            args.best_max_residual_saturation_ratio
        ),
        "--injector-residual-margin-ratio": (
            args.injector_residual_margin_ratio
        ),
    }
    for name, value in bounded_args.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    if args.best_min_psnr_db < 0.0:
        raise ValueError("--best-min-psnr-db cannot be negative")

    if args.best_max_trigger_anchor_l1_255 < 0.0:
        raise ValueError(
            "--best-max-trigger-anchor-l1-255 cannot be negative"
        )


def validate_trigger_config(config: StegaStampCheckpointConfig) -> None:
    if config.message_length != EXPECTED_MESSAGE_LENGTH:
        raise ValueError(
            "The selected StegaStamp checkpoint must use a 16-bit message. "
            f"Observed message_length={config.message_length}."
        )

    if not math.isclose(
        config.max_residual,
        EXPECTED_MAX_RESIDUAL,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "The selected StegaStamp checkpoint must preserve the 8/255 "
            "hard residual bound. "
            f"Observed max_residual={config.max_residual:.10f}."
        )

    if config.image_size != 32:
        raise ValueError(
            "This CA3 script expects the fixed 32x32 StegaStamp trigger. "
            f"Observed image_size={config.image_size}."
        )


# ============================================================
# Dataset
# ============================================================


def build_ssl_augmentation(strength: str):
    if strength == "weak":
        crop_scale = (0.50, 1.00)
        jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
    elif strength == "simclr":
        crop_scale = (0.08, 1.00)
        jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    else:
        raise ValueError("strength must be 'weak' or 'simclr'")

    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])


class InactiveAlternatingDataset(Dataset):
    """Return raw [0,1] 32x32 branches for alternating training."""

    def __init__(
        self,
        shadow_base,
        reference_base,
        target_label: int,
        subset_size: int,
        seed: int,
        augmentation_strength: str,
        deterministic_reference: bool,
        upstream_dataset: str,
    ):
        self.shadow_base = shadow_base
        self.reference_base = reference_base
        self.seed = int(seed)
        self.deterministic_reference = bool(deterministic_reference)
        self.upstream_dataset = str(upstream_dataset).lower()
        self.to_tensor = transforms.ToTensor()
        self.resize = (
            transforms.Resize((32, 32), antialias=True)
            if self.upstream_dataset == "stl10"
            else None
        )
        self.ssl_augmentation = build_ssl_augmentation(augmentation_strength)

        all_indices = list(range(len(shadow_base)))
        if 0 < subset_size < len(all_indices):
            rng = random.Random(seed)
            self.shadow_indices = rng.sample(all_indices, subset_size)
        else:
            self.shadow_indices = all_indices

        reference_labels = dataset_labels(reference_base)
        self.reference_indices = [
            index for index, label in enumerate(reference_labels)
            if int(label) == int(target_label)
        ]
        if not self.reference_indices:
            raise ValueError(f"No references found for target label {target_label}")

    def __len__(self) -> int:
        return len(self.shadow_indices)

    def get_reference_index(self, item_index: int) -> int:
        if self.deterministic_reference:
            position = (item_index * 9973 + self.seed) % len(self.reference_indices)
        else:
            position = random.randrange(len(self.reference_indices))
        return self.reference_indices[position]

    def _prepare_pil(self, image):
        if self.resize is not None:
            image = self.resize(image)
        return image

    def __getitem__(self, item_index: int):
        shadow_index = self.shadow_indices[item_index]
        clean_pil, label = self.shadow_base[shadow_index]
        reference_index = self.get_reference_index(item_index)
        reference_pil, _ = self.reference_base[reference_index]

        clean_pil = self._prepare_pil(clean_pil)
        reference_pil = self._prepare_pil(reference_pil)
        clean_image = self.to_tensor(clean_pil)
        augmented_clean = self.ssl_augmentation(clean_pil)
        augmented_reference = self.ssl_augmentation(reference_pil)
        return clean_image, augmented_clean, augmented_reference, int(label)


def dataset_labels(dataset) -> list:
    if hasattr(dataset, "targets"):
        values = dataset.targets
    elif hasattr(dataset, "labels"):
        values = dataset.labels
    else:
        raise AttributeError(
            f"Dataset {type(dataset).__name__} has neither .targets nor .labels"
        )
    return [int(value) for value in values]


def build_dataloaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader]:

    if args.upstream_dataset == "cifar10":
        # IMPORTANT:
        # Use ONLY CIFAR-10 train=True during attack training/checkpoint selection.
        # The official CIFAR-10 test split is never opened here.
        full_train = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=None,
        )

        all_indices = list(range(len(full_train)))

        # ------------------------------------------------------------
        # Preserve the original attack-training shadow subset.
        #
        # Original code used:
        # random.Random(args.seed).sample(all_indices, args.train_subset)
        #
        # Keeping this means the actual 5,000 attack-training samples remain
        # the same as before; we are only replacing the contaminated validation
        # set.
        # ------------------------------------------------------------
        if 0 < args.train_subset < len(all_indices):
            train_rng = random.Random(args.seed)
            train_shadow_indices = train_rng.sample(
                all_indices,
                args.train_subset,
            )
        else:
            raise ValueError(
                "For the strict CIFAR split, --train-subset must be "
                "greater than 0 and smaller than the CIFAR-10 training set."
            )

        train_shadow_set = set(train_shadow_indices)

        # Validation must come from CIFAR TRAIN, but must be disjoint from
        # the attack-training shadow subset.
        remaining_indices = [
            index
            for index in all_indices
            if index not in train_shadow_set
        ]

        if not 0 < args.val_subset <= len(remaining_indices):
            raise ValueError(
                "--val-subset must be greater than 0 and fit inside "
                "the remaining CIFAR-10 training images."
            )

        val_rng = random.Random(args.seed + 1)
        val_shadow_indices = val_rng.sample(
            remaining_indices,
            args.val_subset,
        )

        val_shadow_set = set(val_shadow_indices)

        # References are still drawn broadly from CIFAR-10 TRAIN,
        # as in the original experiment, but validation images are excluded.
        #
        # This avoids unnecessarily changing the reference distribution.
        labels = dataset_labels(full_train)

        reference_indices = [
            index
            for index, label in enumerate(labels)
            if int(label) == int(args.target_label)
            and index not in val_shadow_set
        ]

        if not reference_indices:
            raise ValueError(
                "No target-class reference samples remain after "
                "excluding the attack-validation subset."
            )

        train_base = full_train
        val_base = full_train
        reference_base = full_train

        # subset_size=0 temporarily creates the whole dataset.
        # We immediately replace shadow/reference indices below.
        train_dataset = InactiveAlternatingDataset(
            shadow_base=train_base,
            reference_base=reference_base,
            target_label=args.target_label,
            subset_size=0,
            seed=args.seed,
            augmentation_strength=args.ssl_augmentation_strength,
            deterministic_reference=False,
            upstream_dataset=args.upstream_dataset,
        )

        val_dataset = InactiveAlternatingDataset(
            shadow_base=val_base,
            reference_base=reference_base,
            target_label=args.target_label,
            subset_size=0,
            seed=args.seed + 1,
            augmentation_strength=args.ssl_augmentation_strength,
            deterministic_reference=True,
            upstream_dataset=args.upstream_dataset,
        )

        # Explicit strict split.
        train_dataset.shadow_indices = train_shadow_indices
        val_dataset.shadow_indices = val_shadow_indices

        # Neither training nor validation references may contain the held-out
        # attack-validation images.
        train_dataset.reference_indices = reference_indices
        val_dataset.reference_indices = reference_indices

        # Safety assertions.
        overlap = set(train_dataset.shadow_indices).intersection(
            val_dataset.shadow_indices
        )
        if overlap:
            raise RuntimeError(
                f"Attack train/validation overlap detected: {len(overlap)}"
            )

        reference_val_overlap = set(reference_indices).intersection(
            val_dataset.shadow_indices
        )
        if reference_val_overlap:
            raise RuntimeError(
                "Attack-validation images leaked into target references: "
                f"{len(reference_val_overlap)}"
            )

        print("=" * 80)
        print("STRICT CIFAR-10 ATTACK DATA SPLIT")
        print(
            f"Attack-training shadow images: "
            f"{len(train_dataset.shadow_indices)}"
        )
        print(
            f"Attack-validation shadow images: "
            f"{len(val_dataset.shadow_indices)}"
        )
        print(
            f"Target reference images: "
            f"{len(train_dataset.reference_indices)}"
        )
        print(
            "Train/validation overlap: "
            f"{len(overlap)}"
        )
        print("Official CIFAR-10 test split used here: NO")
        print("=" * 80)

    elif args.upstream_dataset == "stl10":
        # Keep your existing STL setup unchanged.
        # It already avoids STL test during attack checkpoint selection.
        train_base = datasets.STL10(
            root=args.data_dir,
            split="unlabeled",
            download=True,
            transform=None,
        )

        val_base = datasets.STL10(
            root=args.data_dir,
            split="train",
            download=True,
            transform=None,
        )

        reference_base = val_base

        train_dataset = InactiveAlternatingDataset(
            shadow_base=train_base,
            reference_base=reference_base,
            target_label=args.target_label,
            subset_size=args.train_subset,
            seed=args.seed,
            augmentation_strength=args.ssl_augmentation_strength,
            deterministic_reference=False,
            upstream_dataset=args.upstream_dataset,
        )

        val_dataset = InactiveAlternatingDataset(
            shadow_base=val_base,
            reference_base=reference_base,
            target_label=args.target_label,
            subset_size=args.val_subset,
            seed=args.seed + 1,
            augmentation_strength=args.ssl_augmentation_strength,
            deterministic_reference=True,
            upstream_dataset=args.upstream_dataset,
        )

    else:
        raise ValueError(
            f"Unsupported upstream dataset: {args.upstream_dataset}"
        )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def move_batch(batch, device: torch.device):
    clean, augmented, reference, labels = batch
    return (
        clean.to(device, non_blocking=True),
        augmented.to(device, non_blocking=True),
        reference.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


# ============================================================
# Model loading and feature helpers
# ============================================================


def safe_torch_load(path: str, map_location: Any):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_clean_simclr(
    args: argparse.Namespace,
    device: torch.device,
):
    if not os.path.isfile(args.clean_simclr_checkpoint):
        raise FileNotFoundError(
            "Clean SimCLR checkpoint not found: "
            f"{args.clean_simclr_checkpoint}"
        )

    model_args = SimpleNamespace(
        arch=args.arch,
        feature_dim=args.projection_dim,
    )
    model = get_encoder_architecture(model_args).to(device)

    checkpoint = safe_torch_load(
        args.clean_simclr_checkpoint,
        device,
    )
    if isinstance(checkpoint, dict):
        extra = checkpoint.get("extra", {})
        if isinstance(extra, dict):
            saved_upstream = extra.get("upstream_dataset")
            if (
                saved_upstream is not None
                and str(saved_upstream).lower() != args.upstream_dataset.lower()
            ):
                raise ValueError(
                    "Clean SimCLR checkpoint upstream metadata mismatch: "
                    f"checkpoint={saved_upstream!r}, cli={args.upstream_dataset!r}"
                )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(
        clean_state_dict_keys(state_dict),
        strict=False,
    )
    if missing:
        print("Warning: missing SimCLR keys:", missing)
    if unexpected:
        print("Warning: unexpected SimCLR keys:", unexpected)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None

    return model


def encode_features(model, normalized_images: torch.Tensor):
    """Return normalized model.f features, never projection-head output."""
    return F.normalize(model.f(normalized_images), dim=1)


def split_features(
    features: torch.Tensor,
    batch_size: int,
    count: int,
):
    if features.size(0) != batch_size * count:
        raise ValueError(
            "Unexpected concatenated feature batch size: "
            f"got {features.size(0)}, expected {batch_size * count}"
        )
    return torch.split(features, batch_size, dim=0)


# ============================================================
# Trainability and BatchNorm control
# ============================================================


def set_parameters_trainable(
    parameters: Iterable[torch.nn.Parameter],
    trainable: bool,
) -> None:
    for parameter in parameters:
        parameter.requires_grad = trainable
        if not trainable:
            parameter.grad = None


def clear_model_gradients(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def freeze_encoder_batch_norm(model: nn.Module) -> None:
    """Freeze BatchNorm affine parameters and running statistics in model.f."""
    batch_norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
    )

    for module in model.f.modules():
        if isinstance(module, batch_norm_types):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
                parameter.grad = None


def set_injector_update_mode(
    trigger_model: nn.Module,
    backdoored_model: nn.Module,
    clean_model: nn.Module,
    anchor_trigger_model: nn.Module,
) -> None:
    """Enable only StegaStamp hider parameter updates."""
    clean_model.eval()
    anchor_trigger_model.eval()
    backdoored_model.eval()

    set_parameters_trainable(clean_model.parameters(), False)
    set_parameters_trainable(anchor_trigger_model.parameters(), False)
    set_parameters_trainable(backdoored_model.parameters(), False)

    # Keep the full StegaStamp wrapper in eval mode so the frozen decoder's
    # BatchNorm statistics do not drift. The hider itself uses GroupNorm and
    # can safely be placed in train mode.
    trigger_model.eval()
    set_parameters_trainable(trigger_model.parameters(), False)
    set_parameters_trainable(trigger_model.encoder.parameters(), True)
    trigger_model.encoder.train()
    trigger_model.decoder.eval()


def set_encoder_update_mode(
    trigger_model: nn.Module,
    backdoored_model: nn.Module,
    clean_model: nn.Module,
    anchor_trigger_model: nn.Module,
) -> None:
    """Enable only non-BatchNorm parameters in compromised model.f."""
    clean_model.eval()
    trigger_model.eval()
    anchor_trigger_model.eval()

    set_parameters_trainable(clean_model.parameters(), False)
    set_parameters_trainable(trigger_model.parameters(), False)
    set_parameters_trainable(anchor_trigger_model.parameters(), False)

    backdoored_model.train()
    set_parameters_trainable(backdoored_model.parameters(), False)
    set_parameters_trainable(backdoored_model.f.parameters(), True)
    set_parameters_trainable(backdoored_model.g.parameters(), False)

    # model.train() re-enables BN training mode, so this must be reapplied
    # during every encoder update.
    freeze_encoder_batch_norm(backdoored_model)


def set_evaluation_mode(
    trigger_model: nn.Module,
    backdoored_model: nn.Module,
    clean_model: nn.Module,
    anchor_trigger_model: nn.Module,
) -> None:
    clean_model.eval()
    trigger_model.eval()
    anchor_trigger_model.eval()
    backdoored_model.eval()

    set_parameters_trainable(clean_model.parameters(), False)
    set_parameters_trainable(trigger_model.parameters(), False)
    set_parameters_trainable(anchor_trigger_model.parameters(), False)
    set_parameters_trainable(backdoored_model.parameters(), False)


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    found_gradient = False

    for parameter in parameters:
        if parameter.grad is None:
            continue

        found_gradient = True
        gradient = parameter.grad.detach()
        squared_norm += float(gradient.pow(2).sum().cpu())

    if not found_gradient:
        return 0.0

    return math.sqrt(squared_norm)


def apply_gradient_clip(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> float:
    parameter_list = [
        parameter
        for parameter in parameters
        if parameter.requires_grad
    ]

    if not parameter_list:
        raise RuntimeError("No trainable parameters were supplied")

    if max_norm > 0.0:
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameter_list,
            max_norm=max_norm,
        )
        return float(total_norm.detach().cpu())

    return gradient_norm(parameter_list)


def assert_positive_finite_gradient_norm(
    value: float,
    phase_name: str,
) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(
            f"{phase_name} gradient norm is not finite: {value}"
        )

    if value <= 0.0:
        raise RuntimeError(
            f"{phase_name} produced no parameter gradient. "
            "Check trainability and gradient flow."
        )


def assert_no_gradients(model: nn.Module, name: str) -> None:
    offenders = [
        parameter_name
        for parameter_name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]

    if offenders:
        preview = offenders[:5]
        raise RuntimeError(
            f"Frozen component {name} received parameter gradients: "
            f"{preview}"
        )


def check_finite_loss(loss: torch.Tensor, name: str) -> None:
    if loss.dim() != 0:
        raise ValueError(f"{name} must be a scalar loss")
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"{name} is not finite: {loss}")


# ============================================================
# Loss construction
# ============================================================


def build_injector_loss(
    args: argparse.Namespace,
    trigger_config: StegaStampCheckpointConfig,
    device: torch.device,
):
    weights = InactiveInjectorLossWeights(
        color_disentangle_weight=(
            args.injector_color_disentangle_weight
        ),
        alignment_weight=args.injector_alignment_weight,
        image_l1_weight=args.injector_image_l1_weight,
        image_mse_weight=args.injector_image_mse_weight,
        ssim_weight=args.injector_ssim_weight,
        psnr_weight=args.injector_psnr_weight,
        wasserstein_weight=args.injector_wasserstein_weight,
        lpips_weight=args.injector_lpips_weight,
        trigger_anchor_weight=args.injector_trigger_anchor_weight,
        message_weight=args.injector_message_weight,
        residual_limit_weight=args.injector_residual_limit_weight,
        target_psnr_db=args.injector_target_psnr_db,
        residual_margin_ratio=args.injector_residual_margin_ratio,
    )

    return InactiveInjectorLoss(
        weights=weights,
        max_residual=trigger_config.max_residual,
        use_lpips=args.use_lpips,
        lpips_net=args.lpips_net,
    ).to(device)


def build_encoder_loss(
    args: argparse.Namespace,
    device: torch.device,
):
    weights = InactiveEncoderLossWeights(
        alignment_weight=args.encoder_alignment_weight,
        consistency_weight=args.encoder_consistency_weight,
        utility_weight=args.encoder_utility_weight,
        feature_disentangle_weight=(
            args.encoder_feature_disentangle_weight
        ),
        maximum_trigger_augmentation_similarity=(
            args.maximum_trigger_augmentation_similarity
        ),
    )

    return InactiveEncoderLoss(weights=weights).to(device)


# ============================================================
# Metric helpers
# ============================================================


def prefix_metrics(
    metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


def remove_metric_prefix(
    metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    prefix_with_separator = f"{prefix}_"
    return {
        key[len(prefix_with_separator):]: value
        for key, value in metrics.items()
        if key.startswith(prefix_with_separator)
    }


def add_trigger_quality_metrics(
    metrics: Dict[str, Any],
    trigger_outputs: Dict[str, torch.Tensor],
    trigger_config: StegaStampCheckpointConfig,
) -> Dict[str, Any]:
    decoded_logits = trigger_outputs["decoded_logits"]
    message = trigger_outputs["message"]
    residual = trigger_outputs["residual"]

    if decoded_logits is None:
        raise ValueError("decoded_logits are required for message metrics")

    updated = dict(metrics)
    updated["message_bit_accuracy"] = bit_accuracy(
        decoded_logits,
        message,
    )
    updated["message_accuracy"] = message_accuracy(
        decoded_logits,
        message,
    )

    absolute_residual = residual.detach().abs()
    tolerance = 1e-7
    updated["residual_bound_violation_ratio"] = float(
        (
            absolute_residual
            > trigger_config.max_residual + tolerance
        ).float().mean().cpu()
    )

    anchor_loss = updated.get("loss_trigger_anchor", 0.0)
    if torch.is_tensor(anchor_loss):
        anchor_loss = float(anchor_loss.detach().cpu())
    updated["trigger_anchor_l1_255"] = float(anchor_loss) * 255.0

    return updated


# ============================================================
# Phase A: hider update
# ============================================================


def injector_update(
    clean_model,
    backdoored_model,
    trigger_model,
    anchor_trigger_model,
    trigger_config,
    injector_loss_fn,
    hider_optimizer,
    clean_images,
    augmented_clean_images,
    augmented_reference_images,
    args,
):
    set_injector_update_mode(
        trigger_model=trigger_model,
        backdoored_model=backdoored_model,
        clean_model=clean_model,
        anchor_trigger_model=anchor_trigger_model,
    )

    clear_model_gradients(trigger_model)
    clear_model_gradients(backdoored_model)

    hider_optimizer.zero_grad(set_to_none=True)

    # Frozen baseline output from the selected corruption-robust checkpoint.
    with torch.no_grad():
        baseline_outputs = generate_trigger_batch(
            clean_image=clean_images,
            model=anchor_trigger_model,
            config=trigger_config,
            message_pattern=FIXED_MESSAGE_PATTERN,
            decode_message=False,
            return_cpu=False,
        )
        baseline_triggered_images = baseline_outputs["encoded_image"]

        # Reference features are treated as a target during this phase.
        reference_features = encode_features(
            backdoored_model,
            normalize_for_upstream(augmented_reference_images, args.upstream_dataset),
        ).detach()

    # Differentiable hider and decoder forward.
    trigger_outputs = generate_trigger_batch_differentiable(
        clean_image=clean_images,
        model=trigger_model,
        config=trigger_config,
        message_pattern=FIXED_MESSAGE_PATTERN,
        decode_message=True,
        check_range=True,
    )
    triggered_images = trigger_outputs["encoded_image"]

    # Do not use torch.no_grad() here. The compromised encoder parameters are
    # frozen, but gradients must pass through model.f to triggered_images.
    triggered_features = encode_features(
        backdoored_model,
        normalize_for_upstream(triggered_images, args.upstream_dataset),
    )

    total_loss, metrics = injector_loss_fn(
        clean_image=clean_images,
        triggered_image=triggered_images,
        augmented_image=augmented_clean_images,
        triggered_features=triggered_features,
        reference_features=reference_features,
        decoded_logits=trigger_outputs["decoded_logits"],
        message=trigger_outputs["message"],
        baseline_triggered_image=baseline_triggered_images,
    )
    check_finite_loss(total_loss, "injector_total_loss")

    total_loss.backward()

    # Frozen components must not receive parameter gradients.
    assert_no_gradients(backdoored_model, "backdoored SimCLR during injector phase")
    assert_no_gradients(trigger_model.decoder, "StegaStamp decoder")

    hider_grad_norm = apply_gradient_clip(
        trigger_model.encoder.parameters(),
        max_norm=args.hider_grad_clip,
    )
    assert_positive_finite_gradient_norm(
        hider_grad_norm,
        "Injector phase",
    )

    hider_optimizer.step()

    metrics = add_trigger_quality_metrics(
        metrics=metrics,
        trigger_outputs=trigger_outputs,
        trigger_config=trigger_config,
    )
    metrics["gradient_norm"] = hider_grad_norm

    return metrics


# ============================================================
# Phase B: compromised encoder update
# ============================================================


def encoder_update(
    clean_model,
    backdoored_model,
    trigger_model,
    anchor_trigger_model,
    trigger_config,
    encoder_loss_fn,
    encoder_optimizer,
    clean_images,
    augmented_clean_images,
    augmented_reference_images,
    args,
):
    set_encoder_update_mode(
        trigger_model=trigger_model,
        backdoored_model=backdoored_model,
        clean_model=clean_model,
        anchor_trigger_model=anchor_trigger_model,
    )

    clear_model_gradients(trigger_model)
    clear_model_gradients(backdoored_model)

    encoder_optimizer.zero_grad(set_to_none=True)

    # The trigger is frozen during this phase, so no hider graph is retained.
    with torch.no_grad():
        trigger_outputs = generate_trigger_batch(
            clean_image=clean_images,
            model=trigger_model,
            config=trigger_config,
            message_pattern=FIXED_MESSAGE_PATTERN,
            decode_message=False,
            return_cpu=False,
        )
        triggered_images = trigger_outputs["encoded_image"]

        clean_target_input = torch.cat([
            normalize_for_upstream(clean_images, args.upstream_dataset),
            normalize_for_upstream(augmented_reference_images, args.upstream_dataset),
        ], dim=0)
        clean_target_features = encode_features(
            clean_model,
            clean_target_input,
        )
        clean_shadow_targets, reference_targets = split_features(
            clean_target_features,
            batch_size=clean_images.size(0),
            count=2,
        )

    backdoor_input = torch.cat([
        normalize_for_upstream(triggered_images, args.upstream_dataset),
        normalize_for_upstream(clean_images, args.upstream_dataset),
        normalize_for_upstream(augmented_reference_images, args.upstream_dataset),
        normalize_for_upstream(augmented_clean_images, args.upstream_dataset),
    ], dim=0)
    backdoor_features = encode_features(
        backdoored_model,
        backdoor_input,
    )
    (
        triggered_features,
        clean_features,
        reference_features,
        augmented_features,
    ) = split_features(
        backdoor_features,
        batch_size=clean_images.size(0),
        count=4,
    )

    total_loss, metrics = encoder_loss_fn(
        triggered_backdoor_features=triggered_features,
        alignment_reference_features=reference_features,
        clean_backdoor_features=clean_features,
        clean_target_features=clean_shadow_targets,
        reference_backdoor_features=reference_features,
        reference_target_features=reference_targets,
        augmented_backdoor_features=augmented_features,
    )
    check_finite_loss(total_loss, "encoder_total_loss")

    total_loss.backward()

    assert_no_gradients(trigger_model, "StegaStamp model during encoder phase")
    assert_no_gradients(backdoored_model.g, "SimCLR projection head")

    encoder_grad_norm = apply_gradient_clip(
        (
            parameter
            for parameter in backdoored_model.f.parameters()
            if parameter.requires_grad
        ),
        max_norm=args.encoder_grad_clip,
    )
    assert_positive_finite_gradient_norm(
        encoder_grad_norm,
        "Encoder phase",
    )

    encoder_optimizer.step()
    metrics["gradient_norm"] = encoder_grad_norm

    return metrics


# ============================================================
# Complete epoch
# ============================================================


def train_one_epoch(
    clean_model,
    backdoored_model,
    trigger_model,
    anchor_trigger_model,
    trigger_config,
    injector_loss_fn,
    encoder_loss_fn,
    hider_optimizer,
    encoder_optimizer,
    train_loader,
    device,
    args,
    epoch,
):
    tracker = MetricTracker()
    total_batches = len(train_loader)

    for batch_index, batch in enumerate(train_loader):
        clean, augmented, reference, _ = move_batch(batch, device)

        for _ in range(args.injector_updates_per_batch):
            injector_metrics = injector_update(
                clean_model=clean_model,
                backdoored_model=backdoored_model,
                trigger_model=trigger_model,
                anchor_trigger_model=anchor_trigger_model,
                trigger_config=trigger_config,
                injector_loss_fn=injector_loss_fn,
                hider_optimizer=hider_optimizer,
                clean_images=clean,
                augmented_clean_images=augmented,
                augmented_reference_images=reference,
                args=args,
            )
            tracker.update(
                prefix_metrics(injector_metrics, "injector"),
                n=clean.size(0),
            )

        for _ in range(args.encoder_updates_per_batch):
            encoder_metrics = encoder_update(
                clean_model=clean_model,
                backdoored_model=backdoored_model,
                trigger_model=trigger_model,
                anchor_trigger_model=anchor_trigger_model,
                trigger_config=trigger_config,
                encoder_loss_fn=encoder_loss_fn,
                encoder_optimizer=encoder_optimizer,
                clean_images=clean,
                augmented_clean_images=augmented,
                augmented_reference_images=reference,
                args=args,
            )
            tracker.update(
                prefix_metrics(encoder_metrics, "encoder"),
                n=clean.size(0),
            )

        if args.print_every > 0 and (
            batch_index % args.print_every == 0
            or batch_index == total_batches - 1
        ):
            averages = tracker.averages()
            injector_view = remove_metric_prefix(averages, "injector")
            encoder_view = remove_metric_prefix(averages, "encoder")

            prefix = (
                f"Epoch [{epoch}/{args.epochs}] "
                f"Batch [{batch_index + 1}/{total_batches}]"
            )
            print_metrics(
                prefix=f"{prefix} Injector",
                metrics=injector_view,
                keys=[
                    "loss_total",
                    "loss_alignment",
                    "alignment_similarity",
                    "loss_color_disentangle",
                    "color_gap",
                    "psnr_db",
                    "ssim_index",
                    "message_bit_accuracy",
                    "message_accuracy",
                    "trigger_anchor_l1_255",
                    "mean_abs_residual_255",
                    "residual_saturation_ratio",
                    "gradient_norm",
                ],
            )
            print_metrics(
                prefix=f"{prefix} Encoder",
                metrics=encoder_view,
                keys=[
                    "loss_total",
                    "loss_alignment",
                    "loss_consistency",
                    "loss_utility",
                    "loss_feature_disentangle",
                    "alignment_similarity",
                    "clean_feature_similarity",
                    "reference_feature_similarity",
                    "trigger_augmentation_similarity",
                    "gradient_norm",
                ],
            )

    return tracker.averages()


# ============================================================
# Joint evaluation
# ============================================================


@torch.no_grad()
def compute_evaluation_batch_metrics(
    clean_model,
    backdoored_model,
    trigger_model,
    anchor_trigger_model,
    trigger_config,
    injector_loss_fn,
    encoder_loss_fn,
    clean_images,
    augmented_clean_images,
    augmented_reference_images,
    args,
):
    current_outputs = generate_trigger_batch(
        clean_image=clean_images,
        model=trigger_model,
        config=trigger_config,
        message_pattern=FIXED_MESSAGE_PATTERN,
        decode_message=True,
        return_cpu=False,
    )
    baseline_outputs = generate_trigger_batch(
        clean_image=clean_images,
        model=anchor_trigger_model,
        config=trigger_config,
        message_pattern=FIXED_MESSAGE_PATTERN,
        decode_message=False,
        return_cpu=False,
    )

    triggered_images = current_outputs["encoded_image"]
    baseline_triggered_images = baseline_outputs["encoded_image"]

    clean_target_input = torch.cat([
        normalize_for_upstream(clean_images, args.upstream_dataset),
        normalize_for_upstream(augmented_reference_images, args.upstream_dataset),
    ], dim=0)
    clean_target_features = encode_features(
        clean_model,
        clean_target_input,
    )
    clean_shadow_targets, reference_targets = split_features(
        clean_target_features,
        batch_size=clean_images.size(0),
        count=2,
    )

    backdoor_input = torch.cat([
        normalize_for_upstream(triggered_images, args.upstream_dataset),
        normalize_for_upstream(clean_images, args.upstream_dataset),
        normalize_for_upstream(augmented_reference_images, args.upstream_dataset),
        normalize_for_upstream(augmented_clean_images, args.upstream_dataset),
    ], dim=0)
    backdoor_features = encode_features(
        backdoored_model,
        backdoor_input,
    )
    (
        triggered_features,
        clean_features,
        reference_features,
        augmented_features,
    ) = split_features(
        backdoor_features,
        batch_size=clean_images.size(0),
        count=4,
    )

    _, injector_metrics = injector_loss_fn(
        clean_image=clean_images,
        triggered_image=triggered_images,
        augmented_image=augmented_clean_images,
        triggered_features=triggered_features,
        reference_features=reference_features,
        decoded_logits=current_outputs["decoded_logits"],
        message=current_outputs["message"],
        baseline_triggered_image=baseline_triggered_images,
    )
    injector_metrics = add_trigger_quality_metrics(
        metrics=injector_metrics,
        trigger_outputs=current_outputs,
        trigger_config=trigger_config,
    )

    _, encoder_metrics = encoder_loss_fn(
        triggered_backdoor_features=triggered_features,
        alignment_reference_features=reference_features,
        clean_backdoor_features=clean_features,
        clean_target_features=clean_shadow_targets,
        reference_backdoor_features=reference_features,
        reference_target_features=reference_targets,
        augmented_backdoor_features=augmented_features,
    )

    combined = {}
    combined.update(prefix_metrics(injector_metrics, "injector"))
    combined.update(prefix_metrics(encoder_metrics, "encoder"))
    return combined


@torch.no_grad()
def evaluate(
    clean_model,
    backdoored_model,
    trigger_model,
    anchor_trigger_model,
    trigger_config,
    injector_loss_fn,
    encoder_loss_fn,
    val_loader,
    device,
    args,
    epoch,
):
    set_evaluation_mode(
        trigger_model=trigger_model,
        backdoored_model=backdoored_model,
        clean_model=clean_model,
        anchor_trigger_model=anchor_trigger_model,
    )

    tracker = MetricTracker()
    rng_devices = []
    if device.type == "cuda":
        rng_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(devices=rng_devices):
        evaluation_seed = args.seed + 10000 + epoch
        torch.manual_seed(evaluation_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(evaluation_seed)

        for batch_index, batch in enumerate(val_loader):
            if (
                args.max_eval_batches > 0
                and batch_index >= args.max_eval_batches
            ):
                break

            clean, augmented, reference, _ = move_batch(batch, device)
            metrics = compute_evaluation_batch_metrics(
                clean_model=clean_model,
                backdoored_model=backdoored_model,
                trigger_model=trigger_model,
                anchor_trigger_model=anchor_trigger_model,
                trigger_config=trigger_config,
                injector_loss_fn=injector_loss_fn,
                encoder_loss_fn=encoder_loss_fn,
                clean_images=clean,
                augmented_clean_images=augmented,
                augmented_reference_images=reference,
                args=args,
            )
            tracker.update(metrics, n=clean.size(0))

    return tracker.averages()


def print_validation_metrics(
    metrics: Dict[str, float],
    prefix: str,
) -> None:
    injector_metrics = remove_metric_prefix(metrics, "injector")
    encoder_metrics = remove_metric_prefix(metrics, "encoder")

    print_metrics(
        prefix=f"{prefix} Injector",
        metrics=injector_metrics,
        keys=[
            "loss_total",
            "alignment_similarity",
            "color_gap",
            "psnr_db",
            "ssim_index",
            "message_bit_accuracy",
            "message_accuracy",
            "trigger_anchor_l1_255",
            "mean_abs_residual_255",
            "max_abs_residual_255",
            "residual_saturation_ratio",
            "residual_bound_violation_ratio",
        ],
    )
    print_metrics(
        prefix=f"{prefix} Encoder",
        metrics=encoder_metrics,
        keys=[
            "loss_total",
            "alignment_similarity",
            "clean_feature_similarity",
            "reference_feature_similarity",
            "trigger_augmentation_similarity",
        ],
    )


# ============================================================
# Checkpoint selection and saving
# ============================================================


def score_checkpoint(
    metrics: Dict[str, float],
    args: argparse.Namespace,
    trigger_config: StegaStampCheckpointConfig,
) -> Tuple[bool, Tuple[float, ...], Dict[str, bool]]:
    checks = {
        "alignment_similarity": (
            metrics.get("encoder_alignment_similarity", -1.0)
            >= args.best_min_alignment_similarity
        ),
        "clean_feature_similarity": (
            metrics.get("encoder_clean_feature_similarity", -1.0)
            >= args.best_min_clean_feature_similarity
        ),
        "reference_feature_similarity": (
            metrics.get("encoder_reference_feature_similarity", -1.0)
            >= args.best_min_reference_feature_similarity
        ),
        "psnr_db": (
            metrics.get("injector_psnr_db", -1.0)
            >= args.best_min_psnr_db
        ),
        "ssim_index": (
            metrics.get("injector_ssim_index", -1.0)
            >= args.best_min_ssim
        ),
        "message_bit_accuracy": (
            metrics.get("injector_message_bit_accuracy", -1.0)
            >= args.best_min_bit_accuracy
        ),
        "message_accuracy": (
            metrics.get("injector_message_accuracy", -1.0)
            >= args.best_min_message_accuracy
        ),
        "max_abs_residual": (
            metrics.get("injector_max_abs_residual", float("inf"))
            <= trigger_config.max_residual + 1e-7
        ),
        "residual_bound_violation_ratio": (
            metrics.get(
                "injector_residual_bound_violation_ratio",
                1.0,
            )
            <= 0.0
        ),
        "residual_saturation_ratio": (
            metrics.get(
                "injector_residual_saturation_ratio",
                1.0,
            )
            <= args.best_max_residual_saturation_ratio
        ),
        "trigger_anchor_l1_255": (
            metrics.get(
                "injector_trigger_anchor_l1_255",
                float("inf"),
            )
            <= args.best_max_trigger_anchor_l1_255
        ),
    }

    eligible = all(checks.values())

    score = (
        metrics.get("encoder_alignment_similarity", -1.0),
        metrics.get("injector_message_accuracy", -1.0),
        metrics.get("injector_message_bit_accuracy", -1.0),
        metrics.get("encoder_clean_feature_similarity", -1.0),
        metrics.get("encoder_reference_feature_similarity", -1.0),
        metrics.get("injector_psnr_db", -1.0),
        metrics.get("injector_ssim_index", -1.0),
        -metrics.get("encoder_trigger_augmentation_similarity", 1.0),
        -metrics.get("injector_trigger_anchor_l1_255", float("inf")),
        -metrics.get("injector_residual_saturation_ratio", 1.0),
    )

    return eligible, score, checks


def checkpoint_payload(
    backdoored_model,
    trigger_model,
    hider_optimizer,
    encoder_optimizer,
    trigger_config,
    epoch,
    metrics,
    args,
    best_score,
    best_metrics,
):
    return {
        "backdoored_model_state_dict": backdoored_model.state_dict(),
        "stegastamp_model_state_dict": trigger_model.state_dict(),
        "hider_optimizer_state_dict": hider_optimizer.state_dict(),
        "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
        "epoch": epoch,
        "extra": {
            "stage": "inactive_alternating",
            "metrics": metrics,
            "best_score": list(best_score) if best_score is not None else None,
            "best_metrics": best_metrics,
            "args": vars(args),
            "trigger_config": asdict(trigger_config),
            "fixed_message_pattern": FIXED_MESSAGE_PATTERN,
            "clean_simclr_checkpoint": args.clean_simclr_checkpoint,
            "selected_stegastamp_checkpoint": args.stegastamp_checkpoint,
            "upstream_dataset": args.upstream_dataset,
            "encoder_input_size": 32,
        },
    }


def save_alternating_checkpoint(
    path,
    backdoored_model,
    trigger_model,
    hider_optimizer,
    encoder_optimizer,
    trigger_config,
    epoch,
    metrics,
    args,
    best_score,
    best_metrics,
):
    ensure_dir(os.path.dirname(path))
    torch.save(
        checkpoint_payload(
            backdoored_model=backdoored_model,
            trigger_model=trigger_model,
            hider_optimizer=hider_optimizer,
            encoder_optimizer=encoder_optimizer,
            trigger_config=trigger_config,
            epoch=epoch,
            metrics=metrics,
            args=args,
            best_score=best_score,
            best_metrics=best_metrics,
        ),
        path,
    )


def save_deployable_exports(
    checkpoint_dir,
    tag,
    backdoored_model,
    trigger_model,
    trigger_config,
    epoch,
    metrics,
    args,
):
    encoder_path = os.path.join(
        checkpoint_dir,
        f"backdoored_simclr_encoder_alternating_{tag}.pth",
    )
    trigger_path = os.path.join(
        checkpoint_dir,
        f"stegastamp_inactive_alternating_{tag}.pth",
    )

    torch.save(
        {
            "model_state_dict": backdoored_model.state_dict(),
            "epoch": epoch,
            "extra": {
                "stage": "inactive_alternating_encoder_export",
                "metrics": metrics,
                "args": vars(args),
                "paired_trigger_checkpoint": trigger_path,
                "source_stegastamp_checkpoint": args.stegastamp_checkpoint,
                "upstream_dataset": args.upstream_dataset,
                "encoder_input_size": 32,
            },
        },
        encoder_path,
    )

    trigger_saved_args = dict(vars(args))
    trigger_saved_args.update(asdict(trigger_config))
    # stegastamp_like_trigger.extract_checkpoint_config() reconstructs these
    # two booleans from the original training CLI's disable_* flags.
    trigger_saved_args["disable_spatial_transformer"] = (
        not trigger_config.use_spatial_transformer
    )
    trigger_saved_args["disable_corruption"] = (
        not trigger_config.use_corruption
    )

    torch.save(
        {
            "model_state_dict": trigger_model.state_dict(),
            "epoch": epoch,
            "extra": {
                "stage": "inactive_alternating_trigger_export",
                "metrics": metrics,
                "args": trigger_saved_args,
                "paired_encoder_checkpoint": encoder_path,
                "source_stegastamp_checkpoint": args.stegastamp_checkpoint,
                "fixed_message_pattern": FIXED_MESSAGE_PATTERN,
                "upstream_dataset": args.upstream_dataset,
                "encoder_input_size": 32,
            },
        },
        trigger_path,
    )

    return encoder_path, trigger_path


def load_alternating_resume(
    path,
    backdoored_model,
    trigger_model,
    hider_optimizer,
    encoder_optimizer,
    device,
):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    checkpoint = safe_torch_load(path, device)
    extra = checkpoint.get("extra", {})
    if extra.get("stage") != "inactive_alternating":
        raise ValueError(
            "--resume accepts only a checkpoint created by "
            "train_inactive_alternating.py. Encoder-only pilot "
            "checkpoints must not initialize the main experiment."
        )

    required_keys = (
        "backdoored_model_state_dict",
        "stegastamp_model_state_dict",
        "hider_optimizer_state_dict",
        "encoder_optimizer_state_dict",
    )
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Resume checkpoint is missing key: {key}")

    backdoored_model.load_state_dict(
        checkpoint["backdoored_model_state_dict"],
        strict=True,
    )
    trigger_model.load_state_dict(
        checkpoint["stegastamp_model_state_dict"],
        strict=True,
    )
    hider_optimizer.load_state_dict(
        checkpoint["hider_optimizer_state_dict"]
    )
    encoder_optimizer.load_state_dict(
        checkpoint["encoder_optimizer_state_dict"]
    )

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    saved_best_score = extra.get("best_score")
    best_score = (
        tuple(float(value) for value in saved_best_score)
        if saved_best_score is not None
        else None
    )
    best_metrics = dict(extra.get("best_metrics", {}))

    return start_epoch, best_score, best_metrics


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = parse_args()
    validate_args(args)

    set_random_seed(args.seed)
    device = get_device()

    ensure_dir(args.output_dir)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)

    save_json(vars(args), os.path.join(args.output_dir, "config.json"))
    train_csv_path = os.path.join(log_dir, "train_log.csv")
    validation_csv_path = os.path.join(log_dir, "validation_log.csv")

    print("=" * 80)
    print("CA3 Full Alternating INACTIVE Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Clean SimCLR start: {args.clean_simclr_checkpoint}")
    print(f"Selected StegaStamp start: {args.stegastamp_checkpoint}")
    print("Pilot encoder checkpoints are not used by default.")
    print(f"Fixed trigger message: {FIXED_MESSAGE_PATTERN} (16 bits)")
    print(f"Upstream dataset: {args.upstream_dataset}")
    print(f"Trigger insertion: raw [0, 1], then {args.upstream_dataset} encoder normalization")
    print("Feature source: model.f")
    print("Alternation: separate hider and encoder backward passes")
    print("=" * 80)

    train_loader, val_loader = build_dataloaders(args)

    clean_model = load_clean_simclr(args, device)
    backdoored_model = copy.deepcopy(clean_model).to(device)

    # Load the selected corruption-robust checkpoint as the trainable current
    # trigger. The frozen anchor is copied before any updates or resume state.
    trigger_model, trigger_config, _ = load_stegastamp_model(
        checkpoint_path=args.stegastamp_checkpoint,
        device=device,
        freeze=False,
        strict=True,
    )
    validate_trigger_config(trigger_config)

    anchor_trigger_model = copy.deepcopy(trigger_model).to(device)
    anchor_trigger_model.eval()
    set_parameters_trainable(anchor_trigger_model.parameters(), False)

    injector_loss_fn = build_injector_loss(
        args=args,
        trigger_config=trigger_config,
        device=device,
    )
    encoder_loss_fn = build_encoder_loss(args=args, device=device)

    # Construct optimizers only from the components they are allowed to update.
    hider_optimizer = torch.optim.AdamW(
        trigger_model.encoder.parameters(),
        lr=args.hider_lr,
        weight_decay=args.hider_weight_decay,
    )

    # Establish encoder trainability once before filtering optimizer params.
    set_encoder_update_mode(
        trigger_model=trigger_model,
        backdoored_model=backdoored_model,
        clean_model=clean_model,
        anchor_trigger_model=anchor_trigger_model,
    )
    encoder_trainable_parameters = [
        parameter
        for parameter in backdoored_model.f.parameters()
        if parameter.requires_grad
    ]
    encoder_optimizer = torch.optim.AdamW(
        encoder_trainable_parameters,
        lr=args.encoder_lr,
        weight_decay=args.encoder_weight_decay,
    )

    hider_trainable_count = sum(
        parameter.numel()
        for parameter in trigger_model.encoder.parameters()
    )
    encoder_trainable_count = sum(
        parameter.numel()
        for parameter in encoder_trainable_parameters
    )

    print(f"StegaStamp message length: {trigger_config.message_length}")
    print(
        "StegaStamp hard residual bound: "
        f"{trigger_config.max_residual:.8f} "
        f"({trigger_config.max_residual * 255.0:.2f}/255)"
    )
    print(f"Hider optimizer parameters: {hider_trainable_count:,}")
    print(f"Encoder optimizer parameters: {encoder_trainable_count:,}")
    print(f"Hider learning rate: {args.hider_lr:.2e}")
    print(f"Encoder learning rate: {args.encoder_lr:.2e}")
    print("=" * 80)

    start_epoch = 1
    best_score: Optional[Tuple[float, ...]] = None
    best_metrics: Dict[str, float] = {}

    if args.resume:
        (
            start_epoch,
            best_score,
            best_metrics,
        ) = load_alternating_resume(
            path=args.resume,
            backdoored_model=backdoored_model,
            trigger_model=trigger_model,
            hider_optimizer=hider_optimizer,
            encoder_optimizer=encoder_optimizer,
            device=device,
        )
        print(f"Resumed alternating checkpoint: {args.resume}")
        print(f"Resume start epoch: {start_epoch}")
    else:
        print(
            "Fresh start confirmed: original clean SimCLR encoder + "
            "selected advanced_stegastamp_best.pth"
        )

    # Baseline validation gives a direct before/after comparison.
    initial_metrics = evaluate(
        clean_model=clean_model,
        backdoored_model=backdoored_model,
        trigger_model=trigger_model,
        anchor_trigger_model=anchor_trigger_model,
        trigger_config=trigger_config,
        injector_loss_fn=injector_loss_fn,
        encoder_loss_fn=encoder_loss_fn,
        val_loader=val_loader,
        device=device,
        args=args,
        epoch=max(start_epoch - 1, 0),
    )
    print_validation_metrics(
        initial_metrics,
        prefix="Initial Validation",
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            clean_model=clean_model,
            backdoored_model=backdoored_model,
            trigger_model=trigger_model,
            anchor_trigger_model=anchor_trigger_model,
            trigger_config=trigger_config,
            injector_loss_fn=injector_loss_fn,
            encoder_loss_fn=encoder_loss_fn,
            hider_optimizer=hider_optimizer,
            encoder_optimizer=encoder_optimizer,
            train_loader=train_loader,
            device=device,
            args=args,
            epoch=epoch,
        )

        train_row = {
            "epoch": epoch,
            "hider_lr": hider_optimizer.param_groups[0]["lr"],
            "encoder_lr": encoder_optimizer.param_groups[0]["lr"],
        }
        train_row.update(train_metrics)
        append_metrics_to_csv(train_csv_path, train_row)

        val_metrics: Dict[str, float] = {}
        should_evaluate = (
            args.eval_every > 0
            and epoch % args.eval_every == 0
        )

        if should_evaluate:
            val_metrics = evaluate(
                clean_model=clean_model,
                backdoored_model=backdoored_model,
                trigger_model=trigger_model,
                anchor_trigger_model=anchor_trigger_model,
                trigger_config=trigger_config,
                injector_loss_fn=injector_loss_fn,
                encoder_loss_fn=encoder_loss_fn,
                val_loader=val_loader,
                device=device,
                args=args,
                epoch=epoch,
            )
            print_validation_metrics(
                val_metrics,
                prefix=f"Epoch [{epoch}/{args.epochs}] Validation",
            )

            validation_row = {"epoch": epoch}
            validation_row.update(val_metrics)
            append_metrics_to_csv(
                validation_csv_path,
                validation_row,
            )

            eligible, current_score, checks = score_checkpoint(
                metrics=val_metrics,
                args=args,
                trigger_config=trigger_config,
            )

            if eligible and (
                best_score is None
                or current_score > best_score
            ):
                best_score = current_score
                best_metrics = dict(val_metrics)

                best_path = os.path.join(
                    checkpoint_dir,
                    "inactive_alternating_best.pth",
                )
                save_alternating_checkpoint(
                    path=best_path,
                    backdoored_model=backdoored_model,
                    trigger_model=trigger_model,
                    hider_optimizer=hider_optimizer,
                    encoder_optimizer=encoder_optimizer,
                    trigger_config=trigger_config,
                    epoch=epoch,
                    metrics=val_metrics,
                    args=args,
                    best_score=best_score,
                    best_metrics=best_metrics,
                )
                encoder_export, trigger_export = save_deployable_exports(
                    checkpoint_dir=checkpoint_dir,
                    tag="best",
                    backdoored_model=backdoored_model,
                    trigger_model=trigger_model,
                    trigger_config=trigger_config,
                    epoch=epoch,
                    metrics=val_metrics,
                    args=args,
                )

                print(f"Saved eligible best checkpoint: {best_path}")
                print(f"Saved matched encoder export: {encoder_export}")
                print(f"Saved matched trigger export: {trigger_export}")

            elif not eligible:
                failed_checks = [
                    name
                    for name, passed in checks.items()
                    if not passed
                ]
                print(
                    "Checkpoint not eligible. Failed joint constraints: "
                    + ", ".join(failed_checks)
                )

        # Latest is always resumable, even when no best checkpoint exists.
        latest_metrics = val_metrics if val_metrics else train_metrics
        latest_path = os.path.join(
            checkpoint_dir,
            "inactive_alternating_latest.pth",
        )
        save_alternating_checkpoint(
            path=latest_path,
            backdoored_model=backdoored_model,
            trigger_model=trigger_model,
            hider_optimizer=hider_optimizer,
            encoder_optimizer=encoder_optimizer,
            trigger_config=trigger_config,
            epoch=epoch,
            metrics=latest_metrics,
            args=args,
            best_score=best_score,
            best_metrics=best_metrics,
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            numbered_path = os.path.join(
                checkpoint_dir,
                f"inactive_alternating_epoch_{epoch:04d}.pth",
            )
            save_alternating_checkpoint(
                path=numbered_path,
                backdoored_model=backdoored_model,
                trigger_model=trigger_model,
                hider_optimizer=hider_optimizer,
                encoder_optimizer=encoder_optimizer,
                trigger_config=trigger_config,
                epoch=epoch,
                metrics=latest_metrics,
                args=args,
                best_score=best_score,
                best_metrics=best_metrics,
            )

    final_metrics = evaluate(
        clean_model=clean_model,
        backdoored_model=backdoored_model,
        trigger_model=trigger_model,
        anchor_trigger_model=anchor_trigger_model,
        trigger_config=trigger_config,
        injector_loss_fn=injector_loss_fn,
        encoder_loss_fn=encoder_loss_fn,
        val_loader=val_loader,
        device=device,
        args=args,
        epoch=args.epochs,
    )

    final_path = os.path.join(
        checkpoint_dir,
        "inactive_alternating_final.pth",
    )
    save_alternating_checkpoint(
        path=final_path,
        backdoored_model=backdoored_model,
        trigger_model=trigger_model,
        hider_optimizer=hider_optimizer,
        encoder_optimizer=encoder_optimizer,
        trigger_config=trigger_config,
        epoch=args.epochs,
        metrics=final_metrics,
        args=args,
        best_score=best_score,
        best_metrics=best_metrics,
    )
    final_encoder_export, final_trigger_export = save_deployable_exports(
        checkpoint_dir=checkpoint_dir,
        tag="final",
        backdoored_model=backdoored_model,
        trigger_model=trigger_model,
        trigger_config=trigger_config,
        epoch=args.epochs,
        metrics=final_metrics,
        args=args,
    )

    print("=" * 80)
    print("Alternating INACTIVE training complete")
    print(f"Final resumable checkpoint: {final_path}")
    print(f"Final encoder export: {final_encoder_export}")
    print(f"Final trigger export: {final_trigger_export}")

    if best_score is not None:
        print(
            "Use the matched eligible best pair for downstream evaluation:"
        )
        print(
            os.path.join(
                checkpoint_dir,
                "backdoored_simclr_encoder_alternating_best.pth",
            )
        )
        print(
            os.path.join(
                checkpoint_dir,
                "stegastamp_inactive_alternating_best.pth",
            )
        )
    else:
        print(
            "No epoch satisfied all joint alignment, feature-preservation, "
            "message, stealth, anchor, and residual constraints."
        )
        print(
            "Do not use the final checkpoint as the main downstream model "
            "solely because training completed."
        )


if __name__ == "__main__":
    main()
