"""
Complete downstream evaluator for the CA3 INACTIVE StegaStamp-SimCLR project.

Evaluation protocol
-------------------
1. Load the original clean SimCLR model and train a downstream MLP on clean
   CIFAR-10 training images using frozen, L2-normalized model.f features.
2. For every supplied matched encoder-trigger pair:
   - validate the checkpoint metadata and trigger configuration;
   - train a separate downstream MLP on clean CIFAR-10 training images using
     frozen features from that backdoored encoder;
   - evaluate BA, baseline target rate, ASR and adjusted ASR;
   - evaluate the same trigger with the clean encoder as a sanity control.
3. Repeat downstream-classifier training with isolated, pair-order-invariant
   seeds and aggregate the resulting CA, BA and ASR distributions.
4. Save per-seed JSON results, aggregate JSON/CSV outputs, classifier
   checkpoints, confusion matrices and feature-cosine diagnostics.

Important project constraints enforced here
-------------------------------------------
- CIFAR-10, 32x32.
- Encoder representations come from model.f, never model.g.
- Triggers are inserted while images are raw tensors in [0, 1].
- CIFAR-10 normalization is applied only after trigger insertion.
- Original target-class samples are excluded from ASR.
- The fixed trigger message is the alternating 16-bit pattern.
- The maximum trigger residual must be 8/255.
- Main encoder-trigger pairs must be matched exports from the same stage/epoch.
- Training-time validation alignment is not reported as downstream ASR.

The downstream MLP architecture and adjusted-ASR definition are reused from
train_ssl_badencoder_stegastamp_learned.py in this repository. The evaluator
uses a deterministic train/validation split from CIFAR-10 training data so the
CIFAR-10 test set is not used to select the classifier epoch. Classifier weight
initialisation, DataLoader shuffling and dropout are isolated for every repetition.
All backdoored pairs receive the same classifier initialisation and sample order
within a repetition, independent of their command-line position.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from sim_clr_model import get_encoder_architecture
from stegastamp_like_losses import ssim_index
from stegastamp_like_trigger import (
    StegaStampCheckpointConfig,
    generate_trigger_batch,
    load_stegastamp_model,
)
from stegastamp_like_utils import normalize_cifar10, set_random_seed


FIXED_MESSAGE_PATTERN = "alternating"
EXPECTED_MESSAGE_LENGTH = 16
EXPECTED_IMAGE_SIZE = 32
EXPECTED_MAX_RESIDUAL = 8.0 / 255.0
NUM_CLASSES = 10
CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

ALLOWED_STAGE_PAIRS = {
    "inactive_alternating_encoder_export": "inactive_alternating_trigger_export",
    "inactive_encoder_finish_encoder_export": "inactive_encoder_finish_trigger_export",

    # Redesigned C2 experiment 
    "inactive_alternating_c2_dualphase_encoder_export" : "inactive_alternating_c2_dualphase_trigger_export",
}


@dataclass(frozen=True)
class PairSpec:
    name: str
    encoder_checkpoint: str
    trigger_checkpoint: str


@dataclass(frozen=True)
class ClassifierSeedPlan:
    """Independent random streams used by one downstream classifier run."""

    role: str
    repetition_index: int
    repeat_seed: int
    initialization_seed: int
    dataloader_seed: int
    dropout_seed: int


@dataclass
class ClassificationResult:
    accuracy: float
    total_samples: int
    correct_samples: int
    confusion_matrix: List[List[int]]
    per_class_accuracy: Dict[str, Optional[float]]
    prediction_distribution: List[int]


@dataclass
class TargetRateResult:
    target_rate: float
    target_predictions: int
    total_samples: int
    prediction_distribution: List[int]


class IndexedDataset(Dataset):
    """Expose a deterministic list of indices from a base CIFAR-10 dataset."""

    def __init__(self, base_dataset: Dataset, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return self.base_dataset[self.indices[item]]


class DownstreamMLP(nn.Module):
    """
    CA2-compatible downstream classifier on frozen SSL features.

    Architecture from train_ssl_badencoder_stegastamp_learned.py:
        Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, int] = (512, 256),
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_sizes[1], num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TriggerQualityAccumulator:
    """Aggregate trigger quality without retaining image batches in memory."""

    def __init__(self, max_residual: float) -> None:
        self.max_residual = float(max_residual)
        self.sample_count = 0
        self.pixel_count = 0
        self.psnr_sum = 0.0
        self.ssim_weighted_sum = 0.0
        self.correct_bits = 0
        self.total_bits = 0
        self.correct_messages = 0
        self.abs_residual_sum = 0.0
        self.max_abs_residual = 0.0
        self.saturated_count = 0
        self.bound_violation_count = 0

    @torch.no_grad()
    def update(
        self,
        clean_images: torch.Tensor,
        encoded_images: torch.Tensor,
        residual: torch.Tensor,
        decoded_logits: Optional[torch.Tensor],
        message: torch.Tensor,
    ) -> None:
        batch_size = int(clean_images.size(0))
        residual_abs = residual.detach().abs()

        mse_per_sample = (
            (encoded_images - clean_images)
            .pow(2)
            .flatten(start_dim=1)
            .mean(dim=1)
            .clamp_min(1e-8)
        )
        psnr_per_sample = 10.0 * torch.log10(1.0 / mse_per_sample)
        batch_ssim = float(ssim_index(clean_images, encoded_images).detach().cpu())

        self.sample_count += batch_size
        self.pixel_count += int(residual_abs.numel())
        self.psnr_sum += float(psnr_per_sample.sum().detach().cpu())
        self.ssim_weighted_sum += batch_ssim * batch_size
        self.abs_residual_sum += float(residual_abs.sum().detach().cpu())
        self.max_abs_residual = max(
            self.max_abs_residual,
            float(residual_abs.max().detach().cpu()),
        )
        self.saturated_count += int(
            (residual_abs >= 0.95 * self.max_residual).sum().detach().cpu()
        )
        self.bound_violation_count += int(
            (residual_abs > self.max_residual + 1e-6).sum().detach().cpu()
        )

        if decoded_logits is not None:
            predicted_bits = (torch.sigmoid(decoded_logits) >= 0.5).to(message.dtype)
            bit_matches = predicted_bits.eq(message)
            self.correct_bits += int(bit_matches.sum().detach().cpu())
            self.total_bits += int(bit_matches.numel())
            self.correct_messages += int(
                bit_matches.all(dim=1).sum().detach().cpu()
            )

    def finalize(self) -> Dict[str, float]:
        if self.sample_count <= 0 or self.pixel_count <= 0:
            raise RuntimeError("No samples were supplied to trigger quality accumulator")

        metrics: Dict[str, float] = {
            "psnr_db": self.psnr_sum / self.sample_count,
            "ssim_index": self.ssim_weighted_sum / self.sample_count,
            "mean_abs_residual": self.abs_residual_sum / self.pixel_count,
            "mean_abs_residual_255": (
                self.abs_residual_sum / self.pixel_count * 255.0
            ),
            "max_abs_residual": self.max_abs_residual,
            "max_abs_residual_255": self.max_abs_residual * 255.0,
            "residual_saturation_ratio": self.saturated_count / self.pixel_count,
            "residual_bound_violation_ratio": (
                self.bound_violation_count / self.pixel_count
            ),
        }

        if self.total_bits > 0:
            metrics["message_bit_accuracy"] = self.correct_bits / self.total_bits
            metrics["message_accuracy"] = (
                self.correct_messages / self.sample_count
            )

        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate matched CA3 INACTIVE encoder-trigger pairs on the "
            "CIFAR-10 downstream task"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "ENCODER_CHECKPOINT", "TRIGGER_CHECKPOINT"),
        help=(
            "Matched pair to evaluate. Repeat --pair to compare multiple "
            "pairs in one run."
        ),
    )
    parser.add_argument(
        "--clean-simclr-checkpoint",
        type=str,
        required=True,
        help="Path to the clean SimCLR checkpoint; must be supplied explicitly.",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_inactive_downstream",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not ask torchvision to download CIFAR-10 if it is absent.",
    )

    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument(
        "--message-pattern",
        type=str,
        choices=(FIXED_MESSAGE_PATTERN,),
        default=FIXED_MESSAGE_PATTERN,
    )

    parser.add_argument(
        "--train-subset",
        type=int,
        default=0,
        help="0 uses all 50,000 CIFAR-10 training images before validation split.",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=0,
        help="0 uses all 10,000 CIFAR-10 test images.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=5000,
        help="Held-out part of the selected training set used for MLP selection.",
    )
    parser.add_argument(
        "--max-reference-samples",
        type=int,
        default=1000,
        help=(
            "Maximum CIFAR-10 training images from the target class used for "
            "feature-cosine diagnostics; 0 uses every target reference."
        ),
    )

    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-eval-batch-size", type=int, default=256)
    parser.add_argument("--mlp-lr", type=float, default=1e-4)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-1", type=int, default=512)
    parser.add_argument("--mlp-hidden-2", type=int, default=256)
    parser.add_argument("--mlp-dropout", type=float, default=0.5)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="0 disables early stopping.",
    )
    parser.add_argument("--print-every", type=int, default=1)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Global experiment seed used for the fixed CIFAR-10 split and "
            "stable derivation of all repetition-specific random streams."
        ),
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Number of repeated downstream-classifier runs.",
    )
    parser.add_argument(
        "--repeat-seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help=(
            "Explicit repetition seeds. Supply one value per repetition, for "
            "example: --num-repeats 5 --repeat-seeds 42 43 44 45 46. "
            "If omitted, seeds are seed, seed+1, ... ."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--skip-confusion-plots",
        action="store_true",
    )
    parser.add_argument(
        "--allow-unverified-pair",
        action="store_true",
        help=(
            "Allow checkpoints without the matched-export metadata expected "
            "from the alternating/finishing scripts."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-finish",
        action="store_true",
        help=(
            "Allow a finishing checkpoint whose saved clean/reference "
            "similarity is below 0.98. This is off by default."
        ),
    )

    parser.add_argument(
        "--decision-max-ba-drop-pp",
        type=float,
        default=5.0,
        help="Maximum CA-to-BA drop used by the printed project decision rule.",
    )
    parser.add_argument(
        "--decision-min-asr-percent",
        type=float,
        default=90.0,
        help="Raw ASR target used only by the printed project decision rule.",
    )
    parser.add_argument(
        "--decision-min-adjusted-asr-percent",
        type=float,
        default=90.0,
        help="Adjusted-ASR target used only by the printed project decision rule.",
    )

    args = parser.parse_args()
    args.pairs = [PairSpec(*values) for values in args.pair]
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.target_label < NUM_CLASSES:
        raise ValueError("--target-label must be in [0, 9]")
    if args.target_label != 0:
        print(
            "Warning: the current CA3 experiment design uses target label 0 "
            "(airplane), but a different label was requested."
        )

    positive_values = {
        "--feature-batch-size": args.feature_batch_size,
        "--mlp-epochs": args.mlp_epochs,
        "--mlp-batch-size": args.mlp_batch_size,
        "--mlp-eval-batch-size": args.mlp_eval_batch_size,
        "--mlp-hidden-1": args.mlp_hidden_1,
        "--mlp-hidden-2": args.mlp_hidden_2,
        "--print-every": args.print_every,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")

    nonnegative_values = {
        "--train-subset": args.train_subset,
        "--test-subset": args.test_subset,
        "--validation-size": args.validation_size,
        "--max-reference-samples": args.max_reference_samples,
        "--num-workers": args.num_workers,
        "--early-stopping-patience": args.early_stopping_patience,
        "--mlp-weight-decay": args.mlp_weight_decay,
        "--decision-max-ba-drop-pp": args.decision_max_ba_drop_pp,
    }
    for name, value in nonnegative_values.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative")

    if args.num_repeats <= 0:
        raise ValueError("--num-repeats must be greater than 0")
    if args.repeat_seeds is not None:
        # Friendly behaviour: an explicit seed list can define the repeat count
        # when --num-repeats was left at its backward-compatible default of 1.
        if args.num_repeats == 1 and len(args.repeat_seeds) > 1:
            args.num_repeats = len(args.repeat_seeds)
        if len(args.repeat_seeds) != args.num_repeats:
            raise ValueError(
                "--repeat-seeds must contain exactly --num-repeats values. "
                f"Got {len(args.repeat_seeds)} seeds for "
                f"{args.num_repeats} repetitions."
            )
        if len(set(args.repeat_seeds)) != len(args.repeat_seeds):
            raise ValueError("--repeat-seeds values must be unique")

    if args.mlp_lr <= 0.0:
        raise ValueError("--mlp-lr must be greater than 0")
    if not 0.0 <= args.mlp_dropout < 1.0:
        raise ValueError("--mlp-dropout must be in [0, 1)")
    if args.message_pattern != FIXED_MESSAGE_PATTERN:
        raise ValueError("This evaluator requires the alternating message pattern")

    pair_names = [pair.name for pair in args.pairs]
    if len(set(pair_names)) != len(pair_names):
        raise ValueError("Every --pair NAME must be unique")

    for pair in args.pairs:
        if not pair.name.strip():
            raise ValueError("Pair names cannot be empty")
        if not os.path.isfile(pair.encoder_checkpoint):
            raise FileNotFoundError(
                f"Backdoored encoder checkpoint not found: {pair.encoder_checkpoint}"
            )
        if not os.path.isfile(pair.trigger_checkpoint):
            raise FileNotFoundError(
                f"Trigger checkpoint not found: {pair.trigger_checkpoint}"
            )

    if not os.path.isfile(args.clean_simclr_checkpoint):
        raise FileNotFoundError(
            "Clean SimCLR checkpoint not found: "
            f"{args.clean_simclr_checkpoint}"
        )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def safe_torch_load(path: str, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def clean_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def extract_encoder_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        if not hasattr(checkpoint, "keys"):
            raise TypeError("Checkpoint is not a state dictionary")
        return clean_state_dict_keys(checkpoint)

    for key in (
        "model_state_dict",
        "backdoored_model_state_dict",
        "state_dict",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return clean_state_dict_keys(value)

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return clean_state_dict_keys(checkpoint)

    raise KeyError(
        "Could not find model weights under model_state_dict, "
        "backdoored_model_state_dict or state_dict"
    )


def checkpoint_extra(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("extra"), dict):
        return checkpoint["extra"]
    return {}


def build_simclr(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model_args = SimpleNamespace(
        arch=args.arch,
        feature_dim=args.projection_dim,
    )
    return get_encoder_architecture(model_args).to(device)


def load_simclr_model(
    checkpoint_path: str,
    args: argparse.Namespace,
    device: torch.device,
    role: str,
) -> Tuple[nn.Module, Any]:
    checkpoint = safe_torch_load(checkpoint_path, device)
    state_dict = extract_encoder_state_dict(checkpoint)
    model = build_simclr(args, device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing_f = [key for key in missing if key.startswith("f.")]
        if missing_f:
            raise RuntimeError(
                f"{role} checkpoint does not fully define model.f. "
                f"Missing encoder keys: {missing_f}"
            )
        if missing:
            print(f"Warning: {role} checkpoint missing non-encoder keys: {missing}")
        if unexpected:
            print(f"Warning: {role} checkpoint unexpected keys: {unexpected}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None

    return model, checkpoint


def portable_basename(path_value: Any) -> str:
    text = str(path_value).replace("\\", "/")
    return text.rsplit("/", maxsplit=1)[-1]


def portable_checkpoint_suffix(path_value: Any, depth: int = 3) -> str:
    """
    Return a portable checkpoint-path suffix for matched-pair verification.

    Alternating-stage ablations intentionally reuse the same export filenames
    in different result folders. Comparing only basenames can therefore miss
    an accidental cross-folder encoder/trigger mismatch. Using the last three
    components (result folder / checkpoints / filename) preserves portability
    across absolute-vs-relative paths while distinguishing A/B/C exports.
    """
    text = str(path_value).replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        return ""
    return "/".join(parts[-max(1, int(depth)):])


def extract_saved_metric(extra: Dict[str, Any], names: Sequence[str]) -> Optional[float]:
    metrics = extra.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return float(value)
    return None


def validate_matched_pair_metadata(
    pair: PairSpec,
    encoder_checkpoint: Any,
    trigger_checkpoint: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    encoder_extra = checkpoint_extra(encoder_checkpoint)
    trigger_extra = checkpoint_extra(trigger_checkpoint)
    encoder_stage = encoder_extra.get("stage")
    trigger_stage = trigger_extra.get("stage")
    warnings: List[str] = []

    verified = True
    expected_trigger_stage = ALLOWED_STAGE_PAIRS.get(str(encoder_stage))
    if expected_trigger_stage is None or trigger_stage != expected_trigger_stage:
        verified = False
        warnings.append(
            "Checkpoint stages are not a recognized matched export pair: "
            f"encoder_stage={encoder_stage!r}, trigger_stage={trigger_stage!r}."
        )

    encoder_epoch = (
        int(encoder_checkpoint.get("epoch", -1))
        if isinstance(encoder_checkpoint, dict)
        else -1
    )
    trigger_epoch = (
        int(trigger_checkpoint.get("epoch", -1))
        if isinstance(trigger_checkpoint, dict)
        else -1
    )
    if encoder_epoch < 0 or trigger_epoch < 0 or encoder_epoch != trigger_epoch:
        verified = False
        warnings.append(
            "Encoder and trigger epochs are missing or unequal: "
            f"encoder_epoch={encoder_epoch}, trigger_epoch={trigger_epoch}."
        )

    recorded_trigger = encoder_extra.get("paired_trigger_checkpoint")
    if recorded_trigger is not None:
        if portable_checkpoint_suffix(recorded_trigger) != portable_checkpoint_suffix(
            pair.trigger_checkpoint
        ):
            verified = False
            warnings.append(
                "Encoder metadata points to a different trigger export: "
                f"{recorded_trigger}"
            )
    else:
        verified = False
        warnings.append("Encoder checkpoint has no paired_trigger_checkpoint metadata.")

    recorded_encoder = trigger_extra.get("paired_encoder_checkpoint")
    if recorded_encoder is not None:
        if portable_checkpoint_suffix(recorded_encoder) != portable_checkpoint_suffix(
            pair.encoder_checkpoint
        ):
            verified = False
            warnings.append(
                "Trigger metadata points to a different encoder export: "
                f"{recorded_encoder}"
            )
    else:
        verified = False
        warnings.append("Trigger checkpoint has no paired_encoder_checkpoint metadata.")

    fixed_pattern = trigger_extra.get("fixed_message_pattern")
    if fixed_pattern not in (None, FIXED_MESSAGE_PATTERN):
        verified = False
        warnings.append(
            f"Trigger metadata fixed message is {fixed_pattern!r}, not alternating."
        )

    is_finish = encoder_stage == "inactive_encoder_finish_encoder_export"
    clean_similarity = extract_saved_metric(
        encoder_extra,
        ("clean_feature_similarity", "encoder_clean_feature_similarity"),
    )
    reference_similarity = extract_saved_metric(
        encoder_extra,
        (
            "reference_feature_similarity",
            "encoder_reference_feature_similarity",
        ),
    )

    if is_finish and not args.allow_unsafe_finish:
        if clean_similarity is not None and clean_similarity < 0.98:
            raise ValueError(
                f"Finishing pair {pair.name!r} has saved clean similarity "
                f"{clean_similarity:.6f} below 0.98."
            )
        if reference_similarity is not None and reference_similarity < 0.98:
            raise ValueError(
                f"Finishing pair {pair.name!r} has saved reference similarity "
                f"{reference_similarity:.6f} below 0.98. Use the best_safe "
                "checkpoint, not the unsafe final checkpoint."
            )

    lower_names = (
        portable_basename(pair.encoder_checkpoint).lower(),
        portable_basename(pair.trigger_checkpoint).lower(),
    )
    if is_finish and any("final" in name for name in lower_names):
        warnings.append(
            "A finishing filename contains 'final'. The project handoff says the "
            "epoch-2 final state is unsafe; verify that best_safe was intended."
        )

    if not verified and not args.allow_unverified_pair:
        details = "\n  - ".join(warnings)
        raise ValueError(
            f"Pair {pair.name!r} could not be verified as a matched export.\n"
            f"  - {details}\n"
            "Use the encoder and trigger exports saved together by the same "
            "training stage."
        )

    for warning in warnings:
        print(f"Warning [{pair.name}]: {warning}")

    return {
        "verified": verified,
        "encoder_stage": encoder_stage,
        "trigger_stage": trigger_stage,
        "encoder_epoch": encoder_epoch,
        "trigger_epoch": trigger_epoch,
        "saved_clean_feature_similarity": clean_similarity,
        "saved_reference_feature_similarity": reference_similarity,
        "warnings": warnings,
    }


def validate_trigger_config(
    pair_name: str,
    config: StegaStampCheckpointConfig,
    trigger_checkpoint: Any,
) -> None:
    if config.message_length != EXPECTED_MESSAGE_LENGTH:
        raise ValueError(
            f"Pair {pair_name!r} trigger has message_length={config.message_length}; "
            f"expected {EXPECTED_MESSAGE_LENGTH}."
        )
    if config.image_size != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"Pair {pair_name!r} trigger has image_size={config.image_size}; "
            f"expected {EXPECTED_IMAGE_SIZE}."
        )
    if not math.isclose(
        config.max_residual,
        EXPECTED_MAX_RESIDUAL,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            f"Pair {pair_name!r} trigger max_residual={config.max_residual}; "
            f"expected 8/255={EXPECTED_MAX_RESIDUAL}."
        )

    extra = checkpoint_extra(trigger_checkpoint)
    pattern = extra.get("fixed_message_pattern")
    if pattern not in (None, FIXED_MESSAGE_PATTERN):
        raise ValueError(
            f"Pair {pair_name!r} trigger metadata uses message pattern {pattern!r}."
        )


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "pair"


def make_torch_generator(seed: int) -> torch.Generator:
    """Create an isolated CPU generator without touching global RNG state."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def deterministic_indices(length: int, subset_size: int, seed: int) -> List[int]:
    if subset_size <= 0 or subset_size >= length:
        return list(range(length))
    generator = make_torch_generator(seed)
    return torch.randperm(length, generator=generator)[:subset_size].tolist()


def split_train_validation_indices(
    selected_indices: Sequence[int],
    validation_size: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    indices = torch.tensor(
        [int(index) for index in selected_indices],
        dtype=torch.long,
    )
    if indices.numel() < 2:
        raise ValueError("At least two selected training images are required")

    if validation_size <= 0:
        validation_size = max(1, int(round(0.1 * int(indices.numel()))))
    if validation_size >= int(indices.numel()):
        raise ValueError(
            "--validation-size must be smaller than the selected training set. "
            f"Selected={int(indices.numel())}, validation={validation_size}."
        )

    generator = make_torch_generator(seed)
    order = torch.randperm(int(indices.numel()), generator=generator)
    shuffled = indices[order]
    validation_indices = shuffled[:validation_size].tolist()
    fit_indices = shuffled[validation_size:].tolist()
    return fit_indices, validation_indices


def build_datasets_and_indices(args: argparse.Namespace) -> Dict[str, Any]:
    to_tensor = transforms.ToTensor()
    train_base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=not args.no_download,
        transform=to_tensor,
    )
    test_base = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=not args.no_download,
        transform=to_tensor,
    )

    selected_train = deterministic_indices(
        len(train_base),
        args.train_subset,
        args.seed,
    )
    fit_indices, validation_indices = split_train_validation_indices(
        selected_train,
        args.validation_size,
        args.seed + 11,
    )
    test_indices = deterministic_indices(
        len(test_base),
        args.test_subset,
        args.seed + 23,
    )

    non_target_test_indices = [
        index
        for index in test_indices
        if int(test_base.targets[index]) != args.target_label
    ]
    reference_indices = [
        index
        for index, label in enumerate(train_base.targets)
        if int(label) == args.target_label
    ]
    if args.max_reference_samples > 0:
        reference_indices = deterministic_indices(
            len(reference_indices),
            min(args.max_reference_samples, len(reference_indices)),
            args.seed + 37,
        )
        # deterministic_indices above indexes the local reference list.
        full_target_indices = [
            index
            for index, label in enumerate(train_base.targets)
            if int(label) == args.target_label
        ]
        reference_indices = [full_target_indices[index] for index in reference_indices]

    if not non_target_test_indices:
        raise ValueError("No non-target test samples remain for ASR evaluation")
    if not reference_indices:
        raise ValueError("No target-class reference images were found")

    return {
        "train_base": train_base,
        "test_base": test_base,
        "fit_indices": fit_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "non_target_test_indices": non_target_test_indices,
        "reference_indices": reference_indices,
    }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def make_loader(
    base_dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = make_torch_generator(seed)
    return DataLoader(
        IndexedDataset(base_dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def encode_features(model: nn.Module, normalized_images: torch.Tensor) -> torch.Tensor:
    """Return L2-normalized model.f features; model.g is never used."""
    return F.normalize(model.f(normalized_images), dim=1)


@torch.no_grad()
def extract_clean_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for raw_images, labels in loader:
        raw_images = raw_images.to(device, non_blocking=True)
        if raw_images.min().item() < -1e-6 or raw_images.max().item() > 1.0 + 1e-6:
            raise ValueError("CIFAR-10 loader did not return raw images in [0, 1]")

        normalized_images = normalize_cifar10(raw_images)
        features = encode_features(model, normalized_images)
        all_features.append(features.cpu())
        all_labels.append(labels.long().cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def extract_triggered_features_for_two_encoders(
    clean_model: nn.Module,
    backdoored_model: nn.Module,
    trigger_model: nn.Module,
    trigger_config: StegaStampCheckpointConfig,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    clean_model.eval()
    backdoored_model.eval()
    trigger_model.eval()

    clean_encoder_features: List[torch.Tensor] = []
    backdoored_encoder_features: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    quality = TriggerQualityAccumulator(trigger_config.max_residual)

    for raw_images, labels in loader:
        raw_images = raw_images.to(device, non_blocking=True)
        if raw_images.min().item() < -1e-6 or raw_images.max().item() > 1.0 + 1e-6:
            raise ValueError("Trigger input must be a raw [0, 1] image tensor")

        outputs = generate_trigger_batch(
            clean_image=raw_images,
            model=trigger_model,
            config=trigger_config,
            message_pattern=args.message_pattern,
            decode_message=True,
            return_cpu=False,
            check_range=True,
        )
        triggered_images = outputs["encoded_image"]
        residual = outputs["residual"]
        decoded_logits = outputs["decoded_logits"]
        message = outputs["message"]

        if triggered_images is None or residual is None or message is None:
            raise RuntimeError("Trigger generator returned incomplete outputs")
        if residual.abs().max().item() > EXPECTED_MAX_RESIDUAL + 1e-6:
            raise RuntimeError("Generated trigger exceeded the 8/255 residual bound")

        # Normalization occurs only after the raw trigger has been inserted.
        normalized_triggered = normalize_cifar10(triggered_images)
        clean_features = encode_features(clean_model, normalized_triggered)
        backdoored_features = encode_features(
            backdoored_model,
            normalized_triggered,
        )

        quality.update(
            clean_images=raw_images,
            encoded_images=triggered_images,
            residual=residual,
            decoded_logits=decoded_logits,
            message=message,
        )
        clean_encoder_features.append(clean_features.cpu())
        backdoored_encoder_features.append(backdoored_features.cpu())
        labels_all.append(labels.long().cpu())

    return {
        "clean_encoder_features": torch.cat(clean_encoder_features, dim=0),
        "backdoored_encoder_features": torch.cat(
            backdoored_encoder_features,
            dim=0,
        ),
        "labels": torch.cat(labels_all, dim=0),
        "trigger_quality": quality.finalize(),
    }


def make_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = make_torch_generator(seed)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


@torch.no_grad()
def evaluate_classifier(
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> ClassificationResult:
    loader = make_feature_loader(
        features,
        labels,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    classifier.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    prediction_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    correct = 0
    total = 0

    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        predictions = classifier(batch_features).argmax(dim=1)

        correct += int(predictions.eq(batch_labels).sum().item())
        total += int(batch_labels.numel())
        encoded = batch_labels * NUM_CLASSES + predictions
        confusion += torch.bincount(
            encoded.cpu(),
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        prediction_counts += torch.bincount(
            predictions.cpu(),
            minlength=NUM_CLASSES,
        )

    per_class_accuracy: Dict[str, Optional[float]] = {}
    for class_index, class_name in enumerate(CIFAR10_CLASS_NAMES):
        class_total = int(confusion[class_index].sum().item())
        if class_total == 0:
            per_class_accuracy[class_name] = None
        else:
            per_class_accuracy[class_name] = (
                int(confusion[class_index, class_index].item()) / class_total
            )

    return ClassificationResult(
        accuracy=correct / total,
        total_samples=total,
        correct_samples=correct,
        confusion_matrix=confusion.tolist(),
        per_class_accuracy=per_class_accuracy,
        prediction_distribution=prediction_counts.tolist(),
    )


@torch.no_grad()
def evaluate_target_rate(
    classifier: nn.Module,
    features: torch.Tensor,
    target_label: int,
    batch_size: int,
    device: torch.device,
) -> TargetRateResult:
    dummy_labels = torch.zeros(features.size(0), dtype=torch.long)
    loader = make_feature_loader(
        features,
        dummy_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    classifier.eval()
    target_predictions = 0
    total = 0
    prediction_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)

    for batch_features, _ in loader:
        batch_features = batch_features.to(device)
        predictions = classifier(batch_features).argmax(dim=1)
        target_predictions += int(predictions.eq(target_label).sum().item())
        total += int(predictions.numel())
        prediction_counts += torch.bincount(
            predictions.cpu(),
            minlength=NUM_CLASSES,
        )

    return TargetRateResult(
        target_rate=target_predictions / total,
        target_predictions=target_predictions,
        total_samples=total,
        prediction_distribution=prediction_counts.tolist(),
    )


def stable_seed(*parts: Any) -> int:
    """Derive a stable positive 31-bit seed from explicit experiment fields."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (
        2**31 - 1
    )


def resolve_repeat_seeds(args: argparse.Namespace) -> List[int]:
    if args.repeat_seeds is not None:
        return [int(seed) for seed in args.repeat_seeds]
    return [int(args.seed + index) for index in range(args.num_repeats)]


def classifier_seed_plan(
    global_seed: int,
    repetition_index: int,
    repeat_seed: int,
    role: str,
) -> ClassifierSeedPlan:
    """
    Build explicit independent random streams for one classifier.

    The clean baseline uses role="clean". Every compromised pair uses the same
    role="pair_shared" within a repetition. Consequently, alternating and
    finishing classifiers receive identical initial weights, DataLoader order
    and dropout RNG when their architecture dimensions match. Pair name and
    command-line position are deliberately excluded.
    """
    if role not in {"clean", "pair_shared"}:
        raise ValueError(f"Unsupported classifier seed role: {role}")
    prefix = (
        "ca3_inactive_downstream",
        int(global_seed),
        int(repetition_index),
        int(repeat_seed),
        role,
    )
    return ClassifierSeedPlan(
        role=role,
        repetition_index=int(repetition_index),
        repeat_seed=int(repeat_seed),
        initialization_seed=stable_seed(*prefix, "initialization"),
        dataloader_seed=stable_seed(*prefix, "dataloader_shuffle"),
        dropout_seed=stable_seed(*prefix, "dropout"),
    )


def initialize_mlp_parameters(
    classifier: nn.Module,
    generator: torch.Generator,
) -> None:
    """Reproduce torch Linear defaults using an explicit CPU generator."""
    for module in classifier.modules():
        if not isinstance(module, nn.Linear):
            continue
        nn.init.kaiming_uniform_(
            module.weight,
            a=math.sqrt(5),
            generator=generator,
        )
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(
                module.bias,
                -bound,
                bound,
                generator=generator,
            )


def create_downstream_mlp(
    input_size: int,
    args: argparse.Namespace,
    device: torch.device,
    initialization_seed: int,
) -> nn.Module:
    """Construct and initialise an MLP without changing global RNG state."""
    # nn.Linear constructors initialise parameters automatically. Fork the CPU
    # RNG so those temporary values do not leak into any later operation, then
    # overwrite them using a dedicated generator.
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(0)
        classifier = DownstreamMLP(
            input_size=input_size,
            hidden_sizes=(args.mlp_hidden_1, args.mlp_hidden_2),
            num_classes=NUM_CLASSES,
            dropout=args.mlp_dropout,
        )

    init_generator = make_torch_generator(initialization_seed)
    initialize_mlp_parameters(classifier, init_generator)
    return classifier.to(device)


def cuda_rng_devices(device: torch.device) -> List[int]:
    if device.type != "cuda":
        return []
    if device.index is not None:
        return [int(device.index)]
    return [int(torch.cuda.current_device())]



def train_downstream_mlp(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    classifier_checkpoint_path: str,
    seed_plan: ClassifierSeedPlan,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if train_features.dim() != 2 or validation_features.dim() != 2:
        raise ValueError("Downstream features must be rank-2 tensors")
    if train_features.size(1) != validation_features.size(1):
        raise ValueError("Train and validation feature dimensions differ")

    classifier = create_downstream_mlp(
        input_size=int(train_features.size(1)),
        args=args,
        device=device,
        initialization_seed=seed_plan.initialization_seed,
    )
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = make_feature_loader(
        train_features,
        train_labels,
        batch_size=args.mlp_batch_size,
        shuffle=True,
        seed=seed_plan.dataloader_seed,
    )

    best_validation_accuracy = -1.0
    best_epoch = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    print(
        f"Training downstream MLP for {run_name} "
        f"(repeat seed={seed_plan.repeat_seed})..."
    )

    # Dropout consumes the global CPU/CUDA RNG. Fork and seed those RNG states
    # for the whole training trajectory, then restore the caller's states on
    # exit. This makes the trajectory independent of pair evaluation order.
    with torch.random.fork_rng(
        devices=cuda_rng_devices(device),
        enabled=True,
    ):
        torch.manual_seed(seed_plan.dropout_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed_plan.dropout_seed)

        for epoch in range(1, args.mlp_epochs + 1):
            classifier.train()
            loss_sum = 0.0
            correct = 0
            total = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = classifier(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                predictions = logits.argmax(dim=1)
                batch_count = int(batch_labels.numel())
                loss_sum += float(loss.item()) * batch_count
                correct += int(predictions.eq(batch_labels).sum().item())
                total += batch_count

            validation = evaluate_classifier(
                classifier,
                validation_features,
                validation_labels,
                batch_size=args.mlp_eval_batch_size,
                device=device,
            )
            train_loss = loss_sum / total
            train_accuracy = correct / total
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": validation.accuracy,
                }
            )

            if (
                epoch == 1
                or epoch % args.print_every == 0
                or epoch == args.mlp_epochs
            ):
                print(
                    f"  MLP epoch [{epoch:03d}/{args.mlp_epochs:03d}] "
                    f"loss={train_loss:.6f} "
                    f"train_acc={train_accuracy * 100.0:.2f}% "
                    f"val_acc={validation.accuracy * 100.0:.2f}%"
                )

            if validation.accuracy > best_validation_accuracy:
                best_validation_accuracy = validation.accuracy
                best_epoch = epoch
                best_state_dict = copy.deepcopy(classifier.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"  Early stopping after epoch {epoch}; best epoch was "
                    f"{best_epoch}."
                )
                break

    if best_state_dict is None:
        raise RuntimeError("Downstream MLP did not produce a valid checkpoint")
    classifier.load_state_dict(best_state_dict, strict=True)
    Path(classifier_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "feature_dim": int(train_features.size(1)),
            "architecture": {
                "hidden_1": args.mlp_hidden_1,
                "hidden_2": args.mlp_hidden_2,
                "dropout": args.mlp_dropout,
                "num_classes": NUM_CLASSES,
            },
            "optimizer": {
                "name": "Adam",
                "learning_rate": args.mlp_lr,
                "weight_decay": args.mlp_weight_decay,
            },
            "history": history,
            "seed_plan": asdict(seed_plan),
            "run_name": run_name,
        },
        classifier_checkpoint_path,
    )

    return classifier, {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "history": history,
        "checkpoint": classifier_checkpoint_path,
        "feature_dim": int(train_features.size(1)),
        "seed_plan": asdict(seed_plan),
    }


def compute_adjusted_asr(asr: float, baseline_target_rate: float) -> float:
    """CA2 definition: adjusted ASR = max(0, ASR - baseline target rate)."""
    return max(0.0, asr - baseline_target_rate)


def normalized_centroid(features: torch.Tensor) -> torch.Tensor:
    if features.dim() != 2 or features.size(0) == 0:
        raise ValueError("Cannot form a centroid from empty/non-matrix features")
    return F.normalize(features.float().mean(dim=0, keepdim=True), dim=1)


def mean_cosine_to_centroid(
    features: torch.Tensor,
    centroid: torch.Tensor,
) -> float:
    normalized = F.normalize(features.float(), dim=1)
    normalized_centroid_value = F.normalize(centroid.float(), dim=1)
    return float((normalized @ normalized_centroid_value.t()).mean().item())


def feature_cosine_diagnostics(
    clean_test_features: torch.Tensor,
    pair_clean_test_features: torch.Tensor,
    clean_reference_features: torch.Tensor,
    pair_reference_features: torch.Tensor,
    clean_triggered_features: torch.Tensor,
    pair_triggered_features: torch.Tensor,
    pair_clean_non_target_features: torch.Tensor,
) -> Dict[str, float]:
    clean_target_centroid = normalized_centroid(clean_reference_features)
    pair_target_centroid = normalized_centroid(pair_reference_features)

    return {
        "clean_feature_similarity_pair_vs_clean": float(
            F.cosine_similarity(
                pair_clean_test_features.float(),
                clean_test_features.float(),
                dim=1,
            ).mean().item()
        ),
        "reference_feature_similarity_pair_vs_clean": float(
            F.cosine_similarity(
                pair_reference_features.float(),
                clean_reference_features.float(),
                dim=1,
            ).mean().item()
        ),
        "pair_triggered_to_pair_target_centroid_cosine": (
            mean_cosine_to_centroid(pair_triggered_features, pair_target_centroid)
        ),
        "pair_triggered_to_clean_target_centroid_cosine": (
            mean_cosine_to_centroid(pair_triggered_features, clean_target_centroid)
        ),
        "pair_clean_non_target_to_pair_target_centroid_cosine": (
            mean_cosine_to_centroid(
                pair_clean_non_target_features,
                pair_target_centroid,
            )
        ),
        "clean_triggered_to_clean_target_centroid_cosine": (
            mean_cosine_to_centroid(clean_triggered_features, clean_target_centroid)
        ),
        "pair_triggered_vs_pair_clean_same_image_cosine": float(
            F.cosine_similarity(
                pair_triggered_features.float(),
                pair_clean_non_target_features.float(),
                dim=1,
            ).mean().item()
        ),
    }


def save_confusion_matrix_csv(matrix: Sequence[Sequence[int]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CIFAR10_CLASS_NAMES])
        for class_name, row in zip(CIFAR10_CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])


def save_confusion_matrix_plot(
    matrix: Sequence[Sequence[int]],
    title: str,
    path: str,
) -> None:
    import matplotlib.pyplot as plt

    tensor = torch.tensor(matrix, dtype=torch.float32)
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(tensor.numpy())
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(NUM_CLASSES), CIFAR10_CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(NUM_CLASSES), CIFAR10_CLASS_NAMES)
    fig.colorbar(image, ax=axis)

    threshold = float(tensor.max().item()) / 2.0 if tensor.numel() else 0.0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = int(tensor[row, column].item())
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def classification_result_to_dict(
    result: ClassificationResult,
) -> Dict[str, Any]:
    return {
        "accuracy": result.accuracy,
        "accuracy_percent": result.accuracy * 100.0,
        "total_samples": result.total_samples,
        "correct_samples": result.correct_samples,
        "confusion_matrix": result.confusion_matrix,
        "per_class_accuracy": result.per_class_accuracy,
        "per_class_accuracy_percent": {
            key: None if value is None else value * 100.0
            for key, value in result.per_class_accuracy.items()
        },
        "prediction_distribution": result.prediction_distribution,
    }


def target_rate_result_to_dict(result: TargetRateResult) -> Dict[str, Any]:
    return {
        "target_rate": result.target_rate,
        "target_rate_percent": result.target_rate * 100.0,
        "target_predictions": result.target_predictions,
        "total_samples": result.total_samples,
        "prediction_distribution": result.prediction_distribution,
    }


def write_json(data: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)


def aggregate_series(
    values: Sequence[float],
    repeat_seeds: Sequence[int],
) -> Dict[str, Any]:
    numeric = [float(value) for value in values]
    if len(numeric) != len(repeat_seeds):
        raise ValueError("Metric values and repeat seeds have different lengths")
    if not numeric:
        raise ValueError("Cannot aggregate an empty metric series")

    std_value = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
    return {
        "mean": statistics.fmean(numeric),
        "std_sample": std_value,
        "minimum": min(numeric),
        "maximum": max(numeric),
        "per_seed": [
            {
                "repetition": index + 1,
                "seed": int(seed),
                "value": value,
            }
            for index, (seed, value) in enumerate(zip(repeat_seeds, numeric))
        ],
    }


def pair_metric_values(
    per_repeat_results: Sequence[Dict[str, Any]],
    pair_name: str,
    metric: str,
) -> List[float]:
    values: List[float] = []
    for repeat in per_repeat_results:
        pair = repeat["pairs"][pair_name]
        if metric == "ca_percent":
            value = repeat["clean_baseline"]["ca"]["accuracy_percent"]
        elif metric == "ba_percent":
            value = pair["ba"]["accuracy_percent"]
        elif metric == "ca_to_ba_drop_pp":
            value = pair["ca_to_ba_drop_percentage_points"]
        elif metric == "baseline_target_rate_percent":
            value = pair["baseline_target_rate"]["target_rate_percent"]
        elif metric == "asr_percent":
            value = pair["asr"]["target_rate_percent"]
        elif metric == "adjusted_asr_percent":
            value = pair["adjusted_asr_percent"]
        elif metric == "sanity_asr_percent":
            value = pair["clean_encoder_trigger_sanity_asr"][
                "target_rate_percent"
            ]
        else:
            raise KeyError(f"Unsupported aggregate metric: {metric}")
        values.append(float(value))
    return values


def build_pair_aggregate(
    pair: PairSpec,
    per_repeat_results: Sequence[Dict[str, Any]],
    repeat_seeds: Sequence[int],
    static_pair_data: Dict[str, Any],
) -> Dict[str, Any]:
    metric_names = (
        "ca_percent",
        "ba_percent",
        "ca_to_ba_drop_pp",
        "baseline_target_rate_percent",
        "asr_percent",
        "adjusted_asr_percent",
        "sanity_asr_percent",
    )
    return {
        "pair_name": pair.name,
        "encoder_checkpoint": pair.encoder_checkpoint,
        "trigger_checkpoint": pair.trigger_checkpoint,
        "pair_metadata": static_pair_data["pair_metadata"],
        "trigger_config": static_pair_data["trigger_config"],
        "trigger_quality": static_pair_data["trigger_quality"],
        "feature_cosine_diagnostics": static_pair_data[
            "feature_cosine_diagnostics"
        ],
        "metrics": {
            metric: aggregate_series(
                pair_metric_values(per_repeat_results, pair.name, metric),
                repeat_seeds,
            )
            for metric in metric_names
        },
    }


def make_aggregate_decision(
    pair_aggregates: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    max_drop = float(args.decision_max_ba_drop_pp)
    min_asr = float(args.decision_min_asr_percent)
    min_adjusted = float(args.decision_min_adjusted_asr_percent)

    checks: Dict[str, Dict[str, Any]] = {}
    candidates: List[str] = []
    successful: List[str] = []
    for name, result in pair_aggregates.items():
        metrics = result["metrics"]
        mean_drop = metrics["ca_to_ba_drop_pp"]["mean"]
        mean_asr = metrics["asr_percent"]["mean"]
        mean_adjusted = metrics["adjusted_asr_percent"]["mean"]
        utility_safe = mean_drop <= max_drop
        meets_all = (
            utility_safe
            and mean_asr >= min_asr
            and mean_adjusted >= min_adjusted
        )
        checks[name] = {
            "mean_ba_drop_within_limit": utility_safe,
            "mean_raw_asr_meets_target": mean_asr >= min_asr,
            "mean_adjusted_asr_meets_target": mean_adjusted >= min_adjusted,
            "meets_all_mean_thresholds": meets_all,
        }
        if utility_safe:
            candidates.append(name)
        if meets_all:
            successful.append(name)

    ranked = sorted(
        candidates,
        key=lambda name: (
            pair_aggregates[name]["metrics"]["adjusted_asr_percent"]["mean"],
            -pair_aggregates[name]["metrics"]["adjusted_asr_percent"][
                "std_sample"
            ],
            pair_aggregates[name]["metrics"]["ba_percent"]["mean"],
            pair_aggregates[name]["metrics"]["asr_percent"]["mean"],
        ),
        reverse=True,
    )

    provisional_best = ranked[0] if ranked else None
    genuinely_better: Optional[str] = None
    comparison_status = "no utility-safe pair"
    margin_details: Optional[Dict[str, float]] = None

    if len(ranked) == 1:
        genuinely_better = ranked[0]
        comparison_status = "only one pair passed the mean BA-drop limit"
    elif len(ranked) >= 2:
        top = ranked[0]
        runner_up = ranked[1]
        top_stats = pair_aggregates[top]["metrics"]["adjusted_asr_percent"]
        runner_stats = pair_aggregates[runner_up]["metrics"][
            "adjusted_asr_percent"
        ]
        margin = top_stats["mean"] - runner_stats["mean"]
        one_std_threshold = max(
            top_stats["std_sample"],
            runner_stats["std_sample"],
        )
        margin_details = {
            "mean_adjusted_asr_margin_percentage_points": margin,
            "one_std_heuristic_threshold_percentage_points": one_std_threshold,
        }
        if margin > one_std_threshold:
            genuinely_better = top
            comparison_status = (
                "top mean adjusted ASR exceeds the runner-up by more than the "
                "larger sample standard deviation"
            )
        else:
            comparison_status = (
                "ranking is inconclusive under the one-standard-deviation "
                "stability heuristic"
            )

    if not candidates:
        recommendation = (
            "No pair preserves mean BA within the configured CA-to-BA drop. "
            "Restore representation utility before pursuing higher ASR."
        )
    elif genuinely_better is None:
        recommendation = (
            "Do not claim one checkpoint is better. The repeated-seed adjusted-"
            "ASR distributions overlap under the configured stability heuristic."
        )
    elif successful:
        recommendation = (
            "A utility-safe pair is seed-stably strongest and meets the configured "
            "mean ASR targets. Proceed to robustness and defense evaluation."
        )
    else:
        recommendation = (
            "A utility-safe pair is seed-stably strongest, but its mean attack "
            "metrics remain below the configured target. A new attack-training "
            "stage may be considered only after reviewing these repeated results."
        )

    return {
        "thresholds": {
            "maximum_mean_ca_to_ba_drop_percentage_points": max_drop,
            "minimum_mean_raw_asr_percent": min_asr,
            "minimum_mean_adjusted_asr_percent": min_adjusted,
        },
        "pair_checks": checks,
        "ranking": ranked,
        "provisional_best_pair": provisional_best,
        "genuinely_better_pair": genuinely_better,
        "comparison_status": comparison_status,
        "margin_details": margin_details,
        "pairs_meeting_all_mean_thresholds": successful,
        "selection_rule": (
            "First require mean CA-to-BA drop within the configured limit. Rank "
            "surviving pairs by mean adjusted ASR, then lower adjusted-ASR sample "
            "standard deviation, then mean BA and mean raw ASR. Call the top pair "
            "genuinely better only when its mean adjusted-ASR lead exceeds the "
            "larger of the top two sample standard deviations. This is a practical "
            "stability heuristic, not a formal significance test."
        ),
        "recommendation": recommendation,
    }


def write_aggregate_csv(
    aggregate: Dict[str, Any],
    path: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    for pair_name, pair in aggregate["pairs"].items():
        metrics = pair["metrics"]
        row: Dict[str, Any] = {
            "pair": pair_name,
            "num_repeats": aggregate["protocol"]["num_repeats"],
        }
        for metric_name, stats in metrics.items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std_sample"] = stats["std_sample"]
            row[f"{metric_name}_minimum"] = stats["minimum"]
            row[f"{metric_name}_maximum"] = stats["maximum"]
            row[f"{metric_name}_per_seed"] = json.dumps(
                stats["per_seed"],
                separators=(",", ":"),
            )
        row["trigger_psnr_db"] = pair["trigger_quality"]["psnr_db"]
        row["trigger_ssim"] = pair["trigger_quality"]["ssim_index"]
        row["provisional_best"] = (
            aggregate["decision"]["provisional_best_pair"] == pair_name
        )
        row["genuinely_better"] = (
            aggregate["decision"]["genuinely_better_pair"] == pair_name
        )
        rows.append(row)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["pair"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_aggregate_table(aggregate: Dict[str, Any]) -> None:
    clean = aggregate["clean_baseline_aggregate"]["ca_percent"]
    print("\n" + "=" * 122)
    print("Repeated-seed downstream comparison")
    print("=" * 122)
    print(
        f"Clean CA: {clean['mean']:.2f} +/- {clean['std_sample']:.2f}% "
        f"(min={clean['minimum']:.2f}, max={clean['maximum']:.2f})"
    )
    header = (
        f"{'Pair':<26} {'BA mean+/-sd':>16} {'Drop pp':>13} "
        f"{'ASR mean+/-sd':>16} {'Adj ASR mean+/-sd':>20} "
        f"{'Sanity mean+/-sd':>18}"
    )
    print("-" * 122)
    print(header)
    print("-" * 122)
    for pair_name, pair in aggregate["pairs"].items():
        metrics = pair["metrics"]
        ba = metrics["ba_percent"]
        drop = metrics["ca_to_ba_drop_pp"]
        asr = metrics["asr_percent"]
        adjusted = metrics["adjusted_asr_percent"]
        sanity = metrics["sanity_asr_percent"]
        print(
            f"{pair_name:<26} "
            f"{ba['mean']:6.2f}+/-{ba['std_sample']:<6.2f} "
            f"{drop['mean']:5.2f}+/-{drop['std_sample']:<5.2f} "
            f"{asr['mean']:6.2f}+/-{asr['std_sample']:<6.2f} "
            f"{adjusted['mean']:7.2f}+/-{adjusted['std_sample']:<7.2f} "
            f"{sanity['mean']:6.2f}+/-{sanity['std_sample']:<6.2f}"
        )
    print("-" * 122)
    decision = aggregate["decision"]
    print(f"Provisional best: {decision['provisional_best_pair']}")
    print(f"Genuinely better: {decision['genuinely_better_pair']}")
    print(f"Status: {decision['comparison_status']}")
    print(decision["recommendation"])
    print("=" * 122)


def print_header(
    args: argparse.Namespace,
    device: torch.device,
    repeat_seeds: Sequence[int],
) -> None:
    print("=" * 88)
    print("CA3 INACTIVE Repeated-Seed Downstream Evaluation")
    print("=" * 88)
    print(f"Device: {device}")
    print(f"Clean SimCLR checkpoint: {args.clean_simclr_checkpoint}")
    print("Dataset: CIFAR-10 (32x32)")
    print("Feature source: model.f (L2-normalized); model.g is not used")
    print("Trigger order: raw [0,1] -> trigger -> CIFAR-10 normalization")
    print(
        f"Target: {args.target_label} "
        f"({CIFAR10_CLASS_NAMES[args.target_label]})"
    )
    print("Fixed message: alternating 16-bit")
    print(f"Maximum residual: 8/255 = {EXPECTED_MAX_RESIDUAL:.10f}")
    print(f"Global experiment seed: {args.seed}")
    print(f"Repeat seeds: {list(repeat_seeds)}")
    print(
        "Fairness: all compromised pairs share identical MLP initialisation, "
        "training order and dropout RNG within each repetition."
    )
    print("Pairs:")
    for pair in args.pairs:
        print(f"  - {pair.name}")
        print(f"      encoder: {pair.encoder_checkpoint}")
        print(f"      trigger: {pair.trigger_checkpoint}")
    print("=" * 88)


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def load_saved_classifier(
    state_dict: Dict[str, torch.Tensor],
    feature_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(0)
        classifier = DownstreamMLP(
            input_size=feature_dim,
            hidden_sizes=(args.mlp_hidden_1, args.mlp_hidden_2),
            num_classes=NUM_CLASSES,
            dropout=args.mlp_dropout,
        )
    classifier.load_state_dict(state_dict, strict=True)
    classifier.to(device)
    classifier.eval()
    return classifier


def main() -> None:
    args = parse_args()
    validate_args(args)
    repeat_seeds = resolve_repeat_seeds(args)

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
    print_header(args, device, repeat_seeds)

    data = build_datasets_and_indices(args)
    print("Dataset split")
    print(f"  classifier fit samples: {len(data['fit_indices'])}")
    print(f"  classifier validation samples: {len(data['validation_indices'])}")
    print(f"  clean test samples: {len(data['test_indices'])}")
    print(
        f"  non-target test samples for ASR: "
        f"{len(data['non_target_test_indices'])}"
    )
    print(f"  target reference samples: {len(data['reference_indices'])}")

    fit_loader = make_loader(
        data["train_base"],
        data["fit_indices"],
        args.feature_batch_size,
        args.num_workers,
        args.seed + 101,
    )
    validation_loader = make_loader(
        data["train_base"],
        data["validation_indices"],
        args.feature_batch_size,
        args.num_workers,
        args.seed + 102,
    )
    test_loader = make_loader(
        data["test_base"],
        data["test_indices"],
        args.feature_batch_size,
        args.num_workers,
        args.seed + 103,
    )
    non_target_test_loader = make_loader(
        data["test_base"],
        data["non_target_test_indices"],
        args.feature_batch_size,
        args.num_workers,
        args.seed + 104,
    )
    reference_loader = make_loader(
        data["train_base"],
        data["reference_indices"],
        args.feature_batch_size,
        args.num_workers,
        args.seed + 105,
    )

    print("\n[Clean baseline] Loading encoder and extracting frozen features...")
    clean_model, clean_checkpoint = load_simclr_model(
        args.clean_simclr_checkpoint,
        args,
        device,
        role="clean SimCLR",
    )
    clean_fit_features, fit_labels = extract_clean_features(
        clean_model,
        fit_loader,
        device,
    )
    clean_validation_features, validation_labels = extract_clean_features(
        clean_model,
        validation_loader,
        device,
    )
    clean_test_features, test_labels = extract_clean_features(
        clean_model,
        test_loader,
        device,
    )
    clean_reference_features, reference_labels = extract_clean_features(
        clean_model,
        reference_loader,
        device,
    )
    if not reference_labels.eq(args.target_label).all():
        raise RuntimeError("Reference loader contains non-target labels")

    non_target_mask = test_labels.ne(args.target_label)
    clean_non_target_features = clean_test_features[non_target_mask]

    per_repeat_results: List[Dict[str, Any]] = []
    clean_runtime: List[Dict[str, Any]] = []

    print("\n[Clean baseline] Training one classifier per repetition...")
    for repetition_index, repeat_seed in enumerate(repeat_seeds):
        repeat_number = repetition_index + 1
        repeat_dir = output_dir / "classifiers" / (
            f"repeat_{repeat_number:02d}_seed_{repeat_seed}"
        )
        classifier_path = str(repeat_dir / "downstream_mlp_clean_baseline.pth")
        seed_plan = classifier_seed_plan(
            global_seed=args.seed,
            repetition_index=repetition_index,
            repeat_seed=repeat_seed,
            role="clean",
        )
        classifier, training = train_downstream_mlp(
            clean_fit_features,
            fit_labels,
            clean_validation_features,
            validation_labels,
            args,
            device,
            run_name=f"clean baseline repeat {repeat_number}",
            classifier_checkpoint_path=classifier_path,
            seed_plan=seed_plan,
        )
        ca = evaluate_classifier(
            classifier,
            clean_test_features,
            test_labels,
            args.mlp_eval_batch_size,
            device,
        )
        baseline_rate = evaluate_target_rate(
            classifier,
            clean_non_target_features,
            args.target_label,
            args.mlp_eval_batch_size,
            device,
        )

        confusion_dir = output_dir / "confusion" / (
            f"repeat_{repeat_number:02d}_seed_{repeat_seed}"
        )
        clean_confusion_csv = str(confusion_dir / "clean_ca.csv")
        save_confusion_matrix_csv(ca.confusion_matrix, clean_confusion_csv)
        clean_confusion_png: Optional[str] = None
        if not args.skip_confusion_plots:
            clean_confusion_png = str(confusion_dir / "clean_ca.png")
            save_confusion_matrix_plot(
                ca.confusion_matrix,
                f"Clean SimCLR CA, seed {repeat_seed}",
                clean_confusion_png,
            )

        state_dict_cpu = clone_state_dict_to_cpu(classifier)
        clean_runtime.append(
            {
                "state_dict": state_dict_cpu,
                "seed_plan": seed_plan,
                "ca": ca,
                "baseline_target_rate": baseline_rate,
            }
        )
        per_repeat_results.append(
            {
                "experiment": "ca3_inactive_downstream_repetition",
                "repetition": {
                    "index": repeat_number,
                    "repeat_seed": repeat_seed,
                    "global_experiment_seed": args.seed,
                    "clean_classifier_seed_plan": asdict(seed_plan),
                    "pair_classifier_seed_plan": asdict(
                        classifier_seed_plan(
                            global_seed=args.seed,
                            repetition_index=repetition_index,
                            repeat_seed=repeat_seed,
                            role="pair_shared",
                        )
                    ),
                },
                "clean_baseline": {
                    "clean_simclr_checkpoint": args.clean_simclr_checkpoint,
                    "downstream_classifier_training": training,
                    "ca": classification_result_to_dict(ca),
                    "clean_non_target_baseline_target_rate": (
                        target_rate_result_to_dict(baseline_rate)
                    ),
                    "confusion_matrix_csv": clean_confusion_csv,
                    "confusion_matrix_png": clean_confusion_png,
                },
                "pairs": {},
            }
        )
        print(
            f"[Clean repeat {repeat_number}] seed={repeat_seed} "
            f"CA={ca.accuracy * 100.0:.2f}%"
        )
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    static_pair_data: Dict[str, Dict[str, Any]] = {}

    for pair in args.pairs:
        print("\n" + "-" * 88)
        print(f"[Pair: {pair.name}] Loading and verifying matched checkpoints...")
        backdoored_model, encoder_checkpoint = load_simclr_model(
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
        metadata = validate_matched_pair_metadata(
            pair,
            encoder_checkpoint,
            trigger_checkpoint,
            args,
        )
        validate_trigger_config(pair.name, trigger_config, trigger_checkpoint)
        print(
            f"[Pair: {pair.name}] verified={metadata['verified']} | "
            f"epoch={metadata['encoder_epoch']} | "
            f"stages=({metadata['encoder_stage']}, {metadata['trigger_stage']})"
        )

        print(f"[Pair: {pair.name}] Extracting clean frozen features once...")
        pair_fit_features, pair_fit_labels = extract_clean_features(
            backdoored_model,
            fit_loader,
            device,
        )
        pair_validation_features, pair_validation_labels = extract_clean_features(
            backdoored_model,
            validation_loader,
            device,
        )
        pair_test_features, pair_test_labels = extract_clean_features(
            backdoored_model,
            test_loader,
            device,
        )
        pair_reference_features, pair_reference_labels = extract_clean_features(
            backdoored_model,
            reference_loader,
            device,
        )
        if not torch.equal(pair_fit_labels, fit_labels):
            raise RuntimeError("Clean and backdoored fit labels differ")
        if not torch.equal(pair_validation_labels, validation_labels):
            raise RuntimeError("Clean and backdoored validation labels differ")
        if not torch.equal(pair_test_labels, test_labels):
            raise RuntimeError("Clean and backdoored test labels differ")
        if not pair_reference_labels.eq(args.target_label).all():
            raise RuntimeError("Backdoored reference labels are invalid")

        print(
            f"[Pair: {pair.name}] Generating triggered non-target images once "
            "in [0,1] and extracting frozen features..."
        )
        # The seed is pair-name based, never pair-position based. The current
        # fixed StegaStamp path is deterministic, but this fork prevents a future
        # stochastic trigger implementation from reintroducing order dependence.
        trigger_seed = stable_seed(args.seed, "trigger_eval", pair.name)
        with torch.random.fork_rng(
            devices=cuda_rng_devices(device),
            enabled=True,
        ):
            torch.manual_seed(trigger_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(trigger_seed)
            triggered = extract_triggered_features_for_two_encoders(
                clean_model=clean_model,
                backdoored_model=backdoored_model,
                trigger_model=trigger_model,
                trigger_config=trigger_config,
                loader=non_target_test_loader,
                args=args,
                device=device,
            )
        if triggered["labels"].eq(args.target_label).any():
            raise RuntimeError("ASR set contains original target-class samples")

        pair_non_target_features = pair_test_features[non_target_mask]
        cosine_metrics = feature_cosine_diagnostics(
            clean_test_features=clean_test_features,
            pair_clean_test_features=pair_test_features,
            clean_reference_features=clean_reference_features,
            pair_reference_features=pair_reference_features,
            clean_triggered_features=triggered["clean_encoder_features"],
            pair_triggered_features=triggered["backdoored_encoder_features"],
            pair_clean_non_target_features=pair_non_target_features,
        )
        static_pair_data[pair.name] = {
            "pair_metadata": metadata,
            "trigger_config": asdict(trigger_config),
            "trigger_quality": triggered["trigger_quality"],
            "feature_cosine_diagnostics": cosine_metrics,
        }

        pair_safe_name = sanitize_name(pair.name)
        for repetition_index, repeat_seed in enumerate(repeat_seeds):
            repeat_number = repetition_index + 1
            seed_plan = classifier_seed_plan(
                global_seed=args.seed,
                repetition_index=repetition_index,
                repeat_seed=repeat_seed,
                role="pair_shared",
            )
            classifier_dir = output_dir / "classifiers" / (
                f"repeat_{repeat_number:02d}_seed_{repeat_seed}"
            )
            pair_classifier_path = str(
                classifier_dir / f"downstream_mlp_{pair_safe_name}.pth"
            )
            pair_classifier, pair_training = train_downstream_mlp(
                pair_fit_features,
                pair_fit_labels,
                pair_validation_features,
                pair_validation_labels,
                args,
                device,
                run_name=f"{pair.name} repeat {repeat_number}",
                classifier_checkpoint_path=pair_classifier_path,
                seed_plan=seed_plan,
            )

            ba = evaluate_classifier(
                pair_classifier,
                pair_test_features,
                pair_test_labels,
                args.mlp_eval_batch_size,
                device,
            )
            baseline_target_rate = evaluate_target_rate(
                pair_classifier,
                pair_non_target_features,
                args.target_label,
                args.mlp_eval_batch_size,
                device,
            )
            asr = evaluate_target_rate(
                pair_classifier,
                triggered["backdoored_encoder_features"],
                args.target_label,
                args.mlp_eval_batch_size,
                device,
            )

            clean_classifier = load_saved_classifier(
                clean_runtime[repetition_index]["state_dict"],
                feature_dim=int(clean_fit_features.size(1)),
                args=args,
                device=device,
            )
            sanity_asr = evaluate_target_rate(
                clean_classifier,
                triggered["clean_encoder_features"],
                args.target_label,
                args.mlp_eval_batch_size,
                device,
            )
            clean_baseline_target_rate = clean_runtime[repetition_index][
                "baseline_target_rate"
            ]
            adjusted_asr = compute_adjusted_asr(
                asr.target_rate,
                baseline_target_rate.target_rate,
            )
            sanity_adjusted_asr = compute_adjusted_asr(
                sanity_asr.target_rate,
                clean_baseline_target_rate.target_rate,
            )
            clean_ca = clean_runtime[repetition_index]["ca"]
            ca_to_ba_drop_pp = (clean_ca.accuracy - ba.accuracy) * 100.0

            confusion_dir = output_dir / "confusion" / (
                f"repeat_{repeat_number:02d}_seed_{repeat_seed}"
            )
            pair_confusion_csv = str(
                confusion_dir / f"{pair_safe_name}_ba.csv"
            )
            save_confusion_matrix_csv(ba.confusion_matrix, pair_confusion_csv)
            pair_confusion_png: Optional[str] = None
            if not args.skip_confusion_plots:
                pair_confusion_png = str(
                    confusion_dir / f"{pair_safe_name}_ba.png"
                )
                save_confusion_matrix_plot(
                    ba.confusion_matrix,
                    f"{pair.name} BA, seed {repeat_seed}",
                    pair_confusion_png,
                )

            pair_result: Dict[str, Any] = {
                "pair_name": pair.name,
                "encoder_checkpoint": pair.encoder_checkpoint,
                "trigger_checkpoint": pair.trigger_checkpoint,
                "pair_metadata": metadata,
                "trigger_config": asdict(trigger_config),
                "downstream_classifier_training": pair_training,
                "ba": classification_result_to_dict(ba),
                "ca_to_ba_drop_percentage_points": ca_to_ba_drop_pp,
                "baseline_target_rate": target_rate_result_to_dict(
                    baseline_target_rate
                ),
                "asr": target_rate_result_to_dict(asr),
                "adjusted_asr": adjusted_asr,
                "adjusted_asr_percent": adjusted_asr * 100.0,
                "clean_encoder_trigger_sanity_asr": target_rate_result_to_dict(
                    sanity_asr
                ),
                "clean_encoder_trigger_sanity_adjusted_asr": (
                    sanity_adjusted_asr
                ),
                "clean_encoder_trigger_sanity_adjusted_asr_percent": (
                    sanity_adjusted_asr * 100.0
                ),
                "feature_cosine_diagnostics": cosine_metrics,
                "feature_cosine_note": (
                    "These are deterministic downstream diagnostics, not the "
                    "training-stage validation alignment_similarity."
                ),
                "trigger_quality": triggered["trigger_quality"],
                "confusion_matrix_csv": pair_confusion_csv,
                "confusion_matrix_png": pair_confusion_png,
            }
            per_repeat_results[repetition_index]["pairs"][pair.name] = pair_result

            print(
                f"[Pair: {pair.name} | repeat {repeat_number} | "
                f"seed {repeat_seed}] BA={ba.accuracy * 100.0:.2f}% | "
                f"ASR={asr.target_rate * 100.0:.2f}% | "
                f"adjusted={adjusted_asr * 100.0:.2f}% | "
                f"sanity={sanity_asr.target_rate * 100.0:.2f}%"
            )

            del pair_classifier
            del clean_classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del backdoored_model
        del trigger_model
        del pair_fit_features
        del pair_validation_features
        del pair_test_features
        del pair_reference_features
        del triggered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    protocol: Dict[str, Any] = {
        "dataset": "CIFAR-10",
        "image_size": 32,
        "target_label": args.target_label,
        "target_class_name": CIFAR10_CLASS_NAMES[args.target_label],
        "feature_source": "model.f",
        "projection_head_used": False,
        "feature_l2_normalized": True,
        "trigger_insertion_range": "[0, 1]",
        "normalization_order": (
            "raw image -> optional trigger -> CIFAR-10 normalization"
        ),
        "message_pattern": FIXED_MESSAGE_PATTERN,
        "message_length": EXPECTED_MESSAGE_LENGTH,
        "maximum_residual": EXPECTED_MAX_RESIDUAL,
        "asr_excludes_original_target_class": True,
        "adjusted_asr_definition": "max(0, ASR - baseline target rate)",
        "classifier_architecture": (
            f"{clean_fit_features.size(1)} -> {args.mlp_hidden_1} -> "
            f"{args.mlp_hidden_2} -> {NUM_CLASSES}"
        ),
        "classifier_selection": (
            "best fixed-split CIFAR-10 validation accuracy per repetition"
        ),
        "global_experiment_seed": args.seed,
        "num_repeats": args.num_repeats,
        "repeat_seeds": repeat_seeds,
        "data_split_seed": args.seed + 11,
        "pair_order_invariant_seed_rule": (
            "pair classifier seeds depend on global seed, repetition index and "
            "repeat seed; pair name and command position are excluded"
        ),
        "pair_classifier_fairness": (
            "all pairs share initial weights, DataLoader order and dropout RNG "
            "within each repetition when architecture dimensions match"
        ),
    }
    data_counts = {
        "classifier_fit": len(data["fit_indices"]),
        "classifier_validation": len(data["validation_indices"]),
        "test": len(data["test_indices"]),
        "non_target_test": len(data["non_target_test_indices"]),
        "target_references": len(data["reference_indices"]),
    }

    per_seed_files: List[str] = []
    for repetition_index, result in enumerate(per_repeat_results):
        result["protocol"] = protocol
        result["data_counts"] = data_counts
        repeat_seed = repeat_seeds[repetition_index]
        repeat_number = repetition_index + 1
        per_seed_path = output_dir / "per_seed" / (
            f"repeat_{repeat_number:02d}_seed_{repeat_seed}.json"
        )
        write_json(result, str(per_seed_path))
        per_seed_files.append(str(per_seed_path))

    clean_ca_values = [
        repeat["clean_baseline"]["ca"]["accuracy_percent"]
        for repeat in per_repeat_results
    ]
    clean_rate_values = [
        repeat["clean_baseline"]["clean_non_target_baseline_target_rate"][
            "target_rate_percent"
        ]
        for repeat in per_repeat_results
    ]
    pair_aggregates = {
        pair.name: build_pair_aggregate(
            pair,
            per_repeat_results,
            repeat_seeds,
            static_pair_data[pair.name],
        )
        for pair in args.pairs
    }
    decision = make_aggregate_decision(pair_aggregates, args)
    aggregate: Dict[str, Any] = {
        "experiment": "ca3_inactive_downstream_repeated_seed_aggregate",
        "protocol": protocol,
        "data_counts": data_counts,
        "clean_baseline": {
            "clean_simclr_checkpoint": args.clean_simclr_checkpoint,
            "checkpoint_extra": checkpoint_extra(clean_checkpoint),
        },
        "clean_baseline_aggregate": {
            "ca_percent": aggregate_series(clean_ca_values, repeat_seeds),
            "clean_non_target_target_rate_percent": aggregate_series(
                clean_rate_values,
                repeat_seeds,
            ),
        },
        "pairs": pair_aggregates,
        "decision": decision,
        "per_seed_json_files": per_seed_files,
    }

    aggregate_json = str(output_dir / "inactive_downstream_aggregate.json")
    aggregate_csv = str(output_dir / "inactive_downstream_aggregate.csv")
    legacy_json = str(output_dir / "inactive_downstream_metrics.json")
    legacy_csv = str(output_dir / "inactive_downstream_summary.csv")
    write_json(aggregate, aggregate_json)
    write_json(aggregate, legacy_json)
    write_aggregate_csv(aggregate, aggregate_csv)
    write_aggregate_csv(aggregate, legacy_csv)

    print_aggregate_table(aggregate)
    print(f"Per-seed JSON directory: {output_dir / 'per_seed'}")
    print(f"Aggregate JSON: {aggregate_json}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(
        "Backward-compatible aliases: "
        f"{legacy_json} and {legacy_csv}"
    )


if __name__ == "__main__":
    main()