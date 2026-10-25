import os
import argparse
from typing import Dict, Any, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from stegastamp_like_model import StegaStampCIFAR
from stegastamp_like_losses import (
    StegaStampLoss,
    StegaStampLossWeights
)
from stegastamp_like_corruptions import (
    CORRUPTION_PROFILES,
    build_corruption_config,
    get_corruption_strength
)
from stegastamp_like_utils import (
    set_random_seed,
    get_device,
    ensure_dir,
    create_random_message,
    create_fixed_message,
    repeat_message_to_batch,
    MetricTracker,
    print_metrics,
    append_metrics_to_csv,
    save_json,
    save_checkpoint,
    load_checkpoint,
    save_trigger_visualization,
    count_trainable_parameters,
    count_total_parameters
)


# ============================================================
# train_stegastamp_like.py
#
# Stage 1 / Stage 2 CA3 training:
#
# Train the Advanced StegaStamp V2 trigger generator.
#
# This stage trains:
#   image + message
#   -> advanced stego hider
#   -> triggered image
#   -> optional SSL-style corruption
#   -> decoder
#   -> recovered message
#
# Loss used here:
#   L_stego
#   = L_message
#   + L_stealth
#   + L_residual
#   + L_frequency
#   + L_texture
#
# This file does NOT use INACTIVE disentanglement yet.
#
# The next stage will use:
#   inactive_disentangle_losses.py
#
# Expected image range:
#   [0, 1]
# ============================================================


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Advanced StegaStamp V2 trigger"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for CIFAR-10 dataset"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_stegastamp_ca3",
        help="Directory to save checkpoints, logs, and visualizations"
    )

    parser.add_argument(
        "--train-subset",
        type=int,
        default=0,
        help="Use only this many training samples. 0 means full train set."
    )

    parser.add_argument(
        "--val-subset",
        type=int,
        default=0,
        help="Use only this many validation samples. 0 means full test set."
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    parser.add_argument(
        "--message-length",
        type=int,
        default=16,
        help="Number of hidden message bits"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Image size"
    )

    parser.add_argument(
        "--message-map-size",
        type=int,
        default=8,
        help="Spatial message map size"
    )

    parser.add_argument(
        "--message-channels",
        type=int,
        default=8,
        help="Number of projected message channels"
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base CNN channel width"
    )

    parser.add_argument(
        "--max-residual",
        type=float,
        default=8 / 255,
        help="Maximum residual magnitude"
    )

    parser.add_argument(
        "--disable-spatial-transformer",
        action="store_true",
        help="Disable decoder spatial transformer"
    )

    parser.add_argument(
        "--disable-corruption",
        action="store_true",
        help="Disable corruption layer"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay"
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Use 0 to disable."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--initialize-from",
        type=str,
        default="",
        help=(
            "Start a new training phase from checkpoint model weights only. "
            "Optimizer, scheduler, epoch count, and best metrics are reset."
        )
    )

    parser.add_argument(
        "--best-min-bit-accuracy",
        type=float,
        default=0.90,
        help=(
            "Minimum validation bit accuracy required before checkpoint "
            "selection prioritizes PSNR."
        )
    )

    parser.add_argument(
        "--best-min-message-accuracy",
        type=float,
        default=0.25,
        help=(
            "Minimum validation full-message accuracy required before "
            "checkpoint selection prioritizes PSNR."
        )
    )

    parser.add_argument(
        "--best-min-clean-bit-accuracy",
        type=float,
        default=0.98,
        help="Minimum worst-case clean random/fixed bit accuracy."
    )

    parser.add_argument(
        "--best-min-clean-fixed-message-accuracy",
        type=float,
        default=0.95,
        help="Minimum clean fixed-message full-message accuracy."
    )

    parser.add_argument(
        "--best-min-corrupted-bit-accuracy",
        type=float,
        default=0.90,
        help="Minimum worst-case corrupted random/fixed bit accuracy."
    )

    parser.add_argument(
        "--best-min-clean-psnr-db",
        type=float,
        default=37.0,
        help="Minimum worst-case clean encoded-image PSNR."
    )

    parser.add_argument(
        "--best-min-clean-ssim",
        type=float,
        default=0.985,
        help="Minimum worst-case clean encoded-image SSIM."
    )

    parser.add_argument(
        "--best-max-saturation-ratio",
        type=float,
        default=0.065,
        help="Maximum allowed worst-case residual saturation ratio."
    )

    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=5,
        help=(
            "Freeze the StegaStamp hider for the first N corruption "
            "epochs so the decoder adapts first."
        )
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Resume an interrupted run. Restores model, optimizer, "
            "scheduler, epoch, and best metrics."
        )
    )

    # --------------------------------------------------------
    # Corruption schedule
    # --------------------------------------------------------
    parser.add_argument(
        "--corruption-profile",
        type=str,
        default="combined",
        choices=CORRUPTION_PROFILES,
        help=(
            "Corruption family used for training and corrupted "
            "validation. Use individual profiles for ablation runs."
        )
    )

    parser.add_argument(
        "--initial-corruption-strength",
        type=float,
        default=0.05,
        help="Mild corruption strength used during warmup."
    )

    parser.add_argument(
        "--corruption-warmup-epochs",
        type=int,
        default=2,
        help="Epochs held at the initial corruption strength."
    )

    parser.add_argument(
        "--corruption-ramp-epochs",
        type=int,
        default=13,
        help="Epochs used to ramp to the maximum corruption strength."
    )

    parser.add_argument(
        "--max-corruption-strength",
        type=float,
        default=0.25,
        help="Maximum corruption strength for this training phase."
    )

    parser.add_argument(
        "--eval-corruption-strength",
        type=float,
        default=-1.0,
        help=(
            "Fixed corruption strength used for validation. A negative "
            "value uses --max-corruption-strength."
        )
    )

    # --------------------------------------------------------
    # Loss weights
    # --------------------------------------------------------
    parser.add_argument(
        "--message-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--clean-message-weight",
        type=float,
        default=2.0,
        help=(
            "Additional BCE weight for decoding the clean encoded image "
            "during corruption training."
        )
    )

    parser.add_argument(
        "--image-l1-weight",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--image-mse-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--ssim-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--psnr-weight",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--target-psnr-db",
        type=float,
        default=35.0
    )

    parser.add_argument(
        "--lpips-weight",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--use-lpips",
        action="store_true",
        help="Use LPIPS loss. Requires lpips package."
    )

    parser.add_argument(
        "--residual-l1-weight",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--residual-l2-weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--residual-linf-weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--residual-tv-weight",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--frequency-weight",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--texture-weight",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--residual-mask-l1-weight",
        type=float,
        default=0.01
    )

    # --------------------------------------------------------
    # Logging / saving
    # --------------------------------------------------------
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Print every N batches"
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate every N epochs"
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )

    parser.add_argument(
        "--visualize-every",
        type=int,
        default=5,
        help="Save trigger visualization every N epochs"
    )

    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=20,
        help="Limit validation batches. 0 means full validation set."
    )

    parser.add_argument(
        "--fixed-message-pattern",
        type=str,
        default="alternating",
        help="Pattern used for visualization trigger message"
    )

    return parser.parse_args()


# ============================================================
# Data
# ============================================================

def make_subset(
    dataset,
    subset_size: int,
    seed: int
):
    """
    Create deterministic subset.

    subset_size = 0 means use the full dataset.
    """

    if subset_size <= 0:
        return dataset

    subset_size = min(subset_size, len(dataset))

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:subset_size].tolist()

    return Subset(dataset, indices)


def build_dataloaders(args):
    """
    Build CIFAR-10 train and validation dataloaders.

    Important:
        We use ToTensor only.

    Reason:
        StegaStamp hider works in image space [0, 1].
        Do not normalize before hiding the message.
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_dataset = make_subset(
        dataset=train_dataset,
        subset_size=args.train_subset,
        seed=args.seed
    )

    val_dataset = make_subset(
        dataset=val_dataset,
        subset_size=args.val_subset,
        seed=args.seed + 1
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader


# ============================================================
# Model / loss / optimizer
# ============================================================

def build_model(args, device):
    """
    Build Advanced StegaStamp V2 model.
    """

    model = StegaStampCIFAR(
        message_length=args.message_length,
        image_size=args.image_size,
        message_map_size=args.message_map_size,
        message_channels=args.message_channels,
        base_channels=args.base_channels,
        max_residual=args.max_residual,
        use_spatial_transformer=not args.disable_spatial_transformer,
        use_corruption=not args.disable_corruption
    )

    # Select the corruption family defined in
    # stegastamp_like_corruptions.py.
    if model.corruption is not None:
        model.corruption.config = build_corruption_config(
            args.corruption_profile
        )

    model = model.to(device)

    return model


def build_loss_fn(args, device):
    """
    Build Advanced StegaStamp loss.
    """

    weights = StegaStampLossWeights(
        message_weight=args.message_weight,
        clean_message_weight=args.clean_message_weight,

        image_l1_weight=args.image_l1_weight,
        image_mse_weight=args.image_mse_weight,

        ssim_weight=args.ssim_weight,
        psnr_weight=args.psnr_weight,
        target_psnr_db=args.target_psnr_db,

        lpips_weight=args.lpips_weight,

        residual_l1_weight=args.residual_l1_weight,
        residual_l2_weight=args.residual_l2_weight,
        residual_linf_weight=args.residual_linf_weight,
        residual_tv_weight=args.residual_tv_weight,

        frequency_weight=args.frequency_weight,
        texture_weight=args.texture_weight,

        residual_mask_l1_weight=args.residual_mask_l1_weight
    )

    loss_fn = StegaStampLoss(
        weights=weights,
        max_residual=args.max_residual,
        use_lpips=args.use_lpips
    )

    loss_fn = loss_fn.to(device)

    return loss_fn


def build_optimizer(args, model):
    """
    AdamW optimizer.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    return optimizer


def build_scheduler(args, optimizer):
    """
    Cosine learning rate schedule.
    """

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05
    )

    return scheduler


# ============================================================
# Metric helpers
# ============================================================

def prepare_logging_metrics(
    metrics: Dict[str, Any]
) -> Dict[str, float]:
    """
    Convert metric dictionary to plain floats and add readable residual units.
    """

    clean_metrics = {}

    for key, value in metrics.items():
        if torch.is_tensor(value):
            clean_metrics[key] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            clean_metrics[key] = float(value)

    if "mean_abs_residual" in clean_metrics:
        clean_metrics["mean_abs_residual_255"] = (
            clean_metrics["mean_abs_residual"] * 255.0
        )

    if "max_abs_residual" in clean_metrics:
        clean_metrics["max_abs_residual_255"] = (
            clean_metrics["max_abs_residual"] * 255.0
        )

    return clean_metrics


def prefix_metrics(
    metrics: Dict[str, float],
    prefix: str
) -> Dict[str, float]:
    """
    Add prefix to metric keys.
    """

    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


# ============================================================
# Clean / corrupted message paths
# ============================================================

def set_hider_trainable(model, trainable: bool):
    """Enable or disable gradients for the StegaStamp hider."""

    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def forward_message_paths(
    model,
    image: torch.Tensor,
    message: torch.Tensor,
    apply_corruption: bool,
    corruption_strength: float,
    preserve_clean_decoding: bool
) -> Dict[str, torch.Tensor]:
    """Generate one trigger and decode clean and/or corrupted paths."""

    encoded_image, _, residual_mask = model.encoder.forward_with_mask(
        image,
        message
    )

    residual = encoded_image - image

    if apply_corruption:
        if model.corruption is None:
            raise RuntimeError(
                "Corruption was requested, but the model has no corruption layer."
            )

        corrupted_image = model.corruption(
            encoded_image,
            strength=corruption_strength
        )
    else:
        corrupted_image = encoded_image

    outputs = {
        "encoded_image": encoded_image,
        "residual": residual,
        "residual_mask": residual_mask,
        "corrupted_image": corrupted_image
    }

    if apply_corruption and preserve_clean_decoding:
        decoder_input = torch.cat(
            [corrupted_image, encoded_image],
            dim=0
        )

        all_logits = model.decoder(decoder_input)
        corrupted_logits, clean_logits = torch.split(
            all_logits,
            image.size(0),
            dim=0
        )

        outputs["decoded_logits"] = corrupted_logits
        outputs["clean_decoded_logits"] = clean_logits
    else:
        outputs["decoded_logits"] = model.decoder(corrupted_image)

    return outputs


# ============================================================
# Training one epoch
# ============================================================

def train_one_epoch(
    model,
    loss_fn,
    optimizer,
    train_loader,
    device,
    args,
    epoch: int,
    corruption_strength: float
):
    """
    Train Advanced StegaStamp for one epoch.
    """

    model.train()

    # requires_grad=False stops parameter updates, while eval() also
    # prevents the frozen hider's normalization statistics from drifting.
    hider_is_trainable = any(
        parameter.requires_grad
        for parameter in model.encoder.parameters()
    )

    if not hider_is_trainable:
        model.encoder.eval()

    tracker = MetricTracker()

    total_batches = len(train_loader)

    for batch_index, batch in enumerate(train_loader):
        images, _ = batch

        images = images.to(
            device,
            non_blocking=True
        )

        messages = create_random_message(
            batch_size=images.size(0),
            message_length=args.message_length,
            device=device
        )

        apply_corruption = (
            not args.disable_corruption
            and corruption_strength > 0.0
        )

        outputs = forward_message_paths(
            model=model,
            image=images,
            message=messages,
            apply_corruption=apply_corruption,
            corruption_strength=corruption_strength,
            preserve_clean_decoding=(
                apply_corruption
                and args.clean_message_weight > 0.0
            )
        )

        total_loss, loss_dict = loss_fn(
            outputs=outputs,
            clean_image=images,
            message=messages
        )

        optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip
            )

        optimizer.step()

        metrics = prepare_logging_metrics(loss_dict)

        tracker.update(
            metrics,
            n=images.size(0)
        )

        if (
            args.print_every > 0
            and (
                batch_index % args.print_every == 0
                or batch_index == total_batches - 1
            )
        ):
            current_metrics = tracker.averages()

            print_metrics(
                prefix=(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Batch [{batch_index + 1}/{total_batches}] "
                    f"Corruption {corruption_strength:.3f}"
                ),
                metrics=current_metrics,
                keys=[
                    "loss_total",
                    "loss_message_corrupted",
                    "loss_message_clean",
                    "corrupted_bit_accuracy",
                    "corrupted_message_accuracy",
                    "clean_bit_accuracy",
                    "clean_message_accuracy",
                    "psnr_db",
                    "ssim_index",
                    "mean_abs_residual_255",
                    "max_abs_residual_255",
                    "residual_saturation_ratio"
                ],
                precision=4
            )

    return tracker.averages()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loss_fn,
    val_loader,
    device,
    args,
    apply_corruption: bool,
    corruption_strength: float = 0.0,
    fixed_message: bool = False,
    corruption_seed: int = 0
):
    """
    Evaluate one clean or corrupted message mode.

    Random messages are regenerated deterministically every epoch. The
    corruption RNG is also fixed so checkpoint comparisons are not affected
    by different random corruption samples. Stealth metrics always compare
    the clean image with the uncorrupted encoded image.
    """

    model.eval()

    tracker = MetricTracker()

    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(args.seed + 10_000)

    fixed_message_tensor = None

    if fixed_message:
        fixed_message_tensor = create_fixed_message(
            message_length=args.message_length,
            device=device,
            pattern=args.fixed_message_pattern
        )

    rng_devices = []

    if device.type == "cuda":
        rng_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(corruption_seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(corruption_seed)

        for batch_index, batch in enumerate(val_loader):
            if (
                args.max_eval_batches > 0
                and batch_index >= args.max_eval_batches
            ):
                break

            images, _ = batch

            images = images.to(
                device,
                non_blocking=True
            )

            if fixed_message:
                messages = repeat_message_to_batch(
                    message=fixed_message_tensor,
                    batch_size=images.size(0)
                )
            else:
                messages = torch.randint(
                    low=0,
                    high=2,
                    size=(
                        images.size(0),
                        args.message_length
                    ),
                    generator=eval_generator,
                    device="cpu"
                )

                messages = messages.float().to(device)

            outputs = forward_message_paths(
                model=model,
                image=images,
                message=messages,
                apply_corruption=apply_corruption,
                corruption_strength=corruption_strength,
                preserve_clean_decoding=apply_corruption
            )

            _, loss_dict = loss_fn(
                outputs=outputs,
                clean_image=images,
                message=messages
            )

            metrics = prepare_logging_metrics(loss_dict)

            tracker.update(
                metrics,
                n=images.size(0)
            )

    return tracker.averages()


# ============================================================
# Visualization
# ============================================================

@torch.no_grad()
def save_epoch_visualization(
    model,
    val_loader,
    device,
    args,
    epoch: int
):
    """
    Save clean image, triggered image, residual, abs residual, and mask.
    """

    model.eval()

    images, _ = next(iter(val_loader))

    images = images.to(device)

    max_items = min(8, images.size(0))

    images = images[:max_items]

    fixed_message = create_fixed_message(
        message_length=args.message_length,
        device=device,
        pattern=args.fixed_message_pattern
    )

    messages = repeat_message_to_batch(
        message=fixed_message,
        batch_size=images.size(0)
    )

    outputs = model(
        image=images,
        message=messages,
        apply_corruption=False,
        corruption_strength=0.0
    )

    encoded_images = outputs["encoded_image"]
    residual = outputs["residual"]
    residual_mask = outputs.get("residual_mask", None)

    visualization_dir = os.path.join(
        args.output_dir,
        "visualizations"
    )

    ensure_dir(visualization_dir)

    visualization_path = os.path.join(
        visualization_dir,
        f"epoch_{epoch:04d}.png"
    )

    save_trigger_visualization(
        clean_image=images,
        encoded_image=encoded_images,
        residual=residual,
        residual_mask=residual_mask,
        path=visualization_path,
        max_items=max_items,
        residual_scale=10.0,
        abs_residual_scale=20.0
    )

    print(f"Saved visualization to: {visualization_path}")


# ============================================================
# Checkpoint scoring
# ============================================================

def build_robustness_summary(
    clean_random: Dict[str, float],
    clean_fixed: Dict[str, float],
    corrupted_random: Dict[str, float],
    corrupted_fixed: Dict[str, float]
) -> Dict[str, float]:
    """Build conservative worst-case clean and corrupted metrics."""

    return {
        "clean_bit_accuracy": min(
            clean_random.get("bit_accuracy", 0.0),
            clean_fixed.get("bit_accuracy", 0.0)
        ),
        "clean_random_message_accuracy": clean_random.get(
            "message_accuracy", 0.0
        ),
        "clean_fixed_message_accuracy": clean_fixed.get(
            "message_accuracy", 0.0
        ),
        "corrupted_bit_accuracy": min(
            corrupted_random.get("bit_accuracy", 0.0),
            corrupted_fixed.get("bit_accuracy", 0.0)
        ),
        "corrupted_message_accuracy": min(
            corrupted_random.get("message_accuracy", 0.0),
            corrupted_fixed.get("message_accuracy", 0.0)
        ),
        "clean_psnr_db": min(
            clean_random.get("psnr_db", 0.0),
            clean_fixed.get("psnr_db", 0.0)
        ),
        "clean_ssim": min(
            clean_random.get("ssim_index", 0.0),
            clean_fixed.get("ssim_index", 0.0)
        ),
        "max_saturation_ratio": max(
            clean_random.get("residual_saturation_ratio", 1.0),
            clean_fixed.get("residual_saturation_ratio", 1.0)
        ),
        "max_abs_residual_255": max(
            clean_random.get("max_abs_residual_255", float("inf")),
            clean_fixed.get("max_abs_residual_255", float("inf"))
        )
    }


def corruption_checkpoint_score(
    summary: Dict[str, float],
    args
) -> Tuple[Tuple[float, ...], Dict[str, bool]]:
    """Rank only checkpoints that preserve clean decoding and stealth."""

    checks = {
        "clean_bit_accuracy": (
            summary["clean_bit_accuracy"]
            >= args.best_min_clean_bit_accuracy
        ),
        "clean_random_message_accuracy": (
            summary["clean_random_message_accuracy"]
            >= args.best_min_message_accuracy
        ),
        "clean_fixed_message_accuracy": (
            summary["clean_fixed_message_accuracy"]
            >= args.best_min_clean_fixed_message_accuracy
        ),
        "corrupted_bit_accuracy": (
            summary["corrupted_bit_accuracy"]
            >= args.best_min_corrupted_bit_accuracy
        ),
        "clean_psnr_db": (
            summary["clean_psnr_db"]
            >= args.best_min_clean_psnr_db
        ),
        "clean_ssim": (
            summary["clean_ssim"]
            >= args.best_min_clean_ssim
        ),
        "max_saturation_ratio": (
            summary["max_saturation_ratio"]
            <= args.best_max_saturation_ratio
        ),
        "max_abs_residual": (
            summary["max_abs_residual_255"]
            <= args.max_residual * 255.0 + 1e-3
        )
    }

    eligible = all(checks.values())
    passed_count = sum(int(value) for value in checks.values())

    score = (
        float(eligible),
        float(passed_count),
        summary["corrupted_bit_accuracy"],
        summary["corrupted_message_accuracy"],
        summary["clean_bit_accuracy"],
        summary["clean_fixed_message_accuracy"],
        summary["clean_psnr_db"],
        summary["clean_ssim"],
        -summary["max_saturation_ratio"]
    )

    return score, checks

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.resume and args.initialize_from:
        raise ValueError(
            "Use only one of --resume or --initialize-from, not both."
        )

    if not 0.0 <= args.best_min_bit_accuracy <= 1.0:
        raise ValueError(
            "--best-min-bit-accuracy must be in [0, 1]."
        )
    
    if not 0.0 <= args.best_min_message_accuracy <= 1.0:
        raise ValueError(
            "--best-min-message-accuracy must be in [0, 1]."
        )

    bounded_accuracy_args = {
        "--best-min-clean-bit-accuracy": args.best_min_clean_bit_accuracy,
        "--best-min-clean-fixed-message-accuracy": (
            args.best_min_clean_fixed_message_accuracy
        ),
        "--best-min-corrupted-bit-accuracy": (
            args.best_min_corrupted_bit_accuracy
        ),
        "--best-min-clean-ssim": args.best_min_clean_ssim,
        "--best-max-saturation-ratio": args.best_max_saturation_ratio
    }

    for name, value in bounded_accuracy_args.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")

    if args.best_min_clean_psnr_db < 0.0:
        raise ValueError("--best-min-clean-psnr-db cannot be negative.")

    if not 0.0 <= args.initial_corruption_strength <= 1.0:
        raise ValueError("--initial-corruption-strength must be in [0, 1].")

    if not 0.0 <= args.max_corruption_strength <= 1.0:
        raise ValueError("--max-corruption-strength must be in [0, 1].")

    if args.initial_corruption_strength > args.max_corruption_strength:
        raise ValueError(
            "--initial-corruption-strength cannot exceed "
            "--max-corruption-strength."
        )

    if args.corruption_warmup_epochs < 0:
        raise ValueError(
            "--corruption-warmup-epochs cannot be negative."
        )

    if args.corruption_ramp_epochs < 0:
        raise ValueError(
            "--corruption-ramp-epochs cannot be negative."
        )

    if args.freeze_encoder_epochs < 0:
        raise ValueError(
            "--freeze-encoder-epochs cannot be negative."
        )

    if args.clean_message_weight < 0.0:
        raise ValueError(
            "--clean-message-weight cannot be negative."
        )

    if (
        args.eval_corruption_strength > 1.0
        or args.eval_corruption_strength < -1.0
    ):
        raise ValueError(
            "--eval-corruption-strength must be -1 or a value in [0, 1]."
        )

    set_random_seed(args.seed)

    device = get_device()

    ensure_dir(args.output_dir)

    checkpoint_dir = os.path.join(
        args.output_dir,
        "checkpoints"
    )

    ensure_dir(checkpoint_dir)

    log_dir = os.path.join(
        args.output_dir,
        "logs"
    )

    ensure_dir(log_dir)

    csv_path = os.path.join(
        log_dir,
        "train_log.csv"
    )

    config_path = os.path.join(
        args.output_dir,
        "config.json"
    )

    save_json(
        data=vars(args),
        json_path=config_path
    )

    print("=" * 80)
    print("Advanced StegaStamp V2 Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Message length: {args.message_length}")
    print(f"Max residual: {args.max_residual:.6f}")
    print(f"Max residual in 8-bit scale: {args.max_residual * 255:.2f}")
    print(f"Corruption profile: {args.corruption_profile}")
    print("=" * 80)

    train_loader, val_loader = build_dataloaders(args)

    model = build_model(
        args=args,
        device=device
    )

    loss_fn = build_loss_fn(
        args=args,
        device=device
    )

    optimizer = build_optimizer(
        args=args,
        model=model
    )

    scheduler = build_scheduler(
        args=args,
        optimizer=optimizer
    )

    print(f"Total parameters: {count_total_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    start_epoch = 1
    best_bit_accuracy = 0.0
    best_message_accuracy = 0.0
    best_psnr = 0.0
    best_robust_score: Optional[Tuple[float, ...]] = None
    best_robust_summary: Dict[str, float] = {}

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")

        checkpoint = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=str(device),
            strict=True
        )

        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1

        if "extra" in checkpoint:
            extra = checkpoint["extra"]
            best_bit_accuracy = float(
                extra.get("best_bit_accuracy", 0.0)
            )
            best_message_accuracy = float(
                extra.get("best_message_accuracy", 0.0)
            )
            best_psnr = float(
                extra.get("best_psnr", 0.0)
            )
            saved_score = extra.get("best_robust_score", None)
            if saved_score is not None:
                best_robust_score = tuple(float(x) for x in saved_score)

            best_robust_summary = dict(
                extra.get("best_robust_summary", {})
            )

        print(f"Resume start epoch: {start_epoch}")
    
    elif args.initialize_from:
        print(
            "Starting a new training phase from model weights: "
            f"{args.initialize_from}"
        )

        load_checkpoint(
            path=args.initialize_from,
            model=model,
            optimizer=None,
            scheduler=None,
            map_location=str(device),
            strict=True
        )

        # A new phase uses a fresh optimizer, scheduler and scoring state.
        start_epoch = 1
        best_bit_accuracy = 0.0
        best_message_accuracy = 0.0
        best_psnr = 0.0
        best_robust_score = None
        best_robust_summary = {}

        print(
            "Optimizer, scheduler, epoch count, and best metrics "
            "were reset."
        )

    # Save initial visualization before training.
    save_epoch_visualization(
        model=model,
        val_loader=val_loader,
        device=device,
        args=args,
        epoch=0
    )

    for epoch in range(start_epoch, args.epochs + 1):
        if args.disable_corruption:
            corruption_strength = 0.0
        else:
            corruption_strength = get_corruption_strength(
                epoch=epoch,
                warmup_epochs=args.corruption_warmup_epochs,
                ramp_epochs=args.corruption_ramp_epochs,
                initial_strength=args.initial_corruption_strength,
                max_strength=args.max_corruption_strength
            )

        hider_trainable = (
            args.disable_corruption
            or epoch > args.freeze_encoder_epochs
        )

        set_hider_trainable(
            model=model,
            trainable=hider_trainable
        )

        # Record the learning rate actually used during this epoch.
        epoch_learning_rate = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            args=args,
            epoch=epoch,
            corruption_strength=corruption_strength
        )

        # Update the learning rate for the next epoch.
        scheduler.step()

        row_metrics = {
            "epoch": epoch,
            "corruption_strength": corruption_strength,
            "learning_rate": epoch_learning_rate,
            "hider_trainable": float(hider_trainable),
        }

        row_metrics.update(
            prefix_metrics(train_metrics, "train")
        )

        should_eval = (
            args.eval_every > 0
            and epoch % args.eval_every == 0
        )

        if should_eval:
            eval_corruption_strength = args.eval_corruption_strength
            if eval_corruption_strength < 0.0:
                eval_corruption_strength = args.max_corruption_strength

            val_clean_random = evaluate(
                model=model,
                loss_fn=loss_fn,
                val_loader=val_loader,
                device=device,
                args=args,
                apply_corruption=False,
                corruption_strength=0.0,
                fixed_message=False,
                corruption_seed=args.seed + 30_000
            )

            val_clean_fixed = evaluate(
                model=model,
                loss_fn=loss_fn,
                val_loader=val_loader,
                device=device,
                args=args,
                apply_corruption=False,
                corruption_strength=0.0,
                fixed_message=True,
                corruption_seed=args.seed + 30_000
            )

            val_corrupted_random = evaluate(
                model=model,
                loss_fn=loss_fn,
                val_loader=val_loader,
                device=device,
                args=args,
                apply_corruption=True,
                corruption_strength=eval_corruption_strength,
                fixed_message=False,
                corruption_seed=args.seed + 30_000
            )

            val_corrupted_fixed = evaluate(
                model=model,
                loss_fn=loss_fn,
                val_loader=val_loader,
                device=device,
                args=args,
                apply_corruption=True,
                corruption_strength=eval_corruption_strength,
                fixed_message=True,
                corruption_seed=args.seed + 30_000
            )

            row_metrics.update(prefix_metrics(val_clean_random, "val_clean_random"))
            row_metrics.update(prefix_metrics(val_clean_fixed, "val_clean_fixed"))
            row_metrics.update(prefix_metrics(val_corrupted_random, "val_corrupted_random"))
            row_metrics.update(prefix_metrics(val_corrupted_fixed, "val_corrupted_fixed"))

            validation_modes = [
                ("Clean Random Messages", val_clean_random),
                ("Clean Fixed Message", val_clean_fixed),
                ("Corrupted Random Messages", val_corrupted_random),
                ("Corrupted Fixed Message", val_corrupted_fixed)
            ]

            for mode_name, mode_metrics in validation_modes:
                print_metrics(
                    prefix=f"Epoch [{epoch}/{args.epochs}] Validation {mode_name}:",
                    metrics=mode_metrics,
                    keys=[
                        "loss_total",
                        "loss_message",
                        "bit_accuracy",
                        "message_accuracy",
                        "psnr_db",
                        "ssim_index",
                        "mean_abs_residual_255",
                        "max_abs_residual_255",
                        "residual_saturation_ratio"
                    ],
                    precision=4
                )

            summary = build_robustness_summary(
                clean_random=val_clean_random,
                clean_fixed=val_clean_fixed,
                corrupted_random=val_corrupted_random,
                corrupted_fixed=val_corrupted_fixed
            )

            current_score, constraint_checks = corruption_checkpoint_score(
                summary=summary,
                args=args
            )

            row_metrics.update(prefix_metrics(summary, "checkpoint"))
            row_metrics.update({
                f"constraint_{name}": float(passed)
                for name, passed in constraint_checks.items()
            })

            current_is_eligible = all(constraint_checks.values())
            better = (
                current_is_eligible
                and (
                    best_robust_score is None
                    or current_score > best_robust_score
                )
            )

            if better:
                best_robust_score = current_score
                best_robust_summary = summary
                best_bit_accuracy = summary["corrupted_bit_accuracy"]
                best_message_accuracy = summary["corrupted_message_accuracy"]
                best_psnr = summary["clean_psnr_db"]

                best_path = os.path.join(
                    checkpoint_dir,
                    "advanced_stegastamp_best.pth"
                )

                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=best_bit_accuracy,
                    extra={
                        "best_bit_accuracy": best_bit_accuracy,
                        "best_message_accuracy": best_message_accuracy,
                        "best_psnr": best_psnr,
                        "best_robust_score": list(best_robust_score),
                        "best_robust_summary": best_robust_summary,
                        "args": vars(args)
                    }
                )

                print(f"Saved best corruption-robust checkpoint to: {best_path}")
            elif not current_is_eligible:
                failed = [
                    name
                    for name, passed in constraint_checks.items()
                    if not passed
                ]
                print(
                    "Checkpoint not eligible. Failed constraints: "
                    + ", ".join(failed)
                )


        append_metrics_to_csv(
            csv_path=csv_path,
            metrics=row_metrics
        )

        latest_path = os.path.join(
            checkpoint_dir,
            "advanced_stegastamp_latest.pth"
        )

        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_bit_accuracy,
            extra={
                "best_bit_accuracy": best_bit_accuracy,
                "best_message_accuracy": best_message_accuracy,
                "best_psnr": best_psnr,
                "best_robust_score": (
                    list(best_robust_score)
                    if best_robust_score is not None
                    else None
                ),
                "best_robust_summary": best_robust_summary,
                "args": vars(args)
            }
        )

        if (
            args.save_every > 0
            and epoch % args.save_every == 0
        ):
            epoch_path = os.path.join(
                checkpoint_dir,
                f"advanced_stegastamp_epoch_{epoch:04d}.pth"
            )

            save_checkpoint(
                path=epoch_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_bit_accuracy,
                extra={
                    "best_bit_accuracy": best_bit_accuracy,
                    "best_message_accuracy": best_message_accuracy,
                    "best_psnr": best_psnr,
                    "best_robust_score": (
                        list(best_robust_score)
                        if best_robust_score is not None
                        else None
                    ),
                    "best_robust_summary": best_robust_summary,
                    "args": vars(args)
                }
            )

            print(f"Saved epoch checkpoint to: {epoch_path}")

        if (
            args.visualize_every > 0
            and epoch % args.visualize_every == 0
        ):
            save_epoch_visualization(
                model=model,
                val_loader=val_loader,
                device=device,
                args=args,
                epoch=epoch
            )

    final_path = os.path.join(
        checkpoint_dir,
        "advanced_stegastamp_final.pth"
    )

    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        best_metric=best_bit_accuracy,
        extra={
            "best_bit_accuracy": best_bit_accuracy,
            "best_message_accuracy": best_message_accuracy,
            "best_psnr": best_psnr,
            "best_robust_score": (
                list(best_robust_score)
                if best_robust_score is not None
                else None
            ),
            "best_robust_summary": best_robust_summary,
            "args": vars(args)
        }
    )

    print("=" * 80)
    print("Training complete.")
    print(f"Final checkpoint: {final_path}")
    print(f"Best bit accuracy: {best_bit_accuracy:.4f}")
    print(f"Best message accuracy: {best_message_accuracy:.4f}")
    print(f"Best PSNR: {best_psnr:.4f} dB")
    print("=" * 80)


if __name__ == "__main__":
    main()