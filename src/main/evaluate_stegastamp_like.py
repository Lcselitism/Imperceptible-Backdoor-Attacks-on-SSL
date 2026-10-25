import os
import csv
import json
import argparse
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from stegastamp_like_trigger import (
    load_stegastamp_model,
    generate_trigger_batch,
    prepare_trigger_message
)
from stegastamp_like_losses import ssim_index
from stegastamp_like_corruptions import (
    CORRUPTION_PROFILES,
    DifferentiableImageCorruption,
    build_corruption_config
)
from stegastamp_like_utils import (
    ensure_dir,
    get_device,
    set_random_seed
)


# ============================================================
# evaluate_stegastamp_like.py
#
# Full evaluation for the trained CA3 Advanced StegaStamp trigger.
#
# Each checkpoint is evaluated using four core modes:
#   1. clean random messages
#   2. clean fixed message
#   3. corrupted random messages
#   4. corrupted fixed message
#
# Optional individual corruption profiles can also be evaluated.
#
# Important:
#   - Trigger insertion is performed in raw [0, 1] image space.
#   - Corruption is applied only after the trigger is inserted.
#   - PSNR, SSIM, LPIPS, residual, saturation and mask metrics always
#     compare the clean image against the clean encoded image.
#   - Corruption affects only the image passed to the message decoder.
# ============================================================


DEFAULT_CHECKPOINTS = [
    (
        "./results_stegastamp_corruption_mild/checkpoints/"
        "advanced_stegastamp_best.pth"
    ),
    (
        "./results_stegastamp_corruption_mild/checkpoints/"
        "advanced_stegastamp_final.pth"
    )
]


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CA3 Advanced StegaStamp checkpoints on clean "
            "and corrupted CIFAR-10 test images"
        )
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="One or more Advanced StegaStamp checkpoint paths"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="CIFAR-10 dataset directory"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_stegastamp_corruption_evaluation",
        help="Directory for CSV and JSON evaluation results"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count"
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Maximum evaluation batches. 0 means full test set."
    )

    parser.add_argument(
        "--message-pattern",
        type=str,
        default="alternating",
        choices=[
            "alternating",
            "inverse_alternating",
            "ones",
            "zeros",
            "half"
        ],
        help="Fixed target message pattern"
    )

    parser.add_argument(
        "--corruption-profiles",
        type=str,
        nargs="+",
        default=["combined"],
        choices=CORRUPTION_PROFILES,
        help=(
            "Corruption profiles evaluated after the clean modes. "
            "Use individual profiles for corruption ablation."
        )
    )

    parser.add_argument(
        "--corruption-strengths",
        type=float,
        nargs="+",
        default=[0.25],
        help="One or more fixed corruption strengths in [0, 1]"
    )

    parser.add_argument(
        "--skip-corrupted",
        action="store_true",
        help="Evaluate only clean random and clean fixed modes"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--use-lpips",
        action="store_true",
        help="Evaluate LPIPS. Requires the lpips package."
    )

    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone"
    )

    return parser.parse_args()


# ============================================================
# Metric accumulator
# ============================================================

class EvaluationAccumulator:
    """
    Accumulate exact dataset-level trigger and decoder metrics.
    """

    def __init__(self):
        self.num_samples = 0
        self.num_bits = 0

        self.correct_bits = 0
        self.correct_messages = 0

        self.message_loss_sum = 0.0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0

        self.absolute_residual_sum = 0.0
        self.residual_element_count = 0
        self.saturated_residual_count = 0
        self.maximum_absolute_residual = 0.0

        self.mask_sum = 0.0
        self.mask_element_count = 0
        self.mask_min = float("inf")
        self.mask_max = float("-inf")

        self.corruption_change_sum = 0.0
        self.corruption_element_count = 0
        self.maximum_corruption_change = 0.0

    def update(
        self,
        clean_image: torch.Tensor,
        encoded_image: torch.Tensor,
        residual: torch.Tensor,
        residual_mask: Optional[torch.Tensor],
        decoder_input: torch.Tensor,
        decoded_logits: torch.Tensor,
        message: torch.Tensor,
        max_residual: float,
        lpips_model=None
    ) -> None:
        """
        Add one evaluation batch.

        Stealth metrics use clean_image and encoded_image. The separate
        decoder_input tensor may be either clean encoded or corrupted.
        """

        batch_size = clean_image.size(0)
        message_length = message.size(1)

        predicted_bits = (
            torch.sigmoid(decoded_logits) >= 0.5
        ).to(dtype=message.dtype)

        correct_bit_tensor = predicted_bits == message
        correct_message_tensor = correct_bit_tensor.all(dim=1)

        message_loss = F.binary_cross_entropy_with_logits(
            decoded_logits,
            message,
            reduction="sum"
        )

        mse_per_sample = F.mse_loss(
            encoded_image,
            clean_image,
            reduction="none"
        )

        mse_per_sample = (
            mse_per_sample
            .flatten(start_dim=1)
            .mean(dim=1)
        )

        psnr_per_sample = 10.0 * torch.log10(
            1.0 / torch.clamp(
                mse_per_sample,
                min=1e-8
            )
        )

        batch_ssim = ssim_index(
            clean_image,
            encoded_image
        )

        # Use the true residual after image clamping.
        true_residual = encoded_image - clean_image
        absolute_residual = true_residual.abs()

        saturated_residual = (
            absolute_residual >= 0.95 * max_residual
        )

        corruption_change = (
            decoder_input - encoded_image
        ).abs()

        self.num_samples += batch_size
        self.num_bits += batch_size * message_length

        self.correct_bits += int(
            correct_bit_tensor.sum().item()
        )

        self.correct_messages += int(
            correct_message_tensor.sum().item()
        )

        # BCE reduction="sum" sums across samples and bits.
        self.message_loss_sum += float(
            message_loss.item()
        )

        self.psnr_sum += float(
            psnr_per_sample.sum().item()
        )

        self.ssim_sum += float(
            batch_ssim.item() * batch_size
        )

        self.absolute_residual_sum += float(
            absolute_residual.sum().item()
        )

        self.residual_element_count += (
            absolute_residual.numel()
        )

        self.saturated_residual_count += int(
            saturated_residual.sum().item()
        )

        self.maximum_absolute_residual = max(
            self.maximum_absolute_residual,
            float(absolute_residual.max().item())
        )

        self.corruption_change_sum += float(
            corruption_change.sum().item()
        )

        self.corruption_element_count += (
            corruption_change.numel()
        )

        self.maximum_corruption_change = max(
            self.maximum_corruption_change,
            float(corruption_change.max().item())
        )

        if residual_mask is not None:
            self.mask_sum += float(
                residual_mask.sum().item()
            )

            self.mask_element_count += (
                residual_mask.numel()
            )

            self.mask_min = min(
                self.mask_min,
                float(residual_mask.min().item())
            )

            self.mask_max = max(
                self.mask_max,
                float(residual_mask.max().item())
            )

        if lpips_model is not None:
            clean_lpips = clean_image * 2.0 - 1.0
            encoded_lpips = encoded_image * 2.0 - 1.0

            lpips_values = lpips_model(
                clean_lpips,
                encoded_lpips
            )

            self.lpips_sum += float(
                lpips_values.sum().item()
            )

    def results(
        self,
        lpips_enabled: bool = False
    ) -> Dict[str, float]:
        """
        Return final dataset-level averages.
        """

        if self.num_samples == 0:
            raise RuntimeError(
                "No evaluation samples were processed."
            )

        if self.num_bits == 0:
            raise RuntimeError(
                "No message bits were evaluated."
            )

        if self.residual_element_count == 0:
            raise RuntimeError(
                "No residual elements were evaluated."
            )

        if self.corruption_element_count == 0:
            raise RuntimeError(
                "No decoder-input elements were evaluated."
            )

        results = {
            "num_samples": self.num_samples,

            "bit_accuracy": (
                self.correct_bits / self.num_bits
            ),

            "message_accuracy": (
                self.correct_messages / self.num_samples
            ),

            "message_bce": (
                self.message_loss_sum / self.num_bits
            ),

            "psnr_db": (
                self.psnr_sum / self.num_samples
            ),

            "ssim_index": (
                self.ssim_sum / self.num_samples
            ),

            "mean_abs_residual": (
                self.absolute_residual_sum
                / self.residual_element_count
            ),

            "mean_abs_residual_255": (
                self.absolute_residual_sum
                / self.residual_element_count
                * 255.0
            ),

            "max_abs_residual": (
                self.maximum_absolute_residual
            ),

            "max_abs_residual_255": (
                self.maximum_absolute_residual * 255.0
            ),

            "residual_saturation_ratio": (
                self.saturated_residual_count
                / self.residual_element_count
            ),

            "corruption_mean_abs_change": (
                self.corruption_change_sum
                / self.corruption_element_count
            ),

            "corruption_mean_abs_change_255": (
                self.corruption_change_sum
                / self.corruption_element_count
                * 255.0
            ),

            "corruption_max_abs_change": (
                self.maximum_corruption_change
            ),

            "corruption_max_abs_change_255": (
                self.maximum_corruption_change * 255.0
            )
        }

        if self.mask_element_count > 0:
            results.update({
                "mask_mean": (
                    self.mask_sum
                    / self.mask_element_count
                ),
                "mask_min": self.mask_min,
                "mask_max": self.mask_max
            })

        if lpips_enabled:
            results["lpips"] = (
                self.lpips_sum / self.num_samples
            )

        return results


# ============================================================
# Data
# ============================================================

def build_test_loader(args):
    """
    Build the complete CIFAR-10 test DataLoader.
    """

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
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return loader


# ============================================================
# Optional LPIPS
# ============================================================

def build_lpips_model(
    enabled: bool,
    net: str,
    device: torch.device
):
    """
    Build an optional frozen LPIPS model.
    """

    if not enabled:
        return None

    try:
        import lpips

    except ImportError as error:
        raise ImportError(
            "LPIPS evaluation was requested, but the lpips package "
            "is not installed. Install it with:\n\n"
            "    pip install lpips\n"
        ) from error

    model = lpips.LPIPS(net=net).to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


# ============================================================
# Checkpoint evaluation
# ============================================================

def build_corruption_layer(
    profile: Optional[str],
    image_size: int,
    device: torch.device
):
    """
    Build one external corruption layer for evaluation.

    The layer is constructed externally so that checkpoints trained with
    corruption disabled can still be tested under the same corruptions.
    """

    if profile is None:
        return None

    config = build_corruption_config(
        profile=profile
    )

    corruption = DifferentiableImageCorruption(
        image_size=image_size,
        config=config
    ).to(device)

    corruption.eval()

    return corruption


@torch.no_grad()
def evaluate_checkpoint_mode(
    checkpoint_path: str,
    message_mode: str,
    test_loader,
    device: torch.device,
    args,
    lpips_model=None,
    corruption_profile: Optional[str] = None,
    corruption_strength: float = 0.0,
    corruption_seed: int = 0
) -> Dict[str, Any]:
    """
    Evaluate one clean or corrupted message-decoding mode.

    message_mode:
        "random" or "fixed"

    corruption_profile:
        None for a clean decoder input, otherwise one profile from
        CORRUPTION_PROFILES.
    """

    if message_mode not in ("random", "fixed"):
        raise ValueError(
            "message_mode must be 'random' or 'fixed'"
        )

    if not 0.0 <= corruption_strength <= 1.0:
        raise ValueError(
            "corruption_strength must be in [0, 1]"
        )

    model, config, _ = load_stegastamp_model(
        checkpoint_path=checkpoint_path,
        device=device,
        freeze=True,
        strict=True
    )

    model.eval()

    corruption_layer = build_corruption_layer(
        profile=corruption_profile,
        image_size=config.image_size,
        device=device
    )

    accumulator = EvaluationAccumulator()

    # Resetting the generator for every checkpoint and mode ensures that
    # all checkpoints receive exactly the same random messages.
    random_message_generator = torch.Generator(
        device="cpu"
    )

    random_message_generator.manual_seed(
        args.seed + 20_000
    )

    rng_devices = []

    if device.type == "cuda":
        rng_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]

    # Reset the corruption RNG for fair comparisons across checkpoints.
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(corruption_seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                corruption_seed
            )

        for batch_index, batch in enumerate(test_loader):
            if (
                args.max_batches > 0
                and batch_index >= args.max_batches
            ):
                break

            clean_image, _ = batch

            clean_image = clean_image.to(
                device,
                non_blocking=True
            )

            if message_mode == "random":
                message = torch.randint(
                    low=0,
                    high=2,
                    size=(
                        clean_image.size(0),
                        config.message_length
                    ),
                    generator=random_message_generator,
                    device="cpu"
                )

                message = message.to(
                    device=device,
                    dtype=clean_image.dtype
                )

            else:
                message = prepare_trigger_message(
                    batch_size=clean_image.size(0),
                    message_length=config.message_length,
                    device=device,
                    dtype=clean_image.dtype,
                    message_pattern=args.message_pattern,
                    message=None
                )

            # Generate the clean encoded trigger first. Corruption is not
            # part of the trigger itself and is applied only afterwards.
            outputs = generate_trigger_batch(
                clean_image=clean_image,
                model=model,
                config=config,
                message=message,
                message_pattern=args.message_pattern,
                decode_message=False,
                return_cpu=False,
                check_range=True
            )

            encoded_image = outputs["encoded_image"]

            if corruption_layer is None:
                decoder_input = encoded_image
            else:
                decoder_input = corruption_layer(
                    encoded_image,
                    strength=corruption_strength
                )

            decoded_logits = model.decoder(
                decoder_input
            )

            accumulator.update(
                clean_image=clean_image,
                encoded_image=encoded_image,
                residual=outputs["residual"],
                residual_mask=outputs.get(
                    "residual_mask",
                    None
                ),
                decoder_input=decoder_input,
                decoded_logits=decoded_logits,
                message=outputs["message"],
                max_residual=config.max_residual,
                lpips_model=lpips_model
            )

            if (
                batch_index == 0
                or (batch_index + 1) % 20 == 0
            ):
                decoder_mode = (
                    "clean"
                    if corruption_profile is None
                    else (
                        f"{corruption_profile}@"
                        f"{corruption_strength:.3f}"
                    )
                )

                print(
                    f"  {decoder_mode}, {message_mode}: "
                    f"processed batch {batch_index + 1}/"
                    f"{len(test_loader)}"
                )

    results = accumulator.results(
        lpips_enabled=lpips_model is not None
    )

    decoder_input_mode = (
        "clean"
        if corruption_profile is None
        else "corrupted"
    )

    results.update({
        "checkpoint": checkpoint_path,
        "checkpoint_name": os.path.basename(
            checkpoint_path
        ),
        "decoder_input_mode": decoder_input_mode,
        "corruption_profile": (
            "none"
            if corruption_profile is None
            else corruption_profile
        ),
        "corruption_strength": (
            0.0
            if corruption_profile is None
            else corruption_strength
        ),
        "message_mode": message_mode,
        "message_pattern": (
            args.message_pattern
            if message_mode == "fixed"
            else "deterministic_random"
        ),
        "message_length": config.message_length,
        "max_residual": config.max_residual,
        "max_residual_255": (
            config.max_residual * 255.0
        )
    })

    return results


# ============================================================
# Output helpers
# ============================================================

def print_result(result: Dict[str, Any]) -> None:
    """
    Print one checkpoint and evaluation-mode result.
    """

    print("-" * 80)
    print(f"Checkpoint: {result['checkpoint_name']}")
    print(f"Decoder input: {result['decoder_input_mode']}")
    print(f"Corruption profile: {result['corruption_profile']}")
    print(
        "Corruption strength: "
        f"{result['corruption_strength']:.3f}"
    )
    print(f"Message mode: {result['message_mode']}")
    print(f"Samples: {int(result['num_samples'])}")
    print(f"Bit accuracy: {result['bit_accuracy']:.6f}")
    print(
        "Message accuracy: "
        f"{result['message_accuracy']:.6f}"
    )
    print(f"Message BCE: {result['message_bce']:.6f}")
    print(f"Clean encoded PSNR: {result['psnr_db']:.6f} dB")
    print(f"Clean encoded SSIM: {result['ssim_index']:.6f}")

    if "lpips" in result:
        print(f"Clean encoded LPIPS: {result['lpips']:.6f}")

    print(
        "Mean absolute residual ×255: "
        f"{result['mean_abs_residual_255']:.6f}"
    )

    print(
        "Maximum absolute residual ×255: "
        f"{result['max_abs_residual_255']:.6f}"
    )

    print(
        "Residual saturation ratio: "
        f"{result['residual_saturation_ratio']:.6f}"
    )

    print(
        "Corruption mean absolute change ×255: "
        f"{result['corruption_mean_abs_change_255']:.6f}"
    )

    if "mask_mean" in result:
        print(f"Mask mean: {result['mask_mean']:.6f}")
        print(f"Mask min: {result['mask_min']:.6f}")
        print(f"Mask max: {result['mask_max']:.6f}")


def save_results_csv(
    results: List[Dict[str, Any]],
    csv_path: str
) -> None:
    """
    Save all evaluation rows to CSV.
    """

    if not results:
        return

    ensure_dir(os.path.dirname(csv_path))

    fieldnames = []

    for result in results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)


def save_results_json(
    results: List[Dict[str, Any]],
    json_path: str
) -> None:
    """
    Save all evaluation rows to JSON.
    """

    ensure_dir(os.path.dirname(json_path))

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )


def print_checkpoint_comparison(
    results: List[Dict[str, Any]]
) -> None:
    """
    Print a compact clean/corrupted comparison table.
    """

    print()
    print("=" * 154)
    print("Full CIFAR-10 StegaStamp Clean and Corrupted Comparison")
    print("=" * 154)

    header = (
        f"{'Checkpoint':30s} "
        f"{'Input':9s} "
        f"{'Profile':12s} "
        f"{'Str':>5s} "
        f"{'Msg':6s} "
        f"{'Bit Acc':>9s} "
        f"{'Msg Acc':>9s} "
        f"{'PSNR':>9s} "
        f"{'SSIM':>9s} "
        f"{'Mean R×255':>12s} "
        f"{'Sat Ratio':>10s} "
        f"{'Corr Δ×255':>11s}"
    )

    print(header)
    print("-" * 154)

    for result in results:
        print(
            f"{result['checkpoint_name'][:30]:30s} "
            f"{result['decoder_input_mode']:9s} "
            f"{result['corruption_profile'][:12]:12s} "
            f"{result['corruption_strength']:5.2f} "
            f"{result['message_mode']:6s} "
            f"{result['bit_accuracy']:9.4f} "
            f"{result['message_accuracy']:9.4f} "
            f"{result['psnr_db']:9.4f} "
            f"{result['ssim_index']:9.4f} "
            f"{result['mean_abs_residual_255']:12.4f} "
            f"{result['residual_saturation_ratio']:10.4f} "
            f"{result['corruption_mean_abs_change_255']:11.4f}"
        )

    print("=" * 154)


# ============================================================
# Main
# ============================================================

def validate_args(args) -> None:
    """
    Validate evaluation arguments before loading the dataset.
    """

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0"
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative"
        )

    if args.max_batches < 0:
        raise ValueError(
            "--max-batches cannot be negative"
        )

    for strength in args.corruption_strengths:
        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                "Every --corruption-strengths value must be in [0, 1]"
            )

    for checkpoint_path in args.checkpoints:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )


def main():
    args = parse_args()

    validate_args(args)

    set_random_seed(args.seed)

    device = get_device()

    ensure_dir(args.output_dir)

    print("=" * 80)
    print("CA3 Advanced StegaStamp Clean and Corrupted Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoints: {len(args.checkpoints)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Fixed message pattern: {args.message_pattern}")
    print(f"LPIPS enabled: {args.use_lpips}")
    print(f"Corrupted evaluation skipped: {args.skip_corrupted}")

    if not args.skip_corrupted:
        print(
            "Corruption profiles: "
            f"{', '.join(args.corruption_profiles)}"
        )
        print(
            "Corruption strengths: "
            + ", ".join(
                f"{strength:.3f}"
                for strength in args.corruption_strengths
            )
        )

    print("=" * 80)

    test_loader = build_test_loader(args)

    lpips_model = build_lpips_model(
        enabled=args.use_lpips,
        net=args.lpips_net,
        device=device
    )

    all_results: List[Dict[str, Any]] = []

    for checkpoint_path in args.checkpoints:
        print()
        print(f"Evaluating checkpoint: {checkpoint_path}")

        # ----------------------------------------------------
        # Clean random and clean fixed modes
        # ----------------------------------------------------
        for message_mode in ("random", "fixed"):
            result = evaluate_checkpoint_mode(
                checkpoint_path=checkpoint_path,
                message_mode=message_mode,
                test_loader=test_loader,
                device=device,
                args=args,
                lpips_model=lpips_model,
                corruption_profile=None,
                corruption_strength=0.0,
                corruption_seed=args.seed + 30_000
            )

            all_results.append(result)
            print_result(result)

        # ----------------------------------------------------
        # Corrupted random and corrupted fixed modes
        # ----------------------------------------------------
        if not args.skip_corrupted:
            for profile_index, profile in enumerate(
                args.corruption_profiles
            ):
                for strength_index, strength in enumerate(
                    args.corruption_strengths
                ):
                    # Use one deterministic corruption sequence for both
                    # message modes and all checkpoints at this setting.
                    corruption_seed = (
                        args.seed
                        + 40_000
                        + profile_index * 1_000
                        + strength_index * 100
                    )

                    for message_mode in ("random", "fixed"):
                        result = evaluate_checkpoint_mode(
                            checkpoint_path=checkpoint_path,
                            message_mode=message_mode,
                            test_loader=test_loader,
                            device=device,
                            args=args,
                            lpips_model=lpips_model,
                            corruption_profile=profile,
                            corruption_strength=strength,
                            corruption_seed=corruption_seed
                        )

                        all_results.append(result)
                        print_result(result)

    csv_path = os.path.join(
        args.output_dir,
        "stegastamp_clean_corrupted_comparison.csv"
    )

    json_path = os.path.join(
        args.output_dir,
        "stegastamp_clean_corrupted_comparison.json"
    )

    save_results_csv(
        results=all_results,
        csv_path=csv_path
    )

    save_results_json(
        results=all_results,
        json_path=json_path
    )

    print_checkpoint_comparison(all_results)

    print(f"Saved CSV results to: {csv_path}")
    print(f"Saved JSON results to: {json_path}")
    print("evaluate_stegastamp_like.py evaluation completed.")


if __name__ == "__main__":
    main()