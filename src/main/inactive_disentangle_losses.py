import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stegastamp_like_losses import (
    OptionalLPIPSLoss,
    compute_batch_psnr,
    ssim_index,
)


# ============================================================
# inactive_disentangle_losses.py
#
# CA3 INACTIVE-style losses for:
#   - the corruption-robust Advanced StegaStamp trigger
#   - the clean SimCLR ResNet-18 encoder
#   - the trainable backdoored SimCLR ResNet-18 encoder
#
# Correct image/feature flow:
#
#   raw CIFAR-10 image [0, 1]
#       -> generate_trigger_batch_differentiable(...)
#       -> triggered image [0, 1]
#       -> CIFAR normalization
#       -> SimCLR model.f(image)
#       -> encoder feature, shape (B, 512)
#
# This file separates the two alternating INACTIVE steps:
#
#   1. InactiveInjectorLoss
#      Updates the StegaStamp hider.
#
#   2. InactiveEncoderLoss
#      Updates the compromised SimCLR encoder.
#
# Do not normally update the hider and encoder with one combined
# backward pass. Freeze one component while updating the other.
# ============================================================


# ============================================================
# Basic checks
# ============================================================

def check_image(
    image: torch.Tensor,
    name: str
) -> None:
    """
    Check a raw RGB image batch with shape:
        (B, 3, H, W)
    """

    if not torch.is_tensor(image):
        raise TypeError(
            f"{name} must be a torch.Tensor"
        )

    if image.dim() != 4:
        raise ValueError(
            f"Expected {name} shape (B, 3, H, W), "
            f"but got {tuple(image.shape)}"
        )

    if image.size(0) <= 0:
        raise ValueError(
            f"{name} must contain at least one image"
        )

    if image.size(1) != 3:
        raise ValueError(
            f"Expected {name} to have 3 channels, "
            f"but got {image.size(1)}"
        )

    if not image.is_floating_point():
        raise TypeError(
            f"{name} must be floating point"
        )


def check_image_pair(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    name_a: str,
    name_b: str
) -> None:
    """
    Check two image batches with identical shapes.
    """

    check_image(
        image=image_a,
        name=name_a
    )

    check_image(
        image=image_b,
        name=name_b
    )

    if image_a.shape != image_b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have identical shapes. "
            f"Got {tuple(image_a.shape)} and "
            f"{tuple(image_b.shape)}."
        )


def check_features(
    features: torch.Tensor,
    name: str
) -> None:
    """
    Check encoder features with shape:
        (B, feature_dimension)

    For the current SimCLR ResNet-18 encoder:
        feature_dimension = 512
    """

    if not torch.is_tensor(features):
        raise TypeError(
            f"{name} must be a torch.Tensor"
        )

    if features.dim() != 2:
        raise ValueError(
            f"Expected {name} shape (B, D), "
            f"but got {tuple(features.shape)}"
        )

    if features.size(0) <= 0:
        raise ValueError(
            f"{name} must contain at least one sample"
        )

    if features.size(1) <= 0:
        raise ValueError(
            f"{name} must have a positive feature dimension"
        )

    if not features.is_floating_point():
        raise TypeError(
            f"{name} must be floating point"
        )


def prepare_target_features(
    source_features: torch.Tensor,
    target_features: torch.Tensor
) -> torch.Tensor:
    """
    Validate target features and expand a single target centroid.

    Supported target shapes:

        (B, D):
            One paired reference feature per source sample.

        (1, D):
            One target-class centroid for the whole batch.
    """

    check_features(
        features=source_features,
        name="source_features"
    )

    check_features(
        features=target_features,
        name="target_features"
    )

    if source_features.size(1) != target_features.size(1):
        raise ValueError(
            "source_features and target_features must have "
            "the same feature dimension. "
            f"Got {source_features.size(1)} and "
            f"{target_features.size(1)}."
        )

    if target_features.size(0) == 1:
        target_features = target_features.expand(
            source_features.size(0),
            -1
        )

    elif target_features.size(0) != source_features.size(0):
        raise ValueError(
            "target_features must contain either one centroid "
            "or one reference feature per source sample. "
            f"Got source batch {source_features.size(0)} and "
            f"target batch {target_features.size(0)}."
        )

    return target_features


# ============================================================
# Feature-space losses
# ============================================================

def cosine_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Pull source features toward target/reference features.

    Uses:

        loss = 1 - cosine_similarity(source, target)

    This has the same optimization direction as minimizing
    negative cosine similarity, while remaining non-negative.

    Typical use:

        source_features:
            Backdoored encoder features of triggered images.

        target_features:
            Target-class reference features from the clean encoder.
    """

    target_features = prepare_target_features(
        source_features=source_features,
        target_features=target_features
    )

    source_features = F.normalize(
        source_features,
        p=2,
        dim=1,
        eps=eps
    )

    target_features = F.normalize(
        target_features,
        p=2,
        dim=1,
        eps=eps
    )

    similarity = F.cosine_similarity(
        source_features,
        target_features,
        dim=1,
        eps=eps
    )

    return 1.0 - similarity.mean()


def feature_disentangle_margin_loss(
    triggered_features: torch.Tensor,
    augmented_features: torch.Tensor,
    maximum_similarity: float = 0.50,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Optional CA3 feature-space disentanglement extension.

    This directly discourages excessive similarity between:

        triggered-image features
        SSL-augmented-image features

    The original INACTIVE disentanglement term operates in HSV/HSL
    image space. Therefore, this feature-space term is disabled by
    default.

    Enable it only as a clearly labelled CA3 extension or ablation.

    The loss is:

        ReLU(
            cosine_similarity(triggered, augmented)
            - maximum_similarity
        )

    Minimizing this means that similarity above the chosen maximum
    is penalized.
    """

    check_features(
        features=triggered_features,
        name="triggered_features"
    )

    check_features(
        features=augmented_features,
        name="augmented_features"
    )

    if triggered_features.shape != augmented_features.shape:
        raise ValueError(
            "triggered_features and augmented_features must "
            "have the same shape. "
            f"Got {tuple(triggered_features.shape)} and "
            f"{tuple(augmented_features.shape)}."
        )

    if not -1.0 <= maximum_similarity <= 1.0:
        raise ValueError(
            "maximum_similarity must be in [-1, 1]"
        )

    triggered_features = F.normalize(
        triggered_features,
        p=2,
        dim=1,
        eps=eps
    )

    augmented_features = F.normalize(
        augmented_features,
        p=2,
        dim=1,
        eps=eps
    )

    similarity = F.cosine_similarity(
        triggered_features,
        augmented_features,
        dim=1,
        eps=eps
    )

    return F.relu(
        similarity - maximum_similarity
    ).mean()


# ============================================================
# Differentiable RGB -> HSV / HSL
# ============================================================

def rgb_to_hsv(
    image: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    """
    Convert RGB images in [0, 1] to HSV components in [0, 1].

    Returns:
        hue
        saturation
        value

    Each output has shape:
        (B, 1, H, W)

    The operations are differentiable almost everywhere, allowing
    gradients to reach the StegaStamp hider.
    """

    check_image(
        image=image,
        name="image"
    )

    red = image[:, 0:1]
    green = image[:, 1:2]
    blue = image[:, 2:3]

    rgb = torch.cat(
        [
            red,
            green,
            blue
        ],
        dim=1
    )

    value, max_index = rgb.max(
        dim=1,
        keepdim=True
    )

    minimum = rgb.min(
        dim=1,
        keepdim=True
    ).values

    delta = value - minimum

    safe_delta = torch.clamp(
        delta,
        min=eps
    )

    saturation = delta / torch.clamp(
        value,
        min=eps
    )

    saturation = torch.where(
        value > eps,
        saturation,
        torch.zeros_like(saturation)
    )

    hue_red = torch.remainder(
        (green - blue) / safe_delta,
        6.0
    )

    hue_green = (
        (blue - red) / safe_delta
        + 2.0
    )

    hue_blue = (
        (red - green) / safe_delta
        + 4.0
    )

    hue = torch.where(
        max_index == 0,
        hue_red,
        hue_green
    )

    hue = torch.where(
        max_index == 2,
        hue_blue,
        hue
    )

    hue = hue / 6.0

    # Hue is undefined for grayscale pixels.
    # Set it to zero where chroma is effectively zero.
    hue = torch.where(
        delta > eps,
        hue,
        torch.zeros_like(hue)
    )

    return (
        hue,
        saturation,
        value
    )


def rgb_to_hsl(
    image: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    """
    Convert RGB images in [0, 1] to HSL components in [0, 1].

    Returns:
        hue
        saturation
        lightness
    """

    hue, _, maximum = rgb_to_hsv(
        image=image,
        eps=eps
    )

    minimum = image.min(
        dim=1,
        keepdim=True
    ).values

    delta = maximum - minimum

    lightness = 0.5 * (
        maximum + minimum
    )

    denominator = (
        1.0
        - torch.abs(
            2.0 * lightness - 1.0
        )
    )

    saturation = delta / torch.clamp(
        denominator,
        min=eps
    )

    saturation = torch.where(
        delta > eps,
        saturation,
        torch.zeros_like(saturation)
    )

    return (
        hue,
        saturation,
        lightness
    )


def circular_hue_difference(
    hue_a: torch.Tensor,
    hue_b: torch.Tensor
) -> torch.Tensor:
    """
    Return the shortest hue distance on the circular interval [0, 1].

    For example:

        hue_a = 0.99
        hue_b = 0.01

    These hues should be considered close, not almost one unit apart.
    """

    difference = torch.abs(
        hue_a - hue_b
    )

    return torch.minimum(
        difference,
        1.0 - difference
    )


def rms_distance(
    difference: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute one resolution-independent RMS distance per sample.

    Output shape:
        (B,)
    """

    flattened = difference.flatten(
        start_dim=1
    )

    return torch.sqrt(
        flattened.pow(2).mean(dim=1)
        + eps
    )


# ============================================================
# INACTIVE colour-disentanglement loss
# ============================================================

def inactive_color_disentangle_loss(
    triggered_image: torch.Tensor,
    augmented_image: torch.Tensor,
    use_circular_hue: bool = True,
    eps: float = 1e-8
) -> Tuple[
    torch.Tensor,
    Dict[str, torch.Tensor]
]:
    """
    Enlarge the HSV/HSL gap between triggered and SSL-augmented images.

    The INACTIVE-style colour gap contains:

        HSV hue distance
        HSV saturation distance
        HSV value distance
        HSL lightness distance

    The returned loss is negative:

        loss = -colour_gap

    Therefore, minimizing this loss increases the difference between
    the trigger distribution and the inherent SSL augmentation
    distribution.

    Important:

        This loss operates on raw images in [0, 1].

        Do not pass CIFAR-normalized images into this function.

        This loss updates the StegaStamp hider only when:
            - generate_trigger_batch_differentiable(...) was used
            - the hider parameters have requires_grad=True

        If the hider is frozen, this loss is only a diagnostic value.
    """

    check_image_pair(
        image_a=triggered_image,
        image_b=augmented_image,
        name_a="triggered_image",
        name_b="augmented_image"
    )

    (
        trigger_hue,
        trigger_saturation,
        trigger_value
    ) = rgb_to_hsv(
        image=triggered_image,
        eps=eps
    )

    (
        augment_hue,
        augment_saturation,
        augment_value
    ) = rgb_to_hsv(
        image=augmented_image,
        eps=eps
    )

    (
        _,
        _,
        trigger_lightness
    ) = rgb_to_hsl(
        image=triggered_image,
        eps=eps
    )

    (
        _,
        _,
        augment_lightness
    ) = rgb_to_hsl(
        image=augmented_image,
        eps=eps
    )

    if use_circular_hue:
        hue_difference = circular_hue_difference(
            hue_a=trigger_hue,
            hue_b=augment_hue
        )
    else:
        hue_difference = (
            trigger_hue - augment_hue
        )

    hue_distance = rms_distance(
        difference=hue_difference,
        eps=eps
    )

    saturation_distance = rms_distance(
        difference=(
            trigger_saturation
            - augment_saturation
        ),
        eps=eps
    )

    value_distance = rms_distance(
        difference=(
            trigger_value
            - augment_value
        ),
        eps=eps
    )

    lightness_distance = rms_distance(
        difference=(
            trigger_lightness
            - augment_lightness
        ),
        eps=eps
    )

    color_gap_per_sample = (
        hue_distance
        + saturation_distance
        + value_distance
        + lightness_distance
    )

    color_gap = color_gap_per_sample.mean()

    # Negative sign:
    # minimizing the loss maximizes the colour gap.
    loss = -color_gap

    metrics = {
        "color_gap": color_gap.detach(),

        "hue_distance": (
            hue_distance.mean().detach()
        ),

        "saturation_distance": (
            saturation_distance.mean().detach()
        ),

        "value_distance": (
            value_distance.mean().detach()
        ),

        "lightness_distance": (
            lightness_distance.mean().detach()
        )
    }

    return (
        loss,
        metrics
    )


# ============================================================
# Stealth-preservation helpers
# ============================================================

def empirical_wasserstein_loss(
    image_a: torch.Tensor,
    image_b: torch.Tensor
) -> torch.Tensor:
    """
    Compute an empirical Wasserstein-1 distance per RGB channel.

    For two equally sized one-dimensional empirical distributions,
    Wasserstein-1 is the mean absolute distance between sorted samples.

    Here, each image channel is flattened and treated as a distribution
    of pixel intensities.

    This remains differentiable almost everywhere.
    """

    check_image_pair(
        image_a=image_a,
        image_b=image_b,
        name_a="image_a",
        name_b="image_b"
    )

    flat_a = image_a.flatten(
        start_dim=2
    )

    flat_b = image_b.flatten(
        start_dim=2
    )

    sorted_a = torch.sort(
        flat_a,
        dim=2
    ).values

    sorted_b = torch.sort(
        flat_b,
        dim=2
    ).values

    return torch.abs(
        sorted_a - sorted_b
    ).mean()


def soft_residual_limit_loss(
    residual: torch.Tensor,
    max_residual: float,
    margin_ratio: float = 0.90
) -> torch.Tensor:
    """
    Discourage frequent use of values near the hard residual bound.

    The Advanced StegaStamp architecture already enforces:

        abs(residual) <= 8/255

    This soft penalty begins before the hard limit so later INACTIVE
    fine-tuning does not silently increase residual saturation.
    """

    if max_residual <= 0.0:
        raise ValueError(
            "max_residual must be greater than 0"
        )

    if not 0.0 <= margin_ratio <= 1.0:
        raise ValueError(
            "margin_ratio must be in [0, 1]"
        )

    normalized_residual = (
        residual.abs() / max_residual
    )

    excess = F.relu(
        normalized_residual - margin_ratio
    )

    return excess.pow(2).mean()


# ============================================================
# Loss weights
# ============================================================

@dataclass
class InactiveInjectorLossWeights:
    """
    Weights used when optionally fine-tuning the StegaStamp hider.

    The selected corruption-robust checkpoint is already strong.
    These defaults are deliberately conservative to prevent the trigger
    from drifting away from its existing accuracy-stealth balance.
    """

    # INACTIVE top-level terms.
    color_disentangle_weight: float = 0.10
    alignment_weight: float = 1.00

    # Stealth-preservation terms.
    image_l1_weight: float = 2.00
    image_mse_weight: float = 1.00
    ssim_weight: float = 1.00
    psnr_weight: float = 0.10
    wasserstein_weight: float = 0.10
    lpips_weight: float = 0.00

    # Keep the fine-tuned trigger near the selected starting checkpoint.
    trigger_anchor_weight: float = 1.00

    # Preserve the existing 16-bit message channel.
    message_weight: float = 1.00

    # Avoid increasing residual saturation.
    residual_limit_weight: float = 0.50

    target_psnr_db: float = 37.0
    residual_margin_ratio: float = 0.90


@dataclass
class InactiveEncoderLossWeights:
    """
    Weights used when training the compromised SimCLR encoder.
    """

    # Triggered features -> target reference features.
    alignment_weight: float = 1.0

    # Backdoored reference features -> clean reference features.
    consistency_weight: float = 1.0

    # Backdoored clean features -> clean encoder features.
    utility_weight: float = 1.0

    # Optional CA3 feature-space extension.
    # Keep this at zero for the initial paper-faithful baseline.
    feature_disentangle_weight: float = 0.0

    maximum_trigger_augmentation_similarity: float = 0.50


# ============================================================
# Injector loss
# ============================================================

class InactiveInjectorLoss(nn.Module):
    """
    Loss used when updating the Advanced StegaStamp hider.

    During this step:

        StegaStamp hider:
            trainable

        StegaStamp decoder:
            normally frozen, but gradient may pass through it to the hider

        Backdoored SimCLR encoder:
            parameters frozen, but do not wrap its triggered-image forward
            pass in torch.no_grad(), because gradients must pass through
            the encoder to the triggered image and hider

        Clean SimCLR encoder:
            frozen

    Expected inputs:

        clean_image:
            Raw image in [0, 1].

        triggered_image:
            Output from generate_trigger_batch_differentiable(...).

        augmented_image:
            SSL-augmented clean image in [0, 1].

        triggered_features:
            Frozen compromised-encoder features of triggered images.

        reference_features:
            Target-class reference features.

        baseline_triggered_image:
            Triggered image produced by a frozen copy of the selected
            corruption-robust StegaStamp checkpoint.

        decoded_logits and message:
            Used to preserve 16-bit message recovery.
    """

    def __init__(
        self,
        weights: Optional[
            InactiveInjectorLossWeights
        ] = None,
        max_residual: float = 8 / 255,
        use_lpips: bool = False,
        lpips_net: str = "alex",
        use_circular_hue: bool = True,
        eps: float = 1e-8
    ):
        super().__init__()

        if weights is None:
            weights = (
                InactiveInjectorLossWeights()
            )

        if max_residual <= 0.0:
            raise ValueError(
                "max_residual must be greater than 0"
            )

        self.weights = weights
        self.max_residual = float(
            max_residual
        )

        self.use_circular_hue = bool(
            use_circular_hue
        )

        self.eps = float(eps)

        self.lpips_loss = OptionalLPIPSLoss(
            enabled=(
                use_lpips
                and weights.lpips_weight > 0.0
            ),
            net=lpips_net
        )

        self.message_loss_fn = (
            nn.BCEWithLogitsLoss()
        )

    def forward(
        self,
        clean_image: torch.Tensor,
        triggered_image: torch.Tensor,
        augmented_image: torch.Tensor,
        triggered_features: torch.Tensor,
        reference_features: torch.Tensor,
        decoded_logits: Optional[
            torch.Tensor
        ] = None,
        message: Optional[
            torch.Tensor
        ] = None,
        baseline_triggered_image: Optional[
            torch.Tensor
        ] = None
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor]
    ]:
        """
        Compute the hider/injector objective.
        """

        check_image_pair(
            image_a=clean_image,
            image_b=triggered_image,
            name_a="clean_image",
            name_b="triggered_image"
        )

        check_image_pair(
            image_a=triggered_image,
            image_b=augmented_image,
            name_a="triggered_image",
            name_b="augmented_image"
        )

        # ----------------------------------------------------
        # Feature alignment
        # ----------------------------------------------------
        alignment_loss = cosine_alignment_loss(
            source_features=triggered_features,
            target_features=reference_features,
            eps=self.eps
        )

        # ----------------------------------------------------
        # INACTIVE HSV/HSL disentanglement
        # ----------------------------------------------------
        (
            color_disentangle_loss,
            color_metrics
        ) = inactive_color_disentangle_loss(
            triggered_image=triggered_image,
            augmented_image=augmented_image,
            use_circular_hue=(
                self.use_circular_hue
            ),
            eps=self.eps
        )

        # ----------------------------------------------------
        # Image-space stealth preservation
        # ----------------------------------------------------
        image_l1 = F.l1_loss(
            triggered_image,
            clean_image
        )

        image_mse = F.mse_loss(
            triggered_image,
            clean_image
        )

        ssim_value = ssim_index(
            clean_image,
            triggered_image
        )

        ssim_loss_value = (
            1.0 - ssim_value
        )

        psnr_value = compute_batch_psnr(
            clean_image,
            triggered_image
        )

        psnr_hinge_loss = F.relu(
            self.weights.target_psnr_db
            - psnr_value
        ) / max(
            self.weights.target_psnr_db,
            self.eps
        )

        wasserstein_loss = (
            empirical_wasserstein_loss(
                triggered_image,
                clean_image
            )
        )

        lpips_value = self.lpips_loss(
            clean_image,
            triggered_image
        )

        # ----------------------------------------------------
        # Residual preservation
        # ----------------------------------------------------
        residual = (
            triggered_image - clean_image
        )

        residual_limit_loss = (
            soft_residual_limit_loss(
                residual=residual,
                max_residual=self.max_residual,
                margin_ratio=(
                    self.weights
                    .residual_margin_ratio
                )
            )
        )

        # ----------------------------------------------------
        # Anchor to the previously selected trigger checkpoint
        # ----------------------------------------------------
        if baseline_triggered_image is not None:
            check_image_pair(
                image_a=triggered_image,
                image_b=baseline_triggered_image,
                name_a="triggered_image",
                name_b="baseline_triggered_image"
            )

            trigger_anchor_loss = F.l1_loss(
                triggered_image,
                baseline_triggered_image.detach()
            )

        else:
            trigger_anchor_loss = (
                clean_image.new_tensor(0.0)
            )

        # ----------------------------------------------------
        # Preserve message decoding
        # ----------------------------------------------------
        if (
            decoded_logits is None
            and message is None
        ):
            message_loss = (
                clean_image.new_tensor(0.0)
            )

        elif (
            decoded_logits is None
            or message is None
        ):
            raise ValueError(
                "decoded_logits and message must either "
                "both be supplied or both be omitted"
            )

        else:
            if decoded_logits.shape != message.shape:
                raise ValueError(
                    "decoded_logits and message must "
                    "have identical shapes. "
                    f"Got {tuple(decoded_logits.shape)} and "
                    f"{tuple(message.shape)}."
                )

            message_loss = self.message_loss_fn(
                decoded_logits,
                message
            )

        # ----------------------------------------------------
        # Stealth objective
        # ----------------------------------------------------
        stealth_loss = (
            self.weights.image_l1_weight
            * image_l1

            + self.weights.image_mse_weight
            * image_mse

            + self.weights.ssim_weight
            * ssim_loss_value

            + self.weights.psnr_weight
            * psnr_hinge_loss

            + self.weights.wasserstein_weight
            * wasserstein_loss

            + self.weights.lpips_weight
            * lpips_value

            + self.weights.trigger_anchor_weight
            * trigger_anchor_loss

            + self.weights.message_weight
            * message_loss

            + self.weights.residual_limit_weight
            * residual_limit_loss
        )

        # ----------------------------------------------------
        # Complete injector objective
        # ----------------------------------------------------
        total_loss = (
            stealth_loss

            + self.weights.color_disentangle_weight
            * color_disentangle_loss

            + self.weights.alignment_weight
            * alignment_loss
        )

        # ----------------------------------------------------
        # Logging metrics
        # ----------------------------------------------------
        with torch.no_grad():
            reference_for_metric = (
                prepare_target_features(
                    source_features=(
                        triggered_features
                    ),
                    target_features=(
                        reference_features
                    )
                )
            )

            alignment_similarity = (
                F.cosine_similarity(
                    F.normalize(
                        triggered_features,
                        dim=1,
                        eps=self.eps
                    ),
                    F.normalize(
                        reference_for_metric,
                        dim=1,
                        eps=self.eps
                    ),
                    dim=1,
                    eps=self.eps
                ).mean()
            )

            abs_residual = residual.abs()

            mean_abs_residual = (
                abs_residual.mean()
            )

            max_abs_residual = (
                abs_residual.max()
            )

            residual_saturation_ratio = (
                abs_residual
                >= 0.95 * self.max_residual
            ).float().mean()

        metrics = {
            "loss_total": (
                total_loss.detach()
            ),

            "loss_stealth": (
                stealth_loss.detach()
            ),

            "loss_alignment": (
                alignment_loss.detach()
            ),

            "loss_color_disentangle": (
                color_disentangle_loss.detach()
            ),

            "loss_image_l1": (
                image_l1.detach()
            ),

            "loss_image_mse": (
                image_mse.detach()
            ),

            "loss_ssim": (
                ssim_loss_value.detach()
            ),

            "loss_psnr_hinge": (
                psnr_hinge_loss.detach()
            ),

            "loss_wasserstein": (
                wasserstein_loss.detach()
            ),

            "loss_lpips": (
                lpips_value.detach()
            ),

            "loss_trigger_anchor": (
                trigger_anchor_loss.detach()
            ),

            "loss_message": (
                message_loss.detach()
            ),

            "loss_residual_limit": (
                residual_limit_loss.detach()
            ),

            "alignment_similarity": (
                alignment_similarity.detach()
            ),

            "psnr_db": (
                psnr_value.detach()
            ),

            "ssim_index": (
                ssim_value.detach()
            ),

            "mean_abs_residual": (
                mean_abs_residual.detach()
            ),

            "mean_abs_residual_255": (
                mean_abs_residual.detach()
                * 255.0
            ),

            "max_abs_residual": (
                max_abs_residual.detach()
            ),

            "max_abs_residual_255": (
                max_abs_residual.detach()
                * 255.0
            ),

            "residual_saturation_ratio": (
                residual_saturation_ratio.detach()
            )
        }

        metrics.update(
            color_metrics
        )

        return (
            total_loss,
            metrics
        )


# ============================================================
# Encoder loss
# ============================================================

class InactiveEncoderLoss(nn.Module):
    """
    Loss used when updating the compromised SimCLR encoder.

    During this step:

        Backdoored SimCLR encoder:
            trainable

        Clean SimCLR encoder:
            frozen

        StegaStamp hider:
            frozen

    All feature tensors must come from:

        model.f(normalized_image)

    not from the SimCLR projection head:

        model.g(feature)

    Loss terms:

        alignment:
            Triggered features move toward target-reference features.

        consistency:
            Target-reference images remain stable relative to the
            clean SimCLR encoder.

        utility:
            Clean shadow images preserve their clean SimCLR features.

        optional feature disentanglement:
            Triggered features remain separated from ordinary
            SSL-augmented features.
    """

    def __init__(
        self,
        weights: Optional[
            InactiveEncoderLossWeights
        ] = None,
        eps: float = 1e-8
    ):
        super().__init__()

        if weights is None:
            weights = (
                InactiveEncoderLossWeights()
            )

        self.weights = weights
        self.eps = float(eps)

    def forward(
        self,
        triggered_backdoor_features: torch.Tensor,
        alignment_reference_features: torch.Tensor,
        clean_backdoor_features: torch.Tensor,
        clean_target_features: torch.Tensor,
        reference_backdoor_features: Optional[
            torch.Tensor
        ] = None,
        reference_target_features: Optional[
            torch.Tensor
        ] = None,
        augmented_backdoor_features: Optional[
            torch.Tensor
        ] = None
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor]
    ]:
        """
        Compute the compromised-encoder objective.

        Args:
            triggered_backdoor_features:
                Features of triggered images from the trainable
                backdoored encoder.

            alignment_reference_features:
                Target-class reference features.

                Recommended initial implementation:
                    frozen clean-encoder reference features.

            clean_backdoor_features:
                Features of clean shadow images from the trainable
                backdoored encoder.

            clean_target_features:
                Features of the same clean images from the frozen
                clean encoder.

            reference_backdoor_features:
                Features of target-reference images from the trainable
                backdoored encoder.

            reference_target_features:
                Features of the same reference images from the frozen
                clean encoder.

            augmented_backdoor_features:
                Optional features of ordinary SSL-augmented images.
        """

        # ----------------------------------------------------
        # Trigger -> target feature alignment
        # ----------------------------------------------------
        alignment_loss = cosine_alignment_loss(
            source_features=(
                triggered_backdoor_features
            ),
            target_features=(
                alignment_reference_features
            ),
            eps=self.eps
        )

        # ----------------------------------------------------
        # Clean utility preservation
        # ----------------------------------------------------
        utility_loss = cosine_alignment_loss(
            source_features=(
                clean_backdoor_features
            ),
            target_features=(
                clean_target_features
            ),
            eps=self.eps
        )

        # ----------------------------------------------------
        # Reference consistency
        # ----------------------------------------------------
        if (
            reference_backdoor_features is None
            and reference_target_features is None
        ):
            consistency_loss = (
                triggered_backdoor_features
                .new_tensor(0.0)
            )

        elif (
            reference_backdoor_features is None
            or reference_target_features is None
        ):
            raise ValueError(
                "reference_backdoor_features and "
                "reference_target_features must either "
                "both be supplied or both be omitted"
            )

        else:
            consistency_loss = (
                cosine_alignment_loss(
                    source_features=(
                        reference_backdoor_features
                    ),
                    target_features=(
                        reference_target_features
                    ),
                    eps=self.eps
                )
            )

        # ----------------------------------------------------
        # Optional CA3 feature-space disentanglement
        # ----------------------------------------------------
        if augmented_backdoor_features is None:
            feature_disentangle_loss = (
                triggered_backdoor_features
                .new_tensor(0.0)
            )

        else:
            feature_disentangle_loss = (
                feature_disentangle_margin_loss(
                    triggered_features=(
                        triggered_backdoor_features
                    ),
                    augmented_features=(
                        augmented_backdoor_features
                    ),
                    maximum_similarity=(
                        self.weights
                        .maximum_trigger_augmentation_similarity
                    ),
                    eps=self.eps
                )
            )

        # ----------------------------------------------------
        # Complete encoder objective
        # ----------------------------------------------------
        total_loss = (
            self.weights.alignment_weight
            * alignment_loss

            + self.weights.consistency_weight
            * consistency_loss

            + self.weights.utility_weight
            * utility_loss

            + self.weights.feature_disentangle_weight
            * feature_disentangle_loss
        )

        # ----------------------------------------------------
        # Logging metrics
        # ----------------------------------------------------
        with torch.no_grad():
            alignment_target = (
                prepare_target_features(
                    source_features=(
                        triggered_backdoor_features
                    ),
                    target_features=(
                        alignment_reference_features
                    )
                )
            )

            alignment_similarity = (
                F.cosine_similarity(
                    F.normalize(
                        triggered_backdoor_features,
                        dim=1,
                        eps=self.eps
                    ),
                    F.normalize(
                        alignment_target,
                        dim=1,
                        eps=self.eps
                    ),
                    dim=1,
                    eps=self.eps
                ).mean()
            )

            clean_feature_similarity = (
                F.cosine_similarity(
                    F.normalize(
                        clean_backdoor_features,
                        dim=1,
                        eps=self.eps
                    ),
                    F.normalize(
                        clean_target_features,
                        dim=1,
                        eps=self.eps
                    ),
                    dim=1,
                    eps=self.eps
                ).mean()
            )

            if reference_backdoor_features is None:
                reference_feature_similarity = (
                    total_loss.new_tensor(0.0)
                )

            else:
                reference_feature_similarity = (
                    F.cosine_similarity(
                        F.normalize(
                            reference_backdoor_features,
                            dim=1,
                            eps=self.eps
                        ),
                        F.normalize(
                            reference_target_features,
                            dim=1,
                            eps=self.eps
                        ),
                        dim=1,
                        eps=self.eps
                    ).mean()
                )

            if augmented_backdoor_features is None:
                trigger_augmentation_similarity = (
                    total_loss.new_tensor(0.0)
                )

            else:
                trigger_augmentation_similarity = (
                    F.cosine_similarity(
                        F.normalize(
                            triggered_backdoor_features,
                            dim=1,
                            eps=self.eps
                        ),
                        F.normalize(
                            augmented_backdoor_features,
                            dim=1,
                            eps=self.eps
                        ),
                        dim=1,
                        eps=self.eps
                    ).mean()
                )

        metrics = {
            "loss_total": (
                total_loss.detach()
            ),

            "loss_alignment": (
                alignment_loss.detach()
            ),

            "loss_consistency": (
                consistency_loss.detach()
            ),

            "loss_utility": (
                utility_loss.detach()
            ),

            "loss_feature_disentangle": (
                feature_disentangle_loss.detach()
            ),

            "alignment_similarity": (
                alignment_similarity.detach()
            ),

            "clean_feature_similarity": (
                clean_feature_similarity.detach()
            ),

            "reference_feature_similarity": (
                reference_feature_similarity.detach()
            ),

            "trigger_augmentation_similarity": (
                trigger_augmentation_similarity.detach()
            )
        }

        return (
            total_loss,
            metrics
        )


# ============================================================
# Standalone tests
# ============================================================

def _test_injector_loss() -> None:
    """
    Confirm that:
        - the loss is finite
        - backward works
        - gradients reach the trigger tensor
    """

    torch.manual_seed(42)

    batch_size = 2
    feature_dim = 512
    max_residual = 8 / 255

    clean_image = torch.rand(
        batch_size,
        3,
        32,
        32
    )

    residual_parameter = torch.randn_like(
        clean_image,
        requires_grad=True
    )

    bounded_residual = (
        torch.tanh(residual_parameter)
        * max_residual
    )

    triggered_image = torch.clamp(
        clean_image + bounded_residual,
        0.0,
        1.0
    )

    augmented_image = torch.clamp(
        clean_image * 0.95
        + 0.01 * torch.randn_like(clean_image),
        0.0,
        1.0
    )

    # A simple differentiable projection is used only for this test.
    projection = torch.randn(
        3 * 32 * 32,
        feature_dim
    )

    triggered_features = (
        triggered_image.flatten(start_dim=1)
        @ projection
    )

    reference_features = torch.randn(
        batch_size,
        feature_dim
    )

    decoded_logits = torch.randn(
        batch_size,
        16,
        requires_grad=True
    )

    message = torch.randint(
        low=0,
        high=2,
        size=(batch_size, 16)
    ).float()

    loss_fn = InactiveInjectorLoss(
        max_residual=max_residual,
        use_lpips=False
    )

    total_loss, metrics = loss_fn(
        clean_image=clean_image,
        triggered_image=triggered_image,
        augmented_image=augmented_image,
        triggered_features=triggered_features,
        reference_features=reference_features,
        decoded_logits=decoded_logits,
        message=message,
        baseline_triggered_image=(
            triggered_image.detach()
        )
    )

    total_loss.backward()

    if residual_parameter.grad is None:
        raise RuntimeError(
            "Injector loss did not reach the trigger tensor"
        )

    if not torch.isfinite(total_loss):
        raise RuntimeError(
            "Injector loss is not finite"
        )

    if not torch.isfinite(
        metrics["color_gap"]
    ):
        raise RuntimeError(
            "Colour-gap metric is not finite"
        )

    print("InactiveInjectorLoss test passed")


def _test_encoder_loss() -> None:
    """
    Confirm that:
        - the encoder loss is finite
        - backward works
        - gradients reach trainable features
    """

    torch.manual_seed(123)

    batch_size = 4
    feature_dim = 512

    triggered_features = torch.randn(
        batch_size,
        feature_dim,
        requires_grad=True
    )

    alignment_reference_features = torch.randn(
        batch_size,
        feature_dim
    )

    clean_backdoor_features = torch.randn(
        batch_size,
        feature_dim,
        requires_grad=True
    )

    clean_target_features = torch.randn(
        batch_size,
        feature_dim
    )

    reference_backdoor_features = torch.randn(
        batch_size,
        feature_dim,
        requires_grad=True
    )

    reference_target_features = torch.randn(
        batch_size,
        feature_dim
    )

    augmented_backdoor_features = torch.randn(
        batch_size,
        feature_dim,
        requires_grad=True
    )

    loss_fn = InactiveEncoderLoss()

    total_loss, metrics = loss_fn(
        triggered_backdoor_features=(
            triggered_features
        ),
        alignment_reference_features=(
            alignment_reference_features
        ),
        clean_backdoor_features=(
            clean_backdoor_features
        ),
        clean_target_features=(
            clean_target_features
        ),
        reference_backdoor_features=(
            reference_backdoor_features
        ),
        reference_target_features=(
            reference_target_features
        ),
        augmented_backdoor_features=(
            augmented_backdoor_features
        )
    )

    total_loss.backward()

    if triggered_features.grad is None:
        raise RuntimeError(
            "Encoder loss did not reach triggered features"
        )

    if clean_backdoor_features.grad is None:
        raise RuntimeError(
            "Encoder loss did not reach clean features"
        )

    if not torch.isfinite(total_loss):
        raise RuntimeError(
            "Encoder loss is not finite"
        )

    if not torch.isfinite(
        metrics["alignment_similarity"]
    ):
        raise RuntimeError(
            "Alignment similarity is not finite"
        )

    print("InactiveEncoderLoss test passed")


if __name__ == "__main__":
    _test_injector_loss()
    _test_encoder_loss()

    print(
        "All inactive_disentangle_losses.py tests passed"
    )