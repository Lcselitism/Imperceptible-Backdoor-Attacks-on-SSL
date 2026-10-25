import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# stegastamp_like_losses.py
#
# Advanced CA3 StegaStamp loss file.
#
# This file is for Stage 2 and Stage 3 of the pipeline:
#   1. Train the advanced steganographic trigger.
#   2. Improve stealthiness before adding INACTIVE loss.
#
# Main losses:
#   - message reconstruction loss
#   - image L1 / MSE reconstruction loss
#   - SSIM loss
#   - PSNR hinge loss
#   - residual L1 / L2 / L∞ loss
#   - total variation loss
#   - frequency-domain residual loss
#   - texture-aware residual loss
#   - optional LPIPS loss
#
# Inputs are assumed to be in [0, 1].
# ============================================================


# ============================================================
# Tensor checks
# ============================================================

def check_image_pair(clean_image, encoded_image):
    """
    Check that clean image and encoded image have the same image shape:
        (B, 3, H, W)
    """

    if clean_image.dim() != 4:
        raise ValueError(
            f"Expected clean_image shape (B, C, H, W), "
            f"but got {tuple(clean_image.shape)}"
        )

    if encoded_image.dim() != 4:
        raise ValueError(
            f"Expected encoded_image shape (B, C, H, W), "
            f"but got {tuple(encoded_image.shape)}"
        )

    if clean_image.shape != encoded_image.shape:
        raise ValueError(
            "clean_image and encoded_image must have the same shape. "
            f"Got {tuple(clean_image.shape)} and "
            f"{tuple(encoded_image.shape)}"
        )

    if clean_image.size(1) != 3:
        raise ValueError(
            f"Expected clean_image to have 3 channels, "
            f"but got {clean_image.size(1)}"
        )


def check_message_pair(decoded_logits, message):
    """
    Check that decoded logits and target message have shape:
        (B, message_length)
    """

    if decoded_logits.dim() != 2:
        raise ValueError(
            f"Expected decoded_logits shape (B, message_length), "
            f"but got {tuple(decoded_logits.shape)}"
        )

    if message.dim() != 2:
        raise ValueError(
            f"Expected message shape (B, message_length), "
            f"but got {tuple(message.shape)}"
        )

    if decoded_logits.shape != message.shape:
        raise ValueError(
            "decoded_logits and message must have the same shape. "
            f"Got {tuple(decoded_logits.shape)} and "
            f"{tuple(message.shape)}"
        )


def check_residual(residual, clean_image):
    """
    Check that residual has image shape:
        (B, 3, H, W)
    """

    if residual.dim() != 4:
        raise ValueError(
            f"Expected residual shape (B, C, H, W), "
            f"but got {tuple(residual.shape)}"
        )

    if residual.shape != clean_image.shape:
        raise ValueError(
            "residual and clean_image must have the same shape. "
            f"Got {tuple(residual.shape)} and "
            f"{tuple(clean_image.shape)}"
        )


# ============================================================
# Basic metrics
# ============================================================

@torch.no_grad()
def bit_accuracy(decoded_logits, message):
    """
    Compute bit-level accuracy.

    decoded_logits:
        raw decoder logits.

    message:
        binary message tensor in {0, 1}.
    """

    check_message_pair(decoded_logits, message)

    predicted_bits = (torch.sigmoid(decoded_logits) >= 0.5).float()

    accuracy = (predicted_bits == message).float().mean()

    return accuracy.item()


@torch.no_grad()
def message_accuracy(decoded_logits, message):
    """
    Compute full-message accuracy.

    A message is correct only if all bits are correct.
    """

    check_message_pair(decoded_logits, message)

    predicted_bits = (torch.sigmoid(decoded_logits) >= 0.5).float()

    correct_per_sample = (predicted_bits == message).float().mean(dim=1)

    full_message_correct = (correct_per_sample == 1.0).float().mean()

    return full_message_correct.item()


def mse_to_psnr(mse, eps=1e-8):
    """
    Convert MSE to PSNR.

    Assumes image range [0, 1].
    """

    mse = torch.clamp(mse, min=eps)

    psnr = 10.0 * torch.log10(1.0 / mse)

    return psnr


def compute_batch_psnr(clean_image, encoded_image, eps=1e-8):
    """
    Differentiable batch PSNR.

    Returns:
        scalar mean PSNR over the batch.
    """

    check_image_pair(clean_image, encoded_image)

    mse_per_sample = F.mse_loss(
        encoded_image,
        clean_image,
        reduction="none"
    )

    mse_per_sample = mse_per_sample.flatten(start_dim=1).mean(dim=1)

    psnr_per_sample = mse_to_psnr(
        mse=mse_per_sample,
        eps=eps
    )

    return psnr_per_sample.mean()


# ============================================================
# SSIM loss
# ============================================================

def create_gaussian_window(
    window_size,
    sigma,
    channels,
    device,
    dtype
):
    """
    Create a Gaussian window for SSIM.
    """

    coordinates = torch.arange(
        window_size,
        device=device,
        dtype=dtype
    )

    coordinates = coordinates - window_size // 2

    gaussian_1d = torch.exp(
        -(coordinates ** 2) / (2.0 * sigma ** 2)
    )

    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]

    window = gaussian_2d.view(
        1,
        1,
        window_size,
        window_size
    )

    window = window.repeat(
        channels,
        1,
        1,
        1
    )

    return window


def ssim_index(
    image_a,
    image_b,
    window_size=7,
    sigma=1.5,
    c1=0.01 ** 2,
    c2=0.03 ** 2
):
    """
    Differentiable SSIM index for images in [0, 1].

    Returns:
        scalar mean SSIM.
    """

    check_image_pair(image_a, image_b)

    channels = image_a.size(1)

    window = create_gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        device=image_a.device,
        dtype=image_a.dtype
    )

    padding = window_size // 2

    mu_a = F.conv2d(
        image_a,
        window,
        padding=padding,
        groups=channels
    )

    mu_b = F.conv2d(
        image_b,
        window,
        padding=padding,
        groups=channels
    )

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = F.conv2d(
        image_a * image_a,
        window,
        padding=padding,
        groups=channels
    ) - mu_a_sq

    sigma_b_sq = F.conv2d(
        image_b * image_b,
        window,
        padding=padding,
        groups=channels
    ) - mu_b_sq

    sigma_ab = F.conv2d(
        image_a * image_b,
        window,
        padding=padding,
        groups=channels
    ) - mu_ab

    numerator = (
        (2.0 * mu_ab + c1)
        * (2.0 * sigma_ab + c2)
    )

    denominator = (
        (mu_a_sq + mu_b_sq + c1)
        * (sigma_a_sq + sigma_b_sq + c2)
    )

    ssim_map = numerator / torch.clamp(
        denominator,
        min=1e-8
    )

    return ssim_map.mean()


def ssim_loss(clean_image, encoded_image):
    """
    SSIM loss.

    Lower is better:
        0 means identical by SSIM.
    """

    return 1.0 - ssim_index(clean_image, encoded_image)


# ============================================================
# Residual losses
# ============================================================

def residual_l1_loss(residual):
    """
    Mean absolute residual.
    """

    return residual.abs().mean()


def residual_l2_loss(residual):
    """
    Mean squared residual.
    """

    return (residual ** 2).mean()


def residual_linf_soft_loss(
    residual,
    max_residual,
    margin_ratio=0.90
):
    """
    Soft normalized L-infinity penalty.

    The residual is already hard-bounded by max_residual. This loss
    discourages the hider from frequently using values close to that
    maximum bound.

    The residual is normalized first:

        normalized_abs_residual = abs(residual) / max_residual

    Therefore:
        0.0 means no residual
        1.0 means the full allowed residual

    The penalty starts above margin_ratio.
    """

    if max_residual <= 0.0:
        raise ValueError("max_residual must be greater than 0")

    if not 0.0 <= margin_ratio <= 1.0:
        raise ValueError("margin_ratio must be in [0, 1]")

    normalized_abs_residual = (
        residual.abs() / max_residual
    )

    excess = F.relu(
        normalized_abs_residual - margin_ratio
    )

    return (excess ** 2).mean()


def total_variation_loss(residual):
    """
    Total variation loss.

    Encourages spatial smoothness in the residual.
    """

    vertical_difference = residual[:, :, 1:, :] - residual[:, :, :-1, :]
    horizontal_difference = residual[:, :, :, 1:] - residual[:, :, :, :-1]

    vertical_loss = vertical_difference.abs().mean()
    horizontal_loss = horizontal_difference.abs().mean()

    return vertical_loss + horizontal_loss


# ============================================================
# Frequency-domain loss
# ============================================================

def residual_frequency_loss(
    residual,
    low_frequency_radius_ratio=0.25
):
    """
    Penalize low-frequency residual energy.

    Reason:
        Low-frequency residuals often look like visible color or brightness
        shifts. For stealthiness, the residual should avoid creating obvious
        global image changes.

    This loss computes FFT energy and penalizes the centered low-frequency
    area more strongly.
    """

    check_residual(residual, residual)

    batch_size, channels, height, width = residual.shape

    fft = torch.fft.fft2(
        residual,
        dim=(-2, -1),
        norm="ortho"
    )

    fft = torch.fft.fftshift(
        fft,
        dim=(-2, -1)
    )

    energy = fft.real ** 2 + fft.imag ** 2

    y = torch.arange(
        height,
        device=residual.device,
        dtype=residual.dtype
    )

    x = torch.arange(
        width,
        device=residual.device,
        dtype=residual.dtype
    )

    yy, xx = torch.meshgrid(
        y,
        x,
        indexing="ij"
    )

    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    distance = torch.sqrt(
        (yy - center_y) ** 2
        + (xx - center_x) ** 2
    )

    max_radius = math.sqrt(
        center_y ** 2
        + center_x ** 2
    )

    low_frequency_radius = low_frequency_radius_ratio * max_radius

    low_frequency_mask = (
        distance <= low_frequency_radius
    ).float()

    low_frequency_mask = low_frequency_mask.view(
        1,
        1,
        height,
        width
    )

    low_frequency_energy = (
        energy * low_frequency_mask
    ).mean()

    total_energy = energy.mean()

    return low_frequency_energy + 0.10 * total_energy


# ============================================================
# Texture-aware residual loss
# ============================================================

def rgb_to_grayscale(image):
    """
    Convert RGB image to grayscale.

    Input:
        image in [0, 1], shape (B, 3, H, W)

    Output:
        grayscale image, shape (B, 1, H, W)
    """

    check_image_pair(image, image)

    red = image[:, 0:1]
    green = image[:, 1:2]
    blue = image[:, 2:3]

    gray = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    return gray


def image_gradient_magnitude(image):
    """
    Compute simple image gradient magnitude.

    Used to identify smooth regions and textured regions.
    """

    gray = rgb_to_grayscale(image)

    grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]

    grad_x = F.pad(
        grad_x,
        pad=(0, 1, 0, 0),
        mode="replicate"
    )

    grad_y = F.pad(
        grad_y,
        pad=(0, 0, 0, 1),
        mode="replicate"
    )

    gradient = torch.sqrt(
        grad_x ** 2
        + grad_y ** 2
        + 1e-8
    )

    return gradient


def texture_aware_residual_loss(
    clean_image,
    residual,
    smooth_region_strength=2.0
):
    """
    Penalize residuals more strongly in smooth image regions.

    Idea:
        Residuals are more visible in flat areas.
        Residuals are less visible around edges or textured regions.

    Weight:
        high weight  -> smooth region
        low weight   -> textured/edge region
    """

    check_residual(residual, clean_image)

    gradient = image_gradient_magnitude(clean_image)

    gradient = gradient / torch.clamp(
        gradient.mean(dim=(2, 3), keepdim=True),
        min=1e-6
    )

    texture_weight = torch.exp(-gradient)

    texture_weight = 1.0 + smooth_region_strength * texture_weight

    weighted_residual = texture_weight * residual.abs()

    return weighted_residual.mean()


# ============================================================
# Optional LPIPS loss
# ============================================================

class OptionalLPIPSLoss(nn.Module):
    """
    Optional LPIPS loss wrapper.

    This class does not force the project to install lpips immediately.
    If lpips_weight is 0, this module is not used.

    LPIPS expects image range [-1, 1], so this wrapper converts from [0, 1].
    """

    def __init__(
        self,
        enabled=False,
        net="alex"
    ):
        super().__init__()

        self.enabled = enabled
        self.net = net
        self.loss_fn = None

        if enabled:
            try:
                import lpips

                self.loss_fn = lpips.LPIPS(net=net)

                for parameter in self.loss_fn.parameters():
                    parameter.requires_grad = False

            except ImportError as error:
                raise ImportError(
                    "LPIPS was enabled, but the lpips package is not "
                    "installed. Install it with:\n\n"
                    "    pip install lpips\n\n"
                    "or set lpips_weight=0.0."
                ) from error

    def forward(self, clean_image, encoded_image):
        if not self.enabled:
            return clean_image.new_tensor(0.0)

        clean_scaled = clean_image * 2.0 - 1.0
        encoded_scaled = encoded_image * 2.0 - 1.0

        lpips_value = self.loss_fn(
            clean_scaled,
            encoded_scaled
        )

        return lpips_value.mean()


# ============================================================
# Loss configuration
# ============================================================

@dataclass
class StegaStampLossWeights:
    """
    Loss weights for advanced steganographic trigger training.

    Start conservative. The message loss must be strong enough so the
    decoder actually learns the hidden message.

    Suggested first run:
        message_weight = 1.0
        image_l1_weight = 2.0
        image_mse_weight = 1.0
        ssim_weight = 1.0
        residual_l2_weight = 1.0
        residual_linf_weight = 0.5
        residual_tv_weight = 0.1
        frequency_weight = 0.05
        texture_weight = 0.05
        psnr_weight = 0.1
        lpips_weight = 0.0 first, then 0.05 later
    """

    # Main decoder path. During corruption training, this is the
    # corrupted-image decoder path. 
    message_weight: float = 1.0

    # Additional loss for decoding the clean encoded image during
    # corruption training.
    clean_message_weight: float = 0.0

    image_l1_weight: float = 2.0
    image_mse_weight: float = 1.0

    ssim_weight: float = 1.0
    psnr_weight: float = 0.1
    target_psnr_db: float = 35.0

    lpips_weight: float = 0.0

    residual_l1_weight: float = 0.0
    residual_l2_weight: float = 1.0
    residual_linf_weight: float = 0.5
    residual_tv_weight: float = 0.1

    frequency_weight: float = 0.05
    texture_weight: float = 0.05

    residual_mask_l1_weight: float = 0.01


# ============================================================
# Main advanced loss
# ============================================================

class StegaStampLoss(nn.Module):
    """
    Main loss for training the advanced StegaStamp model.

    Expected model output dictionary:
        {
            "encoded_image": encoded_image,
            "residual": residual,
            "residual_mask": residual_mask,        optional
            "corrupted_image": corrupted_image,
            "decoded_logits": decoded_logits
        }

    Usage:
        loss_fn = StegaStampLoss(max_residual=8/255)
        total_loss, loss_dict = loss_fn(outputs, clean_image, message)
    """

    def __init__(
        self,
        weights: Optional[StegaStampLossWeights] = None,
        max_residual: float = 8 / 255,
        use_lpips: bool = False,
        lpips_net: str = "alex"
    ):
        super().__init__()

        if max_residual <= 0.0:
            raise ValueError("max_residual must be greater than 0")

        if weights is None:
            weights = StegaStampLossWeights()

        self.weights = weights
        self.max_residual = float(max_residual)

        lpips_enabled = (
            use_lpips
            and self.weights.lpips_weight > 0.0
        )

        self.lpips_loss = OptionalLPIPSLoss(
            enabled=lpips_enabled,
            net=lpips_net
        )

        self.message_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clean_image: torch.Tensor,
        message: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and return loss components.

        Returns:
            total_loss:
                scalar tensor used for backprop.

            loss_dict:
                dictionary of detached scalar tensors for logging.
        """

        if "encoded_image" not in outputs:
            raise KeyError("outputs must contain 'encoded_image'")

        if "decoded_logits" not in outputs:
            raise KeyError("outputs must contain 'decoded_logits'")

        encoded_image = outputs["encoded_image"]
        decoded_logits = outputs["decoded_logits"]

        clean_decoded_logits = outputs.get(
            "clean_decoded_logits",
            None
        )

        check_image_pair(clean_image, encoded_image)
        check_message_pair(decoded_logits, message)

        if clean_decoded_logits is not None:
            check_message_pair(
                clean_decoded_logits,
                message
            )

        if "residual" in outputs:
            residual = outputs["residual"]
            check_residual(residual, clean_image)
        else:
            residual = encoded_image - clean_image

        # Use true residual after clamping whenever possible.
        true_residual = encoded_image - clean_image

        residual_mask = outputs.get("residual_mask", None)

        # ----------------------------------------------------
        # Message loss
        # ----------------------------------------------------
        message_loss = self.message_loss_fn(
            decoded_logits,
            message
        )

        if clean_decoded_logits is not None:
            clean_message_loss = self.message_loss_fn(
                clean_decoded_logits,
                message
            )
        else:
            clean_message_loss = clean_image.new_tensor(0.0)

        # ----------------------------------------------------
        # Image-space losses
        # ----------------------------------------------------
        image_l1 = F.l1_loss(
            encoded_image,
            clean_image
        )

        image_mse = F.mse_loss(
            encoded_image,
            clean_image
        )

        ssim_value_loss = ssim_loss(
            clean_image,
            encoded_image
        )

        psnr_value = compute_batch_psnr(
            clean_image,
            encoded_image
        )

        # Hinge-style PSNR loss:
        # If PSNR is already above target, this becomes 0.
        psnr_hinge_loss = F.relu(
            self.weights.target_psnr_db - psnr_value
        ) / self.weights.target_psnr_db

        lpips_value_loss = self.lpips_loss(
            clean_image,
            encoded_image
        )

        # ----------------------------------------------------
        # Residual losses
        # ----------------------------------------------------
        residual_l1 = residual_l1_loss(true_residual)
        residual_l2 = residual_l2_loss(true_residual)

        residual_linf = residual_linf_soft_loss(
            residual=true_residual,
            max_residual=self.max_residual
        )

        residual_tv = total_variation_loss(true_residual)

        frequency_loss = residual_frequency_loss(true_residual)

        texture_loss = texture_aware_residual_loss(
            clean_image=clean_image,
            residual=true_residual
        )

        if residual_mask is not None:
            residual_mask_l1 = residual_mask.mean()
        else:
            residual_mask_l1 = clean_image.new_tensor(0.0)

        # ----------------------------------------------------
        # Weighted total
        # ----------------------------------------------------
        total_loss = (
            self.weights.message_weight * message_loss
            + self.weights.clean_message_weight * clean_message_loss
            + self.weights.image_l1_weight * image_l1
            + self.weights.image_mse_weight * image_mse
            + self.weights.ssim_weight * ssim_value_loss
            + self.weights.psnr_weight * psnr_hinge_loss
            + self.weights.lpips_weight * lpips_value_loss
            + self.weights.residual_l1_weight * residual_l1
            + self.weights.residual_l2_weight * residual_l2
            + self.weights.residual_linf_weight * residual_linf
            + self.weights.residual_tv_weight * residual_tv
            + self.weights.frequency_weight * frequency_loss
            + self.weights.texture_weight * texture_loss
            + self.weights.residual_mask_l1_weight * residual_mask_l1
        )

        # ----------------------------------------------------
        # Logging dictionary
        # ----------------------------------------------------
        with torch.no_grad():
            bit_acc = torch.tensor(
                bit_accuracy(decoded_logits, message),
                device=clean_image.device
            )

            msg_acc = torch.tensor(
                message_accuracy(decoded_logits, message),
                device=clean_image.device
            )

            if clean_decoded_logits is not None:
                clean_bit_acc = torch.tensor(
                    bit_accuracy(
                        clean_decoded_logits,
                        message
                    ),
                    device=clean_image.device
                )

                clean_msg_acc = torch.tensor(
                    message_accuracy(
                        clean_decoded_logits,
                        message
                    ),
                    device=clean_image.device
                )
            else:
                clean_bit_acc = bit_acc
                clean_msg_acc = msg_acc

            abs_residual = true_residual.abs()
            max_abs_residual = abs_residual.max()
            mean_abs_residual = abs_residual.mean()

            # Fraction of residual pixels using at least 95% of the
            # maximum allowed residual.
            residual_saturation_ratio = (
                abs_residual >= 0.95 * self.max_residual
            ).float().mean()

            ssim_value = 1.0 - ssim_value_loss

            if residual_mask is not None:
                mask_mean = residual_mask.mean()
                mask_min = residual_mask.min()
                mask_max = residual_mask.max()
            else:
                mask_mean = clean_image.new_tensor(0.0)
                mask_min = clean_image.new_tensor(0.0)
                mask_max = clean_image.new_tensor(0.0)

            corrupted_image = outputs.get(
                "corrupted_image",
                encoded_image
            )

            corruption_mean_abs_change = (
                corrupted_image - encoded_image
            ).abs().mean()

        loss_dict = {
            "loss_total": total_loss.detach(),

            # Keep the original name for compatibility with the
            # existing training and logging code.
            "loss_message": message_loss.detach(),

            # The main decoder path during corruption training.
            "loss_message_corrupted": message_loss.detach(),

            # Additional clean decoder path.
            "loss_message_clean": clean_message_loss.detach(),

            "loss_image_l1": image_l1.detach(),
            "loss_image_mse": image_mse.detach(),

            "loss_ssim": ssim_value_loss.detach(),
            "ssim_index": ssim_value.detach(),

            "psnr_db": psnr_value.detach(),
            "loss_psnr_hinge": psnr_hinge_loss.detach(),
            "loss_lpips": lpips_value_loss.detach(),

            "loss_residual_l1": residual_l1.detach(),
            "loss_residual_l2": residual_l2.detach(),
            "loss_residual_linf": residual_linf.detach(),
            "loss_residual_tv": residual_tv.detach(),
            "loss_frequency": frequency_loss.detach(),
            "loss_texture": texture_loss.detach(),
            "loss_residual_mask_l1": residual_mask_l1.detach(),

            # Original aliases.
            "bit_accuracy": bit_acc,
            "message_accuracy": msg_acc,

            # Corrupted decoder path.
            "corrupted_bit_accuracy": bit_acc,
            "corrupted_message_accuracy": msg_acc,

            # Clean decoder path.
            "clean_bit_accuracy": clean_bit_acc,
            "clean_message_accuracy": clean_msg_acc,

            "max_abs_residual": max_abs_residual.detach(),
            "mean_abs_residual": mean_abs_residual.detach(),

            "residual_saturation_ratio": (
                residual_saturation_ratio.detach()
            ),

            "mask_mean": mask_mean.detach(),
            "mask_min": mask_min.detach(),
            "mask_max": mask_max.detach(),

            "corruption_mean_abs_change": (
                corruption_mean_abs_change.detach()
            ),

            "corruption_mean_abs_change_255": (
                corruption_mean_abs_change.detach() * 255.0
            ),
        }

        return total_loss, loss_dict


# ============================================================
# Backward-compatible helper function
# ============================================================

def compute_stegastamp_loss(
    outputs,
    clean_image,
    message,
    max_residual=8 / 255,
    weights=None,
    use_lpips=False
):
    """
    Functional wrapper.

    Use this if an older training script prefers a function instead of
    creating StegaStampLoss manually.
    """

    loss_fn = StegaStampLoss(
        weights=weights,
        max_residual=max_residual,
        use_lpips=use_lpips
    )

    loss_fn = loss_fn.to(clean_image.device)

    return loss_fn(
        outputs=outputs,
        clean_image=clean_image,
        message=message
    )


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    message_length = 16

    clean = torch.rand(
        batch_size,
        3,
        32,
        32,
        device=device
    )

    residual = torch.randn_like(clean) * (2 / 255)

    encoded = torch.clamp(
        clean + residual,
        0.0,
        1.0
    )

    true_residual = encoded - clean

    message = torch.randint(
        low=0,
        high=2,
        size=(batch_size, message_length),
        device=device
    ).float()

    decoded_logits = torch.randn(
        batch_size,
        message_length,
        device=device,
        requires_grad=True
    )

    residual_mask = torch.rand(
        batch_size,
        1,
        32,
        32,
        device=device
    )

    outputs = {
        "encoded_image": encoded,
        "residual": true_residual,
        "residual_mask": residual_mask,
        "corrupted_image": encoded,
        "decoded_logits": decoded_logits
    }

    loss_fn = StegaStampLoss(
        max_residual=8 / 255,
        use_lpips=False
    ).to(device)

    total_loss, loss_dict = loss_fn(
        outputs=outputs,
        clean_image=clean,
        message=message
    )

    print("Total loss:", total_loss.item())

    for key, value in loss_dict.items():
        print(key, float(value.detach().cpu()))

    total_loss.backward()

    print("Advanced stegastamp_like_losses.py sanity check passed.")