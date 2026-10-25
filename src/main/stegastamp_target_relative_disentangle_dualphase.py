"""
C2 target-aware disentanglement with continuous augmented-trigger target alignment.

The two C2 branches share the exact same sampled SSL augmentation A_omega:

    z_a  = f(A_omega(x))
    z_ta = f(A_omega(T(x)))

with a frozen target-class centroid c_y.

The redesigned C2 objective has two simple jobs:

    1) separation:
       |s_ta - s_a| should be at least m_sep

    2) continuous target attraction:
       s_ta should keep increasing toward 1 rather than stopping at a hard floor

where

    s_a  = cos(z_a, c_y)
    s_ta = cos(z_ta, c_y)

and

    L_C2 = lambda_sep * ReLU(m_sep - |s_ta - s_a|)
         + lambda_target * (1 - s_ta)

The augmentation-only branch and target centroid are stop-gradient anchors.
Only z_ta is differentiable inside this helper. The caller decides which
parameters receive that gradient: during the hider phase it reaches the
StegaStamp hider through A_omega(T(x)); during the encoder phase it updates
the compromised encoder through the same post-augmentation triggered branch.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _check_feature_matrix(features: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(features):
        raise TypeError(f"{name} must be a torch.Tensor")
    if features.ndim != 2:
        raise ValueError(
            f"{name} must have shape (B, D); got {tuple(features.shape)}"
        )
    if features.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    if not torch.isfinite(features).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def corrected_target_relative_disentangle_loss(
    augmented_clean_features: torch.Tensor,
    augmented_triggered_features: torch.Tensor,
    target_centroid: torch.Tensor,
    separation_margin: float = 0.10,
    separation_weight: float = 0.10,
    target_weight: float = 0.20,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the redesigned two-term C2 loss.

    Args:
        augmented_clean_features:
            f(A_omega(x)); treated as a detached anchor.
        augmented_triggered_features:
            f(A_omega(T(x))); remains differentiable. In the hider phase this
            gradient reaches the trigger generator; in the encoder phase it
            reaches the compromised encoder.
        target_centroid:
            Frozen target-class centroid, shape (1, D) or (B, D).
        separation_margin:
            m_sep in ReLU(m_sep - |s_ta - s_a|).
        separation_weight:
            lambda_sep.
        target_weight:
            lambda_target for the continuous target term (1 - s_ta).

    Returns:
        weighted C2 loss and detached diagnostics.
    """
    _check_feature_matrix(augmented_clean_features, "augmented_clean_features")
    _check_feature_matrix(
        augmented_triggered_features, "augmented_triggered_features"
    )
    _check_feature_matrix(target_centroid, "target_centroid")

    if augmented_clean_features.shape != augmented_triggered_features.shape:
        raise ValueError(
            "augmented_clean_features and augmented_triggered_features must "
            "have the same shape; got "
            f"{tuple(augmented_clean_features.shape)} and "
            f"{tuple(augmented_triggered_features.shape)}"
        )

    batch_size, feature_dim = augmented_triggered_features.shape
    if target_centroid.shape[1] != feature_dim:
        raise ValueError(
            "target_centroid feature dimension does not match input features: "
            f"{target_centroid.shape[1]} vs {feature_dim}"
        )
    if target_centroid.shape[0] not in (1, batch_size):
        raise ValueError(
            "target_centroid must have shape (1, D) or (B, D); got "
            f"{tuple(target_centroid.shape)}"
        )

    if separation_margin < 0.0:
        raise ValueError("separation_margin must be non-negative")
    if separation_weight < 0.0 or target_weight < 0.0:
        raise ValueError("C2 weights must be non-negative")

    # Stop-gradient anchors.
    z_a = F.normalize(augmented_clean_features.detach(), dim=1, eps=eps)
    c_y = F.normalize(target_centroid.detach(), dim=1, eps=eps)
    if c_y.shape[0] == 1:
        c_y = c_y.expand(batch_size, -1)

    # The only differentiable TRAD feature branch. The caller controls whether
    # this gradient updates the hider or the compromised encoder.
    z_ta = F.normalize(augmented_triggered_features, dim=1, eps=eps)

    s_a = F.cosine_similarity(z_a, c_y, dim=1, eps=eps)
    s_ta = F.cosine_similarity(z_ta, c_y, dim=1, eps=eps)
    signed_gap = s_ta - s_a
    abs_gap = signed_gap.abs()

    # Term 1: require target-relevant separation between augmentation-only and
    # trigger+augmentation branches, but stop pushing once margin is satisfied.
    separation_per_sample = F.relu(separation_margin - abs_gap)

    # Term 2: continuously increase augmented-trigger target similarity.
    # Unlike ReLU(tau - s_ta), this remains active even after s_ta > 0.90.
    target_per_sample = 1.0 - s_ta

    separation_loss = separation_per_sample.mean()
    target_loss = target_per_sample.mean()
    total_loss = (
        separation_weight * separation_loss
        + target_weight * target_loss
    )

    with torch.no_grad():
        separation_active = (abs_gap < separation_margin).float()
        separation_satisfied = (abs_gap >= separation_margin).float()
        trigger_above_aug = (signed_gap > 0.0).float()
        wrong_direction = (signed_gap < 0.0).float()

        metrics = {
            "loss_c2_separation": separation_loss.detach(),
            "loss_c2_target": target_loss.detach(),
            "loss_c2_total": total_loss.detach(),
            "c2_aug_target_similarity": s_a.mean().detach(),
            "c2_aug_trigger_target_similarity": s_ta.mean().detach(),
            "c2_signed_target_gap": signed_gap.mean().detach(),
            "c2_abs_target_gap": abs_gap.mean().detach(),
            "c2_separation_hinge_active_fraction": separation_active.mean().detach(),
            "c2_separation_satisfied_fraction": separation_satisfied.mean().detach(),
            "c2_trigger_above_aug_fraction": trigger_above_aug.mean().detach(),
            "c2_wrong_direction_fraction": wrong_direction.mean().detach(),
        }

    return total_loss, metrics
