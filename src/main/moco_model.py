"""MoCo v1 model for the CA3 SSL-framework generalization experiment.

Design choices for this FYP:
- 32x32 small-image ResNet-18 backbone, identical to the existing SimCLR encoder.
- Separate query/key encoders.
- Key encoder updated only by exponential moving average (EMA).
- MoCo v1-style single linear projection head: 512 -> feature_dim (default 128).
- FIFO queue of normalized key projections.
- Downstream features are ALWAYS the 512-D query-backbone representation before
  the projection head.

This file intentionally contains no trigger/backdoor logic. It is only the clean
MoCo baseline required before the finalized C2 attack is mounted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the exact small-image ResNet-18 backbone already used by the active
# SimCLR codebase so the backbone architecture is controlled across frameworks.
from sim_clr_model import SimCLRBase


class MoCoBranch(nn.Module):
    """One MoCo branch = backbone representation + contrastive projection head."""

    def __init__(self, arch: str = "resnet18", feature_dim: int = 128):
        super().__init__()
        self.backbone = SimCLRBase(arch=arch)
        self.backbone_dim = self.backbone.feature_dim  # 512 for ResNet-18

        # MoCo v1 uses a single linear layer for the contrastive representation.
        self.projector = nn.Linear(self.backbone_dim, feature_dim, bias=True)
        self.projection_dim = int(feature_dim)

    def encode(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Return the pre-projection backbone feature used downstream."""
        feature = self.backbone(x)
        if normalize:
            feature = F.normalize(feature, dim=1)
        return feature

    def project(self, feature: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        z = self.projector(feature)
        if normalize:
            z = F.normalize(z, dim=1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(x)
        return F.normalize(self.projector(feature), dim=1)


class MoCo(nn.Module):
    """Single-process MoCo v1 adapted to the CA3 32x32 ResNet-18 setup."""

    def __init__(
        self,
        arch: str = "resnet18",
        feature_dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        super().__init__()

        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        self.encoder_q = MoCoBranch(arch=arch, feature_dim=feature_dim)
        self.encoder_k = MoCoBranch(arch=arch, feature_dim=feature_dim)

        self.backbone_dim = self.encoder_q.backbone_dim
        self.projection_dim = int(feature_dim)
        self.queue_size = int(queue_size)
        self.momentum = float(momentum)
        self.temperature = float(temperature)

        # Initialize key branch from query branch and prevent gradient updates.
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue shape follows the original MoCo convention: [C, K].
        queue = torch.randn(self.projection_dim, self.queue_size)
        queue = F.normalize(queue, dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self) -> None:
        """theta_k <- m * theta_k + (1-m) * theta_q."""
        m = self.momentum
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Insert normalized keys into the FIFO queue with wrap-around support."""
        if keys.ndim != 2 or keys.size(1) != self.projection_dim:
            raise ValueError(
                "keys must have shape [batch, projection_dim], got "
                f"{tuple(keys.shape)}"
            )

        keys = keys.detach()
        batch_size = int(keys.size(0))
        if batch_size <= 0:
            return

        # If an unusually large batch is supplied, only its newest K keys can
        # survive in a K-entry queue.
        if batch_size >= self.queue_size:
            keys = keys[-self.queue_size :]
            batch_size = self.queue_size

        ptr = int(self.queue_ptr.item())
        first = min(self.queue_size - ptr, batch_size)
        second = batch_size - first

        self.queue[:, ptr : ptr + first] = keys[:first].T
        if second > 0:
            self.queue[:, :second] = keys[first:].T

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def encode_query(
        self, x: torch.Tensor, normalize: bool = False
    ) -> torch.Tensor:
        """512-D query-backbone feature for kNN/downstream evaluation."""
        return self.encoder_q.encode(x, normalize=normalize)

    def forward(
        self, im_q: torch.Tensor, im_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return MoCo logits and labels for one pair of augmented batches."""

        # Query branch receives gradients.
        q = self.encoder_q(im_q)  # [N, C], already L2-normalized

        # Key branch is an EMA target and receives no gradient.
        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = self.encoder_k(im_k)  # [N, C], already L2-normalized

        # Positive logits: q dot k for corresponding views.
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(1)

        # Negative logits: q against the stored dictionary.
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.temperature

        # Positive key is always at column 0.
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Queue is updated after its old contents have been used as negatives.
        self.dequeue_and_enqueue(k)

        return logits, labels


def get_moco_architecture(args) -> MoCo:
    """Build MoCo from argparse-style attributes."""
    return MoCo(
        arch=getattr(args, "arch", "resnet18"),
        feature_dim=getattr(args, "feature_dim", 128),
        queue_size=getattr(args, "queue_size", 65536),
        momentum=getattr(args, "moco_momentum", 0.999),
        temperature=getattr(args, "temperature", 0.07),
    )
