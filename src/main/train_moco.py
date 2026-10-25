"""Train the clean CIFAR-10 MoCo baseline for CA3 framework generalization.

This script intentionally performs CLEAN SSL pretraining only. It does not load
StegaStamp, C2, target labels, reference images, or any attack checkpoint.

For the first MoCo experiment the existing repository's CIFAR-10 pair dataset is
reused unchanged. Therefore the clean MoCo views use the same CA3 augmentation
family as the existing clean SimCLR baseline:
    RandomResizedCrop(32, scale=(0.08, 1.0))
    RandomHorizontalFlip(0.5)
    RandomApply(ColorJitter(0.4,0.4,0.4,0.1), p=0.8)
    RandomGrayscale(0.2)
    ToTensor()
    CIFAR-10 normalization

Downstream/kNN features are taken from encoder_q.backbone BEFORE the MoCo
projection head.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_pretraining_dataset
from moco_model import get_moco_architecture


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]
STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD = [0.2603, 0.2566, 0.2713]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_labels(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        values = dataset.targets
    elif hasattr(dataset, "labels"):
        values = dataset.labels
    else:
        raise AttributeError(
            f"Dataset {type(dataset).__name__} has neither .targets nor .labels"
        )
    return [int(value) for value in values]


def knn_predict(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    feature_labels: torch.Tensor,
    classes: int,
    knn_k: int,
    knn_t: float,
) -> torch.Tensor:
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    sim_labels = feature_labels.expand(feature.size(0), -1).gather(
        dim=-1,
        index=sim_indices,
    )
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(
        feature.size(0) * knn_k,
        classes,
        device=feature.device,
    )
    one_hot_label.scatter_(
        dim=-1,
        index=sim_labels.reshape(-1, 1),
        value=1.0,
    )
    one_hot_label = one_hot_label.view(feature.size(0), knn_k, classes)

    pred_scores = torch.sum(
        one_hot_label * sim_weight.unsqueeze(-1),
        dim=1,
    )
    return pred_scores.argsort(dim=-1, descending=True)


@torch.no_grad()
def test_knn(model, memory_loader, test_loader, epoch, args, device) -> float:
    """kNN diagnostic using 512-D query-backbone features, never projection."""
    model.eval()

    classes = len(getattr(memory_loader.dataset, "classes", []))
    if classes <= 0:
        classes = 10

    feature_bank = []
    for data, _ in tqdm(memory_loader, desc="Feature extracting", leave=False):
        data = data.to(device, non_blocking=True)
        feature = model.encode_query(data, normalize=True)
        feature_bank.append(feature)

    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    feature_labels = torch.tensor(
        dataset_labels(memory_loader.dataset),
        device=device,
        dtype=torch.long,
    )

    # Avoid requesting more neighbours than exist in the memory bank.
    effective_k = min(args.knn_k, feature_bank.size(1))

    total_top1 = 0.0
    total_num = 0
    test_bar = tqdm(test_loader, desc="kNN test", leave=False)

    for data, target in test_bar:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        feature = model.encode_query(data, normalize=True)
        pred_labels = knn_predict(
            feature,
            feature_bank,
            feature_labels,
            classes,
            effective_k,
            args.knn_t,
        )

        total_num += data.size(0)
        total_top1 += (pred_labels[:, 0] == target).float().sum().item()
        test_bar.set_description(
            f"kNN Epoch [{epoch}/{args.epochs}] "
            f"Acc@1:{total_top1 / total_num * 100.0:.2f}%"
        )

    return total_top1 / total_num * 100.0


def train_one_epoch(model, loader, optimizer, epoch, args, device) -> float:
    model.train()

    total_loss = 0.0
    total_num = 0
    train_bar = tqdm(loader)

    for im_q, im_k in train_bar:
        im_q = im_q.to(device, non_blocking=True)
        im_k = im_k.to(device, non_blocking=True)

        logits, labels = model(im_q, im_k)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = im_q.size(0)
        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Loss: {total_loss / total_num:.4f}"
        )

    return total_loss / total_num


def normalization_metadata(dataset_name: str) -> dict:
    if dataset_name == "cifar10":
        return {"mean": CIFAR10_MEAN, "std": CIFAR10_STD}
    if dataset_name == "stl10":
        return {"mean": STL10_MEAN, "std": STL10_STD}
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch: int,
    final_loss: float,
    last_knn: float | None,
    args,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        # Save the downstream-relevant encoder explicitly as well. This makes
        # later framework-aware loading unambiguous without using projection.
        "encoder_q_backbone_state_dict": model.encoder_q.backbone.state_dict(),
        "epoch": int(epoch),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "final_ssl_loss": float(final_loss),
        "last_knn_top1": None if last_knn is None else float(last_knn),
        "extra": {
            "stage": "clean_moco_pretraining",
            "ssl_framework": "moco",
            "moco_variant": "v1",
            "implementation": "single_process_ca3_small_image",
            "upstream_dataset": args.dataset,
            "encoder_input_size": 32,
            "normalization": normalization_metadata(args.dataset),
            "arch": args.arch,
            "query_encoder": "small_image_resnet18",
            "key_encoder": "small_image_resnet18_ema",
            "downstream_feature_source": "encoder_q_backbone_pre_projection",
            "backbone_feature_dim": int(model.backbone_dim),
            "projection_head": "linear",
            "projection_dim": int(args.feature_dim),
            "queue_size": int(args.queue_size),
            "moco_momentum": float(args.moco_momentum),
            "temperature": float(args.temperature),
            "optimizer": "SGD",
            "lr": float(args.lr),
            "sgd_momentum": float(args.sgd_momentum),
            "weight_decay": float(args.weight_decay),
            "lr_milestones": [int(v) for v in args.lr_milestones],
            "lr_gamma": float(args.lr_gamma),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "augmentation_profile": args.augmentation_profile,
            "augmentation_pipeline": [
                "RandomResizedCrop(32, scale=(0.08, 1.0))",
                "RandomHorizontalFlip(p=0.5)",
                "RandomApply(ColorJitter(0.4,0.4,0.4,0.1), p=0.8)",
                "RandomGrayscale(p=0.2)",
                "ToTensor()",
                "Normalize(upstream_dataset)",
            ],
            # Original MoCo ShuffleBN depends on multi-GPU local BN subsets.
            # This CA3 baseline is intentionally a single-process implementation.
            "shuffle_bn": False,
            "stl10_pretraining_split": (
                "train+unlabeled" if args.dataset == "stl10" else None
            ),
        },
    }


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    knn: float | None,
    args,
) -> None:
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            final_loss=loss,
            last_knn=knn,
            args=args,
        ),
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean MoCo baseline for CA3 SSL-framework generalization"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10", "stl10"),
        default="cifar10",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help=(
            "Checkpoint directory. Defaults to ./results_moco_<dataset> so an "
            "STL-10 run cannot silently write into the CIFAR-10 result folder."
        ),
    )

    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--feature_dim", type=int, default=128)

    parser.add_argument("--queue_size", type=int, default=65536)
    parser.add_argument("--moco_momentum", type=float, default=0.999)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--optimizer", type=str, choices=("sgd",), default="sgd")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr_milestones",
        type=int,
        nargs="+",
        default=[600, 800],
    )
    parser.add_argument("--lr_gamma", type=float, default=0.1)

    parser.add_argument(
        "--augmentation_profile",
        type=str,
        choices=("ca3_matched",),
        default="ca3_matched",
        help=(
            "Uses the repository's existing CIFAR/STL pair loader. "
            "ca3_matched corresponds to the same crop/flip/jitter/grayscale "
            "family replayed by finalized C2."
        ),
    )

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--knn_k", type=int, default=200)
    parser.add_argument("--knn_t", type=float, default=0.5)
    parser.add_argument("--knn_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    # A periodic checkpoint is useful for a 1000-epoch run. It does not alter
    # optimization and can be disabled with 0.
    parser.add_argument("--save_every", type=int, default=100)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.save_dir is None:
        args.save_dir = f"./results_moco_{args.dataset}"

    if args.dataset != "cifar10":
        print(
            "Note: Step 1 is intended for CIFAR-10 upstream. STL-10 remains "
            "supported by the shared loader for later controlled runs."
        )

    if args.augmentation_profile != "ca3_matched":
        raise ValueError("Only the fixed CA3 matched augmentation profile is allowed")

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("CA3 Clean MoCo Baseline")
    print("=" * 80)
    print("SSL framework: MoCo")
    print("MoCo variant: v1")
    print(f"Upstream dataset: {args.dataset}")
    print("Query encoder: small-image ResNet-18")
    print("Key encoder: small-image ResNet-18, EMA-updated")
    print("Downstream feature: encoder_q backbone, post-GAP / pre-projection (512-D)")
    print(f"Projection head: Linear(512 -> {args.feature_dim})")
    print(f"Queue size: {args.queue_size}")
    print(f"Momentum coefficient: {args.moco_momentum}")
    print(f"Temperature: {args.temperature}")
    print("Optimizer: SGD")
    print(f"Learning rate: {args.lr}")
    print(f"SGD momentum: {args.sgd_momentum}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR milestones: {args.lr_milestones}, gamma={args.lr_gamma}")
    print(f"Augmentation profile: {args.augmentation_profile}")
    print("ShuffleBN: disabled in this single-process CIFAR implementation")
    print("Trigger/backdoor: NONE - clean baseline")
    print(f"Device: {device}")
    print("=" * 80)

    train_data, memory_data, test_data = get_pretraining_dataset(args)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )
    memory_loader = DataLoader(
        memory_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )

    model = get_moco_architecture(args).to(device)

    optimizer = torch.optim.SGD(
        model.encoder_q.parameters(),
        lr=args.lr,
        momentum=args.sgd_momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
    )

    last_loss = float("nan")
    last_knn = None

    for epoch in range(1, args.epochs + 1):
        last_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            args,
            device,
        )

        should_knn = (
            epoch == args.epochs
            or (args.knn_every > 0 and epoch % args.knn_every == 0)
        )
        if should_knn:
            last_knn = test_knn(
                model,
                memory_loader,
                test_loader,
                epoch,
                args,
                device,
            )
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Loss: {last_loss:.4f} "
                f"kNN Acc@1: {last_knn:.2f}% "
                f"LR: {optimizer.param_groups[0]['lr']:.6g}"
            )
        else:
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Loss: {last_loss:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6g}"
            )

        scheduler.step()

        if (
            args.save_every > 0
            and epoch % args.save_every == 0
            and epoch != args.epochs
        ):
            periodic_path = os.path.join(
                args.save_dir,
                f"moco_{args.dataset}_ep{epoch}.pth",
            )
            save_checkpoint(
                periodic_path,
                model,
                optimizer,
                scheduler,
                epoch,
                last_loss,
                last_knn,
                args,
            )
            print(f"Saved periodic checkpoint: {periodic_path}")

    final_path = os.path.join(
        args.save_dir,
        f"moco_{args.dataset}_ep{args.epochs}.pth",
    )
    save_checkpoint(
        final_path,
        model,
        optimizer,
        scheduler,
        args.epochs,
        last_loss,
        last_knn,
        args,
    )

    print("=" * 80)
    print("Clean MoCo pretraining complete")
    print(f"Final SSL loss: {last_loss:.6f}")
    if last_knn is not None:
        print(f"Final kNN Acc@1: {last_knn:.2f}%")
    print(f"Clean checkpoint: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
