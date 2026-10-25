import argparse
import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

from sim_clr_model import get_encoder_architecture

from stegastamp_like_trigger import (
    load_stegastamp_hider,
    apply_stegastamp_trigger_batch
)

from stegastamp_like_utils import (
    set_seed,
    normalize_cifar10_tensor,
    save_json
)


# ============================================================
# tsne_stegastamp_learned_vs_ssl_aug.py
#
# t-SNE visualization for Objective 2.
#
# Purpose:
#   Visualize whether invisible ISSBA / StegaStamp-style triggered images
#   occupy a similar feature space to SSL-augmented images.
#
# Main comparison:
#   Group 0: StegaStamp / ISSBA-style triggered images
#   Group 1: SimCLR-style SSL-augmented images
#
# Feature extractor:
#   Clean SimCLR encoder trained on CIFAR-10.
#
# Default dataset:
#   CIFAR-10 test set.
#
# Why CIFAR-10 here:
#   The clean SimCLR encoder is trained on CIFAR-10, so this visualization
#   directly studies the feature space learned during SSL pretraining.
#
# Output:
#   ./figures/tsne_stegastamp_learned_vs_ssl_aug.png
#   ./results_stegastamp_learned/tsne_metrics.json
# ============================================================


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_RESULT_DIR = "./results_stegastamp_learned"
DEFAULT_FIGURE_DIR = "./figures"

DEFAULT_STEGASTAMP_CHECKPOINT = os.path.join(
    DEFAULT_RESULT_DIR,
    "stegastamp_like_model.pth"
)

DEFAULT_FIGURE_PATH = os.path.join(
    DEFAULT_FIGURE_DIR,
    "tsne_stegastamp_learned_vs_ssl_aug.png"
)

DEFAULT_METRICS_PATH = os.path.join(
    DEFAULT_RESULT_DIR,
    "tsne_metrics.json"
)


# ============================================================
# Minimal args object for sim_clr_model.py
# ============================================================

class SimCLRArgs:
    """
    Your sim_clr_model.py expects an args-style object.
    """

    def __init__(self, arch="resnet18", feature_dim=128):
        self.arch = arch
        self.feature_dim = feature_dim
        self.encoder_usage_info = "cifar10"


# ============================================================
# SimCLR loading
# ============================================================

def clean_state_dict_keys(state_dict):
    """
    Remove DataParallel 'module.' prefix if needed.
    """

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key

        cleaned[new_key] = value

    return cleaned


def load_clean_encoder(checkpoint_path: str, arch: str, projection_dim: int):
    """
    Load clean SimCLR encoder.

    We use model.f(images) as the representation extractor.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Clean SimCLR checkpoint not found: {checkpoint_path}")

    model_args = SimCLRArgs(
        arch=arch,
        feature_dim=projection_dim
    )

    model = get_encoder_architecture(model_args).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if len(missing) > 0:
        print("Warning: missing keys when loading SimCLR checkpoint:")
        print(missing)

    if len(unexpected) > 0:
        print("Warning: unexpected keys when loading SimCLR checkpoint:")
        print(unexpected)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


# ============================================================
# Dataset
# ============================================================

class TSNEComparisonDataset(Dataset):
    """
    Dataset for t-SNE feature comparison.

    For each sampled image, returns:
        1. clean image tensor in [0, 1]
        2. SSL-augmented image tensor in [0, 1]
        3. class label

    The StegaStamp trigger is applied later in batch form on GPU.
    This is faster and cleaner than applying the trigger inside __getitem__.
    """

    def __init__(
        self,
        base_dataset,
        sampled_indices,
        ssl_aug_strength: str = "simclr"
    ):
        self.base_dataset = base_dataset
        self.sampled_indices = sampled_indices
        self.to_tensor = transforms.ToTensor()

        self.ssl_aug_strength = ssl_aug_strength

        self.ssl_aug = self.build_ssl_augmentation(ssl_aug_strength)

    def build_ssl_augmentation(self, ssl_aug_strength: str):
        """
        Build SimCLR-style augmentation.

        This follows the common CIFAR SimCLR augmentations:
            RandomResizedCrop
            HorizontalFlip
            ColorJitter
            RandomGrayscale
            ToTensor
        """

        ssl_aug_strength = ssl_aug_strength.lower()

        if ssl_aug_strength == "weak":
            color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )

            crop_scale = (0.5, 1.0)

        elif ssl_aug_strength == "strong" or ssl_aug_strength == "simclr":
            color_jitter = transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )

            crop_scale = (0.08, 1.0)

        else:
            raise ValueError(
                f"Unsupported ssl_aug_strength: {ssl_aug_strength}. "
                f"Use 'weak', 'strong', or 'simclr'."
            )

        aug = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=crop_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])

        return aug

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        real_idx = self.sampled_indices[idx]

        pil_image, label = self.base_dataset[real_idx]

        clean_tensor = self.to_tensor(pil_image)
        augmented_tensor = self.ssl_aug(pil_image)

        return clean_tensor, augmented_tensor, label


def build_base_dataset(dataset_name: str, data_dir: str):
    """
    Build base dataset for t-SNE.

    Default is CIFAR-10.
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        base_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=None
        )

    elif dataset_name == "stl10":
        base_dataset = torchvision.datasets.STL10(
            root=data_dir,
            split="test",
            download=True,
            transform=None
        )

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Use 'cifar10' or 'stl10'."
        )

    return base_dataset


def sample_indices(
    base_dataset,
    num_samples: int,
    seed: int,
    non_target_only: bool,
    target_label: int
):
    """
    Sample image indices for t-SNE.

    If non_target_only=True:
        only use images whose label is not target_label.
    """

    all_indices = []

    for idx in range(len(base_dataset)):
        _, label = base_dataset[idx]

        if non_target_only:
            if label != target_label:
                all_indices.append(idx)
        else:
            all_indices.append(idx)

    rng = random.Random(seed)

    if num_samples > len(all_indices):
        num_samples = len(all_indices)

    sampled_indices = rng.sample(all_indices, num_samples)

    return sampled_indices


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_features_for_tsne(
    clean_encoder,
    stegastamp_model,
    dataloader,
    message_length: int,
    target_label: int
):
    """
    Extract feature vectors for two groups:

    Group 0:
        StegaStamp / ISSBA-style triggered images.

    Group 1:
        SSL-augmented images.

    Both groups are passed through the clean SimCLR encoder.
    """

    clean_encoder.eval()
    stegastamp_model.eval()

    all_features = []
    all_group_labels = []
    all_class_labels = []

    for clean_images, augmented_images, class_labels in dataloader:
        clean_images = clean_images.to(DEVICE)
        augmented_images = augmented_images.to(DEVICE)

        batch_size = clean_images.size(0)

        # ----------------------------------------------------
        # 1. Apply invisible trigger to clean images.
        # ----------------------------------------------------
        triggered_images = apply_stegastamp_trigger_batch(
            batch_tensor=clean_images,
            model=stegastamp_model,
            message_length=message_length,
            target_label=target_label,
            device=DEVICE,
            return_residual=False,
            return_cpu=False
        )

        # ----------------------------------------------------
        # 2. Normalize before SimCLR encoder.
        # ----------------------------------------------------
        triggered_norm = normalize_cifar10_tensor(triggered_images)
        augmented_norm = normalize_cifar10_tensor(augmented_images)

        # ----------------------------------------------------
        # 3. Extract normalized clean SimCLR features.
        # ----------------------------------------------------
        triggered_features = clean_encoder.f(triggered_norm)
        augmented_features = clean_encoder.f(augmented_norm)

        triggered_features = F.normalize(triggered_features, dim=1)
        augmented_features = F.normalize(augmented_features, dim=1)

        all_features.append(triggered_features.cpu())
        all_features.append(augmented_features.cpu())

        all_group_labels.extend([0] * batch_size)
        all_group_labels.extend([1] * batch_size)

        all_class_labels.extend(class_labels.tolist())
        all_class_labels.extend(class_labels.tolist())

    features = torch.cat(all_features, dim=0)

    return features, np.array(all_group_labels), np.array(all_class_labels)


# ============================================================
# Optional overlap metrics
# ============================================================

def compute_simple_overlap_metrics(features, group_labels):
    """
    Compute simple feature-space statistics before t-SNE.

    These are not the main result, but they help describe the overlap.

    Metrics:
        cosine_centroid_similarity:
            Similarity between group centroids.

        mean_cross_group_similarity:
            Average cosine similarity between a subset of triggered and
            augmented features.

        mean_within_trigger_similarity:
            Average similarity among triggered features.

        mean_within_aug_similarity:
            Average similarity among augmented features.
    """

    features = F.normalize(features, dim=1)

    group_labels_t = torch.tensor(group_labels)

    trigger_features = features[group_labels_t == 0]
    aug_features = features[group_labels_t == 1]

    trigger_centroid = F.normalize(
        trigger_features.mean(dim=0, keepdim=True),
        dim=1
    )

    aug_centroid = F.normalize(
        aug_features.mean(dim=0, keepdim=True),
        dim=1
    )

    centroid_similarity = F.cosine_similarity(
        trigger_centroid,
        aug_centroid
    ).item()

    # Limit pairwise computation for speed.
    max_pair_samples = min(1000, trigger_features.size(0), aug_features.size(0))

    trigger_subset = trigger_features[:max_pair_samples]
    aug_subset = aug_features[:max_pair_samples]

    cross_sim = torch.matmul(trigger_subset, aug_subset.T).mean().item()

    within_trigger_sim = torch.matmul(
        trigger_subset,
        trigger_subset.T
    ).mean().item()

    within_aug_sim = torch.matmul(
        aug_subset,
        aug_subset.T
    ).mean().item()

    return {
        "cosine_centroid_similarity": centroid_similarity,
        "mean_cross_group_similarity": cross_sim,
        "mean_within_trigger_similarity": within_trigger_sim,
        "mean_within_aug_similarity": within_aug_sim
    }


# ============================================================
# t-SNE
# ============================================================

def run_tsne(
    features,
    seed: int,
    perplexity: float,
    learning_rate,
    use_pca: bool,
    pca_dim: int
):
    """
    Run t-SNE.

    Optionally apply PCA first to make t-SNE faster and more stable.
    """

    features_np = features.numpy()

    if use_pca:
        pca_dim = min(pca_dim, features_np.shape[1])

        print(f"Running PCA before t-SNE: {features_np.shape[1]} -> {pca_dim}")

        pca = PCA(
            n_components=pca_dim,
            random_state=seed
        )

        features_np = pca.fit_transform(features_np)

        explained = float(pca.explained_variance_ratio_.sum())
        print(f"PCA explained variance ratio: {explained:.4f}")

    print("Running t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=seed
    )

    tsne_points = tsne.fit_transform(features_np)

    return tsne_points


# ============================================================
# Plotting
# ============================================================

def plot_tsne(
    tsne_points,
    group_labels,
    figure_path: str,
    title: str,
    point_size: int,
    alpha: float
):
    """
    Plot t-SNE result.

    Group labels:
        0 = StegaStamp / ISSBA triggered
        1 = SSL augmented
    """

    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    trigger_points = tsne_points[group_labels == 0]
    aug_points = tsne_points[group_labels == 1]

    plt.figure(figsize=(4, 4))

    plt.scatter(
        trigger_points[:, 0],
        trigger_points[:, 1],
        s=point_size,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.4,
        label="ISSBA / StegaStamp triggered"
    )

    plt.scatter(
        aug_points[:, 0],
        aug_points[:, 1],
        s=point_size,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.4,
        label="SSL augmented"
    )

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE figure to: {figure_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Data.
    parser.add_argument("--data_dir", type=str, default="./data")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "stl10"],
        help="Dataset used for t-SNE image sampling."
    )

    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--non_target_only",
        action="store_true",
        help="Use only non-target samples for t-SNE."
    )

    # Checkpoints.
    parser.add_argument(
        "--clean_encoder",
        type=str,
        default="../code/results/simclr_cifar10_ep1000.pth"
    )

    parser.add_argument(
        "--stegastamp_checkpoint",
        type=str,
        default=DEFAULT_STEGASTAMP_CHECKPOINT
    )

    # SimCLR.
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--projection_dim", type=int, default=128)

    # Trigger.
    parser.add_argument("--target_label", type=int, default=0)

    # SSL augmentation.
    parser.add_argument(
        "--ssl_aug_strength",
        type=str,
        default="simclr",
        choices=["weak", "strong", "simclr"]
    )

    # t-SNE.
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning_rate", default="auto")
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--pca_dim", type=int, default=50)

    # Plot.
    parser.add_argument(
        "--figure_path",
        type=str,
        default=DEFAULT_FIGURE_PATH
    )

    parser.add_argument(
        "--metrics_path",
        type=str,
        default=DEFAULT_METRICS_PATH
    )

    parser.add_argument("--point_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)

    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Using device: {DEVICE}")
    print("Running t-SNE: ISSBA / StegaStamp trigger vs SSL augmentation")
    print(f"Dataset: {args.dataset}")
    print(f"Num samples per group: {args.num_samples}")
    print(f"Clean encoder: {args.clean_encoder}")

    # --------------------------------------------------------
    # Load models.
    # --------------------------------------------------------
    print("\nLoading clean SimCLR encoder...")

    clean_encoder = load_clean_encoder(
        checkpoint_path=args.clean_encoder,
        arch=args.arch,
        projection_dim=args.projection_dim
    )

    print("Loading StegaStamp / ISSBA-style encoder...")

    stegastamp_model, message_length, max_residual = load_stegastamp_hider(
        checkpoint_path=args.stegastamp_checkpoint,
        device=DEVICE,
        freeze=True
    )

    print(f"Message length: {message_length}")
    print(f"Max residual: {max_residual}")

    # --------------------------------------------------------
    # Build dataset.
    # --------------------------------------------------------
    print("\nBuilding t-SNE dataset...")

    base_dataset = build_base_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir
    )

    sampled_indices = sample_indices(
        base_dataset=base_dataset,
        num_samples=args.num_samples,
        seed=args.seed,
        non_target_only=args.non_target_only,
        target_label=args.target_label
    )

    tsne_dataset = TSNEComparisonDataset(
        base_dataset=base_dataset,
        sampled_indices=sampled_indices,
        ssl_aug_strength=args.ssl_aug_strength
    )

    tsne_loader = DataLoader(
        tsne_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Actual samples per group: {len(tsne_dataset)}")
    print(f"Total t-SNE points: {len(tsne_dataset) * 2}")

    # --------------------------------------------------------
    # Extract features.
    # --------------------------------------------------------
    print("\nExtracting features...")

    features, group_labels, class_labels = extract_features_for_tsne(
        clean_encoder=clean_encoder,
        stegastamp_model=stegastamp_model,
        dataloader=tsne_loader,
        message_length=message_length,
        target_label=args.target_label
    )

    print(f"Feature shape: {features.shape}")

    # --------------------------------------------------------
    # Compute simple overlap metrics before t-SNE.
    # --------------------------------------------------------
    overlap_metrics = compute_simple_overlap_metrics(
        features=features,
        group_labels=group_labels
    )

    print("\nFeature-space overlap metrics:")
    for key, value in overlap_metrics.items():
        print(f"{key}: {value:.6f}")

    # --------------------------------------------------------
    # Run t-SNE.
    # --------------------------------------------------------
    tsne_points = run_tsne(
        features=features,
        seed=args.seed,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        use_pca=args.use_pca,
        pca_dim=args.pca_dim
    )

    # --------------------------------------------------------
    # Plot.
    # --------------------------------------------------------
    title = "t-SNE: ISSBA vs SSL Augmentation"

    plot_tsne(
        tsne_points=tsne_points,
        group_labels=group_labels,
        figure_path=args.figure_path,
        title=title,
        point_size=args.point_size,
        alpha=args.alpha
    )

    # --------------------------------------------------------
    # Save metrics.
    # --------------------------------------------------------
    metrics = {
        "setting": "tsne_trigger_vs_ssl_aug",
        "dataset": args.dataset.upper(),
        "target_label": args.target_label,
        "non_target_only": args.non_target_only,
        "num_samples_per_group": len(tsne_dataset),
        "total_points": len(tsne_dataset) * 2,
        "ssl_aug_strength": args.ssl_aug_strength,
        "message_length": message_length,
        "max_residual": max_residual,
        "clean_encoder": args.clean_encoder,
        "stegastamp_checkpoint": args.stegastamp_checkpoint,
        "figure_path": args.figure_path,
        "overlap_metrics": overlap_metrics
    }

    save_json(
        data=metrics,
        path=args.metrics_path
    )

    print(f"Saved t-SNE metrics to: {args.metrics_path}")


if __name__ == "__main__":
    main()