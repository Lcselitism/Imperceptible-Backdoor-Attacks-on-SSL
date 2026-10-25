# -------------------------------------------------------
# CIFAR-10 dataset utilities
# -------------------------------------------------------
# This file provides:
#
# 1. SimCLR pretraining dataset
# 2. INACTIVE shadow (backdoor) dataset
# -------------------------------------------------------

import numpy as np

from torchvision import datasets, transforms

from .backdoor_dataset import (
    BadEncoderDataset,
    BadEncoderTestBackdoor,
    CIFAR10Mem,
)


# -------------------------------------------------------
# SimCLR train transform (CPU / PIL-based, original style)
# -------------------------------------------------------
# This is the standard SimCLR augmentation pipeline:
# PIL image -> augment -> tensor -> normalize
# -------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
])


# -------------------------------------------------------
# Standard clean transform (NO augmentation)
# -------------------------------------------------------
# Used for:
# - memory bank (kNN)
# - clean test set
# - downstream feature extraction
# -------------------------------------------------------
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
])


# -------------------------------------------------------
# Auxiliary transform for INACTIVE shadow dataset
# -------------------------------------------------------
# Used inside BadEncoderDataset for transformed clean/reference
# branches. Keep this close to the original style.
# -------------------------------------------------------
ftt_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
])


# -------------------------------------------------------
# CIFAR-10 dataset for SimCLR pretraining
# -------------------------------------------------------
# Returns TWO augmented views of the SAME image
# exactly like original SimCLR implementations
# -------------------------------------------------------
class CIFAR10Pair(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Convert numpy image to PIL
        img = transforms.ToPILImage()(img)

        # Two independent augmentations
        im_1 = train_transform(img)
        im_2 = train_transform(img)

        return im_1, im_2


# -------------------------------------------------------
# Main CIFAR-10 dataset (SimCLR pipeline)
# -------------------------------------------------------
# Returns:
#   train_data   -> paired augmented views for SimCLR training
#   memory_data  -> normalized clean train set for kNN memory bank
#   test_data    -> normalized clean test set
# -------------------------------------------------------
def get_cifar10_dataset(data_dir):

    train_data = CIFAR10Pair(
        root=data_dir,
        train=True,
        download=True
    )

    memory_data = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=test_transform,
        download=True
    )

    test_data = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )

    return train_data, memory_data, test_data


# -------------------------------------------------------
# CIFAR-10 shadow dataset (INACTIVE)
# -------------------------------------------------------
# Returns:
#   shadow_dataset      -> used during INACTIVE training
#   memory_data         -> clean train split
#   test_data_clean     -> clean test set
#   test_data_backdoor  -> backdoor test set
# -------------------------------------------------------
def get_shadow_cifar10(args):

    class_type = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    train_base = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True
    )

    test_base = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True
    )

    training_data_num = len(train_base)

    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(
        training_data_num,
        training_data_num,
        replace=False
    )

    shadow_dataset = BadEncoderDataset(
        base_dataset=train_base,
        trigger_file=args.trigger_file,
        reference_file=args.reference_file,
        class_type=class_type,
        indices=training_data_sampling_indices,
        transform=None,
        bd_transform=test_transform,
        ftt_transform=ftt_transform,
    )

    memory_data = CIFAR10Mem(
        base_dataset=train_base,
        class_type=class_type,
        transform=test_transform
    )

    test_data_clean = CIFAR10Mem(
        base_dataset=test_base,
        class_type=class_type,
        transform=test_transform
    )

    test_data_backdoor = BadEncoderTestBackdoor(
        base_dataset=test_base,
        trigger_file=args.trigger_file,
        reference_label=args.reference_label,
        transform=test_transform
    )

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor