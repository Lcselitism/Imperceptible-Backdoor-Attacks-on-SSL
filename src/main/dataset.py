from datasets.cifar10_dataset import get_cifar10_dataset, get_shadow_cifar10
from datasets.stl10_dataset import get_stl10_dataset


# -------------------------------------------------------
# Pretraining dataset (SimCLR)
# -------------------------------------------------------
# This is used in train_simclr.py
#
# Returns:
#   train_data   → raw images (no augmentation yet)
#   memory_data  → normalized train set (for kNN)
#   test_data    → normalized test set
#
# SimCLR augmentations will be applied later on GPU
def get_pretraining_dataset(args):

    dataset_name = str(args.dataset).lower()

    if dataset_name == "cifar10":
        return get_cifar10_dataset(args.data_dir)
    if dataset_name == "stl10":
        return get_stl10_dataset(args.data_dir)

    raise ValueError(
        f"Dataset not supported: {args.dataset!r}. Expected 'cifar10' or 'stl10'."
    )


# -------------------------------------------------------
# Shadow / backdoor dataset (INACTIVE)
# -------------------------------------------------------
# This is used in imperative.py
#
# Returns:
#   shadow_dataset       → used for backdoor training
#   memory_data          → clean train split
#   test_data_clean      → clean test set
#   test_data_backdoor   → backdoor test set
#
# This dataset prepares:
# - clean images
# - reference images
# - repeated samples for trigger matching
def get_shadow_dataset(args):

    # Currently only CIFAR-10 is supported
    if args.shadow_dataset == "cifar10":
        return get_shadow_cifar10(args)

    # Safety check
    else:
        raise ValueError("Shadow dataset not supported")