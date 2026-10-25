# -------------------------------------------------------
# Backdoor dataset utilities adapted to torchvision datasets
# -------------------------------------------------------
# This file provides:
#
# 1. BadEncoderDataset
#    Used during INACTIVE-style backdoor training
#
# 2. BadEncoderTestBackdoor
#    Used during backdoor evaluation
#
# 3. CIFAR10Mem
#    Clean dataset wrapper returning (img, label)
#
# 4. ReferenceImg
#    Wrapper around reference .npz images
#
# IMPORTANT DESIGN:
# The learned trigger/filter is NOT applied inside the dataset.
#
# Instead:
# - this dataset prepares the required clean / reference data
# - the actual filter_net is applied later inside training
#   or evaluation code
# -------------------------------------------------------

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os


# -------------------------------------------------------
# Helper: convert image to PIL
# -------------------------------------------------------
# Supports:
# - PIL images
# - numpy arrays
# - torch tensors
# -------------------------------------------------------
def _to_pil(img):
    # Already a PIL image
    if isinstance(img, Image.Image):
        return img

    # Numpy image
    if isinstance(img, np.ndarray):
        # If image is CHW, convert to HWC for PIL
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        return Image.fromarray(img.astype(np.uint8))

    # Torch tensor image
    if torch.is_tensor(img):
        # If tensor is CHW, convert to HWC uint8
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.detach().cpu()
            img = torch.clamp(img, 0.0, 1.0)
            img = (img * 255.0).byte().permute(1, 2, 0).numpy()
            return Image.fromarray(img)

    raise TypeError(f"Unsupported image type: {type(img)}")


# -------------------------------------------------------
# Helper: get raw image + label from a dataset
# -------------------------------------------------------
# We avoid calling dataset.__getitem__ when possible,
# because that may already apply transforms.
# -------------------------------------------------------
def _get_raw_sample(base_dataset, index):
    # CIFAR-style datasets
    if hasattr(base_dataset, "data") and hasattr(base_dataset, "targets"):
        img = base_dataset.data[index]
        target = base_dataset.targets[index]

        if isinstance(target, np.generic):
            target = int(target)

        return _to_pil(img), int(target)

    # STL10-style datasets
    if hasattr(base_dataset, "data") and hasattr(base_dataset, "labels"):
        img = base_dataset.data[index]
        target = base_dataset.labels[index]

        if isinstance(target, np.generic):
            target = int(target)

        return _to_pil(img), int(target)

    # Fallback
    sample = base_dataset[index]

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        img, target = sample[0], sample[1]
        return _to_pil(img), int(target)

    raise ValueError("Unsupported dataset structure for raw sample extraction.")


# -------------------------------------------------------
# Helper: load reference images from .npz
# -------------------------------------------------------
# Expected format:
#   npz["x"] -> images
#   npz["y"] -> labels (optional)
# -------------------------------------------------------
def _load_reference_npz(reference_file):
    if reference_file is None or reference_file == "":
        raise ValueError("reference_file must be provided for BadEncoderDataset.")

    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    ref = np.load(reference_file)

    if "x" not in ref:
        raise ValueError(f"Reference file {reference_file} does not contain key 'x'.")

    ref_images = ref["x"]

    # Labels are optional
    ref_labels = ref["y"] if "y" in ref else None

    return ref_images, ref_labels


# -------------------------------------------------------
# Reference image dataset
# -------------------------------------------------------
# Used for:
# - Baseline-ASR evaluation
# - debugging / target inspection
# -------------------------------------------------------
class ReferenceImg(Dataset):
    def __init__(self, reference_file, transform=None):
        self.data, self.targets = _load_reference_npz(reference_file)
        self.transform = transform

    def __getitem__(self, index):
        img = _to_pil(self.data[index])

        # Some npz files may not contain labels
        if self.targets is None:
            target = -1
        else:
            target = self.targets[index]

            # Handle labels stored as arrays like [3]
            if isinstance(target, np.ndarray):
                target = int(np.array(target).reshape(-1)[0])
            else:
                target = int(target)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


# -------------------------------------------------------
# Main INACTIVE training dataset
# -------------------------------------------------------
# Returns:
#   img_raw
#   img_backdoor_list
#   target_image_list_return
#   target_img_1_list_return
#   im_1
#
# Meaning:
#   img_raw                  -> clean image
#   img_backdoor_list        -> repeated clean copies
#   target_image_list_return -> clean reference images
#   target_img_1_list_return -> augmented reference images
#   im_1                     -> augmented clean image
#
# IMPORTANT:
# The trigger itself is NOT applied here.
# -------------------------------------------------------
class BadEncoderDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        trigger_file,
        reference_file,
        indices,
        class_type,
        transform=None,
        bd_transform=None,
        ftt_transform=None,
    ):
        self.base_dataset = base_dataset
        self.trigger_file = trigger_file
        self.reference_file = reference_file

        # Load reference images
        self.target_image_list, self.target_label_list = _load_reference_npz(reference_file)
        self.target_pil_list = [_to_pil(ref_img) for ref_img in self.target_image_list]

        self.classes = class_type
        self.indices = np.array(indices)

        # Transforms
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

    def __getitem__(self, index):
        # Map current dataset index to actual sample index
        real_index = int(self.indices[index])

        # Raw clean image
        img_pil, _ = _get_raw_sample(self.base_dataset, real_index)

        # ---------------------------------------------------
        # 1. Augmented clean image branch
        # ---------------------------------------------------
        # This is returned as im_1 and later becomes img_trans
        # in imperative.py.
        #
        # IMPORTANT CHANGE:
        # Prefer ftt_transform here, so img_trans is actually
        # an augmented version, not just another clean copy.
        # ---------------------------------------------------
        if self.ftt_transform is not None:
            im_1 = self.ftt_transform(img_pil.copy())
        elif self.transform is not None:
            im_1 = self.transform(img_pil.copy())
        elif self.bd_transform is not None:
            im_1 = self.bd_transform(img_pil.copy())
        else:
            im_1 = img_pil.copy()

        # ---------------------------------------------------
        # 2. Main clean image branch
        # ---------------------------------------------------
        if self.bd_transform is not None:
            img_raw = self.bd_transform(img_pil.copy())
        else:
            img_raw = img_pil.copy()

        # ---------------------------------------------------
        # 3. Repeated clean image list
        # ---------------------------------------------------
        # One copy per reference image.
        # filter_net can later be applied to these copies.
        # ---------------------------------------------------
        img_backdoor_list = []

        for _ in range(len(self.target_image_list)):
            if self.bd_transform is not None:
                img_backdoor = self.bd_transform(img_pil.copy())
            else:
                img_backdoor = img_pil.copy()

            img_backdoor_list.append(img_backdoor)

        # ---------------------------------------------------
        # 4. Reference image lists
        # ---------------------------------------------------
        # target_image_list_return:
        #   clean reference images
        #
        # target_img_1_list_return:
        #   augmented reference images
        # ---------------------------------------------------
        target_image_list_return = []
        target_img_1_list_return = []

        for ref_img_np in self.target_image_list:
            ref_pil = self.target_pil_list[len(target_image_list_return)]

            if self.bd_transform is not None:
                target_image = self.bd_transform(ref_pil.copy())
            else:
                target_image = ref_pil.copy()

            if self.ftt_transform is not None:
                target_img_1 = self.ftt_transform(ref_pil.copy())
            elif self.bd_transform is not None:
                target_img_1 = self.bd_transform(ref_pil.copy())
            else:
                target_img_1 = ref_pil.copy()

            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return (
            img_raw,
            img_backdoor_list,
            target_image_list_return,
            target_img_1_list_return,
            im_1,
        )

    def __len__(self):
        return len(self.indices)


# -------------------------------------------------------
# Backdoor test dataset
# -------------------------------------------------------
# Used during evaluation.
#
# IMPORTANT:
# This dataset does NOT apply the learned trigger itself.
# It only forces every label to the chosen target class.
# -------------------------------------------------------
class BadEncoderTestBackdoor(Dataset):
    def __init__(self, base_dataset, trigger_file, reference_label, transform=None):
        self.base_dataset = base_dataset
        self.trigger_file = trigger_file
        self.target_class = int(reference_label)
        self.test_transform = transform

    def __getitem__(self, index):
        img_pil, _ = _get_raw_sample(self.base_dataset, index)

        if self.test_transform is not None:
            img_backdoor = self.test_transform(img_pil.copy())
        else:
            img_backdoor = img_pil.copy()

        # Force label to target class
        return img_backdoor, self.target_class

    def __len__(self):
        return len(self.base_dataset)


# -------------------------------------------------------
# Clean memory / evaluation dataset
# -------------------------------------------------------
# Standard wrapper returning:
#   (img, label)
# -------------------------------------------------------
class CIFAR10Mem(Dataset):
    def __init__(self, base_dataset, class_type, transform=None):
        self.base_dataset = base_dataset
        self.classes = class_type
        self.transform = transform

        # Expose .targets for convenience
        self.targets = []
        for i in range(len(base_dataset)):
            _, target = _get_raw_sample(base_dataset, i)
            self.targets.append(int(target))

    def __getitem__(self, index):
        img_pil, target = _get_raw_sample(self.base_dataset, index)

        if self.transform is not None:
            img = self.transform(img_pil.copy())
        else:
            img = img_pil.copy()

        return img, target

    def __len__(self):
        return len(self.base_dataset)