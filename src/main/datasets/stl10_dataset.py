# -------------------------------------------------------
# STL-10 dataset utilities for SimCLR pretraining
# -------------------------------------------------------
# The CA3 StegaStamp trigger is fixed at 32x32, so STL-10's
# native 96x96 images are resized to 32x32 BEFORE SimCLR's
# random crop / colour augmentations. This keeps the clean
# STL-10 encoder's input geometry aligned with the later
# StegaStamp + C2 attack pipeline.
# -------------------------------------------------------

from torchvision import datasets, transforms


STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)
IMAGE_SIZE = 32


_resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)

_train_after_resize = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(STL10_MEAN, STL10_STD),
])

_test_transform = transforms.Compose([
    _resize,
    transforms.ToTensor(),
    transforms.Normalize(STL10_MEAN, STL10_STD),
])


class STL10Pair(datasets.STL10):
    """Return two independently augmented 32x32 views of one STL-10 image."""

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)

        # Important: resize first, then sample the SimCLR augmentation.
        image = _resize(image)
        view_1 = _train_after_resize(image)
        view_2 = _train_after_resize(image)
        return view_1, view_2


def get_stl10_dataset(data_dir):
    """
    SimCLR upstream split:
      - training: STL-10 train + unlabeled (labels ignored)
      - kNN memory bank: labeled train split
      - kNN test: labeled test split
    """

    train_data = STL10Pair(
        root=data_dir,
        split="train+unlabeled",
        download=True,
        transform=None,
    )

    memory_data = datasets.STL10(
        root=data_dir,
        split="train",
        download=True,
        transform=_test_transform,
    )

    test_data = datasets.STL10(
        root=data_dir,
        split="test",
        download=True,
        transform=_test_transform,
    )

    return train_data, memory_data, test_data
