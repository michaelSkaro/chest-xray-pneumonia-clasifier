from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Default normalizing and augmenting transformations
NORM_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

AUG_TRANSFORMS = transforms.Compose(
    [
        # transforms.RandomResizedCrop(224),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PneumoniaDataModule(pl.LightningDataModule):
    """Loads data for the pneumonia classifier.

    This module defines the training, validation and test splits,
    data preparation and transforms. The given training and validation
    sets are merged and randomly re-split because the original validation
    set is too small.

    Args:
        data_dir (pathlib.Path):
            The directory containing train, val and test subdirectories.
        batch_size (int, optional):
            The number of samples in a batch. 32 by default.
        num_workers (int, optional):
            Number of subprocesses to use for data loading. The default 0 means
            that the data will be loaded in the main process.
        val_ratio (float, optional):
            The proportion of the validation set. Should be a float number
            between 0 and 1, and is set to 0.1 by default.
        sample_from_train (bool, optional):
            If true, sample val and test sets from the training set. Otherwise
            the given test set is used.
        augment_minority (bool, optional):
            If true, the minority class(es) will be augmented to the same
            sample size as the majority class.
        normalize_transforms (callable, optional):
            A function/transform that takes in an PIL image and returns a
            transformed version.
        augment_transforms (callable, optional):
            A function/transform that takes in an PIL image and returns a
            transformed version. Used to augment the minority class(es).
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 0,
        val_ratio: float = 0.1,
        sample_from_train: bool = False,
        augment_minority: bool = False,
        normalize_transforms: Optional[Callable] = NORM_TRANSFORMS,
        augment_transforms: Optional[Callable] = AUG_TRANSFORMS,
    ):
        super().__init__()

        # Input args
        self.data_dir = data_dir
        self.sample_from_train = sample_from_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio

        self.norm_transforms = normalize_transforms
        self.aug_transforms = augment_transforms

        # Dict mapping class label -> index
        self.class_to_idx = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: str) -> None:
        # make assignments here (val/train/test split)
        # called on every process in DDP

        if self.sample_from_train:
            self._setup_resample_train()
        else:
            self._setup(stage)

    def _setup(self, stage: str):
        if stage == "fit":
            pneumonia_train = ImageFolder(
                str(self.data_dir / "train"), transform=self.norm_transforms
            )
            pneumonia_val = ImageFolder(
                str(self.data_dir / "val"), transform=self.norm_transforms
            )
            dat = ConcatDataset([pneumonia_train, pneumonia_val])
            self.class_to_idx = pneumonia_train.class_to_idx

            # Re-split train and val sets because the original
            # val set was too small (16 samples in total)
            val_sample_num = int(self.val_ratio * len(dat))
            train_sample_num = len(dat) - val_sample_num

            self.pneumonia_train, self.pneumonia_val = random_split(
                dat, [train_sample_num, val_sample_num]
            )

        if stage == "test":
            self.pneumonia_test = ImageFolder(
                str(self.data_dir / "test"), transform=self.norm_transforms
            )
            self.class_to_idx = self.pneumonia_test.class_to_idx

    def _setup_resample_train(self):
        pneumonia_train = ImageFolder(
            str(self.data_dir / "train"), transform=self.norm_transforms
        )
        pneumonia_val = ImageFolder(
            str(self.data_dir / "val"), transform=self.norm_transforms
        )
        dat = ConcatDataset([pneumonia_train, pneumonia_val])
        self.class_to_idx = pneumonia_train.class_to_idx

        # First split a test set
        test_sample_num = int(0.1 * len(dat))
        n = len(dat) - test_sample_num
        dat_n, self.pneumonia_test = random_split(dat, [n, test_sample_num])

        # Re-split train and val sets
        val_sample_num = int(self.val_ratio * n)
        train_sample_num = n - val_sample_num

        self.pneumonia_train, self.pneumonia_val = random_split(
            dat_n, [train_sample_num, val_sample_num]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pneumonia_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pneumonia_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pneumonia_test,
            batch_size=2 * self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def build_dataset_annotation(
    root_dir: Path, seed: Optional[int] = None
) -> pd.DataFrame:
    """Generates annotation table of samples.

    Given a folder, all subdirectories are considered as class labels. Images
    in each subdirectory are counted, and all classes with smaller sample sizes
    are resampled. The returned DataFrame can be used as the input for our torch
    dataset class ImageFolderWithAugmentation.

    Arguments:
        root_dir (Path): directory containing subdirectories with images.
        seed (int): seed number for sampling.

    Returns:
        A pandas DataFrame with columns:
        - filepath (Path): filepath to the image.
        - cls_label (str): class label of the image.
        - aug_image (bool): whether or not the image is resampled.
    """
    dir = root_dir.expanduser().resolve()
    classes = [d.name for d in dir.iterdir() if d.is_dir()]

    # Get file paths for each class label
    img_paths, cls_labels = [], []
    for target_class in classes:
        target_dir = dir / target_class
        target_imgs = [f for f in target_dir.glob("*.jpeg")]
        target_labels = [target_class] * len(target_imgs)
        img_paths.extend(target_imgs)
        cls_labels.extend(target_labels)

    # Arrange results into a DataFrame
    annot = pd.DataFrame(
        {"filepath": img_paths, "cls_label": cls_labels, "aug_image": False}
    )

    # Count number of samples in each class
    class_counts = Counter(cls_labels)
    majority_class = max(class_counts, key=lambda x: class_counts[x])
    majority_class_size = class_counts[majority_class]

    # Mark samples in the minority class(es) that need to be augmented
    for target_class in classes:
        if target_class == majority_class:
            continue

        augment_size = majority_class_size - class_counts[target_class]
        aug_samples = (
            annot.query("cls_label == @target_class")
            .sample(n=augment_size, replace=True, random_state=seed)
            .assign(aug_image=True)
        )
        annot = pd.concat([annot, aug_samples], axis=0, ignore_index=True)

    annot.reset_index(drop=True, inplace=True)
    return annot


class ImageFolderWithAugmentation(Dataset):
    """Augments the minority class(es).

    Resample the minority classes to the same number as the majority class,
    and append the (img_path, class_index) tuple to self.imgs. The class:

      1. Iteratively walks the subdirectories and get paths to images.
      2. Counts number of images in each class.
      3. Augments all minority classes to the same number as the majority.
      4. When given an index, returns a (img, class) tuple that's either
         augmented or just normalized.

    Args:
        df (pd.DataFrame):
            The format should match the output of `build_dataset_annotation`.
        augment_transforms (callable):
            A function/transform that takes in an PIL image and returns a
            transformed version. Used to augment the minority class(es).
        normalizing_transforms (callable):
            Same as augment_transforms, but applied to all other images
            as a normalization.

     Attributes:
        classes (list):
            List of the class names sorted alphabetically.
        class_to_idx (dict):
            Dict with items (class_name, class_index).
        imgs (list):
            List of (image path, class_index) tuples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        augment_transforms: Callable[[Image.Image], Any],
        normalizing_transforms: Callable,
    ):
        self.aug_transforms = augment_transforms
        self.norm_transforms = normalizing_transforms

        self.classes, self.class_to_idx = self.find_classes(df)
        self.imgs, self.aug_samples = self.extract_samples(df)

    @staticmethod
    def find_classes(df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
        classes = df["cls_label"].unique()
        classes.sort()
        classes = classes.tolist()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def extract_samples(
        self, df: pd.DataFrame
    ) -> Tuple[List[Tuple[Any, int]], List[bool]]:
        img_samples = (
            df.assign(
                cls_index=lambda x: [
                    self.class_to_idx[label] for label in x["cls_label"]
                ]
            )
            .filter(["filepath", "cls_index"])
            .to_records(index=False)
        )

        aug_samples = df["aug_image"].tolist()
        return img_samples, aug_samples

    @staticmethod
    def read_pil_image(path: Path) -> Image.Image:
        """
        Same as pil_loader function in torchvision.
        """
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Returns (sample, target) tuple where target is the index of the class label.
        """
        path, target = self.imgs[index]
        augment_sample = self.aug_samples[index]

        sample = self.read_pil_image(path)
        if augment_sample:
            sample = self.aug_transforms(sample)
        else:
            sample = self.norm_transforms(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.imgs)


if __name__ == "__main__":
    # Test build_dataset_annotation
    data_dir = Path("./chest_xray").expanduser().resolve()
    annot = build_dataset_annotation(data_dir / "train")
    print(
        annot.filter(["cls_label", "aug_image"])
        .value_counts()
        .reset_index()
        .pivot(index="cls_label", columns="aug_image", values=0)
        .fillna(0)
        .assign(Total=lambda x: x.sum(axis=1))
        .astype(int)
    )
