# +
from collections import Counter

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

# +
NORM_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
        augment_transforms (callable, optional):
            A function/transform that takes in an PIL image and returns a
            transformed version. Used to augment the minority class(es).
    """

    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=0,
        val_ratio=0.1,
        sample_from_train=False,
        augment_minority=False,
        augment_transforms=None,
    ):
        super().__init__()

        # Input args
        self.data_dir = data_dir
        self.sample_from_train = sample_from_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio

        # Default image transforms
        self.norm_transforms = NORM_TRANSFORMS

        if augment_transforms:
            self.aug_transforms = augment_transforms
        else:
            self.aug_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=30),
                ]
            )

        # Dict mapping class label -> index
        self.class_to_idx = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        if self.sample_from_train:
            self._setup_resample_train()
        else:
            self._setup(stage)

    def _setup(self, stage):
        if stage == "fit":
            pneumonia_train = ImageFolder(
                self.data_dir / "train", transform=self.norm_transforms
            )
            pneumonia_val = ImageFolder(
                self.data_dir / "val", transform=self.norm_transforms
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
                self.data_dir / "test", transform=self.norm_transforms
            )
            self.class_to_idx = self.pneumonia_test.class_to_idx

    def _setup_resample_train(self):
        pneumonia_train = ImageFolder(
            self.data_dir / "train", transform=self.norm_transforms
        )
        pneumonia_val = ImageFolder(
            self.data_dir / "val", transform=self.norm_transforms
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

    def train_dataloader(self):
        return DataLoader(
            self.pneumonia_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.pneumonia_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.pneumonia_test,
            batch_size=2 * self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# +
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class FolderWithAugmentation(ImageFolder):
    """Augment the minority class(es).

    Resample the minority classes to the same number as the majority class,
    and append the (img_path, class_index) tuple to self.imgs. When the getter
    is called, if the index is greater than the original sample size, then an
    augmented version of the appended image is returned.
    
    The code as of now is trying to work with the data that ImageFolder has
    already parsed. If this doesn't work out, then we can just write our own
    torch.Dataset class that:
    
      1. Iteratively walks the subdirectories and get paths to images.
      2. Counts # of images in each class.
      3. Augments all minority classes to the same number as the majority.
      4. When given an index, returns a (img, class) tuple that's either
         augmented or just normalized.
    
    Shouldn't be too difficult :)

    Args:
        root (string):
            Root directory path.
        augment_transforms (callable):
            A function/transform that takes in an PIL image and returns a
            transformed version. Used to augment the minority class(es).
        **kwargs:
            Other args passed to torchvision.datasets.ImageFolder.


     Attributes:
        classes (list):
            List of the class names sorted alphabetically.
        class_to_idx (dict):
            Dict with items (class_name, class_index).
        imgs (list):
            List of (image path, class_index) tuples
    """

    def __init__(self, root, augment_transforms, **kwargs):
        # Normalizing transforms applied to
        self.augment_transforms = augment_transforms
        super(ImageFolder, self).__init__(
            root=root,
            loader=default_loader,
            extensions=IMG_EXTENSIONS,
            transform=NORM_TRANSFORMS,
            **kwargs
        )
        self.imgs = self.samples
        self.majority_class = None
        self.original_length = len(self.imgs)

        self._find_majority_class()
        self._resample_imgs()

    def _find_majority_class(self):
        """Finds index of the majority class."""
        class_counts = Counter((x[1] for x in self.imgs))
        self.majority_class = max(class_counts, key=lambda x: class_counts[x])

    def _resample_imgs(self):
        pass

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


# +
from pathlib import Path

data_dir = Path("../chest_xray").expanduser().resolve()
aug_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
    ]
)
pneumonia_train = FolderWithAugmentation(
    data_dir / "train", augment_transforms=aug_transforms
)