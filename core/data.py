import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PneumoniaDataModule(pl.LightningDataModule):
    """Loads data for the pneumonia classifier.

    This module defines the training, validation and test splits,
    data preparation and transforms. The given training and validation
    sets are merged and randomly re-split because the original validation
    set is too small.

    Args:
      data_dir (pathlib.Path):
        The directory containing train, val and test subdirectories.
      sample_from_train (bool, optional):
        If true, sample val and test sets from the training set. Otherwise
        the given test set is used.
      batch_size (int, optional):
        The number of samples in a batch. 32 by default.
      num_workers (int, optional):
        Number of subprocesses to use for data loading. The default 0 means
        that the data will be loaded in the main process.
      val_ratio (float, optional):
        The proportion of the validation set. Should be a float number
        between 0 and 1, and is set to 0.1 by default.
    """

    def __init__(
        self,
        data_dir,
        sample_from_train=False,
        batch_size=32,
        num_workers=0,
        val_ratio=0.1,
    ):
        super().__init__()

        # Input args
        self.data_dir = data_dir
        self.sample_from_train = sample_from_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio

        # Default image transforms
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
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
                self.data_dir / "train", transform=self.img_transforms
            )
            pneumonia_val = ImageFolder(
                self.data_dir / "val", transform=self.img_transforms
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
                self.data_dir / "test", transform=self.img_transforms
            )
            self.class_to_idx = self.pneumonia_test.class_to_idx

    def _setup_resample_train(self):
        pneumonia_train = ImageFolder(
            self.data_dir / "train", transform=self.img_transforms
        )
        pneumonia_val = ImageFolder(
            self.data_dir / "val", transform=self.img_transforms
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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
