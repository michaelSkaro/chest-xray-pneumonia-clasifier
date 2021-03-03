import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.datasets import ImageFolder


class PneumoniaDataModule(pl.LightningDataModule):
    """
    This module defines the training, validation and test splits,
    data preparation and transforms.
    """

    def __init__(self, data_dir, batch_size=32, num_workers=0, val_ratio=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.class_to_idx = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if stage == "fit":
            pneumonia_train = ImageFolder(self.data_dir / "train")
            pneumonia_val = ImageFolder(self.data_dir / "val")
            dat = ConcatDataset([pneumonia_train, pneumonia_val])

            # Re-split train and val sets because the original
            # val set was too small (16 samples in total)
            val_sample_num = int(self.val_ratio * len(dat))
            train_sample_num = len(dat) - val_sample_num

            self.pneumonia_train, self.pneumonia_val = random_split(
                dat, [train_sample_num, val_sample_num]
            )

        if stage == "test":
            self.pneumonia_test = ImageFolder(self.data_dir / "test")
            self.class_to_idx = self.pneumonia_test.class_to_idx

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
