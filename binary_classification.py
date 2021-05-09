# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CSCI 6380 Mini Project
#
# Chest X-Ray Images (Pneumonia) dataset from Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
#
# The zipfile downloaded from the link above is extracted to a local folder named `chest_xray`. The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
#
# Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
#
# For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.
#
# - Data: https://data.mendeley.com/datasets/rscbjbr9sj/2
# - License: CC BY 4.0
# - Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

# %% [markdown]
# ## TODO
#
# - [X] Data input and simple EDA.
# - [X] Have a baseline model.
# - [X] Generate a random split using just the training set, and see the performance on the "test set". If it's much better, then maybe there's mislabeling in the given test set.
# - [ ] Class imbalance:
#     - [ ] Use a weighted cross entropy loss function to account for the class imbalance.
#     - [X] [Augment](https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch)/Resample the training set to balance the normal and pneumonia samples.
# - [ ] Tune the hyperparameters.
# - [ ] Log images with wrong predictions into TensorBoard and visualize them. Possibly [useful link](https://www.tensorflow.org/tensorboard/image_summaries) from TensorBoard.

# %% tags=[]
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from core.data import PneumoniaDataModule
from core.model import PneumoniaDetector
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import functional as F

# %% [markdown]
# ## Exploratory Data Analysis
#
# Machine learning aside, we first need to check the number of samples under each data category (normal vs. pneumonia). As described above, we simply need to walk the subdirectories and count the number of files.

# %% tags=[]
data_dir = Path("./chest_xray").expanduser().resolve()
labels = ["NORMAL", "PNEUMONIA"]

# Check number of images in each category
label_counts = {}
for data_type in ("train", "val", "test"):
    label_counts[data_type] = {}
    for label in labels:
        label_count = len([x for x in (data_dir / data_type / label).glob("*.jpeg")])
        label_counts[data_type][label] = label_count

label_counts = pd.DataFrame(label_counts).T
label_counts["normal_freq"] = label_counts["NORMAL"] / (
    label_counts["NORMAL"] + label_counts["PNEUMONIA"]
)
label_counts

# %% [markdown]
# The test set is roughly 10% of the entire dataset. Two obvious problems can be found from this table:
#
# 1. The validation set is too small, and
# 2. There's almost three times as many pneumonia samples as normal ones.
#
# The test set should be left aside and only used to assess the performance of the final model. Ignoring the class imbalance problem for now, we can first read in all the training and validation samples, and use a random 90-10 split to generate new training and validation sets. Since we're planning to use [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for this project, we might as well start from the beginning and wrap the data loading code in a [datamodule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html). The core methods of a `datamodule` are:
#
# 1. `prepare_data` downloads data (i.e. writes to disk). In a distributed setting this method is only called from a single process. In our case we skip the definition of this method because the data was already downloaded.
# 2. `setup` contains data operations we might want to perform on every GPU, e.g. apply transforms, perform train/val splits, count frequency of the labels, etc.
# 3. `(train|val|test)_dataloader` generates data loaders for the corresponding datasets. Usually most of the work is already done in `setup`, so we just need to wrap the dataset and return a `DataLoader`. The batch sizes and number of threads to read the data are defined here.
#
# The defined `datamodule` can be found in `./core/data.py`. By default the files are not sorted, so all the normal samples would be retrieved first and then we get all the pneumonia samples. We can grab some images and see if there's any obvious differences between the groups.

# %% tags=[]
dm_test = PneumoniaDataModule(data_dir, batch_size=4)
dm_test.setup("test")

normal_batch = [dm_test.pneumonia_test[i] for i in range(4)]
normal_labels = [normal_batch[i][1] for i in range(4)]
normal_imgs = [normal_batch[i][0] for i in range(4)]

pneumonia_batch = [dm_test.pneumonia_test[i] for i in range(-4, 0)]
pneumonia_labels = [pneumonia_batch[i][1] for i in range(4)]
pneumonia_imgs = [pneumonia_batch[i][0] for i in range(4)]

imgs = normal_imgs + pneumonia_imgs
img_labels = normal_labels + pneumonia_labels
label_idx = {val: key for key, val in dm_test.class_to_idx.items()}

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24, 12))
for i in range(8):
    axs[i // 4, i % 4].imshow(imgs[i].permute(1, 2, 0))
    axs[i // 4, i % 4].text(80, -5, label_idx[img_labels[i]])

# %% [markdown]
# [Here's a pretty good explanation](https://radiologyassistant.nl/chest/chest-x-ray/lung-disease) of what we should be looking for in the chest X-rays. From what I can tell, the images of patients with pneumonia are more cloudy, and the edges of the lung aren't as clear as the ones in the normal images.

# %% [markdown]
# ## Data-Independent Model
#
# We may calculate the metrics of a "model" that always predicts the most common class (which is pneumonia in this case) or a model that gives random guesses. These metrics can be used as a baseline for all future models.

# %% tags=[]
pneumonia_sample_size = label_counts.loc["train", "PNEUMONIA"]
normal_sample_size = label_counts.loc["train", "NORMAL"]
fake_train = [1] * pneumonia_sample_size + [0] * normal_sample_size
common_pred = [1] * (pneumonia_sample_size + normal_sample_size)
random_pred = np.random.default_rng(42).choice(
    2, (pneumonia_sample_size + normal_sample_size)
)

baseline_metrics = pd.DataFrame(
    [
        (
            accuracy_score(fake_train, common_pred),
            precision_score(fake_train, common_pred),
            recall_score(fake_train, common_pred),
            f1_score(fake_train, common_pred),
        ),
        (
            accuracy_score(fake_train, random_pred),
            precision_score(fake_train, random_pred),
            recall_score(fake_train, random_pred),
            f1_score(fake_train, random_pred),
        ),
    ],
    columns=["Accuracy", "Precision", "Recall", "F1"],
    index=["common_class", "random_guess"],
)
baseline_metrics

# %% [markdown]
# ## Baseline Model
#
# The PyTorch library has some pretrained models availabe in the [torchvision](https://pytorch.org/vision/0.8/models.html) package. Here's an article [comparing some of those architectures](https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96), and we'll be using [ResNet18](https://arxiv.org/abs/1512.03385) as the backbone of our feature extractor.
#
# Similar to the `LightningDataModule`, in PyTorch Lightning we define a model using `LightningModule`. It organizes our PyTorch code into the following sections:
#
# 1. `__init__` is the model/system definition.
# 2. `forward` for making inference and predictions.
# 3. `training_step` is the training loop. Stuff like calculating the loss function and model metrics go here. We can also define `training_step_end` and `training_epoch_end` hooks for more fine-grained control.
# 4. `(validation|test)_step` are not required and are very similar to `training_step`. We're defining these as well because we have some custom metrics that we want to log.
# 5. `configure_optimizers` as its name suggests, configures one or multiple optimizers.
#
# We'll try to follow [PyTorch Lightning's style guides](https://pytorch-lightning.readthedocs.io/en/stable/starter/style_guide.html) when defining our model. The model can be found in `./core/model.py`. What we did was simply stripping out the last layer of ResNet, and replace it with a layer that has the same input dimensions but an output of only two nodes. The layers from ResNet are all frozen for faster training.

# %% tags=[]
# Set random seed for pytorch, numpy, python.random
pl.seed_everything(42)

# Init data and model
dm = PneumoniaDataModule(
    data_dir,
    batch_size=64,
    num_workers=16,
)

model = PneumoniaDetector(lr=1e-3)

# setup trainer
logger = TensorBoardLogger("tb_logs", name="pneumonia_bin_classifier")
trainer = pl.Trainer(
    max_epochs=15,
    gpus=1,
    logger=logger,
    deterministic=True,
)

# %% tags=[]
# train model
trainer.fit(model, dm)

# test on best checkpoint
test_metrics = trainer.test(datamodule=dm)


# %% [markdown]
# The baseline model performs really well on the training and validation sets -- accuracy, precision, recall, and F1 score are all >0.95. However, the test metrics are much worse (~0.79 for all metrics except recall). We could closer examine the predictions on the test set using a confusion matrix.

# %% tags=[]
# Evaluate on the test set: confusion matrix
def show_confusion_mat(model, datamod):
    model.freeze()
    y_pred, y_true = [], []

    for batch in datamod.test_dataloader():
        x, y = batch
        y_hat = model(x.cuda())
        pred = F.log_softmax(y_hat, dim=1).argmax(dim=1)
        y_pred.append(pred)
        y_true.append(y)

    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()

    mat = confusion_matrix(y_true, y_pred, labels=sorted(datamod.class_to_idx.values()))
    res = pd.DataFrame(mat, columns=labels, index=labels)
    res.index = [f"True {x}" for x in res.index]
    res.columns = [f"Predicted {x}" for x in res.columns]
    return res


show_confusion_mat(model, dm)

# %% [markdown]
# |                | Predicted normal | Predicted pneumonia |
# |:---------------|-----------------:|--------------------:|
# | True normal    |              122 |                 112 |
# | True pneumonia |                6 |                 384 |
#
# From the confusion matrix, it seems we are getting a lot of false positives where normal X-rays are predicted as pneumonia images. Possible reasons are:
#
# 1. Our model is performing really well, it's just some of the images in the test set are mislabeled.
# 2. We're overfitting to the training set, and the false positives are because of the class imbalance.
#
# Although (2) is the more likely reason, we could easily check (1) by holding out part of the training set as the new "test set" and train on the remaining data. If the performance on the "test set" greatly improves, then *maybe* there actually is mislabeling going on.

# %% [markdown]
# ## Re-split test set

# %%
pl.seed_everything(42)

# Init data and model
dm_split_train = PneumoniaDataModule(
    data_dir,
    sample_from_train=True,
    batch_size=64,
    num_workers=16,
)

model_split_train = PneumoniaDetector(lr=1e-3)

# setup trainer
logger_split_train = TensorBoardLogger(
    "tb_logs", name="pneumonia_bin_classifier_resample_train"
)
trainer_split_train = pl.Trainer(
    max_epochs=15,
    gpus=1,
    logger=logger_split_train,
    deterministic=True,
)

# train model
trainer_split_train.fit(model_split_train, dm_split_train)

# test on best checkpoint
test_metrics_split_train = trainer_split_train.test(datamodule=dm_split_train)

# %% [markdown]
# Okay **that** is interesting. When we ignore the original test set and split the training set into three parts, the train/validation metrics don't change that much, but the test accuracy, precision and recall jump to ~0.96. Let's take a look at the confusion matrix.

# %%
show_confusion_mat(model_split_train, dm_split_train)

# %% [markdown]
# |                | Predicted normal | Predicted pneumonia |
# |:---------------|-----------------:|--------------------:|
# | True normal    |              125 |                  16 |
# | True pneumonia |                6 |                 376 |
#
# The confusion matrix also seems pretty good. The false positive rate is much lower compared to the first model. Maybe there is mis-labeling in the given test set, or that the test set doesn't come from the same distribution as the training set. This problem has been discussed by [some other Kaggle users](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/discussion/122816), but no consensus was made at the end. The [original publication](https://doi.org/10.1016/j.cell.2018.02.010) reported an accuracy of 92.8%, with a sensitivity of 93.2% and a specificity of 90.1% (Figure 6). Perhaps we just need more epochs.

# %% [markdown]
# ## Class imbalance
#
# The high FPR could be because of the class imbalance -- we have way more pneumonia samples than healthy ones. To balance the classes, we could consider resampling the images labeled "normal", or we could augment them with random flips, rotations, etc.

# %% tags=[]
pl.seed_everything(42)

dm_augment = PneumoniaDataModule(
    data_dir,
    batch_size=256,
    num_workers=16,
    augment_minority=True,
    val_ratio=0.2,
)
model_augment = PneumoniaDetector(lr=5e-4)
logger_augment = TensorBoardLogger(
    "tb_logs", name="pneumonia_bin_classifier_augment_train"
)
trainer_aug_train = pl.Trainer(
    max_epochs=50,
    gpus=[2],
    logger=logger_augment,
    deterministic=True,
)

trainer_aug_train.fit(model_augment, dm_augment)
test_metrics_aug_train = trainer_aug_train.test(datamodule=dm_augment)

# %% tags=[]
checkpoint_dir = Path("./model_weights").expanduser().resolve()
checkpoint_dir.mkdir(parents=True, exist_ok=True)

trainer.save_checkpoint(checkpoint_dir / "base_model.pt.ckpt")
trainer_aug_train.save_checkpoint(checkpoint_dir / "aug_model.pt.ckpt")
# Load with:
# model_augment = PneumoniaDetector.load_from_checkpoint(PATH)

# %% tags=[]
show_confusion_mat(model_augment, dm_augment)

# %% [markdown]
# |                | Predicted normal | Predicted pneumonia |
# |:---------------|-----------------:|--------------------:|
# | True normal    |              134 |                 100 |
# | True pneumonia |                4 |                 386 |
#
# The augmentation yields slightly increased test metrics, but those values are still much worse than the training/validation metrics.
