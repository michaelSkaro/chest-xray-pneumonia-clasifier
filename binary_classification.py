# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## CSCI 6380 Mini Project
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
#
# ### TODO
#
# - [ ] Data input and simple EDA.
# - [ ] Have a baseline model.
# - [ ] Use a weighted cross entropy loss function to account for the class imbalance.
# - [ ] Augment/Resample the training set to balance the normal and pneumonia samples.
# - [ ] Tune the hyperparameters.
# - [ ] Log images with wrong predictions into TensorBoard and visualize them. Possibly [useful link](https://www.tensorflow.org/tensorboard/image_summaries) from TensorBoard.

# %%
from pathlib import Path

import pandas as pd

# %% [markdown]
# ### Exploratory Data Analysis
#
# Machine learning aside, we first need to check the number of samples under each data category (normal vs. pneumonia). As described above, we simply need to walk the subdirectories and count the number of files.

# %%
data_dir = Path("./chest_xray").expanduser().absolute()
labels = ["NORMAL", "PNEUMONIA"]

# Check number of images in each category
label_counts = {}
for data_type in ("train", "val", "test"):
    label_counts[data_type] = {}
    for label in labels:
        label_count = len([x for x in (data_dir / data_type / label).glob("*.jpeg")])
        label_counts[data_type][label] = label_count

label_counts = pd.DataFrame(label_counts).T
label_counts

# %% [markdown]
# The test set is roughly 10% of the entire dataset. Two obvious problems can be found from this table:
#
# 1. The validation set is too small, and
# 2. There's almost three times as many pneumonia samples as normal ones.
#
# The test set should be left aside and only used to assess the performance of the final model. Ignoring the class imbalance problem for now, we can first read in all the training and validation samples, and use a random 90-10 split to generate new training and validation sets.
