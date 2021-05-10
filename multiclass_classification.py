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

# %% tags=[]
data_dir = Path("./chest_xray").expanduser().absolute()
checkpoint_file = (
    Path("./model_weights/multiclass_model.pt.ckpt").expanduser().resolve()
)

dm = PneumoniaDataModule(
    data_dir,
    batch_size=256,
    num_workers=16,
    augment_minority=True,
    pneumonia_subclass=True,
    val_ratio=0.2,
)

if not checkpoint_file.is_file():
    pl.seed_everything(42)

    model = PneumoniaDetector(lr=5e-4, class_num=3)
    logger = TensorBoardLogger(
        "tb_logs", name="pneumonia_multi_classifier_augment_train"
    )
    trainer = pl.Trainer(
        max_epochs=50,
        gpus=[1],
        logger=logger,
        deterministic=True,
    )
    trainer.fit(model, dm)
    test_metrics = trainer.test(model=model, datamodule=dm)
    trainer.save_checkpoint(checkpoint_file)
else:
    model = PneumoniaDetector.load_from_checkpoint(checkpoint_file).cuda()
    dm.setup("test")


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

    labels = sorted(datamod.class_to_idx.keys())
    mat = confusion_matrix(y_true, y_pred, labels=sorted(datamod.class_to_idx.values()))
    res = pd.DataFrame(mat, columns=labels, index=labels)
    res.index = [f"True {x}" for x in res.index]
    res.columns = [f"Predicted {x}" for x in res.columns]
    return res


show_confusion_mat(model, dm)

# %% [markdown]
# |                | Predicted normal | Predicted bacterial | Predicted viral |
# |:---------------|-----------------:|--------------------:|----------------:|
# | True normal    |              133 |                  44 |              57 |
# | True bacterial |                4 |                 231 |               7 |
# | True viral     |                0 |                  44 |             104 |
#
# The performance is similar to the binary classifier in terms of differentiating normal and pneumonia images.
