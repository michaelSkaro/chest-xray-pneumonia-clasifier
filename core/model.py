import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import F1, Accuracy, MetricCollection, Precision, Recall
from torch import nn
from torch.nn import functional as F
from torchvision import models


class PneumoniaDetector(pl.LightningModule):
    def __init__(self, lr, **kwargs):
        super().__init__()

        # Hyperparameters (all kwargs) are saved in self.hparams
        self.save_hyperparameters()

        # Use all but the last two layers of ResNet as the feature extractor
        backbone = models.resnet18(pretrained=True)
        num_feat_last_layer = backbone.fc.in_features
        layers = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*layers)

        # Freeze layers from the pre-trained model
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Output layer
        # See https://github.com/PyTorchLightning/lightning-flash
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 2nd to last layer of ResNet
            nn.Flatten(),
            nn.Linear(num_feat_last_layer, 2),
        )

        # Metrics
        metrics = {
            "accuracy": Accuracy(),
            "precision": Precision(is_multiclass=False),
            "recall": Recall(is_multiclass=False),
            "f1": F1(num_classes=2),
        }
        self.train_metrics = self._gen_metric_collection(metrics, "train")
        self.val_metrics = self._gen_metric_collection(metrics, "val")
        self.test_metrics = self._gen_metric_collection(metrics, "test")

    @staticmethod
    def _gen_metric_collection(metrics, prefix):
        res = {}
        for key, val in metrics.items():
            res[f"{prefix}_{key}"] = val
        return MetricCollection(res)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # calls self.forward()

        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)

        pred = F.log_softmax(y_hat, dim=1).argmax(dim=1)
        self.train_metrics(pred, y)
        self.log_dict(self.train_metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        pred = F.log_softmax(y_hat, dim=1).argmax(dim=1)
        self.val_metrics(pred, y)
        self.log_dict(self.val_metrics)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

        pred = F.log_softmax(y_hat, dim=1).argmax(dim=1)
        self.test_metrics(pred, y)
        self.log_dict(self.test_metrics)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
