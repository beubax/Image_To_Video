from typing import Any, Callable, List, Union
from vit.utils import load_pretrained_weights
from vit.vision_transformer_graph import vit_base
from vit.vision_transformer_point import vit_base as vit_base_point
from vit.vision_transformer_trajectory import vit_base as vit_base_trajectory
import torch
from torch import nn, optim
from torchmetrics.functional import accuracy, f1_score
import lightning.pytorch as pl

class VideoLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr: float = 3e-4,
        weight_decay: float = 0,
        weight_path: str = None,
        max_epochs: int = None,
        label_smoothing: float = 0.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        point_cloud_classify: bool = False,
        testing: bool = False,
        trajectory: bool = False,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.num_classes = num_classes
        if point_cloud_classify:
            self.model = vit_base_point(
                num_classes=self.num_classes
            )
        elif testing:
             self.model = vit_base_point(
                num_classes=self.num_classes
            )
        elif trajectory:
            self.model = vit_base_trajectory(
                num_classes=self.num_classes
            )
        else:
            self.model = vit_base(
                num_classes=self.num_classes
            )

        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path), strict=False)
        else:
            load_pretrained_weights(
                self.model, model_name="vit_base", patch_size=16)
            

        if point_cloud_classify:
            for name, param in self.model.named_parameters():
                if 'point_cloud_classify' not in name:
                        param.requires_grad = False
        elif testing:
            for name, param in self.model.named_parameters():
                if 'pretrained_video_classifier' not in name and 'head' not in name:
                        param.requires_grad = False
        elif trajectory:
            for name, param in self.model.named_parameters():
                if 'temporal' not in name and 'head' not in name and 'proj_q' not in name and 'proj_kv' not in name: 
                        param.requires_grad = False
        else:
            for name, param in self.model.named_parameters():
                if 'temporal' not in name and 'head' not in name and 'point_cloud_tokenize' not in name and 'graph_transformer' not in name or 'flow_model' in name: 
                        param.requires_grad = False
  
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)
        self.log("train_f1", f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True, sync_dist=True)
        self.log("val_f1", f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes), prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        y_hat = self(x)
        y_pred = torch.softmax(y_hat, dim=-1)

        return {"y": y, "y_pred": torch.argmax(y_pred, dim=-1), "y_prob": y_pred}