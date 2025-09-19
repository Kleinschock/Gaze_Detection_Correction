import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pandas as pd


from .models import get_model
from .config import GESTURE_LABELS

class GestureLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training gesture recognition models.

    This module encapsulates the model, training, validation, and testing logic,
    making the training process cleaner and more organized.
    """
    def __init__(self, model_name: str, input_size: int, model_params: dict, learning_rate: float, weight_decay: float, class_balancing_strategy: str, class_weights: torch.Tensor = None):
        super().__init__()
        # Saves all __init__ arguments (model_name, learning_rate, etc.) to self.hparams
        # This is automatically logged by PyTorch Lightning's loggers (e.g., WandbLogger)
        self.save_hyperparameters()

        # --- Backward Compatibility Layer ---
        # This logic ensures that we can load older checkpoints that used
        # 'hidden_size' and 'num_layers' instead of the new 'hidden_sizes' list.
        params = self.hparams.model_params.copy() # Work on a copy
        if 'hidden_sizes' not in params:
            if 'hidden_size' in params:
                num_layers = params.get('num_layers', 1)
                params['hidden_sizes'] = [params['hidden_size']] * num_layers
                # Clean up old keys
                del params['hidden_size']
                if 'num_layers' in params:
                    del params['num_layers']
        # --- End Compatibility Layer ---

        self.model = get_model(name=self.hparams.model_name, input_size=self.hparams.input_size, num_classes=len(GESTURE_LABELS), model_params=params)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Initialize metrics for each phase
        self.train_accuracy = Accuracy(task="multiclass", num_classes=len(GESTURE_LABELS))
        self.val_accuracy = Accuracy(task="multiclass", num_classes=len(GESTURE_LABELS))
        self.test_accuracy = Accuracy(task="multiclass", num_classes=len(GESTURE_LABELS))
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=len(GESTURE_LABELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        Logs loss and accuracy for the training set.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_accuracy(logits, y)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        Logs loss and accuracy for the validation set.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_accuracy(logits, y)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_confusion_matrix.update(logits, y)

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        Computes, plots, and logs the confusion matrix.
        """
        # Ensure we are using a WandbLogger
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return

        # Compute the confusion matrix
        cm = self.val_confusion_matrix.compute().cpu().numpy()
        
        # Create a figure and plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GESTURE_LABELS, yticklabels=GESTURE_LABELS, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # Log the plot to wandb
        self.logger.experiment.log({
            "validation/confusion_matrix": wandb.Image(fig)
        })
        
        # Close the plot to free up memory
        plt.close(fig)

        # Reset the confusion matrix for the next epoch
        self.val_confusion_matrix.reset()

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.
        Logs accuracy for the test set.
        """
        x, y = batch
        logits = self(x)
        
        # Log metrics
        self.test_accuracy(logits, y)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        """
        Configures the optimizer for the training process.
        Uses AdamW, which is often a good default choice.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


class SpotterLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the binary gesture spotter model.
    """
    def __init__(self, model_name: str, input_size: int, model_params: dict, learning_rate: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()

        # The spotter is always a binary classifier (motion vs. no motion)
        self.model = get_model(name=model_name, input_size=input_size, num_classes=1, model_params=model_params)
        
        # Use BCEWithLogitsLoss for binary classification, which is more numerically stable
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize metrics for binary classification
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1) # Squeeze to match target shape
        loss = self.criterion(logits, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_accuracy(torch.sigmoid(logits), y)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_accuracy(torch.sigmoid(logits), y)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        
        self.test_accuracy(torch.sigmoid(logits), y)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
