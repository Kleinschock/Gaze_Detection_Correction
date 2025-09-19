import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

class GestureSpotter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GestureSpotter, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the input for the FFN
        x = x.view(x.size(0), -1)
        return self.network(x)

class GestureSpotterLightningModule(pl.LightningModule):
    def __init__(self, input_size, hidden_size, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = GestureSpotter(input_size, hidden_size, 1)
        self.criterion = nn.BCELoss()
        
        # Metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.test_precision = BinaryPrecision()

        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_binary = (y > 0).float().unsqueeze(1) # Convert to binary: 0=idle, 1=gesture
        y_hat = self(x)
        loss = self.criterion(y_hat, y_binary)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('train_recall', self.train_recall(y_hat, y_binary), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_binary = (y > 0).float().unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y_binary)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_accuracy(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('val_precision', self.val_precision(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall(y_hat, y_binary), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_binary = (y > 0).float().unsqueeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y_binary)

        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_acc', self.test_accuracy(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('test_precision', self.test_precision(y_hat, y_binary), on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall(y_hat, y_binary), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
