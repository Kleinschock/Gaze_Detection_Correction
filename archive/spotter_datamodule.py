import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from typing import List

from .data_loader import GestureDataset

class SpotterDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the gesture spotting task.

    This module loads the gesture dataset but transforms the labels into a
    binary format: 0 for 'idle' and 1 for any other gesture. This is
    specifically for training the binary 'spotter' model.
    """
    def __init__(self, data_root: str, batch_size: int):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None

    def prepare_data(self):
        """Data is assumed to be already present at data_root."""
        pass

    def setup(self, stage: str = None):
        """
        Assigns train/val/test datasets for DataLoaders.
        This method is called on every GPU in a distributed setup.
        """
        if not self.full_dataset:
            # Use a fixed seed for reproducible splits
            torch.manual_seed(42)
            
            # Load the full dataset
            self.full_dataset = GestureDataset(self.data_root)

            # --- Create a 3-way Split: 80% Train, 10% Val, 10% Test ---
            test_split = 0.1
            val_split = 0.1
            
            test_size = int(test_split * len(self.full_dataset))
            val_size = int(val_split * len(self.full_dataset))
            train_size = len(self.full_dataset) - val_size - test_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.full_dataset, [train_size, val_size, test_size]
            )

            print(f"Spotter dataset split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation, {len(self.test_dataset)} test samples.")

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True, # Drop last for consistent batch sizes
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
