import torch
import pytorch_lightning as pl
import numpy as np
import wandb
import os
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset
from collections import Counter
from typing import List

from .data_loader import GestureDataset
from .config import (
    GESTURE_LABELS, MAX_SEQ_LENGTH, PREPROCESSING_STRATEGY, SCALER_PATH, MODEL_DIR,
    FINETUNING_CONFIG
)
from .lightning_module import SpotterLightningModule
from .scaler import StandardScaler

def moving_average(data, window_size):
    """Applies a simple moving average filter."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

class GestureDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the gesture recognition dataset.

    This module handles all data-related tasks: loading, splitting,
    and creating DataLoaders for training, validation, and testing.
    It also calculates class weights to handle data imbalance.
    
    In 'segmentation' mode, it uses a pre-trained spotter model to
    segment the data stream into periods of activity before feeding
    them to the main classifier.
    """
    def __init__(self, data_root: str, batch_size: int, model_type: str = 'gru', balancing_strategy: str = None, segmentation_artifact: str = None):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.model_type = model_type
        self.balancing_strategy = balancing_strategy
        self.segmentation_artifact = segmentation_artifact
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.scaler = None

    @property
    def train_labels(self) -> List[int]:
        """Returns the labels of the training dataset."""
        if not self.train_dataset:
            return []
        # This now works universally as self.train_dataset is always a Subset
        return [self.train_dataset.dataset.labels[i] for i in self.train_dataset.indices]

    def prepare_data(self):
        """
        Downloads or prepares data.
        In this case, the data is assumed to be already present at data_root.
        """
        pass

    def setup(self, stage: str = None):
        """
        Assigns train/val/test datasets. This method is responsible for:
        1. Splitting the data into training, validation, and test sets.
        2. If strategy is 'standardize', fitting the scaler ONLY on the training data.
        3. Saving the fitted scaler for inference.
        4. Assigning the scaler to all datasets to ensure consistent transformation.
        """
        torch.manual_seed(42)

        # --- Dataset Instantiation & Splitting ---
        if FINETUNING_CONFIG['ENABLED']:
            print("--- Setting up Data for Fine-Tuning ---")
            # For fine-tuning, we use the specialized dataset for training
            # but validate against the original, standard dataset.
            finetune_dataset = GestureDataset(
                self.data_root,
                balancing_strategy=self.balancing_strategy,
                scaler=None # Scaler will be assigned later
            )
            # Wrap the fine-tuning dataset in a Subset to ensure it has the same
            # attributes (.indices, .dataset) as the standard split datasets.
            self.train_dataset = Subset(finetune_dataset, range(len(finetune_dataset)))
            
            # Create a standard dataset for validation and testing
            standard_dataset = GestureDataset(
                self.data_root,
                model_type=self.model_type,
                balancing_strategy=None, # No balancing on val/test
                scaler=None,
                force_standard_load=True # IMPORTANT: Override fine-tuning mix
            )
            
            # Split the standard dataset into validation and test sets (50/50 split of 20% of data)
            val_split = 0.5
            if len(standard_dataset) == 0:
                raise ValueError("The standard dataset for validation/testing is empty.")
            
            val_size = int(val_split * len(standard_dataset))
            test_size = len(standard_dataset) - val_size
            
            if val_size <= 0 or test_size <= 0:
                raise ValueError(f"Standard dataset is too small to be split. Val: {val_size}, Test: {test_size}")

            # We only need val and test from this split
            _, self.val_dataset, self.test_dataset = random_split(
                standard_dataset, [0, val_size, test_size]
            )
            print(f"Fine-tuning setup: {len(self.train_dataset)} training samples (fine-tune mix).")
            print(f"Validation/Test setup: {len(self.val_dataset)} validation, {len(self.test_dataset)} test samples (standard data).")

        else:
            # Standard (non-finetuning) setup
            if self.segmentation_artifact:
                print(f"Running in segmentation mode with artifact: {self.segmentation_artifact}")
                full_dataset = self._get_segmented_dataset()
            else:
                print("Running in standard mode (no segmentation).")
                full_dataset = GestureDataset(
                    self.data_root,
                    model_type=self.model_type,
                    balancing_strategy=self.balancing_strategy,
                    scaler=None
                )

            # Create a 3-way Split: 80% Train, 10% Val, 10% Test
            test_split = 0.1
            val_split = 0.1
            if len(full_dataset) == 0:
                raise ValueError("The initial dataset is empty. Check data paths and loading logic.")

            test_size = int(test_split * len(full_dataset))
            val_size = int(val_split * len(full_dataset))
            train_size = len(full_dataset) - val_size - test_size

            if train_size <= 0 or val_size <= 0 or test_size <= 0:
                 raise ValueError(f"Dataset is too small to be split. Train: {train_size}, Val: {val_size}, Test: {test_size}")

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
            print(f"Dataset split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation, {len(self.test_dataset)} test samples.")

        # --- Scaler Fitting and Assignment (The Correct Way) ---
        if PREPROCESSING_STRATEGY == 'standardize':
            if stage == 'fit':
                print("Setting up for training: Fitting StandardScaler on the training split...")
                
                # Extract raw sequences ONLY from the training dataset to prevent data leakage
                # This now works universally as self.train_dataset is always a Subset
                train_indices = self.train_dataset.indices
                raw_train_data = np.array([self.train_dataset.dataset.sequences[i] for i in train_indices])

                # Reshape for scaler: (n_samples, seq_len, n_features) -> (n_samples * seq_len, n_features)
                if raw_train_data.ndim == 3:
                    n_samples, seq_len, n_features = raw_train_data.shape
                else:
                    # Handle case where data might be empty or malformed
                    print("Warning: Training data for scaler is empty or malformed.")
                    return
                
                reshaped_data = raw_train_data.reshape(n_samples * seq_len, n_features)

                self.scaler = StandardScaler().fit(reshaped_data)
                
                # Save the fitted scaler to be used for inference later
                os.makedirs(MODEL_DIR, exist_ok=True)
                self.scaler.save(SCALER_PATH)
                print(f"StandardScaler fitted on training data and saved to {SCALER_PATH}")

            elif stage in ('test', 'predict'):
                print(f"Setting up for inference: Loading pre-fitted StandardScaler from {SCALER_PATH}")
                if not os.path.exists(SCALER_PATH):
                    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}. Please run training first to fit and save the scaler.")
                self.scaler = StandardScaler().load(SCALER_PATH)
            
            # IMPORTANT: Assign the fitted/loaded scaler to all dataset splits
            if self.scaler:
                # This now works universally as all datasets are Subsets
                self.train_dataset.dataset.scaler = self.scaler
                self.val_dataset.dataset.scaler = self.scaler
                self.test_dataset.dataset.scaler = self.scaler
                print("Scaler assigned to train, validation, and test datasets.")

    def _get_segmented_dataset(self):
        """
        Uses the spotter model to segment the data and returns a new dataset
        containing only the detected gesture segments.
        """
        # 1. Download and load the spotter model from wandb artifacts
        run = wandb.init(project="gesture-recognition", job_type="dataset_creation")
        artifact = run.use_artifact(self.segmentation_artifact, type='model')
        artifact_dir = artifact.download()
        spotter_model_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        
        print(f"Loading spotter model from: {spotter_model_path}")
        spotter_model = GestureSpotterLightningModule.load_from_checkpoint(spotter_model_path)
        spotter_model.eval()
        spotter_model.freeze()

        # 2. Load the full, unsegmented dataset
        raw_dataset = GestureDataset(self.data_root)
        raw_loader = DataLoader(raw_dataset, batch_size=self.batch_size, shuffle=False)

        # 3. Perform frame-wise prediction with the spotter
        all_scores = []
        all_features = []
        all_labels = []
        print("Running spotter model for frame-wise prediction...")
        with torch.no_grad():
            for features, labels in raw_loader:
                scores = spotter_model(features).squeeze().cpu().numpy()
                all_scores.extend(scores)
                all_features.extend(features.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_scores = np.array(all_scores)
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        # 4. Smooth scores and identify active segments
        # These parameters can be tuned
        smoothing_window = 5
        activity_threshold = 0.5 

        smoothed_scores = moving_average(all_scores, smoothing_window)
        active_frames = smoothed_scores > activity_threshold
        
        # Find contiguous blocks of active frames
        active_indices = np.where(active_frames)[0]
        
        if len(active_indices) == 0:
            print("Warning: No active frames detected by the spotter model.")
            return TensorDataset(torch.empty(0, MAX_SEQ_LENGTH, all_features.shape[2]), torch.empty(0, dtype=torch.long))

        # 5. Create new dataset from segmented data
        # We will extract sequences of MAX_SEQ_LENGTH from the active regions
        segmented_features = []
        segmented_labels = []
        
        print(f"Extracting segments from {len(active_indices)} active frames...")
        # Simple strategy: create a sequence starting from each active frame
        for idx in active_indices:
            # Ensure we don't go out of bounds
            if idx + MAX_SEQ_LENGTH <= len(all_features):
                seq_features = all_features[idx : idx + MAX_SEQ_LENGTH]
                # Use the label of the last frame in the sequence as the representative label
                seq_label = all_labels[idx + MAX_SEQ_LENGTH - 1]
                
                # We only want to classify actual gestures, not idle periods
                if seq_label != GESTURE_LABELS["idle"]:
                    segmented_features.append(seq_features)
                    segmented_labels.append(seq_label)

        if not segmented_features:
            print("Warning: No valid gesture segments found after filtering for non-idle labels.")
            return TensorDataset(torch.empty(0, MAX_SEQ_LENGTH, all_features.shape[2]), torch.empty(0, dtype=torch.long))

        # Convert to tensors and create a new TensorDataset
        final_features = torch.tensor(np.array(segmented_features), dtype=torch.float32)
        final_labels = torch.tensor(np.array(segmented_labels), dtype=torch.long)
        
        print(f"Created segmented dataset with {len(final_features)} samples.")
        
        # Close the temporary wandb run
        run.finish()

        return TensorDataset(final_features, final_labels)

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        drop_last = True if self.model_type == 'ffnn' else False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=4, # Use multiple workers to speed up data loading
            pin_memory=True, # Speeds up data transfer to the GPU
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        drop_last = True if self.model_type == 'ffnn' else False
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test DataLoader."""
        drop_last = True if self.model_type == 'ffnn' else False
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
