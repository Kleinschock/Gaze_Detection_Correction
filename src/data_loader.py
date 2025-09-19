import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math

# Import constants from the configuration file
from .config import (
    GESTURE_LABELS, MAX_SEQ_LENGTH, VALIDATION_SPLIT, BATCH_SIZE,
    WINDOW_STRIDE, LABELING_STRATEGY, FEATURES_TO_USE,
    PREPROCESSING_STRATEGY, SELECTED_KEYPOINTS, USE_DEDICATED_IDLE_DATA,
    FINETUNING_CONFIG
)
from .class_balancing import undersample_data
from .preprocessing import preprocess_sequence

# --- Ebene 1: Dynamische Keypoint-Auswahl aus der Konfiguration ---
# FEATURES_TO_USE is imported from config.py. We parse it to get the list of
# keypoints and the columns for coordinates, ignoring confidence scores for geometry.
KEYPOINT_COORDINATE_COLUMNS = [f for f in FEATURES_TO_USE if f.endswith(('_x', '_y', '_z'))]


class GestureDataset(Dataset):
    """
    PyTorch Dataset for gesture recognition. Implements a scientifically-grounded
    feature engineering pipeline:
    1.  Selects 14 relevant body keypoints.
    2.  Applies position and size normalization.
    3.  Engineers dynamic features (velocity, acceleration).
    4.  Engineers geometric/biomechanical features (angles, distances).
    5.  Pads/truncates sequences to a fixed length.
    """
    def __init__(self, data_root: str, model_type: str, balancing_strategy: str = None, scaler=None, debug: bool = False, force_standard_load: bool = False):
        self.data_root = data_root
        self.model_type = model_type
        self.balancing_strategy = balancing_strategy
        self.scaler = scaler
        self.sequences = []
        self.labels = []
        self._load_data(debug, force_standard_load)

        if self.balancing_strategy == 'undersample':
            self.sequences, self.labels = undersample_data(self.sequences, self.labels)

    def _load_data(self, debug: bool, force_standard_load: bool):
        """
        Scans the data directory, finds gesture sequences within each CSV,
        and populates the samples and labels lists.
        Supports multiple strategies: standard, dedicated idle, and fine-tuning.
        """
        print("Scanning data files...")
        if FINETUNING_CONFIG['ENABLED'] and not force_standard_load:
            print("Using FINETUNING data strategy.")
            self._load_finetuning_strategy(debug)
        elif USE_DEDICATED_IDLE_DATA:
            print("Using dedicated idle data strategy for GRU/FFNN models.")
            self._load_dedicated_idle_strategy(debug)
        else:
            print("Using standard data strategy for GRU model.")
            self._load_standard_strategy(debug)

        if not debug:
            print(f"Found {len(self.sequences)} total sequences.")

    def _process_csv_file(self, csv_path, gesture_label):
        """Helper to process a single CSV file and return its windows and labels."""
        file_sequences = []
        file_labels = []
        df = pd.read_csv(csv_path)

        if 'ground_truth' in df.columns and 'is_gesture' not in df.columns:
            df.rename(columns={'ground_truth': 'is_gesture'}, inplace=True)
        
        if 'is_gesture' in df.columns and df['is_gesture'].dtype == 'object':
            df['is_gesture'] = df['is_gesture'].apply(lambda x: x.strip().lower() != 'idle')

        if 'is_gesture' not in df.columns:
            return [], []
        
        for col in KEYPOINT_COORDINATE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        
        keypoint_data = df[KEYPOINT_COORDINATE_COLUMNS].values.astype(np.float32)
        
        for i in range(0, len(keypoint_data) - MAX_SEQ_LENGTH + 1, WINDOW_STRIDE):
            sequence = keypoint_data[i:i + MAX_SEQ_LENGTH]
            gesture_frames_in_sequence = df['is_gesture'][i:i + MAX_SEQ_LENGTH]
            
            is_gesture_window = False
            if LABELING_STRATEGY == 'any':
                is_gesture_window = gesture_frames_in_sequence.any()
            elif LABELING_STRATEGY == 'majority':
                gesture_ratio = gesture_frames_in_sequence.mean()
                is_gesture_window = gesture_ratio > 0.5
            
            current_label = gesture_label if is_gesture_window else GESTURE_LABELS['idle']
            
            file_sequences.append(sequence)
            file_labels.append(current_label)
            
        return file_sequences, file_labels

    def _load_standard_strategy(self, debug: bool):
        """Loads all data, including idle parts of gesture files."""
        for gesture_name, label in GESTURE_LABELS.items():
            gesture_dir = os.path.join(self.data_root, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue
            for fname in sorted(os.listdir(gesture_dir)):
                if not fname.endswith('.csv'):
                    continue
                csv_path = os.path.join(gesture_dir, fname)
                try:
                    sequences, labels = self._process_csv_file(csv_path, label)
                    self.sequences.extend(sequences)
                    self.labels.extend(labels)
                except Exception as e:
                    if debug: print(f"ERROR processing {fname}: {e}")

    def _load_dedicated_idle_strategy(self, debug: bool):
        """
        Loads gesture data by first creating all windows and then filtering out
        the ones labeled as 'idle'. Then, separately loads the dedicated idle data.
        """
        # 1. Process gesture files, creating windows and then filtering out idle ones.
        for gesture_name, label in GESTURE_LABELS.items():
            if gesture_name == 'idle':
                continue  # Skip the main idle folder for now
            
            gesture_dir = os.path.join(self.data_root, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue

            for fname in sorted(os.listdir(gesture_dir)):
                if not fname.endswith('.csv'):
                    continue
                csv_path = os.path.join(gesture_dir, fname)
                try:
                    sequences, labels = self._process_csv_file(csv_path, label)
                    # Filter out the windows that were labeled as idle from this gesture file
                    pure_gesture_sequences = [seq for seq, lbl in zip(sequences, labels) if lbl != GESTURE_LABELS['idle']]
                    pure_gesture_labels = [lbl for lbl in labels if lbl != GESTURE_LABELS['idle']]
                    self.sequences.extend(pure_gesture_sequences)
                    self.labels.extend(pure_gesture_labels)
                except Exception as e:
                    if debug: print(f"ERROR processing {fname}: {e}")

        # 2. Process the dedicated 'idle' folder.
        idle_dir = os.path.join(self.data_root, 'idle')
        if os.path.isdir(idle_dir):
            for fname in sorted(os.listdir(idle_dir)):
                if not fname.endswith('.csv'):
                    continue
                csv_path = os.path.join(idle_dir, fname)
                try:
                    sequences, labels = self._process_csv_file(csv_path, GESTURE_LABELS['idle'])
                    self.sequences.extend(sequences)
                    self.labels.extend(labels)
                except Exception as e:
                    if debug: print(f"ERROR processing {fname}: {e}")

    def __len__(self):
        return len(self.sequences)

    def _load_finetuning_strategy(self, debug: bool):
        """
        Loads data specifically for fine-tuning.
        1. Loads all data from the specified idle files.
        2. Loads all gesture data and samples a fraction of it.
        3. Combines them to create the fine-tuning dataset.
        """
        # 1. Load all data from the dedicated idle files for fine-tuning
        idle_dir = os.path.join(self.data_root, 'idle')
        if os.path.isdir(idle_dir):
            for fname in FINETUNING_CONFIG['IDLE_FILES_TO_USE']:
                if not fname.endswith('.csv'):
                    continue
                csv_path = os.path.join(idle_dir, fname)
                if not os.path.exists(csv_path):
                    if debug: print(f"WARNING: Idle file not found for fine-tuning: {csv_path}")
                    continue
                try:
                    sequences, labels = self._process_csv_file(csv_path, GESTURE_LABELS['idle'])
                    self.sequences.extend(sequences)
                    self.labels.extend(labels)
                except Exception as e:
                    if debug: print(f"ERROR processing {fname}: {e}")

        # 2. Load all gesture data to prepare for sampling
        all_gesture_sequences = []
        all_gesture_labels = []
        for gesture_name, label in GESTURE_LABELS.items():
            if gesture_name == 'idle':
                continue
            
            gesture_dir = os.path.join(self.data_root, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue

            for fname in sorted(os.listdir(gesture_dir)):
                if not fname.endswith('.csv'):
                    continue
                csv_path = os.path.join(gesture_dir, fname)
                try:
                    sequences, labels = self._process_csv_file(csv_path, label)
                    # Filter out idle windows from gesture files to get pure gestures
                    pure_gesture_sequences = [seq for seq, lbl in zip(sequences, labels) if lbl != GESTURE_LABELS['idle']]
                    pure_gesture_labels = [lbl for lbl in labels if lbl != GESTURE_LABELS['idle']]
                    all_gesture_sequences.extend(pure_gesture_sequences)
                    all_gesture_labels.extend(pure_gesture_labels)
                except Exception as e:
                    if debug: print(f"ERROR processing {fname}: {e}")

        # 3. Sample a fraction of the gesture data to prevent catastrophic forgetting
        sample_fraction = FINETUNING_CONFIG['GESTURE_DATA_SAMPLE_FRACTION']
        if sample_fraction > 0 and len(all_gesture_sequences) > 0:
            num_to_sample = int(len(all_gesture_sequences) * sample_fraction)
            if num_to_sample > 0:
                indices = list(range(len(all_gesture_sequences)))
                sampled_indices = random.sample(indices, num_to_sample)
                
                sampled_gesture_sequences = [all_gesture_sequences[i] for i in sampled_indices]
                sampled_gesture_labels = [all_gesture_labels[i] for i in sampled_indices]

                self.sequences.extend(sampled_gesture_sequences)
                self.labels.extend(sampled_gesture_labels)
                print(f"Sampled {len(sampled_gesture_sequences)} gesture sequences for fine-tuning.")

    def __getitem__(self, idx: int):
        raw_coords = self.sequences[idx]
        label = self.labels[idx]
        seq_len = raw_coords.shape[0]

        # Use the centralized preprocessing pipeline, passing the scaler if it exists
        final_features = preprocess_sequence(raw_coords, scaler=self.scaler)

        # Pad or Truncate to MAX_SEQ_LENGTH
        if seq_len < MAX_SEQ_LENGTH:
            padding = np.zeros((MAX_SEQ_LENGTH - seq_len, final_features.shape[1]), dtype=np.float32)
            final_features = np.vstack([final_features, padding])
        elif seq_len > MAX_SEQ_LENGTH:
            final_features = final_features[:MAX_SEQ_LENGTH, :]

        return torch.from_numpy(final_features), torch.tensor(label, dtype=torch.long)


def get_data_loaders(data_root: str, model_type: str = 'gru', balancing_strategy: str = None):
    """
    Creates and returns the training, validation, and test data loaders.
    This function ensures a consistent and non-overlapping data split.
    """
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)

    dataset = GestureDataset(data_root, model_type=model_type, balancing_strategy=balancing_strategy)

    # --- Create a 3-way Split: 80% Train, 10% Val, 10% Test ---
    test_split = 0.1  # Hardcoding 10% for test set
    val_split = 0.1   # Hardcoding 10% for validation set
    
    test_size = int(test_split * len(dataset))
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples.")

    # FFNN requires flattened data, so drop_last is safer to ensure consistent batch sizes
    drop_last = True if model_type == 'ffnn' else False
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last)

    return train_loader, val_loader, test_loader
