import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import GESTURE_LABELS, FEATURES_TO_USE, USE_DEDICATED_IDLE_DATA
from .preprocessing import preprocess_sequence
from .class_balancing import undersample_data

# --- Dynamic Keypoint Selection from Config ---
KEYPOINT_COORDINATE_COLUMNS = [f for f in FEATURES_TO_USE if f.endswith(('_x', '_y', '_z'))]

# This window size must be at least 3 to calculate acceleration
SPOTTER_WINDOW_SIZE = 3

class SpotterDataset(Dataset):
    """
    PyTorch Dataset for the gesture spotter model.
    This dataset provides individual frames labeled as either "motion" (1) or "no motion" (0).
    It uses a small sliding window to enable dynamic feature calculation (velocity, etc.).
    """
    def __init__(self, data_root: str, scaler=None, include_chaotic_idle: bool = True, balancing_strategy: str = None):
        self.data_root = data_root
        self.scaler = scaler
        self.sequences = []
        self.labels = []
        self._load_data(include_chaotic_idle)

        if balancing_strategy == 'undersample' and len(self.labels) > 0:
            print("Applying undersampling to the spotter dataset...")
            self.sequences, self.labels = undersample_data(self.sequences, self.labels)
            print(f"Dataset size after undersampling: {len(self.sequences)} frames.")

    def _load_data(self, include_chaotic_idle: bool):
        """
        Scans data directories, creates small overlapping windows of frames,
        and assigns a binary label corresponding to the last frame of each window.
        """
        print("Scanning data files for spotter model...")

        if USE_DEDICATED_IDLE_DATA:
            print("Using dedicated idle data strategy.")
            # 1. Process all gesture files, but filter out their internal 'idle' sections.
            for gesture_name in GESTURE_LABELS.keys():
                if gesture_name == 'idle':
                    continue  # Skip the main idle folder for now
                
                gesture_dir = os.path.join(self.data_root, gesture_name)
                if not os.path.isdir(gesture_dir):
                    continue

                for fname in sorted(os.listdir(gesture_dir)):
                    if not fname.endswith('.csv'):
                        continue
                    csv_path = os.path.join(gesture_dir, fname)
                    self._process_file(csv_path, filter_internal_idle=True)
            
            # 2. Process the dedicated 'idle' folder.
            idle_dir = os.path.join(self.data_root, 'idle')
            if os.path.isdir(idle_dir):
                for fname in sorted(os.listdir(idle_dir)):
                    if not fname.endswith('.csv'):
                        continue
                    csv_path = os.path.join(idle_dir, fname)
                    self._process_file(csv_path, force_idle=True)

        else:
            print("Using standard data strategy (all data included).")
            # Original logic: process all folders including their idle parts.
            for gesture_name in GESTURE_LABELS.keys():
                gesture_dir = os.path.join(self.data_root, gesture_name)
                if not os.path.isdir(gesture_dir):
                    continue

                for fname in sorted(os.listdir(gesture_dir)):
                    if not fname.endswith('.csv'):
                        continue
                    csv_path = os.path.join(gesture_dir, fname)
                    self._process_file(csv_path)

        if include_chaotic_idle:
            # This file contains more challenging "idle" examples
            chaotic_idle_path = os.path.join('archive', 'idle', 'idle_confusion.csv')
            if os.path.exists(chaotic_idle_path):
                print(f"Processing chaotic idle file: {chaotic_idle_path}")
                self._process_file(chaotic_idle_path, force_idle=True)
            else:
                print(f"WARNING: Chaotic idle file not found at {chaotic_idle_path}")

        print(f"Found {len(self.sequences)} total frames for spotter training.")
        if len(self.labels) > 0:
            motion_perc = (sum(self.labels) / len(self.labels)) * 100
            print(f"Data distribution: {motion_perc:.2f}% motion, {100-motion_perc:.2f}% no motion.")

    def _process_file(self, csv_path: str, force_idle: bool = False, filter_internal_idle: bool = False):
        """Helper function to read a CSV and extract feature windows and labels."""
        try:
            df = pd.read_csv(csv_path)

            if filter_internal_idle and 'is_gesture' in df.columns:
                # For gesture files, remove the rows marked as 'idle' before any processing
                df = df[df['is_gesture'] != 'idle'].reset_index(drop=True)
                if df.empty:
                    return # Skip file if it becomes empty after filtering
            
            if 'ground_truth' in df.columns and 'is_gesture' not in df.columns:
                df.rename(columns={'ground_truth': 'is_gesture'}, inplace=True)
            
            if 'is_gesture' not in df.columns and not force_idle:
                return
            
            # Add missing columns with 0.0 to prevent KeyErrors
            for col in KEYPOINT_COORDINATE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0

            if force_idle:
                is_gesture_col = pd.Series([False] * len(df))
            else:
                is_gesture_col = df['is_gesture'].astype(str).str.strip().str.lower() != 'idle'

            keypoint_data = df[KEYPOINT_COORDINATE_COLUMNS].values.astype(np.float32)
            
            # Create small overlapping windows to calculate dynamic features
            for i in range(len(keypoint_data) - SPOTTER_WINDOW_SIZE + 1):
                sequence = keypoint_data[i : i + SPOTTER_WINDOW_SIZE]
                # The label corresponds to the LAST frame in the window
                label = int(is_gesture_col[i + SPOTTER_WINDOW_SIZE - 1])
                
                self.sequences.append(sequence)
                self.labels.append(label)

        except Exception as e:
            print(f"ERROR processing file {os.path.basename(csv_path)}: {e}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        raw_sequence = self.sequences[idx]
        label = self.labels[idx]

        # Preprocess the short sequence to get dynamic features
        processed_sequence = preprocess_sequence(raw_sequence, scaler=self.scaler)

        # We only need the features for the last frame of the sequence
        final_features = processed_sequence[-1, :]

        return torch.from_numpy(final_features), torch.tensor(label, dtype=torch.float32)
