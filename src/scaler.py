import numpy as np
import json

class StandardScaler:
    """
    A stateful scaler to compute and apply Z-score normalization (standardization).
    It calculates the mean and standard deviation on a training dataset and can
    save/load these parameters to be used consistently during validation, testing,
    and live inference.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray):
        """
        Computes the mean and standard deviation for each feature across the dataset.
        Args:
            data (np.ndarray): The training data, expected shape (num_samples, num_features).
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array for fitting, but got shape {data.shape}")
        
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # Add a small epsilon to std to prevent division by zero for features with no variance
        self.std_[self.std_ == 0] = 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the stored mean and standard deviation to transform the data.
        Args:
            data (np.ndarray): The data to transform.
        Returns:
            np.ndarray: The standardized data.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() before transform().")
        
        return (data - self.mean_) / self.std_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        A convenience method to fit the scaler and then transform the data.
        """
        self.fit(data)
        return self.transform(data)

    def save(self, path: str):
        """
        Saves the scaler's mean and standard deviation to a JSON file.
        Args:
            path (str): The file path to save to.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot save an unfitted scaler.")
        
        with open(path, 'w') as f:
            json.dump({
                'mean': self.mean_.tolist(),
                'std': self.std_.tolist()
            }, f, indent=4)

    def load(self, path: str):
        """
        Loads the scaler's mean and standard deviation from a JSON file.
        Args:
            path (str): The file path to load from.
        """
        with open(path, 'r') as f:
            params = json.load(f)
            self.mean_ = np.array(params['mean'])
            self.std_ = np.array(params['std'])
        return self
