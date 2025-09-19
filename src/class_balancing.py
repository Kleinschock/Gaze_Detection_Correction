import torch
import numpy as np
from collections import Counter
from typing import List, Tuple

def undersample_data(sequences: List[np.ndarray], labels: List[int]) -> Tuple[List[np.ndarray], List[int]]:
    """
    Performs random undersampling to balance the dataset.

    This function identifies the minority class and reduces the number of samples
    in all other classes to match the number of samples in the minority class.
    The selection of samples to keep is done randomly.

    Args:
        sequences (List[np.ndarray]): The list of all data sequences.
        labels (List[int]): The list of all corresponding labels.

    Returns:
        Tuple[List[np.ndarray], List[int]]: A tuple containing the undersampled
                                            sequences and labels.
    """
    print("Performing undersampling...")
    class_counts = Counter(labels)
    minority_class_size = min(class_counts.values())
    print(f"Minority class size: {minority_class_size}")

    balanced_sequences = []
    balanced_labels = []
    
    # Group indices by class
    class_indices = {label: [] for label in class_counts.keys()}
    for i, label in enumerate(labels):
        class_indices[label].append(i)

    # Undersample each class to the size of the minority class
    for label, indices in class_indices.items():
        if len(indices) > minority_class_size:
            # Randomly choose indices to keep
            chosen_indices = np.random.choice(indices, minority_class_size, replace=False)
        else:
            chosen_indices = indices
        
        for i in chosen_indices:
            balanced_sequences.append(sequences[i])
            balanced_labels.append(labels[i])

    print(f"Undersampling complete. New dataset size: {len(balanced_labels)}")
    return balanced_sequences, balanced_labels

def calculate_inverse_frequency_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Calculates class weights based on the inverse frequency of samples.

    This method is a common starting point for addressing class imbalance.
    The weight for a class is inversely proportional to its frequency in the
    training data.

    Formula:
        Weight_i = Total_Samples / (Num_Classes * Samples_in_Class_i)

    Args:
        labels (List[int]): A list of all labels in the training dataset.
        num_classes (int): The total number of unique classes.

    Returns:
        torch.Tensor: A tensor of weights for each class, ordered by class index.
    """
    class_counts = Counter(labels)
    total_samples = len(labels)

    weights = []
    for i in range(num_classes):
        # Get the count for the class, default to 1 to avoid division by zero
        # if a class is not present in a particular batch of data (though unlikely for the whole dataset).
        count = class_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Calculated class weights using inverse frequency: {class_weights.numpy().round(2)}")
    return class_weights
