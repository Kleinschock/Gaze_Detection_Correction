import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import itertools
import json
from collections import Counter
import copy
import argparse
import optuna

from .data_loader import GestureDataset
from .models import get_model
from .config import (
    DATA_ROOT,
    DEVICE,
    GESTURE_LABELS,
    DEFAULT_MODEL_TYPE
)

class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.inf
        self.delta = delta
        self.best_model_wts = None

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter == 1:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping):
    """Helper function to train and validate a model with early stopping."""
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        
        epoch_acc = correct / total if total > 0 else 0
        early_stopping(epoch_acc, model)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_wts)
            break
            
    return early_stopping.best_score

def run_grid_search():
    """
    Main function to run the hyperparameter tuning process using Grid Search.
    """
    print("Starting hyperparameter tuning with Grid Search...")
    print(f"Using device: {DEVICE}")

    # --- Define Hyperparameter Search Space (Refined for Efficiency) ---
    param_grid = {
        'learning_rate': [0.001, 0.0005],
        'batch_size': [16, 32],
        'hidden_size': [128, 256],
        'num_layers': [2],
        'dropout': [0.4, 0.5],
        'weight_decay': [1e-4, 1e-5],
        'optimizer': ['adamw'] # AdamW is generally preferred
    }

    # --- Load Data ---
    dataset = GestureDataset(DATA_ROOT)
    num_classes = len(GESTURE_LABELS)
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    results = []

    # --- Generate all combinations of hyperparameters ---
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Testing {len(param_combinations)} parameter combinations with {k_folds}-fold cross-validation.")

    for i, params in enumerate(param_combinations):
        print(f"\n--- Combination {i+1}/{len(param_combinations)} ---")
        print(f"Parameters: {params}")

        fold_accuracies = []
        
        # --- K-Fold Cross-Validation ---
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            # Create data loaders for the current fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=val_subsampler)

            # --- Class Weighting for Imbalanced Datasets (within the fold) ---
            train_labels = [dataset.labels[i] for i in train_ids]
            class_counts = Counter(train_labels)
            total_samples = len(train_labels)
            
            weights = []
            for i in range(num_classes):
                count = class_counts.get(i, 1)
                weight = total_samples / (num_classes * count)
                weights.append(weight)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
            # -----------------------------------------

            # Initialize model with current hyperparameters
            model_params = {
                'hidden_size': params['hidden_size'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout']
            }
            model = get_model(DEFAULT_MODEL_TYPE, num_classes, model_params).to(DEVICE)
            
            # Select optimizer and apply L2 regularization
            if params['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            else:
                optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # Initialize Early Stopping for the current fold
            early_stopping = EarlyStopping(patience=7, verbose=False)

            # Train and validate
            val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, DEVICE, epochs=30, early_stopping=early_stopping)
            fold_accuracies.append(val_acc)
            print(f"  Fold {fold+1}/{k_folds}, Val Accuracy: {val_acc:.4f}")

        avg_accuracy = np.mean(fold_accuracies)
        results.append({'params': params, 'accuracy': avg_accuracy})
        print(f"  Average Validation Accuracy: {avg_accuracy:.4f}")

    # --- Find and Print the Best Hyperparameters ---
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print("\n--- Hyperparameter Tuning Complete ---")
    print("Best Hyperparameters Found:")
    print(json.dumps(best_result['params'], indent=4))
    print(f"Best Average Validation Accuracy: {best_result['accuracy']:.4f}")

def objective(trial, dataset, num_classes, kfold):
    """Objective function for Optuna to optimize."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.2, 0.6),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    }

    fold_accuracies = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=val_subsampler)

        train_labels = [dataset.labels[i] for i in train_ids]
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

        model_params = {
            'hidden_size': params['hidden_size'],
            'num_layers': params['num_layers'],
            'dropout': params['dropout']
        }
        model = get_model(DEFAULT_MODEL_TYPE, num_classes, model_params).to(DEVICE)
        
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        early_stopping = EarlyStopping(patience=7, verbose=False)

        val_acc = train_and_validate(model, train_loader, val_loader, criterion, optimizer, DEVICE, epochs=30, early_stopping=early_stopping)
        fold_accuracies.append(val_acc)

    avg_accuracy = np.mean(fold_accuracies)
    return avg_accuracy

def run_bayesian_search(n_trials=50):
    """
    Main function to run the hyperparameter tuning process using Bayesian Optimization with Optuna.
    """
    print("Starting hyperparameter tuning with Bayesian Search (Optuna)...")
    print(f"Using device: {DEVICE}")

    dataset = GestureDataset(DATA_ROOT)
    num_classes = len(GESTURE_LABELS)
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dataset, num_classes, kfold), n_trials=n_trials)

    print("\n--- Hyperparameter Tuning Complete ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Gesture Recognition Model")
    parser.add_argument('--method', type=str, default='bayesian', choices=['grid', 'bayesian'],
                        help='Tuning method to use: "grid" for Grid Search or "bayesian" for Bayesian Optimization.')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for Bayesian Optimization.')
    args = parser.parse_args()

    if args.method == 'grid':
        run_grid_search()
    else:
        run_bayesian_search(n_trials=args.n_trials)
