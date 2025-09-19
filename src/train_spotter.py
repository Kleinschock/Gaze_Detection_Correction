import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Import spotter-specific components
from .spotter_data_loader import SpotterDataset
from .lightning_module import SpotterLightningModule
from .scaler import StandardScaler

# Import shared components and configs
from .config import (
    DATA_ROOT,
    MODEL_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    FEATURES_TO_USE,
    FEATURE_ENGINEERING_CONFIG,
    PREPROCESSING_STRATEGY,
    MODEL_PARAMS,
    SCALER_PATH
)

def get_spotter_input_size() -> int:
    """
    Calculates the spotter model's input size based on the feature configuration.
    The spotter operates on individual frames, not sequences.
    """
    num_coordinate_features = len([f for f in FEATURES_TO_USE if f.endswith(('_x', '_y', '_z'))])
    num_features = num_coordinate_features
    
    if FEATURE_ENGINEERING_CONFIG.get('velocity'):
        num_features += num_coordinate_features
    if FEATURE_ENGINEERING_CONFIG.get('acceleration'):
        num_features += num_coordinate_features
        
    return num_features

def run_spotter_training(config: dict):
    """
    Runs the training process for the spotter model with a given configuration.

    Args:
        config (dict): A dictionary containing training parameters like
                       learning_rate, batch_size, epochs, etc.
    """
    # 1. --- Initialize Dataset and DataLoaders ---
    scaler = None
    if PREPROCESSING_STRATEGY == 'standardize':
        print("Standardization strategy is enabled. A scaler will be used.")
        # Placeholder for future scaler logic within a DataModule
        pass

    full_dataset = SpotterDataset(
        data_root=DATA_ROOT,
        scaler=scaler,
        balancing_strategy=config.get('balancing_strategy', 'none')
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset split: {train_size} training, {val_size} validation frames.")

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4, persistent_workers=True)

    # 2. --- Initialize LightningModule ---
    input_size = get_spotter_input_size()
    print(f"Creating Spotter model with input size: {input_size}")

    # Prepare model parameters from the config
    model_params = {
        'hidden_sizes': config['hidden_sizes'],
        'dropout': config['dropout_rate']
    }

    lightning_module = SpotterLightningModule(
        model_name='spotter',
        input_size=input_size,
        model_params=model_params,
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # 3. --- Configure Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="spotter_model-{epoch:02d}-{val_acc:.4f}",
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        verbose=True,
        mode='max'
    )

    # 4. --- Initialize Logger and Trainer ---
    # If this is part of a sweep, wandb.init() is handled by the sweep agent
    wandb_logger = WandbLogger(
        project="gesture-recognition-spotter",
        log_model="all",
        config=config  # Log the complete configuration for the run
    )

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='auto',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        log_every_n_steps=10
    )

    # 5. --- Start Training ---
    print("Starting spotter model training...")
    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\nTraining complete.")
    print(f"Best spotter model saved at: {checkpoint_callback.best_model_path}")


def main():
    """
    Main function to train the gesture spotter model with default parameters.
    """
    parser = argparse.ArgumentParser(description="Train a gesture spotter model.")
    parser.add_argument(
        '--balancing_strategy',
        type=str,
        default='none',
        choices=['none', 'undersample'],
        help="Choose the class balancing strategy: 'none' or 'undersample'."
    )
    args = parser.parse_args()

    # Get default spotter parameters from the main config file
    spotter_params = MODEL_PARAMS.get('spotter', {})

    # Prepare the configuration dictionary for a standalone run
    default_config = {
        "balancing_strategy": args.balancing_strategy,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "weight_decay": spotter_params.get('weight_decay', 1e-5),
        "dropout_rate": spotter_params.get('dropout_rate', 0.5),
        "hidden_sizes": spotter_params.get('hidden_sizes', [256, 128])
    }

    # Run the training process
    run_spotter_training(config=default_config)

if __name__ == '__main__':
    main()
