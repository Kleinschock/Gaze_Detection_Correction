import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from .lightning_datamodule import GestureDataModule
from .lightning_module import GestureLightningModule
from .class_balancing import calculate_inverse_frequency_weights
from .config import (
    DATA_ROOT,
    MODEL_DIR,
    DEFAULT_MODEL_TYPE,
    GESTURE_LABELS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    FEATURES_TO_USE,
    FEATURE_ENGINEERING_CONFIG,
    MAX_SEQ_LENGTH,
    WINDOW_STRIDE,
    LABELING_STRATEGY,
    PREPROCESSING_STRATEGY,
    MODEL_PARAMS
)


def get_input_size(model_type: str) -> int:
    """
    Calculates the model's input size based on the current feature configuration.
    This ensures the model architecture always matches the data pipeline.
    """
    # Start with the number of raw coordinate features (e.g., 23 keypoints * 3 coords = 69)
    num_coordinate_features = len([f for f in FEATURES_TO_USE if f.endswith(('_x', '_y', '_z'))])
    
    num_features = num_coordinate_features
    
    # Add features if they are enabled in the config
    if FEATURE_ENGINEERING_CONFIG.get('velocity'):
        num_features += num_coordinate_features
    if FEATURE_ENGINEERING_CONFIG.get('acceleration'):
        num_features += num_coordinate_features
        
    # For FFNN, the input is the flattened sequence
    if model_type == 'ffnn':
        return num_features * MAX_SEQ_LENGTH
    
    # For GRU, the input is the number of features per time step
    return num_features


def run_training(config: dict, finetune_checkpoint_path: str = None):
    """
    Main function to train or fine-tune a model using PyTorch Lightning.
    Accepts a config dictionary and an optional checkpoint path for fine-tuning.
    """
    print(f"Using device: {DEVICE}")
    
    # 1. --- Initialize DataModule ---
    data_module = GestureDataModule(
        data_root=DATA_ROOT,
        batch_size=config.get("batch_size", BATCH_SIZE),
        model_type=config.get("model_type", DEFAULT_MODEL_TYPE),
        balancing_strategy=config.get("balancing_strategy") if config.get("balancing_strategy") != 'weighted_loss' else None,
    )
    data_module.setup()

    # 2. --- Configure Class Balancing Strategy ---
    class_weights = None
    if config.get("balancing_strategy") == 'weighted_loss':
        print("Applying 'weighted_loss' strategy.")
        class_weights = calculate_inverse_frequency_weights(
            labels=data_module.train_labels,
            num_classes=len(GESTURE_LABELS)
        ).to(DEVICE)
    elif config.get("balancing_strategy") == 'undersample':
        print("Applying 'undersample' strategy. Loss will not be weighted.")
    else:
        print("No class balancing strategy applied.")

    # 3. --- Initialize or Load LightningModule ---
    if finetune_checkpoint_path:
        print(f"Loading model from checkpoint for fine-tuning: {finetune_checkpoint_path}")
        lightning_module = GestureLightningModule.load_from_checkpoint(
            finetune_checkpoint_path,
            learning_rate=config.get("learning_rate"),
            class_weights=class_weights  # Pass weights in case they are needed
        )
        model_type = lightning_module.hparams.model_name
        print(f"Fine-tuning {model_type.upper()} model.")
    else:
        model_type = config.get("model_type", DEFAULT_MODEL_TYPE)
        input_size = get_input_size(model_type=model_type)
        print(f"Creating new {model_type.upper()} model with input size: {input_size}")

        model_params = MODEL_PARAMS.get(model_type, {}).copy()
        if 'dropout_rate' in config:
            model_params['dropout_rate'] = config['dropout_rate']
        if 'hidden_sizes' in config:
            model_params['hidden_sizes'] = config['hidden_sizes']

        lightning_module = GestureLightningModule(
            model_name=model_type,
            input_size=input_size,
            model_params=model_params,
            learning_rate=config.get("learning_rate"),
            weight_decay=config.get("weight_decay", 1e-4),
            class_balancing_strategy=config.get("balancing_strategy"),
            class_weights=class_weights
        )

    # 4. --- Configure Callbacks ---
    prefix = "finetuned" if finetune_checkpoint_path else ""
    model_name = f"{prefix}_{model_type}_main_model" if prefix else f"{model_type}_main_model"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename=f"{model_name}-{{epoch:02d}}-{{val_acc:.4f}}",
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

    # 5. --- Initialize Logger and Trainer ---
    tags = ["finetune"] if finetune_checkpoint_path else ["train"]
    wandb_logger = WandbLogger(project="gesture-recognition", log_model="all", tags=tags)
    wandb_logger.watch(lightning_module.model)
    
    trainer = pl.Trainer(
        max_epochs=config.get("epochs", EPOCHS),
        accelerator='auto',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        log_every_n_steps=10
    )

    # 6. --- Start Training ---
    print(f"Starting {'fine-tuning' if finetune_checkpoint_path else 'training'} for {model_type.upper()} model...")
    trainer.fit(lightning_module, datamodule=data_module)

    print("\nTraining complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_score

def main():
    """
    Main function to train a gesture recognition model using PyTorch Lightning.
    This function handles command-line arguments for a single training run.
    """
    parser = argparse.ArgumentParser(description="Train a gesture recognition model with PyTorch Lightning.")
    parser.add_argument(
        '--model',
        choices=['ffnn', 'gru'],
        default=DEFAULT_MODEL_TYPE,
        help=f"Choose the model architecture. Defaults to '{DEFAULT_MODEL_TYPE}'."
    )
    parser.add_argument(
        '--balancing_strategy',
        type=str,
        default='weighted_loss',
        choices=['none', 'weighted_loss', 'undersample'],
        help="Choose the class balancing strategy: 'none', 'weighted_loss', or 'undersample'."
    )
    args = parser.parse_args()

    # Assemble configuration from defaults and command-line arguments
    # This config is passed to run_training and is also used by wandb sweeps.
    config = {
        "model_type": args.model,
        "balancing_strategy": args.balancing_strategy,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "weight_decay": 1e-4, # A sensible default
        # Pass model-specific params for this run
        "dropout_rate": MODEL_PARAMS.get(args.model, {}).get('dropout_rate'),
        "hidden_sizes": MODEL_PARAMS.get(args.model, {}).get('hidden_sizes'),
        # Log other important info
        "max_seq_length": MAX_SEQ_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "labeling_strategy": LABELING_STRATEGY,
        "preprocessing_strategy": PREPROCESSING_STRATEGY,
        "feature_engineering": FEATURE_ENGINEERING_CONFIG,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
    }
    
    run_training(config)

if __name__ == '__main__':
    main()
