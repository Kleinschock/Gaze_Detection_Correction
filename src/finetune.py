import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from .train import run_training
from .config import (
    FINETUNING_CONFIG,
    BATCH_SIZE,
    EPOCHS,
    MODEL_PARAMS,
    MAX_SEQ_LENGTH,
    WINDOW_STRIDE,
    LABELING_STRATEGY,
    PREPROCESSING_STRATEGY,
    FEATURE_ENGINEERING_CONFIG,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA
)

def main():
    """
    Main function to fine-tune a pre-trained gesture recognition model.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a gesture recognition model.")
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help="Path to the model checkpoint to be fine-tuned."
    )
    parser.add_argument(
        '--balancing_strategy',
        type=str,
        default='weighted_loss',
        choices=['none', 'weighted_loss', 'undersample'],
        help="Choose the class balancing strategy for the fine-tuning data."
    )
    args = parser.parse_args()

    # --- Enable the fine-tuning data strategy ---
    # This is a global flag used by the data loader.
    FINETUNING_CONFIG['ENABLED'] = True

    # --- Assemble the configuration for the fine-tuning run ---
    # We use the dedicated fine-tuning learning rate and other parameters.
    config = {
        "balancing_strategy": args.balancing_strategy,
        "learning_rate": FINETUNING_CONFIG['FINETUNE_LEARNING_RATE'],
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "weight_decay": 1e-5, # Use a small weight decay for fine-tuning
        # Log other important info for reproducibility
        "finetuning_config": FINETUNING_CONFIG,
        "max_seq_length": MAX_SEQ_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "labeling_strategy": LABELING_STRATEGY,
        "preprocessing_strategy": PREPROCESSING_STRATEGY,
        "feature_engineering": FEATURE_ENGINEERING_CONFIG,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
    }

    print("--- Starting Fine-Tuning Session ---")
    print(f"Checkpoint to fine-tune: {args.checkpoint_path}")
    print(f"Data sampling fraction (gestures): {FINETUNING_CONFIG['GESTURE_DATA_SAMPLE_FRACTION']}")
    print(f"Learning Rate: {config['learning_rate']}")
    
    # Call the main training function, passing the checkpoint path
    run_training(config, finetune_checkpoint_path=args.checkpoint_path)

    print("\n--- Fine-Tuning Session Complete ---")

if __name__ == '__main__':
    main()
