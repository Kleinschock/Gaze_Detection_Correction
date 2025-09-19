import argparse
import os
import pytorch_lightning as pl
import wandb
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from .spotter_datamodule import SpotterDataModule
from .spotter import GestureSpotterLightningModule
from ..src.config import (
    DATA_ROOT,
    MODEL_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    INPUT_SIZE
)

def main():
    """
    Main function to train the gesture spotter model.
    """
    parser = argparse.ArgumentParser(description="Train a gesture spotter model.")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate.')
    args = parser.parse_args()

    # 1. --- Initialize DataModule ---
    data_module = SpotterDataModule(
        data_root=DATA_ROOT,
        batch_size=args.batch_size
    )

    # 2. --- Initialize LightningModule ---
    # Hyperparameters for the FFN spotter model
    hidden_size = 256 
    
    lightning_module = GestureSpotterLightningModule(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        learning_rate=args.lr
    )

    # 3. --- Configure Callbacks ---
    model_name = "spotter_model"
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
        patience=10,
        verbose=True,
        mode='max'
    )

    # 4. --- Initialize Logger and Trainer ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"spotter-training-{timestamp}"

    wandb_logger = WandbLogger(
        project="gesture-recognition",
        name=run_name,
        log_model=False,  # We will log the model manually as an artifact
        config=vars(args)
    )
    wandb_logger.experiment.config.update({"hidden_size": hidden_size})

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        log_every_n_steps=10
    )

    # 5. --- Start Training ---
    print("Starting training for Gesture Spotter model...")
    trainer.fit(lightning_module, datamodule=data_module)

    print("\nSpotter training complete.")
    print(f"Best spotter model saved at: {checkpoint_callback.best_model_path}")

    # 6. --- Log Best Model as a wandb Artifact ---
    # This is the crucial step to link the two pipeline stages.
    best_model_path = checkpoint_callback.best_model_path
    if os.path.exists(best_model_path):
        print(f"Logging best model to wandb as an artifact: {best_model_path}")
        artifact = wandb.Artifact(
            name='spotter-model',
            type='model',
            description='Trained gesture spotter model for detecting activity vs. resting frames.',
            metadata=trainer.logged_metrics
        )
        artifact.add_file(best_model_path)
        wandb_logger.experiment.log_artifact(artifact)
        print("Artifact logged successfully.")
    else:
        print(f"Warning: Could not find best model at path: {best_model_path}")

    # Finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
