import argparse
import wandb
import yaml
from .train import run_training
from .train_spotter import run_spotter_training
from .config import DEFAULT_MODEL_TYPE, EPOCHS

# 1. --- Define Sweep Configurations ---

# Configuration for the main GRU gesture classification model
sweep_config_gru = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.01},
        'batch_size': {'values': [16, 32, 64, 128]},
        'dropout_rate': {'min': 0.1, 'max': 0.6},
        'weight_decay': {'min': 1e-6, 'max': 1e-3},
        'hidden_sizes': {
            'values': [
                [128], [256], [512],
                [128, 64], [256, 128], [512, 256],
                [256, 256], [256, 128, 64]
            ]
        }
    }
}

# Configuration for the main FFNN gesture classification model
sweep_config_ffnn = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.01},
        'batch_size': {'values': [16, 32, 64, 128]},
        'dropout_rate': {'min': 0.1, 'max': 0.6},
        'weight_decay': {'min': 1e-6, 'max': 1e-3},
        'hidden_sizes': {
            'values': [
                # Smaller, GRU-comparable sizes
                [256, 128],
                [512, 256],
                [256, 128, 64],
                # Medium to Large sizes
                [1024],
                [1024, 512],
                [2048, 1024],
                [1024, 512, 256]
            ]
        }
    }
}

# Configuration for the Spotter (FFNN) model
sweep_config_spotter = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.01},
        'batch_size': {'values': [32, 64, 128, 256]},
        'dropout_rate': {'min': 0.1, 'max': 0.5},
        'weight_decay': {'min': 1e-6, 'max': 1e-4},
        'hidden_sizes': {
            'values': [
                [64], [128], [256],
                [64, 32], [128, 64], [256, 128]
            ]
        }
    }
}


# 2. --- Define Training Functions for Sweeps ---

def train_sweep_classifier(model_type: str):
    """
    Generic wrapper for classifier (GRU or FFNN) training logic for a wandb sweep.
    """
    def sweep_func():
        run = None
        try:
            run = wandb.init()
            config = wandb.config
            training_config = {
                "model_type": model_type,
                "balancing_strategy": 'weighted_loss',
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": EPOCHS,
                "weight_decay": config.weight_decay,
                "dropout_rate": config.dropout_rate,
                "hidden_sizes": config.hidden_sizes
            }
            run_training(config=training_config)
        except Exception as e:
            print(f"An error occurred during the {model_type} model sweep run: {e}")
        finally:
            if run:
                run.finish()
    return sweep_func

def train_sweep_spotter():
    """ Wrapper for the spotter model training logic for a wandb sweep. """
    run = None
    try:
        run = wandb.init()
        config = wandb.config
        training_config = {
            "balancing_strategy": 'undersample', # Good default for spotter
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": EPOCHS,
            "weight_decay": config.weight_decay,
            "dropout_rate": config.dropout_rate,
            "hidden_sizes": config.hidden_sizes
        }
        run_spotter_training(config=training_config)
    except Exception as e:
        print(f"An error occurred during the spotter sweep run: {e}")
    finally:
        if run:
            run.finish()


# 3. --- Main script to launch the sweep ---
def main():
    """
    Main function to initialize and run a hyperparameter sweep for a specified model.
    """
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep with Weights & Biases.")
    parser.add_argument(
        '--model_type',
        type=str,
        default='gru',
        choices=['gru', 'ffnn', 'spotter'],
        help="The type of model to run the sweep for: 'gru', 'ffnn', or 'spotter'."
    )
    parser.add_argument('--count', type=int, default=10, help="Number of runs to execute in the sweep.")
    args = parser.parse_args()

    if args.model_type == 'gru':
        project_name = "gesture-sweep-gru"
        sweep_config = sweep_config_gru
        sweep_function = train_sweep_classifier(model_type='gru')
        print("Initializing hyperparameter sweep for the GRU gesture model...")
    elif args.model_type == 'ffnn':
        project_name = "gesture-sweep-ffnn"
        sweep_config = sweep_config_ffnn
        sweep_function = train_sweep_classifier(model_type='ffnn')
        print("Initializing hyperparameter sweep for the FFNN gesture model...")
    else:  # spotter
        project_name = "gesture-sweep-spotter"
        sweep_config = sweep_config_spotter
        sweep_function = train_sweep_spotter
        print("Initializing hyperparameter sweep for the SPOTTER model...")

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Sweep initialized. ID: {sweep_id} | Project: {project_name}")
    print(f"Running {args.count} trials...")

    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_function, count=args.count)

    print(f"Hyperparameter sweep for '{args.model_type}' model complete.")

if __name__ == '__main__':
    main()
