import argparse
import sys
import os

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Main entry point for the Gesture Control Project.
    
    This script orchestrates the different components of the project:
    - 'train': To train a model (FFNN or GRU).
    - 'evaluate': To compare the performance of the trained models.
    - 'run': To launch the live gesture recognition application.
    """
    parser = argparse.ArgumentParser(
        description="Gesture-based Presentation Controller using PyTorch.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Train Sub-command ---
    parser_train = subparsers.add_parser('train', help='Train a gesture recognition model.')
    parser_train.add_argument(
        '--model', 
        choices=['ffnn', 'gru'], 
        default=None,  # Default will be handled by the train script's config
        help='Specify the model to train (ffnn or gru). Defaults to the one in config.'
    )

    # --- Evaluate Sub-command ---
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate and compare trained models.')

    # --- Tune Sub-command ---
    parser_tune = subparsers.add_parser('tune', help='Run hyperparameter tuning.')
    
    # --- Run Live Sub-command ---
    parser_run = subparsers.add_parser('run', help='Run the live gesture recognition application.')
    parser_run.add_argument(
        '--model', 
        choices=['ffnn', 'gru'], 
        default='gru', 
        help='Specify the model to use for live recognition (default: gru).'
    )

    args = parser.parse_args()

    if args.command == 'train':
        # We need to temporarily modify sys.argv for the train script's argparse
        from src import train as train_script
        original_argv = sys.argv
        sys.argv = [original_argv[0]] 
        if args.model:
            sys.argv.extend(['--model', args.model])
        print(f"Executing training script for model: {args.model or 'default'}...")
        train_script.main()
        sys.argv = original_argv # Restore

    elif args.command == 'evaluate':
        from src import evaluate as evaluate_script
        print("Executing evaluation script...")
        evaluate_script.main()

    elif args.command == 'tune':
        from src import tuning as tuning_script
        print("Executing hyperparameter tuning script...")
        tuning_script.run_tuning()

    elif args.command == 'run':
        from src.run_live import LiveGestureController
        print(f"Starting live gesture controller with {args.model.upper()} model...")
        controller = LiveGestureController(model_type=args.model)
        controller.run()

if __name__ == '__main__':
    # To make this runnable without arguments, we can set a default command.
    # If no command is provided, default to 'run'.
    if len(sys.argv) == 1:
        print("No command specified. Defaulting to 'run'.")
        print("Usage: python main.py {train|evaluate|tune|run} [--model MODEL_TYPE]")
        # To run without args, we inject the 'run' command.
        sys.argv.append('run')

    main()
