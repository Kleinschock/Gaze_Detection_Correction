import os
import argparse
import glob
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from .data_loader import get_data_loaders
from .models import get_model
from .lightning_module import GestureLightningModule
from .config import (
    DATA_ROOT,
    MODEL_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    DEVICE,
    GESTURE_LABELS
)

def evaluate_model(model, loader, device):
    """Evaluates a model on a given data loader and returns true labels and predictions."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(y.numpy())
    return y_true, y_pred

def generate_report(y_true, y_pred, model_name):
    """Generates a classification report and a confusion matrix."""
    report = classification_report(
        y_true, y_pred, target_names=list(GESTURE_LABELS.keys()), labels=list(range(len(GESTURE_LABELS))), output_dict=True
    )
    # Separate accuracy from the main report for clarity
    accuracy = report.pop('accuracy')
    df_report = pd.DataFrame(report).transpose()
    
    print(f"\n--- Classification Report for {model_name.upper()} ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(df_report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=GESTURE_LABELS.keys(), yticklabels=GESTURE_LABELS.keys())
    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    return df_report, accuracy


def find_latest_model_path(model_type: str) -> str:
    """Finds the latest PyTorch Lightning checkpoint for a given model type."""
    search_pattern = os.path.join(MODEL_DIR, f"{model_type}_main_model-*.ckpt")

    # For the baseline, we assume a different naming convention might be used
    if model_type == 'ffnn':
        search_pattern = os.path.join(MODEL_DIR, f"{model_type}_baseline-*.ckpt")

    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None

    # Find the file with the most recent modification time
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
    
def main():
    """Main function to evaluate trained models."""
    parser = argparse.ArgumentParser(description="Evaluate gesture recognition models.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to a specific model checkpoint (.ckpt) to evaluate. If not provided, all models will be evaluated."
    )
    args = parser.parse_args()

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"Using device: {DEVICE}")
    
    # 1. Load Test Data
    # Assuming 'gru' data loading is suitable for all models.
    _, _, test_loader = get_data_loaders(DATA_ROOT, model_type='gru')

    if args.model_path:
        # --- Evaluate a single, specified model ---
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"Error: Model path not found: {model_path}")
            return

        model_name = os.path.basename(model_path).split('.ckpt')[0]
        print(f"\n--- Evaluating Single Model: {model_name.upper()} ---")
        print(f"Loading model from: {model_path}")

        try:
            lightning_module = GestureLightningModule.load_from_checkpoint(
                model_path,
                map_location=DEVICE
            )
            model = lightning_module.model.to(DEVICE)
        except Exception as e:
            print(f"Error loading checkpoint {model_path}: {e}.")
            return

        y_true, y_pred = evaluate_model(model, test_loader, DEVICE)
        generate_report(y_true, y_pred, model_name)

    else:
        # --- Original behavior: Evaluate and compare all models ---
        print("\n--- Evaluating and Comparing All Models ---")
        model_types = ['ffnn', 'gru']
        all_reports = {}
        all_accuracies = {}

        for model_type in model_types:
            print(f"\n--- Evaluating {model_type.upper()} Model ---")
            
            model_path = find_latest_model_path(model_type)
            
            if not model_path or not os.path.exists(model_path):
                fallback_path = os.path.join(MODEL_DIR, f"{model_type}_baseline.pth")
                if model_type == 'ffnn' and os.path.exists(fallback_path):
                     print(f"Warning: Could not find .ckpt for {model_type}. Using legacy {fallback_path}.")
                     try:
                        sample_x, _ = next(iter(test_loader))
                        input_size = sample_x.shape[-1]
                        num_classes = len(GESTURE_LABELS)
                        model = get_model(model_type, input_size, num_classes).to(DEVICE)
                        model.load_state_dict(torch.load(fallback_path, map_location=DEVICE, weights_only=True))
                     except Exception as e:
                        print(f"Failed to load legacy model: {e}. Skipping evaluation.")
                        continue
                else:
                    print(f"Warning: Model checkpoint not found for {model_type}. Skipping evaluation.")
                    continue
            else:
                print(f"Loading model from: {model_path}")
                try:
                    lightning_module = GestureLightningModule.load_from_checkpoint(
                        model_path,
                        map_location=DEVICE
                    )
                    model = lightning_module.model.to(DEVICE)
                except Exception as e:
                    print(f"Error loading checkpoint {model_path}: {e}. Skipping.")
                    continue

            y_true, y_pred = evaluate_model(model, test_loader, DEVICE)
            report_df, accuracy = generate_report(y_true, y_pred, model_type)
            all_reports[model_type] = report_df
            all_accuracies[model_type] = accuracy

        if all_reports:
            summary_data = {
                model: {
                    'accuracy': all_accuracies.get(model, 0.0),
                    'f1-score (macro avg)': data.loc['macro avg', 'f1-score']
                } for model, data in all_reports.items()
            }
            summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
            
            comparison_path = os.path.join(RESULTS_DIR, 'performance_comparison.csv')
            summary_df.to_csv(comparison_path)
            
            print("\n--- Performance Summary ---")
            print(summary_df)
            print(f"\nConsolidated performance comparison saved to {comparison_path}")
        else:
            print("\nNo models were evaluated. Please train the models first.")

if __name__ == '__main__':
    main()
