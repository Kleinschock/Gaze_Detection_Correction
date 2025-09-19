import os
import re
from collections import defaultdict

MODEL_DIR = "models"
TOP_N = 10

def get_model_files():
    """Returns a list of all .ckpt files in the models directory."""
    if not os.path.isdir(MODEL_DIR):
        print(f"Error: Directory '{MODEL_DIR}' not found.")
        return []
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".ckpt")]

def parse_filename(filename):
    """Parses a model filename to extract its type and validation accuracy."""
    # Handle finetuned models first
    if filename.startswith("finetuned_"):
        return "finetuned", 1.0  # Keep all finetuned, assign max accuracy

    # Regex to find validation accuracy
    # This regex specifically looks for a float (digits.digits) to avoid being too greedy.
    match = re.search(r"val_acc=([\d]+\.[\d]+)", filename)
    if not match:
        return None, None
    
    accuracy = float(match.group(1))
    
    if filename.startswith("gru_"):
        model_type = "gru"
    elif filename.startswith("ffnn_"):
        model_type = "ffnn"
    else:
        model_type = "unknown"
        
    return model_type, accuracy

def identify_files_to_delete():
    """Identifies which model files to keep and which to delete based on specified criteria."""
    all_files = get_model_files()
    if not all_files:
        return []

    # Group files by model type
    grouped_files = defaultdict(list)
    for filename in all_files:
        model_type, accuracy = parse_filename(filename)
        if model_type:
            grouped_files[model_type].append((accuracy, filename))

    # Identify files to keep
    files_to_keep = set()

    # 1. Keep all finetuned models
    for _, filename in grouped_files.get("finetuned", []):
        files_to_keep.add(filename)

    # 2. Keep top N of GRU and FFNN models
    for model_type in ["gru", "ffnn"]:
        # Sort by accuracy, descending
        sorted_models = sorted(grouped_files.get(model_type, []), key=lambda x: x[0], reverse=True)
        # Add the top N to the keep set
        for _, filename in sorted_models[:TOP_N]:
            files_to_keep.add(filename)

    # Identify files to delete
    all_files_set = set(all_files)
    files_to_delete = all_files_set - files_to_keep
    
    return sorted(list(files_to_delete))

if __name__ == "__main__":
    files_to_delete = identify_files_to_delete()
    
    if not files_to_delete:
        print("No model files to delete.")
    else:
        print("The following model files will be deleted:")
        for f in files_to_delete:
            # Print the full path for the deletion command
            print(os.path.join(MODEL_DIR, f))
