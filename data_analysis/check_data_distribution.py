import os

# Define the root directory for the training data
DATA_ROOT = 'Data/train'

def analyze_data_distribution(data_root):
    """
    Analyzes and prints the distribution of data files in the specified directory.

    Args:
        data_root (str): The path to the root directory of the training data.
    """
    print(f"Analyzing data distribution in: {data_root}\n")

    if not os.path.isdir(data_root):
        print(f"Error: Directory not found at '{data_root}'")
        return

    # Get the list of gesture labels (subdirectories)
    try:
        gesture_labels = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
    except FileNotFoundError:
        print(f"Error: Cannot access directory '{data_root}'")
        return

    if not gesture_labels:
        print("No gesture subdirectories found.")
        return

    print("Found gesture classes:")
    total_files = 0
    for label in sorted(gesture_labels):
        class_path = os.path.join(data_root, label)
        try:
            # Count only .csv files
            files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
            file_count = len(files)
            print(f"- {label}: {file_count} samples")
            total_files += file_count
        except FileNotFoundError:
            print(f"- {label}: Directory not found")

    print(f"\nTotal number of samples: {total_files}")
    print("\nThis count reflects the number of .csv files in each gesture's subdirectory.")

if __name__ == "__main__":
    analyze_data_distribution(DATA_ROOT)
