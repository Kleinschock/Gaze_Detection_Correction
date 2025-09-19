import os
import pandas as pd
from collections import Counter

# Define the root directory for the training data
DATA_ROOT = 'Data/train'

def count_gestures_from_csv(data_root):
    """
    Counts the occurrences of each gesture from the 'ground_truth' column in CSV files.

    Args:
        data_root (str): The path to the root directory of the training data.
    """
    print(f"Analyzing gesture counts from CSV files in: {data_root}\n")

    if not os.path.isdir(data_root):
        print(f"Error: Directory not found at '{data_root}'")
        return

    gesture_counts = Counter()
    total_rows = 0

    for subdir, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'ground_truth' in df.columns:
                        # Get value counts from the 'ground_truth' column
                        counts = df['ground_truth'].value_counts()
                        gesture_counts.update(counts.to_dict())
                        total_rows += len(df)
                    else:
                        print(f"Warning: 'ground_truth' column not found in {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not gesture_counts:
        print("No gestures found or 'ground_truth' column missing in all CSV files.")
        return

    print("Gesture counts from 'ground_truth' column:")
    for gesture, count in sorted(gesture_counts.items()):
        print(f"- {gesture}: {count} occurrences")

    print(f"\nTotal number of rows processed: {total_rows}")
    print(f"Total number of unique gestures: {len(gesture_counts)}")

if __name__ == "__main__":
    count_gestures_from_csv(DATA_ROOT)
