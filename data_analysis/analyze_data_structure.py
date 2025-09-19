import os
import pandas as pd
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_ROOT, GESTURE_LABELS

def analyze_data():
    """
    Analyzes the structure of the training data.
    - Counts the number of recordings for each gesture.
    - Reads a sample file from each gesture to report on its structure.
    """
    print("--- Starting Data Structure Analysis ---")
    
    # We use the keys from GESTURE_LABELS to ensure we only analyze configured gestures.
    # This also implicitly includes the 'idle' gesture if it's part of the label mapping.
    gesture_folders = [label for label in GESTURE_LABELS.keys() if label != 'no_gesture']
    
    # Explicitly add the 'idle' folder if it exists, as it's handled separately
    if os.path.exists(os.path.join(DATA_ROOT, 'idle')):
        if 'idle' not in gesture_folders:
            gesture_folders.append('idle')

    for gesture in gesture_folders:
        gesture_path = os.path.join(DATA_ROOT, gesture)
        if not os.path.isdir(gesture_path):
            print(f"\nWarning: Directory not found for gesture '{gesture}'. Skipping.")
            continue

        try:
            # Get all CSV files in the directory
            recordings = [f for f in os.listdir(gesture_path) if f.endswith('.csv')]
            num_recordings = len(recordings)

            print(f"\n--- Gesture: {gesture.upper()} ---")
            print(f"Found {num_recordings} recordings.")

            if num_recordings > 0:
                # Analyze the first recording as a sample
                sample_file_path = os.path.join(gesture_path, recordings[0])
                try:
                    df = pd.read_csv(sample_file_path)
                    num_frames = len(df)
                    num_columns = len(df.columns)
                    
                    # The last column is the 'is_idle' flag
                    num_features = num_columns - 1 
                    
                    print(f"Sample file '{recordings[0]}' analysis:")
                    print(f"  - Number of frames (rows): {num_frames}")
                    print(f"  - Total columns: {num_columns}")
                    print(f"  - Number of raw features (x, y, z per keypoint): {num_features}")
                    print(f"  - Column names: {df.columns.tolist()}")

                except Exception as e:
                    print(f"Error reading or analyzing sample file {recordings[0]}: {e}")

        except Exception as e:
            print(f"Error processing directory for gesture '{gesture}': {e}")

    print("\n--- Data Structure Analysis Finished ---")

if __name__ == '__main__':
    analyze_data()
