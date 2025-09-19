import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Configuration ---
DATA_ROOT = 'Data/train'
GESTURE_LABELS = {
    "swipe_left": 0, "swipe_right": 1, "rotate": 2,
    "flip": 3, "rolling": 4, "idle": 5
}
LABEL_TO_GESTURE = {v: k for k, v in GESTURE_LABELS.items()}
OUTPUT_PLOT_PATH = 'results/gesture_length_distribution.png'
OUTLIER_REPORT_PATH = 'results/outlier_report.csv'

def analyze_and_report_outliers():
    """
    Scans gesture data, calculates statistics, generates a distribution plot,
    and creates a detailed report of outliers based on the IQR method.
    """
    # --- Pass 1: Collect all gesture segments and their info ---
    gesture_segments = defaultdict(list)
    print("Starting analysis: Pass 1 - Collecting all gesture segments...")

    for gesture_folder in os.listdir(DATA_ROOT):
        folder_path = os.path.join(DATA_ROOT, gesture_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith('_with_ground_truth.csv'):
                continue
            
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if 'ground_truth' not in df.columns:
                continue

            current_gesture_label = None
            current_gesture_length = 0
            
            # Add a sentinel row to ensure the last gesture is always processed
            sentinel = pd.DataFrame([{'ground_truth': -1}], index=[len(df)])
            df = pd.concat([df, sentinel])

            for index, row in df.iterrows():
                try:
                    label = int(row['ground_truth'])
                except (ValueError, TypeError):
                    label_str = str(row['ground_truth']).strip()
                    label = GESTURE_LABELS.get(label_str, -1)

                if label == current_gesture_label:
                    current_gesture_length += 1
                else:
                    if current_gesture_label is not None and current_gesture_label != GESTURE_LABELS['idle']:
                        gesture_name = LABEL_TO_GESTURE.get(current_gesture_label, "Unknown")
                        end_row = index - 1
                        start_row = end_row - current_gesture_length + 1
                        gesture_segments[gesture_name].append({
                            'filename': filename,
                            'start_row': start_row,
                            'end_row': end_row,
                            'length': current_gesture_length
                        })
                    
                    current_gesture_label = label
                    current_gesture_length = 1

    # --- Pass 2: Calculate stats, find outliers, and generate reports ---
    print("Analysis: Pass 2 - Calculating statistics and identifying outliers...")
    
    all_plot_data = []
    outlier_report_data = []
    summary_data = []

    for gesture, segments in gesture_segments.items():
        lengths = [s['length'] for s in segments]
        if not lengths:
            continue

        # Prepare data for plotting
        for seg in segments:
            all_plot_data.append({'Gesture': gesture, 'Length (frames)': seg['length']})

        # Calculate statistics for summary table
        series = pd.Series(lengths)
        summary_data.append({
            'Gesture': gesture, 'Count': series.count(), 'Mean': series.mean(),
            'Std Dev': series.std(), 'Min': series.min(), '25%': series.quantile(0.25),
            'Median (50%)': series.quantile(0.50), '75%': series.quantile(0.75), 'Max': series.max()
        })

        # Calculate outlier bounds using the IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify and record outliers
        for segment in segments:
            if segment['length'] < lower_bound or segment['length'] > upper_bound:
                outlier_report_data.append({
                    'Gesture': gesture,
                    'Filename': segment['filename'],
                    # Add 2 for human-readable row number (1 for header, 1 for 0-based index)
                    'Start Row': segment['start_row'] + 2,
                    'End Row': segment['end_row'] + 2,
                    'Length': segment['length'],
                    'Reason': 'Too Short' if segment['length'] < lower_bound else 'Too Long'
                })

    # --- Output Generation ---

    # Print statistical summary
    print("\n--- Gesture Length Statistics (in frames) ---")
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save and print outlier report
    if outlier_report_data:
        print("\n--- Outlier Report ---")
        outlier_df = pd.DataFrame(outlier_report_data)
        print(outlier_df.to_string(index=False))
        os.makedirs(os.path.dirname(OUTLIER_REPORT_PATH), exist_ok=True)
        outlier_df.to_csv(OUTLIER_REPORT_PATH, index=False)
        print(f"\nOutlier report saved to {OUTLIER_REPORT_PATH}")
    else:
        print("\nNo outliers found based on the 1.5 * IQR rule.")

    # Generate and save the plot
    print(f"\nGenerating and saving plot to {OUTPUT_PLOT_PATH}...")
    plot_df = pd.DataFrame(all_plot_data)
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='Gesture', y='Length (frames)', data=plot_df, hue='Gesture', inner='quartile', palette='muted', legend=False)
    sns.stripplot(x='Gesture', y='Length (frames)', data=plot_df, color='k', alpha=0.3, jitter=True)
    plt.title('Distribution of Gesture Lengths (with individual data points)')
    plt.ylabel('Length (Number of Frames)')
    plt.xlabel('Gesture Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH)
    
    print("Analysis complete.")

if __name__ == '__main__':
    analyze_and_report_outliers()
