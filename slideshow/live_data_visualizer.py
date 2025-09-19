import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import filedialog, messagebox


def load_session_data(session_path):
    """Load all data files from a session directory."""
    data = {}

    # Load unprocessed data if it exists
    unprocessed_path = os.path.join(session_path, "unprocessed_data.npy")
    if os.path.exists(unprocessed_path):
        data["unprocessed"] = np.load(unprocessed_path)
        print(f"Loaded unprocessed data: {data['unprocessed'].shape}")
    else:
        print("No unprocessed data found.")

    # Load preprocessed data if it exists
    preprocessed_path = os.path.join(session_path, "preprocessed_data.npy")
    if os.path.exists(preprocessed_path):
        data["preprocessed"] = np.load(preprocessed_path)
        print(f"Loaded preprocessed data: {data['preprocessed'].shape}")
    else:
        print("No preprocessed data found.")

    # Load predictions if they exist
    predictions_path = os.path.join(session_path, "predictions.json")
    if os.path.exists(predictions_path):
        with open(predictions_path, 'r') as f:
            data["predictions"] = json.load(f)
        print(f"Loaded {len(data['predictions'])} prediction records.")
    else:
        print("No prediction data found.")

    return data


def convert_predictions_to_dataframe(predictions):
    """Convert the prediction JSON data to a pandas DataFrame for easier analysis."""
    # Extract basic information
    df_rows = []
    gesture_classes = set()

    # First pass to gather all gesture classes
    for pred in predictions:
        for gesture in pred["all_probabilities"].keys():
            gesture_classes.add(gesture)

    # Second pass to create DataFrame rows
    for pred in predictions:
        row = {
            "timestamp": pred["timestamp"],
            "predicted_gesture": pred["predicted_gesture"],
            "confidence": pred["confidence"],
        }

        # Add all probability values
        for gesture in gesture_classes:
            row[f"prob_{gesture}"] = pred["all_probabilities"].get(gesture, 0)

        df_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(df_rows)

    # Convert timestamp to datetime and add frame number
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit='s')
        df["frame"] = range(len(df))

    return df


def plot_prediction_timeline(df, output_dir=None):
    """
    Plot a timeline of gesture predictions with confidence.
    """
    plt.figure(figsize=(14, 8))

    # Get unique gesture classes for coloring
    gesture_classes = df["predicted_gesture"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(gesture_classes)))
    color_map = dict(zip(gesture_classes, colors))

    # Plot scatter points colored by gesture
    for gesture in gesture_classes:
        gesture_df = df[df["predicted_gesture"] == gesture]
        plt.scatter(gesture_df["frame"], gesture_df["confidence"],
                    c=[color_map[g] for g in gesture_df["predicted_gesture"]],
                    label=gesture, alpha=0.7, s=50)

    plt.xlabel("Frame Number")
    plt.ylabel("Confidence")
    plt.title("Gesture Prediction Timeline")
    plt.legend(title="Gesture Type")
    plt.grid(True, alpha=0.3)

    # Add horizontal line at common confidence threshold
    plt.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label="Typical Threshold (0.6)")

    if output_dir:
        plt.savefig(os.path.join(output_dir, "prediction_timeline.png"), dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_gesture_distribution(df, output_dir=None):
    """
    Plot the distribution of detected gestures.
    """
    plt.figure(figsize=(10, 6))

    # Count occurrences of each gesture
    gesture_counts = df["predicted_gesture"].value_counts()

    # Plot as bar chart
    gesture_counts.plot(kind='bar', color=plt.cm.tab10(np.linspace(0, 1, len(gesture_counts))))
    plt.xlabel("Gesture Type")
    plt.ylabel("Count")
    plt.title("Distribution of Detected Gestures")
    plt.grid(True, axis='y', alpha=0.3)

    # Add count labels on top of bars
    for i, count in enumerate(gesture_counts):
        plt.text(i, count + 0.5, str(count), ha='center')

    if output_dir:
        plt.savefig(os.path.join(output_dir, "gesture_distribution.png"), dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_probability_heatmap(df, output_dir=None):
    """
    Create a heatmap showing probability values over time for each gesture.
    """
    # Extract probability columns
    prob_cols = [col for col in df.columns if col.startswith("prob_")]

    # Create a matrix of probability values
    prob_matrix = df[prob_cols].values

    # Get gesture names (without the "prob_" prefix)
    gesture_names = [col[5:] for col in prob_cols]

    plt.figure(figsize=(14, 8))
    plt.imshow(prob_matrix.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')

    plt.yticks(np.arange(len(gesture_names)), gesture_names)
    plt.xlabel("Frame Number")
    plt.ylabel("Gesture Class")
    plt.title("Probability Heatmap Over Time")

    if output_dir:
        plt.savefig(os.path.join(output_dir, "probability_heatmap.png"), dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_confidence_histogram(df, output_dir=None):
    """
    Plot histogram of confidence scores.
    """
    plt.figure(figsize=(10, 6))

    plt.hist(df["confidence"], bins=20, alpha=0.7, color='steelblue')
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Confidence Scores")
    plt.grid(True, alpha=0.3)

    # Add vertical line at common threshold
    plt.axvline(x=0.6, color='r', linestyle='--', alpha=0.7, label="Typical Threshold (0.6)")
    plt.legend()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "confidence_histogram.png"), dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_feature_visualization(unprocessed_data, preprocessed_data, frame_idx=0, output_dir=None):
    """
    Visualize the difference between unprocessed and preprocessed features.
    """
    try:
        if frame_idx >= len(unprocessed_data):
            print(f"Warning: Frame index {frame_idx} exceeds data length. Using first frame.")
            frame_idx = 0

        # Get a single window of unprocessed data
        unprocessed_window = unprocessed_data[frame_idx]

        # Check dimensions and reshape if needed
        if len(unprocessed_window.shape) > 2:
            print(f"Warning: Unexpected unprocessed data shape: {unprocessed_window.shape}. Attempting to reshape.")
            if unprocessed_window.shape[0] == 1:
                unprocessed_window = unprocessed_window[0]

        window_size, feature_count = unprocessed_window.shape

        # Prepare the figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot unprocessed features
        im1 = axes[0].imshow(unprocessed_window.T, aspect='auto', cmap='coolwarm')
        axes[0].set_title("Unprocessed Features (Window)")
        axes[0].set_xlabel("Window Position")
        axes[0].set_ylabel("Feature Index")
        plt.colorbar(im1, ax=axes[0], label="Feature Value")

        # Plot preprocessed data if available
        if preprocessed_data is not None:
            try:
                preprocessed_features = preprocessed_data[frame_idx]

                # Check dimensions and reshape if needed
                if len(preprocessed_features.shape) > 1:
                    print(f"Reshaping preprocessed data from {preprocessed_features.shape}")
                    preprocessed_features = preprocessed_features.flatten()

                im2 = axes[1].plot(preprocessed_features, 'g-', linewidth=1.5)
                axes[1].set_title(f"Preprocessed Features (After PCA) - {len(preprocessed_features)} components")
                axes[1].set_xlabel("PCA Component")
                axes[1].set_ylabel("Value")
                axes[1].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Error plotting preprocessed data: {e}")
                axes[1].set_title("Error plotting preprocessed data")
                axes[1].text(0.5, 0.5, f"Error: {str(e)}",
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[1].transAxes)

        if output_dir:
            try:
                output_path = os.path.join(output_dir, f"feature_visualization_frame{frame_idx}.png")
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Saved feature visualization to {output_path}")
            except Exception as e:
                print(f"Error saving feature visualization: {e}")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in feature visualization: {e}")
        import traceback
        traceback.print_exc()


def create_prediction_animation(df, output_path=None):
    """Create an animation showing how prediction probabilities change over time."""
    try:
        # Get probability columns
        prob_cols = [col for col in df.columns if col.startswith("prob_")]
        gesture_names = [col[5:] for col in prob_cols]

        # Prepare the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set the limits
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.5, len(gesture_names) - 0.5)

        # Prepare the bars
        bars = ax.barh(range(len(gesture_names)), [0] * len(gesture_names),
                       color=plt.cm.tab10(np.linspace(0, 1, len(gesture_names))))

        # Add gesture names as y-tick labels
        ax.set_yticks(range(len(gesture_names)))
        ax.set_yticklabels(gesture_names)

        # Add grid and labels
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlabel("Probability")
        ax.set_title("Gesture Recognition Probabilities")
    except Exception as e:
        print(f"Error creating animation: {e}")
        print("Skipping animation and continuing with other visualizations...")



    # Add frame counter text
    frame_text = ax.text(0.98, 0.02, "Frame: 0", transform=ax.transAxes,
                         ha='right', va='bottom', fontsize=12)

    # Add threshold line
    threshold_line = ax.axvline(x=0.6, color='r', linestyle='--', alpha=0.7, label="Threshold (0.6)")
    ax.legend()

    def update(frame):
        if frame < len(df):
            # Update the bars
            for i, col in enumerate(prob_cols):
                bars[i].set_width(df.iloc[frame][col])

            # Highlight the predicted gesture
            predicted = df.iloc[frame]["predicted_gesture"]
            for i, gesture in enumerate(gesture_names):
                if gesture == predicted:
                    bars[i].set_facecolor('gold')
                    bars[i].set_edgecolor('black')
                else:
                    bars[i].set_facecolor(plt.cm.tab10(i / len(gesture_names)))
                    bars[i].set_edgecolor(None)

            # Update frame counter
            frame_text.set_text(f"Frame: {frame} | Prediction: {predicted} ({df.iloc[frame]['confidence']:.2f})")

        # Convert bars to list if it's a tuple
        bar_list = list(bars) if isinstance(bars, tuple) else bars
        return bar_list + [frame_text, threshold_line]

        # Create the animation with blit=False to avoid certain backend issues
        anim = FuncAnimation(fig, update, frames=min(len(df), 500), interval=100, blit=False)

        if output_path:
            try:
                # Limit to max 200 frames for GIF to keep file size reasonable
                frames_to_save = min(len(df), 200)
                print(f"Saving animation with {frames_to_save} frames to {output_path}...")

                # Use pillow for GIF export with reduced frames
                reduced_anim = FuncAnimation(fig, update, frames=frames_to_save, interval=100, blit=False)
                reduced_anim.save(output_path, writer='pillow', fps=10)
                print(f"Animation saved to {output_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Continuing with other visualizations...")

        plt.tight_layout()
        plt.show()

def find_sessions():
    """Find all available session directories."""
    base_dir = Path("collected_data")
    if not base_dir.exists():
        print(f"Error: Data directory '{base_dir}' not found.")
        return []

    sessions = list(base_dir.glob("session_*"))
    # Sort by creation time (newest first)
    sessions.sort(key=os.path.getctime, reverse=True)
    return sessions


def select_session_gui():
    """Use a simple GUI dialog to select a session directory."""
    # Hide the main Tkinter window
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select a directory
    session_path = filedialog.askdirectory(
        title="Select Session Directory",
        initialdir="collected_data"
    )

    # Clean up
    root.destroy()

    if not session_path:
        return None

    return Path(session_path)


def select_output_directory_gui():
    """Use a GUI dialog to select an output directory for visualizations."""
    try:
        # Hide the main Tkinter window
        root = tk.Tk()
        root.withdraw()

        # Ask the user to select a directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Visualizations (Cancel to skip saving)",
            initialdir="."
        )

        # Clean up
        root.destroy()

        if not output_dir:
            return None

        # Create the directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        return output_path
    except Exception as e:
        print(f"Error selecting output directory: {e}")
        # Fallback: create a directory in the current location
        fallback_dir = Path("visualization_output")
        fallback_dir.mkdir(exist_ok=True)
        print(f"Using fallback directory: {fallback_dir}")
        return fallback_dir


def select_session_console():
    """Select a session using console input."""
    sessions = find_sessions()
    if not sessions:
        print("No sessions found.")
        return None

    print("Available sessions:")
    for i, session in enumerate(sessions):
        # Try to get a timestamp from the session name
        try:
            timestamp = int(session.name.split("_")[1])
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i + 1}. {session.name} - {date_str}")
        except:
            print(f"{i + 1}. {session.name}")

    try:
        selection = int(input("\nSelect a session number (or 0 for latest): "))
        if selection == 0:
            return sessions[0]  # Latest session
        elif 1 <= selection <= len(sessions):
            return sessions[selection - 1]
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Invalid input.")
        return None


def analyze_session(session_path, output_dir=None):
    """Analyze a single session with all visualizations."""
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")

    # Load the data
    data = load_session_data(session_path)
    if not data:
        print("No data found in the selected session.")
        return

    # Check if we have prediction data
    if "predictions" in data:
        try:
            # Convert to DataFrame for easier analysis
            df = convert_predictions_to_dataframe(data["predictions"])

            print("\nPrediction Summary:")
            print(f"Total frames: {len(df)}")

            # Calculate some basic statistics
            gesture_counts = df["predicted_gesture"].value_counts()
            print("\nGesture distribution:")
            for gesture, count in gesture_counts.items():
                print(f"  {gesture}: {count} frames ({count / len(df) * 100:.1f}%)")

            print(f"\nAverage confidence: {df['confidence'].mean():.4f}")
            print(f"Min confidence: {df['confidence'].min():.4f}")
            print(f"Max confidence: {df['confidence'].max():.4f}")

            # Create visualizations
            print("\nGenerating visualizations...")

            # Use try-except blocks for each visualization to ensure one failure doesn't stop the process
            try:
                print("Creating prediction timeline plot...")
                plot_prediction_timeline(df, output_dir)
            except Exception as e:
                print(f"Error creating prediction timeline: {e}")

            try:
                print("Creating gesture distribution plot...")
                plot_gesture_distribution(df, output_dir)
            except Exception as e:
                print(f"Error creating gesture distribution: {e}")

            try:
                print("Creating probability heatmap...")
                plot_probability_heatmap(df, output_dir)
            except Exception as e:
                print(f"Error creating probability heatmap: {e}")

            try:
                print("Creating confidence histogram...")
                plot_confidence_histogram(df, output_dir)
            except Exception as e:
                print(f"Error creating confidence histogram: {e}")

            # Animation is the most likely to fail, so we'll handle it separately
            try:
                print("Creating animation...")
                if output_dir:
                    animation_path = os.path.join(output_dir, "prediction_animation.gif")
                    create_prediction_animation(df, animation_path)
                else:
                    create_prediction_animation(df)
            except Exception as e:
                print(f"Error creating animation: {e}")
                print("Skipping animation generation...")

        except Exception as e:
            print(f"Error analyzing prediction data: {e}")
            import traceback
            traceback.print_exc()

    # Visualize feature data if available
    if "unprocessed" in data:
        max_frame = len(data["unprocessed"]) - 1

        # Sample visualization of the first frame
        plot_feature_visualization(
            data["unprocessed"],
            data.get("preprocessed"),  # Might be None
            frame_idx=0,
            output_dir=output_dir
        )

        # Create a simple dialog for frame selection
        def create_frame_selector():
            # Create a small window for frame selection
            selector = tk.Tk()
            selector.title("Select Frame to Visualize")
            selector.geometry("400x150")

            tk.Label(selector, text=f"Enter frame number (0-{max_frame}):").pack(pady=10)

            frame_var = tk.StringVar()
            entry = tk.Entry(selector, textvariable=frame_var, width=10)
            entry.pack(pady=5)
            entry.focus_set()

            def view_frame():
                try:
                    frame_idx = int(frame_var.get())
                    if 0 <= frame_idx <= max_frame:
                        plot_feature_visualization(
                            data["unprocessed"],
                            data.get("preprocessed"),
                            frame_idx=frame_idx,
                            output_dir=output_dir
                        )
                    else:
                        messagebox.showerror("Error", f"Frame index out of range (0-{max_frame})")
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid number")

            def on_done():
                selector.destroy()

            button_frame = tk.Frame(selector)
            button_frame.pack(pady=10)

            tk.Button(button_frame, text="View Frame", command=view_frame).pack(side=tk.LEFT, padx=10)
            tk.Button(button_frame, text="Done", command=on_done).pack(side=tk.LEFT, padx=10)

            # Handle Enter key
            selector.bind('<Return>', lambda event: view_frame())

            selector.mainloop()

        # Ask if user wants to view more frames
        root = tk.Tk()
        root.withdraw()
        view_more = messagebox.askyesno("Feature Visualization",
                                        "Would you like to view more specific frames?")
        root.destroy()

        if view_more:
            create_frame_selector()

    print("\nVisualization complete.")


def run_visualization_tool():
    """Main function to run the visualization tool with GUI."""
    try:
        print("=" * 60)
        print("Gesture Recognition Data Visualization Tool")
        print("=" * 60)

        # Check if we have any sessions
        sessions = find_sessions()
        if not sessions:
            print("No data sessions found in the 'collected_data' directory.")
            input("Press Enter to exit...")
            return

        # Ask if user wants to use GUI or console
        try:
            use_gui = input("Use graphical interface for selection? (y/n, default: y): ").lower() != 'n'
        except:
            print("Error reading input, defaulting to GUI interface.")
            use_gui = True

        # Select session
        session_path = None
        try:
            if use_gui:
                session_path = select_session_gui()
            else:
                session_path = select_session_console()
        except Exception as e:
            print(f"Error selecting session: {e}")
            # Fallback to the latest session
            if sessions:
                session_path = sessions[0]
                print(f"Falling back to the latest session: {session_path}")

        if not session_path:
            print("No session selected. Exiting.")
            return

        print(f"\nSelected session: {session_path}")

        # Ask about saving visualizations
        save_output = False
        try:
            save_output = input("Save visualizations to disk? (y/n, default: n): ").lower() == 'y'
        except:
            print("Error reading input, defaulting to not saving visualizations.")

        output_dir = None
        if save_output:
            try:
                if use_gui:
                    output_dir = select_output_directory_gui()
                    if output_dir:
                        print(f"Visualizations will be saved to: {output_dir}")
                    else:
                        print("No output directory selected. Visualizations will not be saved.")
                else:
                    try:
                        output_path = input("Enter output directory path (or press Enter to use default): ")
                        if output_path:
                            output_dir = Path(output_path)
                        else:
                            # Create a default output directory based on session name
                            output_dir = Path(session_path) / "output"
                    except:
                        print("Error reading input. Using default output directory.")
                        output_dir = Path(session_path) / "output"

                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Visualizations will be saved to: {output_dir}")
            except Exception as e:
                print(f"Error setting up output directory: {e}")
                # Create a fallback directory
                output_dir = Path(session_path) / "output"
                os.makedirs(output_dir, exist_ok=True)
                print(f"Using fallback output directory: {output_dir}")

        # Run the analysis
        analyze_session(session_path, output_dir)

    except Exception as e:
        print(f"Unexpected error in visualization tool: {e}")
        import traceback
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()


if __name__ == "__main__":
    run_visualization_tool()