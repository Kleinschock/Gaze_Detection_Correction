import numpy as np
import pandas as pd

def calculate_velocity(sequence_df):
    """
    Calculates the velocity of each keypoint.
    Velocity is calculated as the difference in position between consecutive frames.
    """
    velocity = sequence_df.diff().fillna(0)
    velocity.columns = [f'{col}_vel' for col in sequence_df.columns]
    return velocity

def calculate_acceleration(sequence_df):
    """
    Calculates the acceleration of each keypoint.
    Acceleration is calculated as the difference in velocity between consecutive frames.
    """
    # First, calculate velocity
    velocity = sequence_df.diff().fillna(0)
    # Then, calculate acceleration from velocity
    acceleration = velocity.diff().fillna(0)
    acceleration.columns = [f'{col}_acc' for col in sequence_df.columns]
    return acceleration

# Placeholder for future geometric features
def calculate_distances(sequence_df):
    """
    Placeholder function to calculate specific distances between keypoints.
    """
    # Example: distance between hand and shoulder
    # This would require specific column names to be identified
    distances = pd.DataFrame()
    return distances

def calculate_angles(sequence_df):
    """
    Placeholder function to calculate specific joint angles.
    """
    # Example: elbow angle
    # This would require specific column names for shoulder, elbow, wrist
    angles = pd.DataFrame()
    return angles
