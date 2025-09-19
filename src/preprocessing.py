import numpy as np
import pandas as pd

# Import from our project files
from .config import PREPROCESSING_STRATEGY, SELECTED_KEYPOINTS, FEATURE_ENGINEERING_CONFIG
from . import feature_engineering as fe

def center_on_shoulders(keypoints, kp_map):
    """Applies position invariance by centering on the shoulder midpoint."""
    torso_anchor = (keypoints[..., kp_map['left_shoulder'], :] + keypoints[..., kp_map['right_shoulder'], :]) / 2.0
    # Add a new axis for broadcasting
    return keypoints - torso_anchor[..., np.newaxis, :]

def scale_by_shoulder_width(keypoints, kp_map):
    """Applies size invariance by scaling based on shoulder width."""
    shoulder_dist = np.linalg.norm(keypoints[..., kp_map['left_shoulder'], :] - keypoints[..., kp_map['right_shoulder'], :], axis=-1, keepdims=True)
    # Add a new axis for broadcasting
    scale = shoulder_dist[..., :, np.newaxis] + 1e-8
    return keypoints / scale

def rotate_to_body_coordinates(keypoints, kp_map):
    """Applies rotation invariance by aligning to a body-centric coordinate system."""
    # Vector from right to left shoulder as the new X-axis
    x_axis = keypoints[..., kp_map['left_shoulder'], :] - keypoints[..., kp_map['right_shoulder'], :]
    x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True) + 1e-8

    # Vector from shoulder midpoint to nose as the "up" vector
    mid_shoulders = (keypoints[..., kp_map['left_shoulder'], :] + keypoints[..., kp_map['right_shoulder'], :]) / 2
    y_approx = keypoints[..., kp_map['nose'], :] - mid_shoulders
    
    # New Z-axis via cross product
    z_axis = np.cross(x_axis, y_approx, axis=-1)
    z_axis /= np.linalg.norm(z_axis, axis=-1, keepdims=True) + 1e-8
    
    # Final Y-axis is orthogonal to X and Z
    y_axis = np.cross(z_axis, x_axis, axis=-1)
    y_axis /= np.linalg.norm(y_axis, axis=-1, keepdims=True) + 1e-8

    # Stack axes into rotation matrices
    rotation_matrices = np.stack([x_axis, y_axis, z_axis], axis=-1)
    
    # Apply rotation via batched matrix multiplication
    return np.einsum('...ij,...kj->...ki', rotation_matrices, keypoints)

def preprocess_sequence(sequence_data, scaler=None):
    """
    The main, centralized preprocessing pipeline.
    Applies a sequence of transformations to raw keypoint data based on the
    global PREPROCESSING_STRATEGY and FEATURE_ENGINEERING_CONFIG from the config file.
    """
    kp_map = {name: i for i, name in enumerate(SELECTED_KEYPOINTS)}
    
    # Check if we're processing a single frame or a sequence
    is_single_frame = len(sequence_data.shape) == 2

    # Reshape to (..., num_keypoints, 3) for normalization functions
    num_kp = len(SELECTED_KEYPOINTS)
    keypoints = sequence_data.reshape(-1, num_kp, 3)

    # --- Normalization / Standardization ---
    if PREPROCESSING_STRATEGY == 'body_centric':
        keypoints = center_on_shoulders(keypoints, kp_map)
        keypoints = scale_by_shoulder_width(keypoints, kp_map)
        keypoints = rotate_to_body_coordinates(keypoints, kp_map)

    elif PREPROCESSING_STRATEGY == 'standardize':
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when using the 'standardize' strategy.")
        
        keypoints_flat = keypoints.reshape(keypoints.shape[0], -1)
        standardized_flat = scaler.transform(keypoints_flat)
        keypoints = standardized_flat.reshape(keypoints.shape)
    else:
        pass # No normalization applied

    # --- Feature Engineering ---
    # Convert to a DataFrame for easier feature calculation and concatenation
    positions_flat = keypoints.reshape(keypoints.shape[0], -1)
    
    # Create column names for the DataFrame
    columns = [f'{kp}_{coord}' for kp in SELECTED_KEYPOINTS for coord in ['x', 'y', 'z']]
    positions_df = pd.DataFrame(positions_flat, columns=columns)

    # A list to hold all feature DataFrames, starting with the normalized positions
    all_features = [positions_df]

    # Dynamically calculate and add features based on the config
    if FEATURE_ENGINEERING_CONFIG.get('velocity'):
        velocity_df = fe.calculate_velocity(positions_df)
        all_features.append(velocity_df)

    if FEATURE_ENGINEERING_CONFIG.get('acceleration'):
        acceleration_df = fe.calculate_acceleration(positions_df)
        all_features.append(acceleration_df)

    if FEATURE_ENGINEERING_CONFIG.get('distances'):
        distances_df = fe.calculate_distances(positions_df)
        if not distances_df.empty:
            all_features.append(distances_df)

    if FEATURE_ENGINEERING_CONFIG.get('angles'):
        angles_df = fe.calculate_angles(positions_df)
        if not angles_df.empty:
            all_features.append(angles_df)

    # Combine all features into a single DataFrame, then convert to numpy array
    final_features_df = pd.concat(all_features, axis=1)
    final_features = final_features_df.to_numpy(dtype=np.float32)

    # If input was a single frame, remove the extra dimensions before returning
    if is_single_frame:
        return final_features.squeeze()
        
    return final_features
