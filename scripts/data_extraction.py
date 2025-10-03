"""
Data extraction and preprocessing for gaze prediction.
Handles fixation data extraction from video and gaze position data.
"""

import numpy as np
import pandas as pd
import cv2
import os


def extract_fixation_data(gaze_folder="000", export_number="000"):
    """
    Extract ONE sample per fixation (not per frame).
    For each fixation:
      - Extract the middle frame as the representative image
      - Calculate median gaze position across ALL frames in the fixation
    
    Returns:
        list of dicts with keys: 'frame', 'gaze_x', 'gaze_y', 'fixation_id', 'frame_number'
    """
    print("Extracting fixation data (one sample per fixation)...")
    
    # Load fixation data
    fixations = pd.read_csv(os.path.join(gaze_folder, "FixationData.csv"))
    
    # Load gaze positions
    gaze_data_path = os.path.join(gaze_folder, 'gaze_positions.csv')
    gaze_data = pd.read_csv(gaze_data_path)
    
    # Load world video
    world_video_path = os.path.join(gaze_folder, 'world.mp4')
    cap = cv2.VideoCapture(world_video_path)
    
    training_data = []
    filtered_count = 0
    
    # Loop through each fixation
    for idx, fix in fixations.iterrows():
        start_frame = int(fix.StartFrame)
        end_frame = int(fix.EndFrame)
        
        print(f"Processing fixation {idx+1}/{len(fixations)}: frames {start_frame}-{end_frame}")
        
        # Get the middle frame as representative image
        middle_frame = (start_frame + end_frame) // 2
        
        # Read the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  WARNING: Could not read frame {middle_frame}")
            continue
        
        # Collect ALL gaze positions across this entire fixation
        fixation_gaze_data = gaze_data[
            (gaze_data.world_index >= start_frame) & 
            (gaze_data.world_index <= end_frame)
        ]
        
        if len(fixation_gaze_data) == 0:
            print(f"  WARNING: No gaze data for this fixation")
            continue
        
        # Calculate median gaze position across ALL samples in this fixation
        gaze_x_values = fixation_gaze_data.norm_pos_x.values
        gaze_y_values = fixation_gaze_data.norm_pos_y.values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(gaze_x_values) | np.isnan(gaze_y_values))
        gaze_x_values = gaze_x_values[valid_mask]
        gaze_y_values = gaze_y_values[valid_mask]
        
        if len(gaze_x_values) == 0:
            filtered_count += 1
            print(f"  WARNING: All gaze data is NaN for this fixation")
            continue
        
        # Median gaze across the entire fixation
        gaze_x = np.median(gaze_x_values)
        gaze_y = 1 - np.median(gaze_y_values)  # Flip Y
        
        # FILTER: Skip gaze coordinates outside [0, 1] range
        if gaze_x < 0 or gaze_x > 1 or gaze_y < 0 or gaze_y > 1:
            filtered_count += 1
            print(f"  Filtered: gaze_x={gaze_x:.4f}, gaze_y={gaze_y:.4f} (out of range)")
            continue
        
        # DIAGNOSTIC: Print first few valid samples
        if len(training_data) < 5:
            print(f"  Valid sample {len(training_data)}: gaze_x={gaze_x:.4f}, gaze_y={gaze_y:.4f} "
                  f"(from {len(gaze_x_values)} gaze samples)")
        
        training_data.append({
            'frame': frame,
            'gaze_x': float(gaze_x),
            'gaze_y': float(gaze_y),
            'fixation_id': idx,
            'frame_number': middle_frame,
            'n_frames': end_frame - start_frame + 1,
            'n_gaze_samples': len(gaze_x_values)
        })
    
    cap.release()
    
    print(f"\nExtracted {len(training_data)} fixations (one sample per fixation)")
    print(f"Filtered out {filtered_count} fixations (NaN or out of [0,1] range)")
    
    if len(training_data) > 0:
        avg_frames = np.mean([d['n_frames'] for d in training_data])
        avg_samples = np.mean([d['n_gaze_samples'] for d in training_data])
        print(f"Average frames per fixation: {avg_frames:.1f}")
        print(f"Average gaze samples per fixation: {avg_samples:.1f}")
    
    return training_data


def save_dataset(data, output_folder):
    """
    Save extracted data to disk to avoid re-processing.
    """
    print(f"Saving dataset to {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    
    # Save images
    image_folder = os.path.join(output_folder, 'images')
    os.makedirs(image_folder, exist_ok=True)
    
    labels = []
    for i, item in enumerate(data):
        # Save image
        img_path = os.path.join(image_folder, f'fixation_{i:06d}.jpg')
        cv2.imwrite(img_path, item['frame'])
        
        # Save label
        labels.append({
            'image_path': img_path,
            'gaze_x': item['gaze_x'],
            'gaze_y': item['gaze_y'],
            'fixation_id': item['fixation_id'],
            'frame_number': item['frame_number'],
            'n_frames': item['n_frames'],
            'n_gaze_samples': item['n_gaze_samples']
        })
    
    # Save labels CSV
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(os.path.join(output_folder, 'labels.csv'), index=False)
    
    print("Dataset saved!")


def load_saved_dataset(dataset_folder):
    """
    Load a previously saved dataset and filter out any out-of-range values.
    """
    print(f"Loading dataset from {dataset_folder}...")
    
    labels_path = os.path.join(dataset_folder, 'labels.csv')
    labels_df = pd.read_csv(labels_path)
    
    data = []
    filtered_count = 0
    
    for _, row in labels_df.iterrows():
        gaze_x = row['gaze_x']
        gaze_y = row['gaze_y']
        
        # Filter out-of-range values
        if gaze_x < 0 or gaze_x > 1 or gaze_y < 0 or gaze_y > 1:
            filtered_count += 1
            continue
        
        data.append({
            'image_path': row['image_path'],
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'fixation_id': row.get('fixation_id', -1),
            'frame_number': row['frame_number']
        })
    
    print(f"Loaded {len(data)} valid fixations")
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} fixations with out-of-range gaze values")
    
    return data


def split_data(data, train_ratio=0.8):
    """
    Split data into train and validation sets.
    Since we have one sample per fixation, random splitting is now appropriate.
    """
    n_train = int(len(data) * train_ratio)
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    
    train_data = [data[i] for i in indices[:n_train]]
    val_data = [data[i] for i in indices[n_train:]]
    
    print(f"Train fixations: {len(train_data)}, Val fixations: {len(val_data)}")
    
    return train_data, val_data