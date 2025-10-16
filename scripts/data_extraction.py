import numpy as np
import pandas as pd
import cv2
import os


def extract_fixation_frames(gaze_folder="000"):
    """
    Extract all frames for each fixation (not just the middle one).
    Each frame within a fixation is paired with the fixation’s median gaze position.

    Returns:
        list of dicts with keys:
        'frame', 'gaze_x', 'gaze_y', 'fixation_id', 'frame_number'
    """
    print("Extracting all fixation frames...")

    # Load fixation data and gaze data
    fixations = pd.read_csv(os.path.join(gaze_folder, "FixationData.csv"))
    gaze_data = pd.read_csv(os.path.join(gaze_folder, "gaze_positions.csv"))
    world_video_path = os.path.join(gaze_folder, "world.mp4")
    cap = cv2.VideoCapture(world_video_path)

    all_data = []
    filtered_count = 0

    for idx, fix in fixations.iterrows():
        start_frame = int(fix.StartFrame)
        end_frame = int(fix.EndFrame)
        n_frames = end_frame - start_frame + 1

        print(f"Processing fixation {idx+1}/{len(fixations)}: frames {start_frame}-{end_frame}")

        # Get gaze samples for this fixation
        fixation_gaze_data = gaze_data[
            (gaze_data.world_index >= start_frame) & 
            (gaze_data.world_index <= end_frame)
        ]

        if len(fixation_gaze_data) == 0:
            print(f"  WARNING: No gaze data for fixation {idx}")
            continue

        # Median gaze position across fixation
        gaze_x_values = fixation_gaze_data.norm_pos_x.dropna().values
        gaze_y_values = fixation_gaze_data.norm_pos_y.dropna().values
        if len(gaze_x_values) == 0:
            print(f"  WARNING: All gaze values NaN for fixation {idx}")
            filtered_count += 1
            continue

        gaze_x = np.median(gaze_x_values)
        gaze_y = 1 - np.median(gaze_y_values)  # Flip Y

        # Skip invalid gaze coordinates
        if not (0 <= gaze_x <= 1 and 0 <= gaze_y <= 1):
            filtered_count += 1
            print(f"  Filtered: invalid gaze ({gaze_x:.3f}, {gaze_y:.3f})")
            continue

        # Extract every frame in the fixation
        for frame_number in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                print(f"  WARNING: Could not read frame {frame_number}")
                continue

            all_data.append({
                'frame': frame,
                'gaze_x': float(gaze_x),
                'gaze_y': float(gaze_y),
                'fixation_id': idx,
                'frame_number': frame_number
            })

    cap.release()
    print(f"\nExtracted {len(all_data)} frame samples total.")
    print(f"Filtered out {filtered_count} fixations (NaN or out of range).")
    return all_data, len(fixations)


def split_and_save_dataset(data, n_fixations, train_ratio=0.8, output_folder="dataset"):
    """
    Split data by fixation index (chronologically), not randomly.
    First train_ratio% fixations → training, remainder → validation.
    """
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")

    os.makedirs(os.path.join(train_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_folder, "images"), exist_ok=True)

    train_labels, val_labels = [], []

    cutoff_fixation = int(train_ratio * n_fixations)
    print(f"Splitting fixations: first {cutoff_fixation}/{n_fixations} for training.")

    for i, item in enumerate(data):
        folder = train_folder if item["fixation_id"] < cutoff_fixation else val_folder
        img_folder = os.path.join(folder, "images")

        img_path = os.path.join(img_folder, f'frame_{i:06d}.jpg')
        cv2.imwrite(img_path, item["frame"])

        label_entry = {
            'image_path': img_path,
            'gaze_x': item['gaze_x'],
            'gaze_y': item['gaze_y'],
            'fixation_id': item['fixation_id'],
            'frame_number': item['frame_number']
        }

        if folder == train_folder:
            train_labels.append(label_entry)
        else:
            val_labels.append(label_entry)

    # Save labels
    pd.DataFrame(train_labels).to_csv(os.path.join(train_folder, "labels.csv"), index=False)
    pd.DataFrame(val_labels).to_csv(os.path.join(val_folder, "labels.csv"), index=False)

    print(f"Saved {len(train_labels)} training samples and {len(val_labels)} validation samples.")

def load_train_val_data(dataset_folder):
    """
    Load previously saved training and validation datasets.

    Args:
        dataset_folder (str): Path to the dataset root folder that contains
                              'train/' and 'val/' subfolders.

    Returns:
        (train_data, val_data): Two lists of dicts with keys:
            'image_path', 'gaze_x', 'gaze_y', 'fixation_id', 'frame_number'
    """
    train_folder = os.path.join(dataset_folder, "train")
    val_folder = os.path.join(dataset_folder, "val")

    train_labels_path = os.path.join(train_folder, "labels.csv")
    val_labels_path = os.path.join(val_folder, "labels.csv")

    if not os.path.exists(train_labels_path) or not os.path.exists(val_labels_path):
        raise FileNotFoundError(
            f"Could not find 'labels.csv' in {train_folder} or {val_folder}. "
            f"Make sure you ran the extraction and saving pipeline first."
        )

    def load_split(labels_path):
        df = pd.read_csv(labels_path)
        data = []
        filtered = 0
        for _, row in df.iterrows():
            gaze_x, gaze_y = row["gaze_x"], row["gaze_y"]
            if not (0 <= gaze_x <= 1 and 0 <= gaze_y <= 1):
                filtered += 1
                continue
            data.append({
                "image_path": row["image_path"],
                "gaze_x": float(gaze_x),
                "gaze_y": float(gaze_y),
                "fixation_id": int(row["fixation_id"]),
                "frame_number": int(row["frame_number"])
            })
        if filtered > 0:
            print(f"Filtered out {filtered} samples with invalid gaze coords from {labels_path}")
        return data

    print(f"Loading train data from {train_labels_path}...")
    train_data = load_split(train_labels_path)

    print(f"Loading val data from {val_labels_path}...")
    val_data = load_split(val_labels_path)

    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples.")
    return train_data, val_data


# data, n_fixations = extract_fixation_frames("../input_data/session_1")
# split_and_save_dataset(data, n_fixations, train_ratio=0.8, output_folder="../processed_data/session_1")
