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

def extract_fixation_frames_per_frame(gaze_folder="000"):
    """
    Extract all frames for each fixation, pairing each frame with the specific gaze data for that frame.
    
    Rules:
      - If multiple gaze points exist for a frame, take the median.
      - If a frame has no gaze points, use the most recent valid gaze point within the fixation.
      - If no previous valid gaze point exists, use the fixation's overall median.
      - Skip frames where the chosen gaze is invalid (not in [0, 1]).
    
    Returns:
        list of dicts with keys:
        'frame', 'gaze_x', 'gaze_y', 'fixation_id', 'frame_number'
    """
    print("Extracting per-frame gaze samples...")

    fixations = pd.read_csv(os.path.join(gaze_folder, "FixationData.csv"))
    gaze_data = pd.read_csv(os.path.join(gaze_folder, "gaze_positions.csv"))
    world_video_path = os.path.join(gaze_folder, "world.mp4")
    cap = cv2.VideoCapture(world_video_path)

    all_data = []
    filtered_count = 0

    for idx, fix in fixations.iterrows():
        start_frame = int(fix.StartFrame)
        end_frame = int(fix.EndFrame)
        fixation_gaze_data = gaze_data[
            (gaze_data.world_index >= start_frame) &
            (gaze_data.world_index <= end_frame)
        ]

        if len(fixation_gaze_data) == 0:
            print(f"  WARNING: No gaze data for fixation {idx}")
            continue

        # Compute fixation-level median as fallback
        fix_gaze_x_vals = fixation_gaze_data.norm_pos_x.dropna().values
        fix_gaze_y_vals = fixation_gaze_data.norm_pos_y.dropna().values
        if len(fix_gaze_x_vals) == 0:
            print(f"  WARNING: All gaze values NaN for fixation {idx}")
            filtered_count += 1
            continue

        fixation_median_x = np.median(fix_gaze_x_vals)
        fixation_median_y = 1 - np.median(fix_gaze_y_vals)  # Flip Y

        if not (0 <= fixation_median_x <= 1 and 0 <= fixation_median_y <= 1):
            print(f"  Filtered fixation {idx}: invalid median gaze ({fixation_median_x:.3f}, {fixation_median_y:.3f})")
            filtered_count += 1
            continue

        print(f"Processing fixation {idx+1}/{len(fixations)}: frames {start_frame}-{end_frame}")

        last_valid_gaze = None
        for frame_number in range(start_frame, end_frame + 1):
            # All gaze samples for this frame
            frame_gaze_samples = fixation_gaze_data[
                fixation_gaze_data.world_index == frame_number
            ]

            if len(frame_gaze_samples) > 0:
                gx = frame_gaze_samples.norm_pos_x.dropna().values
                gy = frame_gaze_samples.norm_pos_y.dropna().values

                if len(gx) > 0:
                    gaze_x = np.median(gx)
                    gaze_y = 1 - np.median(gy)  # Flip Y
                    last_valid_gaze = (gaze_x, gaze_y)
                else:
                    # No valid samples for this frame
                    if last_valid_gaze is not None:
                        gaze_x, gaze_y = last_valid_gaze
                    else:
                        gaze_x, gaze_y = fixation_median_x, fixation_median_y
            else:
                # No samples for this frame
                if last_valid_gaze is not None:
                    gaze_x, gaze_y = last_valid_gaze
                else:
                    gaze_x, gaze_y = fixation_median_x, fixation_median_y

            # Skip invalid gaze coordinates
            if not (0 <= gaze_x <= 1 and 0 <= gaze_y <= 1):
                print(f"  Frame {frame_number}: invalid gaze ({gaze_x:.3f}, {gaze_y:.3f}) — skipped")
                filtered_count += 1
                continue

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
    print(f"\nExtracted {len(all_data)} per-frame samples total.")
    print(f"Filtered out {filtered_count} frames (invalid gaze or missing data).")
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

def extract_all_valid_gaze_frames(gaze_folder="000"):
    """
    Extract every video frame that has at least one valid gaze point.
    
    Rules:
      - Take all frames with valid gaze data from gaze_positions.csv.
      - If multiple gaze samples correspond to the same frame, use their median.
      - Skip frames with no valid gaze or with invalid gaze coordinates.
    
    Returns:
        list of dicts with keys:
            'frame', 'gaze_x', 'gaze_y', 'frame_number'
    """
    print("Extracting all frames with valid gaze data...")

    gaze_data = pd.read_csv(os.path.join(gaze_folder, "gaze_positions.csv"))
    world_video_path = os.path.join(gaze_folder, "world.mp4")
    cap = cv2.VideoCapture(world_video_path)

    # Drop invalid rows
    gaze_data = gaze_data.dropna(subset=["world_index", "norm_pos_x", "norm_pos_y"])
    gaze_data = gaze_data[(gaze_data.norm_pos_x.between(0, 1)) & (gaze_data.norm_pos_y.between(0, 1))]

    # Group by frame index
    grouped = gaze_data.groupby("world_index")
    all_data = []
    skipped = 0

    for frame_number, group in grouped:
        gaze_x = np.median(group.norm_pos_x.values)
        gaze_y = 1 - np.median(group.norm_pos_y.values)  # flip Y to image coordinates

        # Skip invalid gaze
        if not (0 <= gaze_x <= 1 and 0 <= gaze_y <= 1):
            skipped += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: Could not read frame {frame_number}")
            skipped += 1
            continue

        all_data.append({
            "frame": frame,
            "gaze_x": float(gaze_x),
            "gaze_y": float(gaze_y),
            "frame_number": int(frame_number)
        })

    cap.release()
    print(f"\nExtracted {len(all_data)} valid frames total.")
    print(f"Skipped {skipped} frames (invalid gaze or unreadable).")
    return all_data


# data, n_fixations = extract_fixation_frames("../input_data/session_1")
# split_and_save_dataset(data, n_fixations, train_ratio=0.8, output_folder="../processed_data/session_1")
