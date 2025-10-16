"""
Main execution script for gaze prediction training or validation.
Use --no-train to skip training and only run validation + visualization.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

# Import custom modules
from data_extraction import extract_fixation_frames, load_train_val_data, split_and_save_dataset, extract_fixation_frames_per_frame
from dataset import GazeDataset, MultiFrameGazeDataset
from model import SimpleGazeNet, GazeNetResNet, FrozenResNetBackbone, TinyGazeNet, UNetResNet18Gaze, UNetResNet18MultiFrameGaze, LightUNetResNet18MultiFrameGaze
from training import train_model
from diagnostics import check_data_quality, visualize_batch, baseline_mean_predictor, diagnose_model
from visualization import validate_and_visualize


def main():
    """
    Main execution function.
    """
    # -------- Parse arguments --------
    parser = argparse.ArgumentParser(description="Train or validate gaze prediction model.")
    parser.add_argument("--no-train", action="store_true", help="Skip training and only run validation/visualization.")
    args = parser.parse_args()

    # -------- Configuration --------
    GAZE_DATA_FOLDER = "../input_data/session_1/"         # Raw gaze data + video
    DATASET_FOLDER = "../processed_data/session_1/gaze_data"       # Where to save/load processed dataset

    BEST_MODEL_FILE_NAME = "../models/heatmap_softmax_multi_image_gaze.pth" 
    VALIDATION_PREDICTION_OUTPUT = "../predictions/session_1_validation_predictions/heatmap_softmax_multi_image_gaze"

    BATCH_SIZE = 16
    EPOCHS = 50000
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.8  # portion of fixations used for training

    # -------- Data Preparation --------
    if not os.path.exists(DATASET_FOLDER) or not os.path.exists(os.path.join(DATASET_FOLDER, "train")) or not os.path.exists(os.path.join(DATASET_FOLDER, "train", "images")):
        print("=== EXTRACTING DATA (ALL FRAMES PER FIXATION) ===")
        data, n_fixations = extract_fixation_frames_per_frame(GAZE_DATA_FOLDER)
        split_and_save_dataset(data, n_fixations, train_ratio=TRAIN_RATIO, output_folder=DATASET_FOLDER)
    else:
        print("=== USING EXISTING DATASET ===")

    train_data, val_data = load_train_val_data(DATASET_FOLDER)
    check_data_quality(train_data + val_data)

    train_dataset = MultiFrameGazeDataset(train_data, load_from_disk=True)
    val_dataset = MultiFrameGazeDataset(val_data, load_from_disk=True)

    # -------- Dataloader Setup --------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== USING DEVICE: {device} ===")

    if device.type == 'cuda':
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,              
            pin_memory=True,         
            persistent_workers=True,   
            prefetch_factor=4,         
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    # -------- Diagnostics: Visualize Batch --------
    # visualize_batch(train_loader, n_samples=4)
    
    # -------- Model Setup --------
    print(f"\n=== CREATING MODEL (device: {device}) ===")

    model = UNetResNet18MultiFrameGaze().to(device)
    
    if os.path.exists(BEST_MODEL_FILE_NAME):
        print(f"Loading saved model from {BEST_MODEL_FILE_NAME}...")
        model.load_state_dict(torch.load(BEST_MODEL_FILE_NAME, map_location=device))
        print("Model loaded successfully!")
    else:
        print("No saved model found, starting from scratch...")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    # -------- Baseline and Diagnostics --------
    baseline_error = baseline_mean_predictor(val_loader, device)
    diagnose_model(model, train_loader, val_loader, device)
    
    # -------- Train or Skip --------
    if not args.no_train:
        print("\n=== TRAINING ===")
        model = train_model(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            best_model_file_name=BEST_MODEL_FILE_NAME,
        )
    else:
        print("\n=== SKIPPING TRAINING (Validation Only Mode) ===")

    # -------- Validation + Visualization --------
    print("\n=== GENERATING VALIDATION VISUALIZATIONS ===")
    validate_and_visualize(model, val_dataset, device, output_folder=VALIDATION_PREDICTION_OUTPUT)
    
    print("\n=== DONE ===")
    print(f"Best model saved to {BEST_MODEL_FILE_NAME}")
    print(f"Validation predictions saved to: {VALIDATION_PREDICTION_OUTPUT}")


if __name__ == "__main__":
    main()
