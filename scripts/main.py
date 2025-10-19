"""
Main execution script for gaze prediction training or validation.
Use --no-train to skip training and only run validation + visualization.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Import custom modules
from data_extraction import extract_fixation_frames, load_train_val_data, split_and_save_dataset, extract_fixation_frames_per_frame, extract_all_valid_gaze_frames
from dataset import GazeDataset, MultiFrameGazeDataset
from model import SimpleGazeNet, GazeNetResNet, FrozenResNetBackbone, TinyGazeNet, UNetResNet18Gaze, UNetResNet18MultiFrameGaze, LightUNetResNet18MultiFrameGaze, UNetTemporalAttentionGaze
from training import train_model
from diagnostics import check_data_quality, visualize_batch, baseline_mean_predictor, diagnose_model
from visualization import validate_and_visualize
from stitch_images_to_videos import stitch_images_to_video



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

    BATCH_SIZE = 64 * 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.9  # portion of fixations used for training
    NUM_FRAMES = 1 # Number of frames to use as temporal context
    USE_NORMALIZED_COORDINATES = True


    BEST_MODEL_FILE_NAME = f"../models/UNetTemporalAttentionGaze_{NUM_FRAMES}_frames_MSE.pth"
    VALIDATION_PREDICTION_OUTPUT = f"../predictions/session_1_validation_predictions/UNetTemporalAttentionGaze_{NUM_FRAMES}_frames_MSE"

    OUTPUT_FRAME_RATE=15

    # -------- Data Preparation --------
    if not os.path.exists(DATASET_FOLDER) or not os.path.exists(os.path.join(DATASET_FOLDER, "train")) or not os.path.exists(os.path.join(DATASET_FOLDER, "train", "images")):
        print("=== EXTRACTING DATA (ALL FRAMES PER FIXATION) ===")
        data, num_frames = extract_all_valid_gaze_frames(GAZE_DATA_FOLDER)
        split_and_save_dataset(data, num_frames, train_ratio=TRAIN_RATIO, output_folder=DATASET_FOLDER)
    else:
        print("=== USING EXISTING DATASET ===")

    train_data, val_data = load_train_val_data(DATASET_FOLDER)
    check_data_quality(train_data + val_data)

    train_dataset = MultiFrameGazeDataset(train_data, num_frames=NUM_FRAMES, load_from_disk=True)
    val_dataset = MultiFrameGazeDataset(val_data, num_frames=NUM_FRAMES, load_from_disk=True)

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
    for mode in ["heatmap", "mlp"]:
        print("MODE: " + mode)
        for criterion in [nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()]:
            for RUN_NUM in range(4):
                print("RUN NUMBER: " + str(RUN_NUM))
                for NUM_FRAMES in [1, 2, 4, 8]:
                    print("USING NUM_FRAMES: " + str(NUM_FRAMES))
                    print(f"\n=== CREATING MODEL (device: {device}) ===")

                    model = UNetTemporalAttentionGaze(num_frames=NUM_FRAMES, mode=mode).to(device)
                    
                    # if os.path.exists(BEST_MODEL_FILE_NAME):
                    #     print(f"Loading saved model from {BEST_MODEL_FILE_NAME}...")
                    #     model.load_state_dict(torch.load(BEST_MODEL_FILE_NAME, map_location=device))
                    #     print("Model loaded successfully!")
                    # else:
                    #     print("No saved model found, starting from scratch...")

                    # n_params = sum(p.numel() for p in model.parameters())
                    # print(f"Model has {n_params:,} parameters")

                    # # -------- Baseline and Diagnostics --------
                    # baseline_error = baseline_mean_predictor(val_loader, device)
                    # diagnose_model(model, train_loader, val_loader, device, use_normalized=USE_NORMALIZED_COORDINATES)
                    
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
                            use_normalized=USE_NORMALIZED_COORDINATES,
                            criterion=criterion
                        )
                    else:
                        print("\n=== SKIPPING TRAINING (Validation Only Mode) ===")

    # # -------- Validation + Visualization --------
    # print("\n=== GENERATING VALIDATION VISUALIZATIONS ===")
    # # model.load_state_dict(torch.load(BEST_MODEL_FILE_NAME, map_location=device))
    # validate_and_visualize(model, val_dataset, device, output_folder=VALIDATION_PREDICTION_OUTPUT, use_normalized=USE_NORMALIZED_COORDINATES)

    # print("\n=== STITCHING IMAGES TO VIDEO ===")
    # stitch_images_to_video(VALIDATION_PREDICTION_OUTPUT, frame_rate=OUTPUT_FRAME_RATE)

    # print("\n=== DONE ===")
    # print(f"Best model saved to {BEST_MODEL_FILE_NAME}")
    # print(f"Validation predictions saved to: {VALIDATION_PREDICTION_OUTPUT}")


if __name__ == "__main__":
    main()
