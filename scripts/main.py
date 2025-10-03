"""
Main execution script for gaze prediction training.
Imports all modules and orchestrates the training pipeline.
"""

import os
import torch
from torch.utils.data import DataLoader

# Import custom modules
from data_extraction import extract_fixation_data, save_dataset, load_saved_dataset, split_data
from dataset import GazeDataset
from model import SimpleGazeNet, GazeNetResNet, FrozenResNetBackbone, TinyGazeNet, UNetResNet18Gaze
from training import train_model
from diagnostics import check_data_quality, visualize_batch, baseline_mean_predictor, diagnose_model
from visualization import validate_and_visualize


def main():
    """
    Main execution function.
    """
    # Configuration
    GAZE_DATA_FOLDER = "../input_data/session_1/" # Where gaze data lives
    DATASET_FOLDER = "../input_data/session_1/"  # Where to save/load dataset

    BEST_MODEL_FILE_NAME = "../models/heatmap_softmax.pth" # Where to save good models
    VALIDATION_PREDICTION_OUTPUT = "../predictions/session1_validation_predictions/heatmap_softmax" # where to save prediction visualizations

    BATCH_SIZE = 16
    EPOCHS = 50000
    LEARNING_RATE = 0.001
    
    # Check if we should extract data or load existing
    if not os.path.exists(os.path.join(DATASET_FOLDER, 'labels.csv')):
        print("=== EXTRACTING DATA ===")
        data = extract_fixation_data(GAZE_DATA_FOLDER)
        save_dataset(data, DATASET_FOLDER)
        load_from_disk = False
    else:
        print("=== LOADING EXISTING DATASET ===")
        data = load_saved_dataset(DATASET_FOLDER)
        load_from_disk = True
    
    # ========== DIAGNOSTIC: Check data quality ==========
    check_data_quality(data)
    
    # Split data
    train_data, val_data = split_data(data, train_ratio=0.8)
    
    # Create datasets and dataloaders
    train_dataset = GazeDataset(train_data, load_from_disk=load_from_disk)
    val_dataset = GazeDataset(val_data, load_from_disk=load_from_disk)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ========== DIAGNOSTIC: Visualize batch ==========
    visualize_batch(train_loader, n_samples=4)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== CREATING MODEL (device: {device}) ===")

    # DEFINE THE MODEL HERE
    model = UNetResNet18Gaze().to(device)
    
    if os.path.exists(BEST_MODEL_FILE_NAME):
        print(f"Loading saved model from {BEST_MODEL_FILE_NAME}...")
        model.load_state_dict(torch.load(BEST_MODEL_FILE_NAME, map_location=device))
        print("Model loaded successfully!")
    else:
        print("No saved model found, starting from scratch...")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    
    # ========== DIAGNOSTIC: Baseline and model checks ==========
    baseline_error = baseline_mean_predictor(val_loader, device)
    diagnose_model(model, train_loader, val_loader, device)
    
    # Train (uncomment to enable training)
    print("\n=== TRAINING ===")

    # COMMENT THIS LINE OUT IF YOU WANT TO GENERATE THE IMAGES IN THE VISUALIZATION STEP ONLY
    # model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=device, best_model_file_name=BEST_MODEL_FILE_NAME)
    
    # ========== VALIDATION WITH VISUALIZATION ==========
    print("\n=== GENERATING VALIDATION VISUALIZATIONS ===")
    validate_and_visualize(model, val_dataset, device, output_folder=VALIDATION_PREDICTION_OUTPUT)
    
    print("\n=== DONE ===")
    print(f"Best model saved to {BEST_MODEL_FILE_NAME}")
    print("Validation predictions saved to: validation_predictions/")


if __name__ == "__main__":
    main()