import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: PREPARE DATA
# ============================================================================

def extract_fixation_data(subject_folder, gaze_folder="000", export_number="000"):
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
    fixations = pd.read_csv(os.path.join(subject_folder, "FixationData.csv"))
    
    # Load gaze positions
    gaze_data_path = os.path.join(subject_folder, gaze_folder, 'exports', export_number, 'gaze_positions.csv')
    gaze_data = pd.read_csv(gaze_data_path)
    
    # Load world video
    world_video_path = os.path.join(subject_folder, gaze_folder, 'world.mp4')
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


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def check_data_quality(data):
    """Diagnose data issues"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    gaze_x = [item['gaze_x'] for item in data]
    gaze_y = [item['gaze_y'] for item in data]
    
    print(f"\nGaze X Statistics:")
    print(f"  Min: {min(gaze_x):.4f}")
    print(f"  Max: {max(gaze_x):.4f}")
    print(f"  Mean: {np.mean(gaze_x):.4f}")
    print(f"  Std: {np.std(gaze_x):.4f}")
    print(f"  Unique values: {len(set(gaze_x))}")
    
    print(f"\nGaze Y Statistics:")
    print(f"  Min: {min(gaze_y):.4f}")
    print(f"  Max: {max(gaze_y):.4f}")
    print(f"  Mean: {np.mean(gaze_y):.4f}")
    print(f"  Std: {np.std(gaze_y):.4f}")
    print(f"  Unique values: {len(set(gaze_y))}")
    
    # Check if all values are the same
    if np.std(gaze_x) < 0.01:
        print("\n⚠️  WARNING: Gaze X positions have very low variance!")
    if np.std(gaze_y) < 0.01:
        print("⚠️  WARNING: Gaze Y positions have very low variance!")
    
    # Verify all values are in [0, 1] range
    if min(gaze_x) >= 0 and max(gaze_x) <= 1:
        print("\n✓ All Gaze X values are in [0, 1] range")
    else:
        print("\n⚠️  ERROR: Some Gaze X values are outside [0, 1] range!")
    
    if min(gaze_y) >= 0 and max(gaze_y) <= 1:
        print("✓ All Gaze Y values are in [0, 1] range")
    else:
        print("⚠️  ERROR: Some Gaze Y values are outside [0, 1] range!")
    
    print(f"\nTotal fixations: {len(data)}")
    print("First 10 fixations:")
    for i in range(min(10, len(data))):
        print(f"  [{i}] gaze_x={data[i]['gaze_x']:.4f}, gaze_y={data[i]['gaze_y']:.4f}")
    
    print("="*60 + "\n")


def visualize_batch(dataloader, n_samples=4, save_path='batch_visualization.png'):
    """Visualize a batch to check data"""
    print("\n" + "="*60)
    print("BATCH VISUALIZATION")
    print("="*60)
    
    images, gazes = next(iter(dataloader))
    
    fig, axes = plt.subplots(1, min(n_samples, len(images)), figsize=(15, 3))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(min(n_samples, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Gaze: ({gazes[i][0]:.3f}, {gazes[i][1]:.3f})", fontsize=10)
        axes[i].axis('off')
        
        # Draw gaze point
        gaze_x_px = gazes[i][0].item() * 224
        gaze_y_px = gazes[i][1].item() * 224
        axes[i].plot(gaze_x_px, gaze_y_px, 'r+', markersize=15, markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()
    print("="*60 + "\n")


def baseline_mean_predictor(val_loader, device):
    """Calculate baseline error by predicting mean gaze position"""
    print("\n" + "="*60)
    print("BASELINE PREDICTOR (Mean Gaze)")
    print("="*60)
    
    all_gazes = []
    for _, gazes in val_loader:
        all_gazes.append(gazes)
    all_gazes = torch.cat(all_gazes)
    mean_gaze = all_gazes.mean(dim=0)
    
    print(f"Mean gaze position: ({mean_gaze[0]:.4f}, {mean_gaze[1]:.4f})")
    
    # Calculate error
    errors = torch.norm((all_gazes - mean_gaze) * torch.tensor([1920, 1080]), dim=1)
    print(f"Baseline error (always predict mean): {errors.mean().item():.1f} pixels")
    print(f"Baseline error std: {errors.std().item():.1f} pixels")
    print("="*60 + "\n")
    
    return errors.mean().item()


def diagnose_model(model, train_loader, val_loader, device):
    """Run comprehensive diagnostics before training"""
    print("\n" + "="*60)
    print("MODEL DIAGNOSTICS")
    print("="*60)
    
    # Check a single batch
    images, gazes = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Gazes: {gazes.shape}")
    print(f"\nGaze range in batch:")
    print(f"  Min: {gazes.min().item():.4f}")
    print(f"  Max: {gazes.max().item():.4f}")
    print(f"  Mean: {gazes.mean().item():.4f}")
    print(f"  Std: {gazes.std().item():.4f}")
    
    # Check model forward pass
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device))
    
    print(f"\nModel predictions (current weights):")
    print(f"  Min: {preds.min().item():.4f}")
    print(f"  Max: {preds.max().item():.4f}")
    print(f"  Mean: {preds.mean().item():.4f}")
    print(f"  Std: {preds.std().item():.4f}")
    
    # Check if predictions are stuck
    if preds.std().item() < 0.01:
        print("\n⚠️  WARNING: Model predictions have very low variance!")
        print("   This suggests the model might be stuck or saturated.")
    
    # Calculate initial loss
    criterion = nn.MSELoss()
    loss = criterion(preds, gazes.to(device))
    print(f"\nInitial loss on first batch: {loss.item():.4f}")
    
    # Check gradients
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    preds = model(images.to(device))
    loss = criterion(preds, gazes.to(device))
    loss.backward()
    
    # Check gradient norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"Gradient norm: {total_norm:.4f}")
    if total_norm < 1e-7:
        print("⚠️  WARNING: Gradients are extremely small - learning will be very slow!")
    elif total_norm > 100:
        print("⚠️  WARNING: Gradients are very large - may need gradient clipping!")
    
    print("="*60 + "\n")


# ============================================================================
# STEP 2: CREATE DATASET CLASS
# ============================================================================

class GazeDataset(Dataset):
    """
    PyTorch Dataset for gaze prediction.
    """
    def __init__(self, data, transform=None, load_from_disk=False):
        self.data = data
        self.load_from_disk = load_from_disk
        
        # Default transform: resize and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        if self.load_from_disk:
            image = cv2.imread(item['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(item['frame'], cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(image)
        image = self.transform(image)
        
        # Get gaze label
        gaze = torch.tensor([item['gaze_x'], item['gaze_y']], dtype=torch.float32)
        
        return image, gaze


# ============================================================================
# STEP 3: DEFINE MODEL
# ============================================================================

class SimpleGazeNet(nn.Module):
    """
    Simple CNN for gaze prediction.
    Input: 224x224x3 image
    Output: (x, y) gaze coordinates normalized [0, 1]
    """
    def __init__(self):
        super(SimpleGazeNet, self).__init__()
        
        # Simple CNN backbone
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# ============================================================================
# STEP 4: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    for images, gaze_targets in train_loader:
        images = images.to(device)
        gaze_targets = gaze_targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        gaze_pred = model(images)
        loss = criterion(gaze_pred, gaze_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0
    total_error_pixels = 0
    
    with torch.no_grad():
        for images, gaze_targets in val_loader:
            images = images.to(device)
            gaze_targets = gaze_targets.to(device)
            
            gaze_pred = model(images)
            loss = criterion(gaze_pred, gaze_targets)
            
            total_loss += loss.item()
            
            # Calculate pixel error (assuming 1920x1080 image)
            error_pixels = torch.norm((gaze_pred - gaze_targets) * torch.tensor([1920, 1080]).to(device), dim=1)
            total_error_pixels += error_pixels.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    avg_error_pixels = total_error_pixels / len(val_loader.dataset)
    
    return avg_loss, avg_error_pixels


def validate_and_visualize(model, val_dataset, device, output_folder='validation_predictions'):
    """
    Validate the model and save visualizations of predictions.
    
    Args:
        model: trained model
        val_dataset: validation dataset
        device: torch device
        output_folder: folder to save visualization images
    """
    print("\n" + "="*60)
    print("VALIDATION WITH VISUALIZATION")
    print("="*60)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving predictions to: {output_folder}/")
    
    model.eval()
    total_error_pixels = 0
    
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            # Get single sample
            image_tensor, gaze_target = val_dataset[idx]
            
            # Get original image for visualization
            item = val_dataset.data[idx]
            if val_dataset.load_from_disk:
                orig_image = cv2.imread(item['image_path'])
            else:
                orig_image = item['frame'].copy()
            
            # Get image dimensions
            h, w = orig_image.shape[:2]
            
            # Make prediction
            image_batch = image_tensor.unsqueeze(0).to(device)
            gaze_pred = model(image_batch).squeeze(0).cpu()
            
            # Calculate error
            error_pixels = torch.norm((gaze_pred - gaze_target) * torch.tensor([w, h]))
            total_error_pixels += error_pixels.item()
            
            # Convert normalized coordinates to pixel coordinates
            pred_x = int(gaze_pred[0].item() * w)
            pred_y = int(gaze_pred[1].item() * h)
            
            # Draw prediction on image (red circle)
            cv2.circle(orig_image, (pred_x, pred_y), radius=8, color=(0, 0, 255), thickness=-1)
            cv2.circle(orig_image, (pred_x, pred_y), radius=10, color=(0, 0, 255), thickness=2)
            
            # Optional: draw ground truth (green circle) for comparison
            gt_x = int(gaze_target[0].item() * w)
            gt_y = int(gaze_target[1].item() * h)
            cv2.circle(orig_image, (gt_x, gt_y), radius=8, color=(0, 255, 0), thickness=-1)
            cv2.circle(orig_image, (gt_x, gt_y), radius=10, color=(0, 255, 0), thickness=2)
            
            # Add text with error
            text = f"Error: {error_pixels.item():.1f}px"
            cv2.putText(orig_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Save image
            output_path = os.path.join(output_folder, f'val_{idx:06d}.jpg')
            cv2.imwrite(output_path, orig_image)
            
            # Print progress every 50 images
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(val_dataset)} images...")
    
    avg_error_pixels = total_error_pixels / len(val_dataset)
    
    print(f"\nValidation complete!")
    print(f"Average error: {avg_error_pixels:.1f} pixels")
    print(f"Saved {len(val_dataset)} images to: {output_folder}/")
    print(f"  Red circle = Prediction")
    print(f"  Green circle = Ground Truth")
    print("="*60 + "\n")
    
    return avg_error_pixels


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """
    Full training loop.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_error_px = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Error={val_error_px:.1f}px")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gaze_model.pth')
            print(f"  → New best model saved!")
    
    print("\nTraining complete!")
    return model


# ============================================================================
# STEP 5: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    # Configuration
    SUBJECT_FOLDER = "./"  # Change this to your path
    GAZE_FOLDER = "000"
    DATASET_FOLDER = "./"  # Where to save/load dataset
    BATCH_SIZE = 16
    EPOCHS = 50000
    LEARNING_RATE = 0.001
    
    # Check if we should extract data or load existing
    if not os.path.exists(os.path.join(DATASET_FOLDER, 'labels.csv')):
        print("=== EXTRACTING DATA ===")
        data = extract_fixation_data(SUBJECT_FOLDER, GAZE_FOLDER)
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
    model = SimpleGazeNet().to(device)
    
    checkpoint_path = './best_gaze_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading saved model from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("No saved model found, starting from scratch...")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    
    # ========== DIAGNOSTIC: Baseline and model checks ==========
    baseline_error = baseline_mean_predictor(val_loader, device)
    diagnose_model(model, train_loader, val_loader, device)
    
    # Train
    print("\n=== TRAINING ===")
    # model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=device)
    
    # ========== VALIDATION WITH VISUALIZATION ==========
    print("\n=== GENERATING VALIDATION VISUALIZATIONS ===")
    validate_and_visualize(model, val_dataset, device, output_folder='validation_predictions')
    
    print("\n=== DONE ===")
    print("Best model saved to: best_gaze_model.pth")
    print("Validation predictions saved to: validation_predictions/")


if __name__ == "__main__":
    main()