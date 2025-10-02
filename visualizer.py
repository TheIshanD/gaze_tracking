import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

# ============================================================================
# MODEL DEFINITION (must match training script)
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
# DATASET CLASS
# ============================================================================

class GazeDataset(Dataset):
    """
    PyTorch Dataset for gaze prediction.
    """
    def __init__(self, data, transform=None):
        self.data = data
        
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
        image = cv2.imread(item['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(image)
        image = self.transform(image)
        
        # Get gaze label (if available)
        if 'gaze_x' in item and 'gaze_y' in item:
            gaze = torch.tensor([item['gaze_x'], item['gaze_y']], dtype=torch.float32)
        else:
            gaze = torch.tensor([0.0, 0.0], dtype=torch.float32)  # Dummy values
        
        return image, gaze


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def load_dataset_from_folder(images_folder, labels_csv=None):
    """
    Load dataset from images folder and optional labels CSV.
    
    Args:
        images_folder: path to folder containing images
        labels_csv: optional path to labels.csv file with ground truth
    
    Returns:
        list of dicts with image_path and optionally gaze_x, gaze_y
    """
    print(f"Loading images from: {images_folder}")
    
    data = []
    
    # Check if labels CSV exists
    has_labels = False
    labels_dict = {}
    if labels_csv and os.path.exists(labels_csv):
        print(f"Loading labels from: {labels_csv}")
        labels_df = pd.read_csv(labels_csv)
        has_labels = True
        
        # Create a dictionary for quick lookup
        for _, row in labels_df.iterrows():
            labels_dict[row['image_path']] = {
                'gaze_x': row['gaze_x'],
                'gaze_y': row['gaze_y']
            }
    else:
        print("No labels.csv found - will only show predictions")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_folder) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {len(image_files)} images")
    
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        
        item = {'image_path': img_path}
        
        # Add labels if available
        if has_labels and img_path in labels_dict:
            item['gaze_x'] = labels_dict[img_path]['gaze_x']
            item['gaze_y'] = labels_dict[img_path]['gaze_y']
        
        data.append(item)
    
    return data, has_labels


def visualize_predictions(model, dataset, device, output_folder='predictions', has_labels=False):
    """
    Generate predictions and save visualizations.
    
    Args:
        model: trained model
        dataset: dataset to visualize
        device: torch device
        output_folder: folder to save visualization images
        has_labels: whether ground truth labels are available
    """
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving predictions to: {output_folder}/")
    
    model.eval()
    total_error_pixels = 0
    valid_error_count = 0
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            # Get single sample
            image_tensor, gaze_target = dataset[idx]
            
            # Get original image for visualization
            item = dataset.data[idx]
            orig_image = cv2.imread(item['image_path'])
            
            # Get image dimensions
            h, w = orig_image.shape[:2]
            
            # Make prediction
            image_batch = image_tensor.unsqueeze(0).to(device)
            gaze_pred = model(image_batch).squeeze(0).cpu()
            
            # Convert normalized coordinates to pixel coordinates
            pred_x = int(gaze_pred[0].item() * w)
            pred_y = int(gaze_pred[1].item() * h)
            
            # Draw prediction on image (red circle)
            cv2.circle(orig_image, (pred_x, pred_y), radius=8, color=(0, 0, 255), thickness=-1)
            cv2.circle(orig_image, (pred_x, pred_y), radius=10, color=(0, 0, 255), thickness=2)
            
            # Add prediction text
            pred_text = f"Pred: ({gaze_pred[0].item():.3f}, {gaze_pred[1].item():.3f})"
            cv2.putText(orig_image, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If labels available, draw ground truth and calculate error
            if has_labels and 'gaze_x' in item:
                gt_x = int(gaze_target[0].item() * w)
                gt_y = int(gaze_target[1].item() * h)
                
                # Draw ground truth (green circle)
                cv2.circle(orig_image, (gt_x, gt_y), radius=8, color=(0, 255, 0), thickness=-1)
                cv2.circle(orig_image, (gt_x, gt_y), radius=10, color=(0, 255, 0), thickness=2)
                
                # Calculate error
                error_pixels = torch.norm((gaze_pred - gaze_target) * torch.tensor([w, h]))
                total_error_pixels += error_pixels.item()
                valid_error_count += 1
                
                # Add error text
                error_text = f"Error: {error_pixels.item():.1f}px"
                cv2.putText(orig_image, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Add ground truth text
                gt_text = f"GT: ({gaze_target[0].item():.3f}, {gaze_target[1].item():.3f})"
                cv2.putText(orig_image, gt_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Save image
            img_name = os.path.basename(item['image_path'])
            output_path = os.path.join(output_folder, f'pred_{img_name}')
            cv2.imwrite(output_path, orig_image)
            
            # Print progress every 50 images
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)} images...")
    
    print(f"\nVisualization complete!")
    print(f"Saved {len(dataset)} images to: {output_folder}/")
    print(f"  Red circle = Prediction")
    
    if has_labels and valid_error_count > 0:
        avg_error_pixels = total_error_pixels / valid_error_count
        print(f"  Green circle = Ground Truth")
        print(f"\nAverage error: {avg_error_pixels:.1f} pixels")
    else:
        print(f"  (No ground truth labels available)")
    
    print("="*60 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - visualize predictions only.
    """
    # Configuration
    MODEL_PATH = './best_gaze_model.pth'
    IMAGES_FOLDER = './images'
    LABELS_CSV = './labels.csv'  # Optional
    OUTPUT_FOLDER = './predictions'
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please make sure 'best_gaze_model.pth' exists in the current directory.")
        return
    
    # Check if images folder exists
    if not os.path.exists(IMAGES_FOLDER):
        print(f"ERROR: Images folder not found at {IMAGES_FOLDER}")
        print("Please make sure './images' folder exists in the current directory.")
        return
    
    # Load dataset
    data, has_labels = load_dataset_from_folder(IMAGES_FOLDER, LABELS_CSV)
    
    if len(data) == 0:
        print("ERROR: No images found in the images folder!")
        return
    
    # Create dataset
    dataset = GazeDataset(data)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== LOADING MODEL (device: {device}) ===")
    model = SimpleGazeNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded successfully from: {MODEL_PATH}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    
    # Generate predictions
    visualize_predictions(model, dataset, device, OUTPUT_FOLDER, has_labels)
    
    print("\n=== DONE ===")
    print(f"All predictions saved to: {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()