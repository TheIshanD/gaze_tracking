"""
Visualization utilities for model predictions.
"""

import os
import torch
import cv2

def validate_and_visualize(model, val_dataset, device, output_folder='validation_predictions', use_normalized=True):
    """
    Validate the model and save visualizations of predictions.
    
    Args:
        model: trained model
        val_dataset: validation dataset
        device: torch device
        output_folder: folder to save visualization images
        use_normalized: If True, error reported in normalized space. If False, in pixel space.
    """
    print("\n" + "="*60)
    print("VALIDATION WITH VISUALIZATION")
    print("="*60)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving predictions to: {output_folder}/")
    
    model.eval()
    total_error = 0
    
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
            if use_normalized:
                error = torch.norm(gaze_pred - gaze_target)
            else:
                error = torch.norm((gaze_pred - gaze_target) * torch.tensor([w, h]))
            total_error += error.item()
            
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
            if use_normalized:
                text = f"Error: {error.item():.4f} (normalized units)"
            else:
                text = f"Error: {error.item():.1f}px"
            cv2.putText(orig_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Save image
            output_path = os.path.join(output_folder, f'val_{idx:06d}.jpg')
            cv2.imwrite(output_path, orig_image)
            
            # Print progress every 50 images
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(val_dataset)} images...")
    
    avg_error = total_error / len(val_dataset)
    
    print(f"\nValidation complete!")
    if use_normalized:
        print(f"Average error: {avg_error:.4f} (normalized units)")
    else:
        print(f"Average error: {avg_error:.1f} pixels")
    print(f"Saved {len(val_dataset)} images to: {output_folder}/")
    print(f"  Red circle = Prediction")
    print(f"  Green circle = Ground Truth")
    print("="*60 + "\n")
    
    return avg_error