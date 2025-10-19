"""
Diagnostic and visualization utilities for gaze prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
    errors = torch.norm((all_gazes - mean_gaze), dim=1)
    print(f"Baseline error (always predict mean): {errors.mean().item():.6f} normalized units")
    print(f"Baseline error std: {errors.std().item():.6f} normalized units")
    print("="*60 + "\n")
    
    return errors.mean().item()


def diagnose_model(model, train_loader, val_loader, device, use_normalized=True):
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
    
    # Scale to pixel space if not normalized
    if not use_normalized:
        dims = torch.tensor([1280, 720]).to(device)
        preds_display = preds * dims
    else:
        preds_display = preds
    
    print(f"\nModel predictions (current weights):")
    print(f"  Min: {preds_display.min().item():.4f}")
    print(f"  Max: {preds_display.max().item():.4f}")
    print(f"  Mean: {preds_display.mean().item():.4f}")
    print(f"  Std: {preds_display.std().item():.4f}")
    
    # Check if predictions are stuck
    if preds_display.std().item() < 0.01:
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
    if not use_normalized:
        dims = torch.tensor([1280, 720]).to(device)
        preds_display = preds * dims
    else:
        preds_display = preds

    loss = criterion(preds_display, gazes.to(device))
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