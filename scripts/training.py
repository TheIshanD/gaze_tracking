"""
Training and validation functions for gaze prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    for images, gaze_targets in train_loader:
        images = images.to(device, non_blocking=True)
        gaze_targets = gaze_targets.to(device, non_blocking=True)
        
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


def validate(model, val_loader, criterion, device, use_normalized=True):
    """
    Validate the model.
    
    Args:
        use_normalized: If True, coordinates are in normalized space (0-1).
                       If False, coordinates are in pixel space.
    """
    model.eval()
    total_loss = 0
    total_error_pixels = 0
    
    dims = torch.tensor([1280, 720]).to(device)
    
    with torch.no_grad():
        for images, gaze_targets in val_loader:
            images = images.to(device)
            gaze_targets = gaze_targets.to(device)
            
            gaze_pred = model(images)
            
            if use_normalized:
                # Loss and error calculation in normalized space
                loss = criterion(gaze_pred, gaze_targets)
                
                # Error reported in normalized space
                error_pixels = torch.norm(gaze_pred - gaze_targets, dim=1)
            else:
                # Denormalize predictions and targets to pixel space
                gaze_pred_pixels = gaze_pred * dims
                gaze_targets_pixels = gaze_targets * dims
                
                # Loss and error calculation in pixel space
                loss = criterion(gaze_pred_pixels, gaze_targets_pixels)
                
                # Error already in pixel space
                error_pixels = torch.norm(gaze_pred_pixels - gaze_targets_pixels, dim=1)
            
            total_loss += loss.item()
            total_error_pixels += error_pixels.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    avg_error_pixels = total_error_pixels / len(val_loader.dataset)
    
    return avg_loss, avg_error_pixels

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda', best_model_file_name='best_gaze_model.pth', use_normalized=True, criterion=nn.MSELoss()):
    """
    Full training loop.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')

    error_units = " normalized units" if use_normalized else "px"
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_error_px = validate(model, val_loader, criterion, device, use_normalized=use_normalized)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.6f}, "
              f"Val Loss={val_loss:.6f}, "
              f"Val Error={val_error_px:.6f}{error_units}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_file_name)
            print(f"  → New best model saved!")
    
    print("\nTraining complete!")
    return model