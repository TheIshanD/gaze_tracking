"""
Neural network model for gaze prediction.
"""

import torch
import torch.nn as nn
import torchvision.models as models


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
    
    def get_loss_function(self):
        return nn.MSELoss()


class GazeNetResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()  # remove final layer
        self.backbone = base
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.regressor(x)

    def get_loss_function(self):
        # You can use MSELoss, L1Loss, or SmoothL1Loss
        return nn.MSELoss()


class FrozenResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(FrozenResNetBackbone, self).__init__()
        
        # Load pretrained ResNet (e.g., ResNet18)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the original classification head (fc layer)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Replace with identity to get feature vector
        
        # Add your regression head
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # e.g., predicting (x, y) gaze coordinates
        )

        # 🔒 Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ✅ Only the regression head remains trainable
        for param in self.regressor.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output

    def get_loss_function(self):
        return nn.MSELoss()

import torch
import torch.nn as nn

class TinyGazeNet(nn.Module):
    """
    Lightweight CNN for gaze prediction.
    Input: 224x224x3 image
    Output: (x, y) gaze coordinates normalized [0, 1]
    Designed for very small datasets (~300 fixations)
    """
    def __init__(self):
        super(TinyGazeNet, self).__init__()
        
        # Small CNN backbone
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Small regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 64),  # reduced from 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
            nn.Sigmoid()  # output normalized [0,1]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
    def get_loss_function(self):
        # SmoothL1 is robust for small datasets with noisy gaze labels
        return nn.MSELoss()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class UNetResNet18Gaze(nn.Module):
    """
    UNet-style gaze prediction network:
    - Backbone: ResNet18 pretrained on ImageNet
    - Output: heatmap of gaze probability
    - Spatial softmax converts heatmap to normalized (x, y) gaze coordinate
    """

    def __init__(self, pretrained=True, heatmap_size=(28, 28)):
        super().__init__()
        self.heatmap_size = heatmap_size

        # -------- Backbone: ResNet18 encoder --------
        backbone = resnet18(pretrained=pretrained)
        self.initial = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.encoder1 = backbone.layer1  # 64 channels
        self.encoder2 = backbone.layer2  # 128 channels
        self.encoder3 = backbone.layer3  # 256 channels
        self.encoder4 = backbone.layer4  # 512 channels

        # -------- Decoder (upsampling with activations) --------
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final heatmap output
        self.heatmap_conv = nn.Conv2d(32, 1, kernel_size=1)  # 1-channel heatmap

    def forward(self, x):
        # -------- Encoder --------
        x0 = self.initial(x)     # after conv1+bn+relu+maxpool
        x1 = self.encoder1(x0)   # 64
        x2 = self.encoder2(x1)   # 128
        x3 = self.encoder3(x2)   # 256
        x4 = self.encoder4(x3)   # 512

        # -------- Decoder with skip connections --------
        d4 = self.up4(x4)
        # Handle potential spatial dimension mismatch
        if d4.shape[2:] != x3.shape[2:]:
            d4 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d4 = d4 + x3
        
        d3 = self.up3(d4)
        if d3.shape[2:] != x2.shape[2:]:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + x2
        
        d2 = self.up2(d3)
        if d2.shape[2:] != x1.shape[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + x1
        
        d1 = self.up1(d2)  # no skip for x0

        # Heatmap output
        heatmap = self.heatmap_conv(d1)  # shape [B, 1, H, W]
        heatmap = F.interpolate(heatmap, size=self.heatmap_size, mode='bilinear', align_corners=False)

        # -------- Spatial softmax to extract gaze keypoint --------
        gaze = self.spatial_softmax_2d(heatmap)

        return gaze
        # , heatmap  # gaze: [B, 2], heatmap: [B, 1, H, W]

    @staticmethod
    def spatial_softmax_2d(heatmap):
        """
        Convert heatmap [B, 1, H, W] -> normalized (x, y) coordinates [0,1]
        """
        B, C, H, W = heatmap.shape
        assert C == 1, "Heatmap should have 1 channel"

        # Flatten spatial dimensions
        heatmap_flat = heatmap.view(B, -1)
        softmax = F.softmax(heatmap_flat, dim=1).view(B, 1, H, W)

        # Create coordinate grids
        pos_x = torch.linspace(0, 1, W, device=heatmap.device)
        pos_y = torch.linspace(0, 1, H, device=heatmap.device)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')  # HxW

        # Expected value: sum of position weighted by probability
        # Sum over spatial dimensions [1, 2] to preserve batch dimension
        exp_x = torch.sum(softmax[:, 0] * grid_x, dim=[1, 2])  # [B]
        exp_y = torch.sum(softmax[:, 0] * grid_y, dim=[1, 2])  # [B]

        gaze = torch.stack([exp_x, exp_y], dim=1)  # [B, 2]
        return gaze

    def get_loss_function(self):
        """
        MSE loss between predicted gaze keypoint and ground-truth gaze.
        """
        return nn.MSELoss()
