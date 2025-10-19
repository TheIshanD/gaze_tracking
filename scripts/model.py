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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class UNetResNet18MultiFrameGaze(nn.Module):
    """
    Multi-frame UNet-style gaze prediction network.
    - Input: 4 RGB frames (past 3 + current)
    - Backbone: ResNet18 pretrained on ImageNet (modified for 12-channel input)
    - Output: gaze heatmap and normalized (x, y) coordinates
    """

    def __init__(self, pretrained=True, heatmap_size=(28, 28), num_frames=4):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.num_frames = num_frames
        self.in_channels = 3 * num_frames  # 4 frames × 3 channels

        # -------- Backbone: ResNet18 encoder --------
        backbone = resnet18(weights='DEFAULT')

        # Modify first conv layer to accept multi-frame input
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            self.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

        # Initialize new weights for added channels
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight  # copy pretrained RGB weights
            if self.in_channels > 3:
                for i in range(3, self.in_channels):
                    backbone.conv1.weight[:, i:i+1] = old_conv.weight[:, i % 3:i % 3 + 1]

        # Store encoder blocks
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
        self.heatmap_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        """
        Input shape: [B, num_frames, 3, H, W]
        """
        B, _, H, W = x.shape
        T = self.num_frames
        C = 3 # RGB channels
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"

        # Merge temporal + channel dimensions
        x = x.view(B, T * C, H, W)

        # -------- Encoder --------
        x0 = self.initial(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # -------- Decoder --------
        d4 = self.up4(x4)
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

        d1 = self.up1(d2)

        # -------- Heatmap & Gaze --------
        heatmap = self.heatmap_conv(d1)
        heatmap = F.interpolate(heatmap, size=self.heatmap_size, mode='bilinear', align_corners=False)
        gaze = self.spatial_softmax_2d(heatmap)

        return gaze

    @staticmethod
    def spatial_softmax_2d(heatmap):
        """
        Convert heatmap [B, 1, H, W] -> normalized (x, y)
        """
        B, C, H, W = heatmap.shape
        assert C == 1, "Heatmap should have 1 channel"

        # Flatten spatial dims
        heatmap_flat = heatmap.view(B, -1)
        softmax = F.softmax(heatmap_flat, dim=1).view(B, 1, H, W)

        # Create coordinate grids
        pos_x = torch.linspace(0, 1, W, device=heatmap.device)
        pos_y = torch.linspace(0, 1, H, device=heatmap.device)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')

        # Expected value
        exp_x = torch.sum(softmax[:, 0] * grid_x, dim=[1, 2])
        exp_y = torch.sum(softmax[:, 0] * grid_y, dim=[1, 2])
        return torch.stack([exp_x, exp_y], dim=1)

    def get_loss_function(self):
        return nn.MSELoss()
    
class LightUNetResNet18MultiFrameGaze(nn.Module):
    """
    Multi-frame U-Net-style gaze prediction network with a frozen ResNet18 encoder.
    - Input: [B, num_frames, 3, H, W]
    - Output: normalized gaze coordinates (x, y)
    """

    def __init__(self, pretrained=True, heatmap_size=(28, 28), num_frames=4, unfreeze_layers=0):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.num_frames = num_frames
        self.in_channels = 3 * num_frames

        # -------- Backbone: ResNet18 --------
        backbone = resnet18(pretrained=pretrained)

        # Modify first conv to accept multi-frame input
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            self.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Initialize new conv1 weights
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = old_conv.weight
            for i in range(3, self.in_channels):
                backbone.conv1.weight[:, i:i+1] = old_conv.weight[:, i % 3:i % 3 + 1]

        # Encoder stages
        self.initial = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.enc1 = backbone.layer1  # 64
        self.enc2 = backbone.layer2  # 128
        self.enc3 = backbone.layer3  # 256
        self.enc4 = backbone.layer4  # 512

        # -------- Lightweight decoder --------
        self.up4 = self._upsample_block(512, 128)
        self.up3 = self._upsample_block(128, 64)
        self.up2 = self._upsample_block(64, 32)

        # Final heatmap conv
        self.heatmap_head = nn.Conv2d(32, 1, kernel_size=1)

        # Freeze encoder parameters by default
        self.freeze_encoder(unfreeze_layers)

    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def freeze_encoder(self, unfreeze_layers=0):
        """
        Freezes encoder layers. You can unfreeze deeper layers by setting `unfreeze_layers > 0`.
        unfreeze_layers: 0=all frozen, 1=unfreeze layer4, 2=layer3–4, 3=layer2–4, 4=all trainable
        """
        encoders = [self.enc1, self.enc2, self.enc3, self.enc4]
        for enc in encoders:
            for param in enc.parameters():
                param.requires_grad = False

        # Selectively unfreeze last N layers
        if unfreeze_layers > 0:
            for enc in encoders[-unfreeze_layers:]:
                for param in enc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        """
        Input: [B, num_frames, 3, H, W]
        Output: [B, 2] normalized (x, y)
        """
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)

        # -------- Encoder --------
        x0 = self.initial(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # -------- Decoder --------
        d4 = self.up4(x4)
        d4 = d4 + F.interpolate(x3, size=d4.shape[2:], mode="bilinear", align_corners=False)

        d3 = self.up3(d4)
        d3 = d3 + F.interpolate(x2, size=d3.shape[2:], mode="bilinear", align_corners=False)

        d2 = self.up2(d3)
        d2 = d2 + F.interpolate(x1, size=d2.shape[2:], mode="bilinear", align_corners=False)

        # -------- Heatmap and Gaze --------
        heatmap = self.heatmap_head(d2)
        heatmap = F.interpolate(heatmap, size=self.heatmap_size, mode="bilinear", align_corners=False)
        gaze = self.spatial_softmax_2d(heatmap)
        return gaze

    @staticmethod
    def spatial_softmax_2d(heatmap):
        """Convert [B,1,H,W] → normalized (x, y)."""
        B, _, H, W = heatmap.shape
        heatmap_flat = heatmap.view(B, -1)
        softmax = F.softmax(heatmap_flat, dim=1).view(B, 1, H, W)

        pos_x = torch.linspace(0, 1, W, device=heatmap.device)
        pos_y = torch.linspace(0, 1, H, device=heatmap.device)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')

        exp_x = torch.sum(softmax[:, 0] * grid_x, dim=[1, 2])
        exp_y = torch.sum(softmax[:, 0] * grid_y, dim=[1, 2])
        return torch.stack([exp_x, exp_y], dim=1)

    def get_loss_function(self):
        return nn.MSELoss()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class UNetTemporalAttentionGaze(nn.Module):
    """
    Hybrid Temporal Fusion Gaze Model with Temporal Attention
    - Input: num_frames RGB frames (B, num_frames * 3, H, W)
    - Encoder: ResNet18 (shared across frames)
    - Temporal Fusion: Transformer attention (if num_frames > 1)
    - Decoder: UNet-style upsampling
    - Output: heatmap + gaze (via softmax) or direct (x, y) MLP regression
    """

    def __init__(self, num_frames=4, heatmap_size=(28, 28), pretrained=True, mode="heatmap"):
        super().__init__()
        assert mode in ["heatmap", "mlp"], "mode must be 'heatmap' or 'mlp'"
        self.num_frames = num_frames
        self.heatmap_size = heatmap_size
        self.mode = mode

        # ---------------- Encoder ----------------
        base = resnet18(weights='DEFAULT' if pretrained else None)
        self.encoder = nn.Sequential(*list(base.children())[:-2])  # remove avgpool, fc
        self.feature_dim = 512  # ResNet18 final conv output channels

        # ---------------- Temporal Attention (only if T > 1) ----------------
        if num_frames > 1:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim, nhead=8, dim_feedforward=1024, batch_first=True
            )
            self.temporal_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        else:
            self.temporal_attention = None  # skip transformer entirely

        # ---------------- Decoder (UNet-style) ----------------
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

        # Heatmap output head
        self.heatmap_conv = nn.Conv2d(32, 1, kernel_size=1)

        # ---------------- MLP Head (for direct gaze regression) ----------------
        self.mlp_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        """
        x: [B, T*3, H, W]
        Returns:
            if mode == "heatmap": (gaze_xy, heatmap)
            if mode == "mlp": gaze_xy
        """
        B, C, H, W = x.shape
        C_per_frame = C // self.num_frames
        x = x.view(B, self.num_frames, C_per_frame, H, W)

        # ----- Encode each frame -----
        feats = [self.encoder(x[:, t]) for t in range(self.num_frames)]  # list of [B, 512, H', W']
        feats = torch.stack(feats, dim=1)  # [B, T, 512, H', W']
        Hf, Wf = feats.shape[-2:]

        # ----- Temporal Fusion -----
        if self.temporal_attention is not None:
            # Flatten spatial dims for attention
            feats_flat = feats.flatten(3).permute(0, 3, 1, 2)  # [B, HW, T, C]
            feats_flat = feats_flat.reshape(B * Hf * Wf, self.num_frames, self.feature_dim)

            fused = self.temporal_attention(feats_flat)  # [B*HW, T, C]
            fused = fused.mean(dim=1)                    # average over time
            fused = fused.view(B, Hf, Wf, self.feature_dim).permute(0, 3, 1, 2)  # [B, C, H', W']
        else:
            # No temporal fusion → just use single frame’s features
            fused = feats[:, 0]  # [B, C, H', W']

        # ----- Output -----
        if self.mode == "mlp":
            pooled = F.adaptive_avg_pool2d(fused, 1).view(B, self.feature_dim)
            gaze = self.mlp_head(pooled)
            return gaze

        # Decode via UNet upsampling
        d4 = self.up4(fused)
        d3 = self.up3(d4)
        d2 = self.up2(d3)
        d1 = self.up1(d2)

        heatmap = self.heatmap_conv(d1)
        heatmap = F.interpolate(heatmap, size=self.heatmap_size, mode='bilinear', align_corners=False)

        gaze = self.spatial_softmax_2d(heatmap)
        return gaze

    @staticmethod
    def spatial_softmax_2d(heatmap):
        """Convert [B, 1, H, W] heatmap → normalized gaze (x, y) in [0, 1]"""
        B, _, H, W = heatmap.shape
        softmax = F.softmax(heatmap.view(B, -1), dim=1).view(B, 1, H, W)

        pos_x = torch.linspace(0, 1, W, device=heatmap.device)
        pos_y = torch.linspace(0, 1, H, device=heatmap.device)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')

        exp_x = torch.sum(softmax[:, 0] * grid_x, dim=[1, 2])
        exp_y = torch.sum(softmax[:, 0] * grid_y, dim=[1, 2])
        return torch.stack([exp_x, exp_y], dim=1)

    def get_loss_function(self):
        return nn.MSELoss()
