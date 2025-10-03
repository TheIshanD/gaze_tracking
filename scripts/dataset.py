"""
PyTorch Dataset class for gaze prediction.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image


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