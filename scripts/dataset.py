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

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2
from PIL import Image

class MultiFrameGazeDataset(Dataset):
    """
    PyTorch Dataset for gaze prediction using the current frame and the past (num_frames - 1) frames.
    Each sample returns a (num_frames * 3)-channel tensor and a normalized gaze coordinate (x, y).
    """
    def __init__(self, data, num_frames=4, transform=None, load_from_disk=False):
        """
        Args:
            data: list of dicts with keys {"frame_number", "gaze_x", "gaze_y", "image_path" or "frame"}
            num_frames: total number of frames per sample (default=4: past 3 + current)
            transform: torchvision transform applied per frame
            load_from_disk: whether to read images from disk (True) or use in-memory arrays (False)
        """
        self.data = sorted(data, key=lambda x: x["frame_number"])  # ensure chronological order
        self.num_frames = num_frames
        self.load_from_disk = load_from_disk

        if not self.load_from_disk:
            for item in self.data:
                if "image_path" not in item:
                    raise ValueError(f"Item missing 'image_path': {item}")
                frame = cv2.imread(item["image_path"])
                if frame is None:
                    raise ValueError(f"Failed to load image: {item['image_path']}")
                # Convert to RGB upfront to save memory and conversion time
                item["frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # Default transform: resize and normalize (applied per frame)
        if transform is None:
            self.base_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.base_transform = transform

    def __len__(self):
        return len(self.data)

    def _load_image(self, item):
        if self.load_from_disk:
            image = cv2.imread(item["image_path"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = item["frame"]  # already RGB
        return Image.fromarray(image)

    def _get_past_frames(self, idx):
        """
        Collect up to (num_frames - 1) past frames.
        If not enough valid past frames (start or cuts), repeat the earliest available frame.
        """
        current_item = self.data[idx]
        current_frame_num = current_item["frame_number"]
        past_frames = [current_item]
        last_valid = current_item

        for offset in range(1, self.num_frames):
            prev_idx = idx - offset
            if prev_idx < 0:
                # Start of dataset — pad with earliest available valid frame
                past_frames.append(last_valid)
                continue

            prev_item = self.data[prev_idx]
            if abs(current_frame_num - prev_item["frame_number"]) > self.num_frames:
                # Too far back — reuse last valid (either current or most recent)
                past_frames.append(last_valid)
            else:
                past_frames.append(prev_item)
                last_valid = prev_item

        # Reverse to chronological order (oldest → newest)
        past_frames = list(reversed(past_frames))
        return past_frames

    def __getitem__(self, idx):
        # Get N chronological frames (past + current)
        frame_items = self._get_past_frames(idx)

        transformed_frames = []
        for item in frame_items:
            img = self._load_image(item)
            img_t = self.base_transform(img)
            transformed_frames.append(img_t)

        # Stack into (num_frames * 3)-channel tensor
        stacked = torch.cat(transformed_frames, dim=0)  # [3*num_frames, H, W]

        # Gaze label (same for all frames)
        gaze = torch.tensor(
            [frame_items[-1]["gaze_x"], frame_items[-1]["gaze_y"]],
            dtype=torch.float32
        )

        return stacked, gaze
