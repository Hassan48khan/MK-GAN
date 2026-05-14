"""
Data Loader for MK-GAN
Supports CAMUS, EchoNet-Dynamic, HMC-QU, MCE datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Callable


# ============================================================================
# Echocardiographic Dataset
# ============================================================================

class EchoDataset(Dataset):
    """
    Generic echocardiographic segmentation dataset
    Works with CAMUS, EchoNet-Dynamic, HMC-QU, and MCE
    """
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 img_size: int = 256,
                 mode: str = 'train',
                 augment: bool = True):
        """
        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing segmentation masks
            img_size: Target image size
            mode: 'train', 'val', or 'test'
            augment: Whether to apply augmentation
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.mode = mode
        self.augment = augment and (mode == 'train')

        # Collect image paths
        self.image_paths = sorted([
            p for p in self.image_dir.glob('*')
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"[{mode.upper()}] Found {len(self.image_paths)} images in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load grayscale image"""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load and binarize mask"""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        return mask

    def _augment_pair(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentation to image-mask pair"""
        # Random horizontal flip
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Random rotation (-15 to +15 degrees)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        # Random brightness/contrast (for images only)
        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-10, 10)
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        return image, mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [-1, 1]"""
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # Apply augmentation
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        # Normalize image
        image = self._normalize(image)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        image_tensor = image_tensor.repeat(3, 1, 1)  # (3, H, W) grayscale to RGB

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        # Replicate to 3 channels to match generator output
        mask_tensor = mask_tensor.repeat(3, 1, 1)

        return image_tensor, mask_tensor


# ============================================================================
# CAMUS Dataset (Multi-class: LVendo, LVmyo, LAtr)
# ============================================================================

class CAMUSDataset(Dataset):
    """
    CAMUS dataset with three structures: LVendo, LVmyo, LAtr
    Handles 2CH and 4CH views with ED and ES phases
    """
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 img_size: int = 256,
                 mode: str = 'train',
                 view: str = 'both',  # '2CH', '4CH', or 'both'
                 phase: str = 'both',  # 'ED', 'ES', or 'both'
                 augment: bool = True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.mode = mode
        self.view = view
        self.phase = phase
        self.augment = augment and (mode == 'train')

        # Filter by view and phase
        all_paths = sorted(self.image_dir.glob('*.png'))
        self.image_paths = self._filter_paths(all_paths)

        print(f"[{mode.upper()}] CAMUS: {len(self.image_paths)} images "
              f"(view={view}, phase={phase})")

    def _filter_paths(self, paths):
        """Filter paths by view and phase"""
        filtered = []
        for p in paths:
            name = p.stem.lower()

            # Filter by view
            if self.view == '2CH' and '2ch' not in name:
                continue
            if self.view == '4CH' and '4ch' not in name:
                continue

            # Filter by phase
            if self.phase == 'ED' and 'ed' not in name:
                continue
            if self.phase == 'ES' and 'es' not in name:
                continue

            filtered.append(p)

        return filtered

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Load multi-class mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_NEAREST)

        # Separate into 3 channels: LVendo, LVmyo, LAtr
        mask_lvendo = (mask == 1).astype(np.float32)
        mask_lvmyo = (mask == 2).astype(np.float32)
        mask_latr = (mask == 3).astype(np.float32)

        mask_3ch = np.stack([mask_lvendo, mask_lvmyo, mask_latr], axis=0)

        # Augmentation
        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                mask_3ch = mask_3ch[:, :, ::-1].copy()

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        # Convert to tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        mask_tensor = torch.from_numpy(mask_3ch)

        return image_tensor, mask_tensor


# ============================================================================
# Helper Functions
# ============================================================================

def get_dataloader(image_dir: str,
                   mask_dir: str,
                   batch_size: int = 1,
                   img_size: int = 256,
                   num_workers: int = 4,
                   mode: str = 'train',
                   shuffle: bool = None,
                   dataset_type: str = 'standard') -> DataLoader:
    """
    Create DataLoader for echocardiographic data

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        batch_size: Batch size
        img_size: Image size (default 256)
        num_workers: Number of worker processes
        mode: 'train', 'val', or 'test'
        shuffle: Whether to shuffle (auto-set based on mode if None)
        dataset_type: 'standard' or 'camus' for multi-class

    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (mode == 'train')

    if dataset_type == 'camus':
        dataset = CAMUSDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            img_size=img_size,
            mode=mode
        )
    else:
        dataset = EchoDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            img_size=img_size,
            mode=mode
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )

    return dataloader


if __name__ == "__main__":
    # Quick test
    print("DataLoader module loaded successfully!")
    print("Available datasets: EchoDataset, CAMUSDataset")
    print("Helper function: get_dataloader()")
