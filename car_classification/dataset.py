"""
Dataset and data augmentation utilities
"""

import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import Config


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CarDataset(Dataset):
    """
    Car classification dataset
    
    Expected directory structure:
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                ...
        test/
            img1.jpg
            img2.jpg
    """
    
    def __init__(
        self,
        root: str,
        classes: List[str],
        transform: Optional[Callable] = None,
        is_test: bool = False,
    ):
        self.root = Path(root)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.is_test = is_test
        self.samples: List[Tuple[Path, int]] = []
        
        self._load_samples()
    
    def _load_samples(self):
        """Load image paths and labels"""
        if self.is_test:
            # Test: just image paths, no labels
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for p in sorted(self.root.glob(ext)):
                    self.samples.append((p, -1))
        else:
            # Train: class folder structure
            for cls_name in self.classes:
                cls_dir = self.root / cls_name
                if not cls_dir.exists():
                    continue
                label = self.class_to_idx[cls_name]
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for p in sorted(cls_dir.glob(ext)):
                        self.samples.append((p, label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image on error
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return path.stem, image
        return image, label
    
    def get_labels(self) -> List[int]:
        """Get all labels for stratified split"""
        return [label for _, label in self.samples]


def get_train_transform(cfg: Config) -> Callable:
    """Training augmentation pipeline"""
    return T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.1, scale=(0.02, 0.2)),
    ])


def get_val_transform(cfg: Config) -> Callable:
    """Validation transform (no augmentation)"""
    return T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_tta_transforms(cfg: Config) -> List[Callable]:
    """Test-time augmentation transforms"""
    size = cfg.img_size
    base = T.Resize((size, size))
    
    transforms_list = [
        # Original
        T.Compose([base, T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # Horizontal flip
        T.Compose([base, T.Lambda(TF.hflip), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # Vertical flip
        T.Compose([base, T.Lambda(TF.vflip), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # Rotation +10
        T.Compose([base, T.Lambda(lambda x: TF.rotate(x, 10)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # Rotation -10
        T.Compose([base, T.Lambda(lambda x: TF.rotate(x, -10)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # Brightness up
        T.Compose([base, T.Lambda(lambda x: TF.adjust_brightness(x, 1.1)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        # Brightness down
        T.Compose([base, T.Lambda(lambda x: TF.adjust_brightness(x, 0.9)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    ]
    
    return transforms_list[:cfg.tta_transforms]


# ==================== CutMix & MixUp ====================

def rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
    """Generate random bounding box for CutMix"""
    cut_ratio = math.sqrt(1 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    
    cx = random.randint(0, width)
    cy = random.randint(0, height)
    
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(width, cx + cut_w // 2)
    y2 = min(height, cy + cut_h // 2)
    
    return x1, y1, x2, y2


def cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation
    
    Args:
        images: (B, C, H, W) tensor
        labels: (B,) tensor
        alpha: Beta distribution parameter
    
    Returns:
        mixed_images, labels1, labels2, lambda
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)
    
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    
    _, _, H, W = images.shape
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    
    images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
    
    # Adjust lambda based on actual area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    
    return images, labels, shuffled_labels, lam


def mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp augmentation
    
    Args:
        images: (B, C, H, W) tensor
        labels: (B,) tensor
        alpha: Beta distribution parameter
    
    Returns:
        mixed_images, labels1, labels2, lambda
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)
    
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    
    mixed_images = lam * images + (1 - lam) * shuffled_images
    
    return mixed_images, labels, shuffled_labels, lam


def apply_mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float]:
    """
    Randomly apply CutMix or MixUp
    
    Returns:
        images, labels1, labels2 (or None), lambda (1.0 if no mix)
    """
    r = random.random()
    
    if r < cfg.cutmix_prob:
        return cutmix(images, labels, cfg.cutmix_alpha)
    elif r < cfg.cutmix_prob + cfg.mixup_prob:
        return mixup(images, labels, cfg.mixup_alpha)
    else:
        return images, labels, None, 1.0


# ==================== DataLoader Factory ====================

def create_dataloaders(
    cfg: Config,
    train_indices: List[int],
    val_indices: List[int],
    classes: List[str],
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Create full datasets
    train_dataset = CarDataset(
        root=cfg.train_dir,
        classes=classes,
        transform=get_train_transform(cfg),
        is_test=False,
    )
    
    val_dataset = CarDataset(
        root=cfg.train_dir,
        classes=classes,
        transform=get_val_transform(cfg),
        is_test=False,
    )
    
    # Filter by indices
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    
    return train_loader, val_loader


def create_test_loader(
    cfg: Config,
    classes: List[str],
    transform: Optional[Callable] = None,
) -> DataLoader:
    """Create test dataloader"""
    if transform is None:
        transform = get_val_transform(cfg)
    
    test_dataset = CarDataset(
        root=cfg.test_dir,
        classes=classes,
        transform=transform,
        is_test=True,
    )
    
    return DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
