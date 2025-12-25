"""
Utility functions
"""

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_classes_from_dir(train_dir: str) -> List[str]:
    """Get class names from directory structure"""
    train_path = Path(train_dir)
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    return classes


def get_classes_from_submission(sample_submission_path: str) -> List[str]:
    """Get class names from sample submission"""
    df = pd.read_csv(sample_submission_path)
    return df.columns[1:].tolist()


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def setup_logging(
    output_dir: str,
    name: str = "train",
) -> logging.Logger:
    """Setup logging to file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"{name}_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def find_best_checkpoints(
    output_dir: str,
    metric: str = "logloss",
    top_k: int = 3,
) -> List[str]:
    """Find best checkpoints by metric in filename"""
    output_path = Path(output_dir)
    checkpoints = list(output_path.glob("*.pth"))
    
    # Parse metric from filename (e.g., "epoch_10_logloss_0.1234.pth")
    scored_checkpoints = []
    for ckpt in checkpoints:
        parts = ckpt.stem.split("_")
        for i, part in enumerate(parts):
            if part == metric and i + 1 < len(parts):
                try:
                    score = float(parts[i + 1])
                    scored_checkpoints.append((score, str(ckpt)))
                except ValueError:
                    pass
    
    # Sort by score (lower is better for logloss)
    scored_checkpoints.sort(key=lambda x: x[0])
    
    return [ckpt for _, ckpt in scored_checkpoints[:top_k]]


def print_config(cfg) -> str:
    """Pretty print configuration"""
    lines = ["=" * 50, "Configuration", "=" * 50]
    for key, value in vars(cfg).items():
        lines.append(f"  {key}: {value}")
    lines.append("=" * 50)
    return "\n".join(lines)
