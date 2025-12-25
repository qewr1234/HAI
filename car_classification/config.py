"""
Configuration for Car Classification
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    """Training and inference configuration"""
    
    # Paths
    train_dir: str = "./data/train"
    test_dir: str = "./data/test"
    output_dir: str = "./outputs"
    sample_submission: str = "./data/sample_submission.csv"
    
    # Model
    model_name: str = "convnext_base.fb_in22k_ft_in1k"
    num_classes: Optional[int] = None  # Auto-detected from data
    pretrained: bool = True
    
    # Training
    img_size: int = 384
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    
    # Augmentation
    cutmix_prob: float = 0.3
    mixup_prob: float = 0.3
    cutmix_alpha: float = 1.0
    mixup_alpha: float = 0.4
    
    # Regularization
    use_rdrop: bool = True
    rdrop_lambda: float = 5e-3
    dropout: float = 0.0
    
    # EMA & SWA
    use_ema: bool = True
    ema_decay: float = 0.9998
    use_swa: bool = True
    swa_start_epoch: int = 20
    swa_lr: float = 1e-5
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "onecycle"
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    
    # Validation
    val_ratio: float = 0.15
    
    # Inference
    tta_transforms: int = 5
    save_top_k: int = 3
    
    # Pseudo Labeling
    use_pseudo_label: bool = False
    pseudo_threshold: float = 0.95
    pseudo_start_epoch: int = 15
    
    # System
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True
    gradient_accumulation: int = 1
    
    # Logging
    log_interval: int = 50
    save_interval: int = 5
    
    def __post_init__(self):
        """Create output directory if not exists"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# Preset configurations
CONFIGS = {
    "base": Config(),
    
    "light": Config(
        model_name="convnext_small.fb_in22k_ft_in1k",
        img_size=224,
        batch_size=64,
        epochs=20,
        use_rdrop=False,
        use_swa=False,
        tta_transforms=3,
    ),
    
    "heavy": Config(
        model_name="convnext_large.fb_in22k_ft_in1k",
        img_size=384,
        batch_size=16,
        epochs=40,
        gradient_accumulation=2,
        tta_transforms=7,
    ),
}


def get_config(name: str = "base") -> Config:
    """Get configuration by name"""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
