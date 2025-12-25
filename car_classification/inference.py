"""
Inference utilities with Test-Time Augmentation (TTA)
"""

from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import Config
from model import create_model, load_checkpoint
from dataset import CarDataset, get_val_transform, get_tta_transforms


class Inferencer:
    """
    Inference with support for:
    - Test-Time Augmentation (TTA)
    - Multi-checkpoint ensemble
    """
    
    def __init__(
        self,
        cfg: Config,
        classes: List[str],
        device: torch.device,
    ):
        self.cfg = cfg
        self.classes = classes
        self.device = device
        self.models: List[nn.Module] = []
    
    def load_models(self, checkpoint_paths: List[str]):
        """Load multiple checkpoints for ensemble"""
        self.models = []
        
        for ckpt_path in checkpoint_paths:
            model = create_model(self.cfg, num_classes=len(self.classes))
            model = load_checkpoint(model, ckpt_path, self.device)
            model.to(self.device)
            model.eval()
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    @torch.no_grad()
    def predict_single(
        self,
        dataloader: DataLoader,
        model: nn.Module,
    ) -> np.ndarray:
        """Predict with single model"""
        all_probs = []
        
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            if isinstance(batch, (list, tuple)):
                # Test dataset returns (filename, image)
                _, images = batch
            else:
                images = batch
            
            images = images.to(self.device)
            
            with autocast(enabled=self.cfg.amp):
                logits = model(images)
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        
        return np.vstack(all_probs)
    
    @torch.no_grad()
    def predict_with_tta(
        self,
        model: nn.Module,
    ) -> tuple:
        """Predict with Test-Time Augmentation"""
        tta_transforms = get_tta_transforms(self.cfg)
        
        all_probs = None
        all_ids = None
        
        for i, transform in enumerate(tta_transforms):
            print(f"TTA {i+1}/{len(tta_transforms)}")
            
            dataset = CarDataset(
                root=self.cfg.test_dir,
                classes=self.classes,
                transform=transform,
                is_test=True,
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
            )
            
            probs = []
            ids = []
            
            for filenames, images in tqdm(dataloader, leave=False):
                images = images.to(self.device)
                
                with autocast(enabled=self.cfg.amp):
                    logits = model(images)
                
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
                ids.extend(filenames)
            
            probs = np.vstack(probs)
            
            if all_probs is None:
                all_probs = probs
                all_ids = ids
            else:
                all_probs += probs
        
        # Average TTA predictions
        all_probs /= len(tta_transforms)
        
        return all_ids, all_probs
    
    def predict_ensemble(self) -> tuple:
        """Predict with model ensemble + TTA"""
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")
        
        ensemble_probs = None
        all_ids = None
        
        for i, model in enumerate(self.models):
            print(f"\nModel {i+1}/{len(self.models)}")
            
            ids, probs = self.predict_with_tta(model)
            
            if ensemble_probs is None:
                ensemble_probs = probs
                all_ids = ids
            else:
                ensemble_probs += probs
        
        # Average ensemble predictions
        ensemble_probs /= len(self.models)
        
        return all_ids, ensemble_probs
    
    def create_submission(
        self,
        ids: List[str],
        probs: np.ndarray,
        output_path: str,
        sample_submission_path: Optional[str] = None,
    ):
        """Create submission CSV"""
        # Create DataFrame
        df = pd.DataFrame(probs, columns=self.classes)
        df.insert(0, "ID", ids)
        
        # Reorder to match sample submission if provided
        if sample_submission_path is not None:
            sample = pd.read_csv(sample_submission_path)
            df = df.set_index("ID").loc[sample["ID"]].reset_index()
        
        # Save
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        
        return df


def run_inference(
    cfg: Config,
    checkpoint_paths: List[str],
    classes: List[str],
    output_path: str,
    sample_submission_path: Optional[str] = None,
):
    """
    Run full inference pipeline
    
    Args:
        cfg: Configuration
        checkpoint_paths: List of checkpoint paths for ensemble
        classes: List of class names
        output_path: Output CSV path
        sample_submission_path: Optional sample submission for column ordering
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    inferencer = Inferencer(cfg, classes, device)
    inferencer.load_models(checkpoint_paths)
    
    ids, probs = inferencer.predict_ensemble()
    
    submission = inferencer.create_submission(
        ids, probs, output_path, sample_submission_path
    )
    
    return submission
