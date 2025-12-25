"""
Training and validation logic
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score

from config import Config
from model import (
    EMA, SWA,
    cross_entropy_with_smoothing,
    mixup_criterion,
    rdrop_loss,
    rdrop_mixup_loss,
)
from dataset import apply_mixup_cutmix


class Trainer:
    """
    Model trainer with support for:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - EMA / SWA
    - CutMix / MixUp
    - R-Drop regularization
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        cfg: Config,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        
        # AMP scaler
        self.scaler = GradScaler() if cfg.amp else None
        
        # EMA
        self.ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
        
        # SWA
        self.swa = SWA(model) if cfg.use_swa else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.history: List[Dict] = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Apply MixUp / CutMix
            images, labels1, labels2, lam = apply_mixup_cutmix(
                images, labels, self.cfg
            )
            use_mix = labels2 is not None
            
            # Forward pass with optional AMP
            with autocast(enabled=self.cfg.amp):
                if self.cfg.use_rdrop:
                    # R-Drop: two forward passes
                    logits1 = self.model(images)
                    logits2 = self.model(images)
                    
                    if use_mix:
                        loss = rdrop_mixup_loss(
                            logits1, logits2, labels1, labels2, lam,
                            alpha=self.cfg.rdrop_lambda,
                            smoothing=self.cfg.label_smoothing,
                        )
                    else:
                        loss = rdrop_loss(
                            logits1, logits2, labels,
                            alpha=self.cfg.rdrop_lambda,
                            smoothing=self.cfg.label_smoothing,
                        )
                else:
                    # Standard forward pass
                    logits = self.model(images)
                    
                    if use_mix:
                        loss = mixup_criterion(
                            logits, labels1, labels2, lam,
                            smoothing=self.cfg.label_smoothing,
                        )
                    else:
                        loss = cross_entropy_with_smoothing(
                            logits, labels,
                            smoothing=self.cfg.label_smoothing,
                        )
            
            # Gradient accumulation
            loss = loss / self.cfg.gradient_accumulation
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.cfg.gradient_accumulation == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update()
                
                # Update scheduler (if step-based)
                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Accumulate metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size * self.cfg.gradient_accumulation
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss / total_samples:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        # Update scheduler (if epoch-based)
        if not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()
        
        # Update SWA
        if self.swa is not None and epoch >= self.cfg.swa_start_epoch:
            self.swa.update()
        
        return {
            'train_loss': total_loss / total_samples,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        use_ema: bool = True,
    ) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        # Apply EMA weights for evaluation
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        all_logits = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0
        
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with autocast(enabled=self.cfg.amp):
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        # Restore original weights
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        probs = F.softmax(all_logits, dim=1).numpy()
        preds = all_logits.argmax(dim=1).numpy()
        labels_np = all_labels.numpy()
        
        val_loss = total_loss / total_samples
        val_logloss = log_loss(labels_np, probs, labels=list(range(probs.shape[1])))
        val_accuracy = accuracy_score(labels_np, preds)
        
        return {
            'val_loss': val_loss,
            'val_logloss': val_logloss,
            'val_accuracy': val_accuracy,
        }
    
    def save_checkpoint(
        self,
        path: str,
        metrics: Optional[Dict] = None,
    ):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': vars(self.cfg),
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])


def create_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Create optimizer with optional layer-wise lr decay"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    steps_per_epoch: int,
):
    """Create learning rate scheduler"""
    if cfg.scheduler == "cosine":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epochs,
            eta_min=cfg.min_lr,
        )
    elif cfg.scheduler == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=cfg.lr * 3,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch // cfg.gradient_accumulation,
            pct_start=cfg.warmup_epochs / cfg.epochs,
            anneal_strategy='cos',
            final_div_factor=cfg.lr / cfg.min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
