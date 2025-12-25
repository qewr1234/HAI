"""
Model definition and utilities
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import Config


class ClassificationModel(nn.Module):
    """
    Image classification model using timm backbone
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head"""
        return self.backbone.forward_features(x)


class EMA:
    """
    Exponential Moving Average for model weights
    
    Usage:
        ema = EMA(model, decay=0.9998)
        
        # During training
        optimizer.step()
        ema.update()
        
        # During evaluation
        ema.apply_shadow()
        evaluate(model)
        ema.restore()
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state for saving"""
        return {
            'shadow': self.shadow,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state"""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class SWA:
    """
    Stochastic Weight Averaging
    
    Usage:
        swa = SWA(model)
        
        # After each epoch (starting from swa_start_epoch)
        swa.update()
        
        # At the end of training
        swa.apply()
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.swa_model = copy.deepcopy(model)
        self.n_averaged = 0
        
        # Initialize with zeros
        for param in self.swa_model.parameters():
            param.data.zero_()
    
    def update(self):
        """Update SWA weights"""
        self.n_averaged += 1
        for swa_param, param in zip(
            self.swa_model.parameters(), 
            self.model.parameters()
        ):
            swa_param.data += (param.data - swa_param.data) / self.n_averaged
    
    def apply(self):
        """Apply SWA weights to model"""
        for param, swa_param in zip(
            self.model.parameters(),
            self.swa_model.parameters()
        ):
            param.data = swa_param.data.clone()
    
    def get_model(self) -> nn.Module:
        """Get SWA model"""
        return self.swa_model


# ==================== Loss Functions ====================

def cross_entropy_with_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Cross entropy loss with label smoothing"""
    return F.cross_entropy(logits, targets, label_smoothing=smoothing)


def mixup_criterion(
    logits: torch.Tensor,
    targets1: torch.Tensor,
    targets2: torch.Tensor,
    lam: float,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Loss for MixUp/CutMix"""
    loss1 = F.cross_entropy(logits, targets1, label_smoothing=smoothing)
    loss2 = F.cross_entropy(logits, targets2, label_smoothing=smoothing)
    return lam * loss1 + (1 - lam) * loss2


def rdrop_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 5e-3,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    R-Drop loss: Cross entropy + KL divergence between two forward passes
    
    Paper: https://arxiv.org/abs/2106.14448
    """
    ce_loss = 0.5 * (
        F.cross_entropy(logits1, targets, label_smoothing=smoothing) +
        F.cross_entropy(logits2, targets, label_smoothing=smoothing)
    )
    
    p1 = F.log_softmax(logits1, dim=1)
    p2 = F.softmax(logits2, dim=1)
    q1 = F.log_softmax(logits2, dim=1)
    q2 = F.softmax(logits1, dim=1)
    
    kl_loss = 0.5 * (
        F.kl_div(p1, p2, reduction='batchmean') +
        F.kl_div(q1, q2, reduction='batchmean')
    )
    
    return ce_loss + alpha * kl_loss


def rdrop_mixup_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    targets1: torch.Tensor,
    targets2: torch.Tensor,
    lam: float,
    alpha: float = 5e-3,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """R-Drop loss combined with MixUp/CutMix"""
    ce_loss = 0.5 * (
        mixup_criterion(logits1, targets1, targets2, lam, smoothing) +
        mixup_criterion(logits2, targets1, targets2, lam, smoothing)
    )
    
    p1 = F.log_softmax(logits1, dim=1)
    p2 = F.softmax(logits2, dim=1)
    kl_loss = F.kl_div(p1, p2, reduction='batchmean')
    
    return ce_loss + alpha * kl_loss


# ==================== Model Factory ====================

def create_model(cfg: Config, num_classes: int) -> nn.Module:
    """Create model from config"""
    model = ClassificationModel(
        model_name=cfg.model_name,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
    )
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """Load model checkpoint"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=strict)
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    ema: Optional[EMA] = None,
    **kwargs,
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs,
    }
    
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    torch.save(checkpoint, path)
