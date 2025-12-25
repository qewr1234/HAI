#!/usr/bin/env python3
"""
Main training script for Car Classification

Usage:
    python train.py --config base
    python train.py --config light --epochs 15
    python train.py --train_dir ./data/train --test_dir ./data/test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from config import Config, get_config
from dataset import CarDataset, get_val_transform, create_dataloaders
from model import create_model
from trainer import Trainer, create_optimizer, create_scheduler
from inference import run_inference
from utils import (
    seed_everything,
    get_device,
    get_classes_from_dir,
    get_classes_from_submission,
    count_parameters,
    format_time,
    setup_logging,
    print_config,
    EarlyStopping,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Car Classification Training")
    
    # Config preset
    parser.add_argument("--config", type=str, default="base",
                        help="Config preset: base, light, heavy")
    
    # Override config
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sample_submission", type=str, default=None)
    
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--no_rdrop", action="store_true")
    
    # Training control
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation")
    
    return parser.parse_args()


def update_config(cfg: Config, args) -> Config:
    """Update config with command line arguments"""
    if args.train_dir:
        cfg.train_dir = args.train_dir
    if args.test_dir:
        cfg.test_dir = args.test_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.sample_submission:
        cfg.sample_submission = args.sample_submission
    
    if args.model_name:
        cfg.model_name = args.model_name
    if args.img_size:
        cfg.img_size = args.img_size
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    
    if args.seed:
        cfg.seed = args.seed
    if args.no_amp:
        cfg.amp = False
    if args.no_ema:
        cfg.use_ema = False
    if args.no_rdrop:
        cfg.use_rdrop = False
    
    return cfg


def train(cfg: Config):
    """Main training function"""
    
    # Setup
    seed_everything(cfg.seed)
    device = get_device()
    logger = setup_logging(cfg.output_dir, "train")
    
    logger.info(print_config(cfg))
    
    # Get classes
    if Path(cfg.sample_submission).exists():
        classes = get_classes_from_submission(cfg.sample_submission)
    else:
        classes = get_classes_from_dir(cfg.train_dir)
    
    cfg.num_classes = len(classes)
    logger.info(f"Number of classes: {cfg.num_classes}")
    
    # Save classes
    np.save(Path(cfg.output_dir) / "classes.npy", np.array(classes))
    
    # Create dataset for splitting
    full_dataset = CarDataset(
        root=cfg.train_dir,
        classes=classes,
        transform=get_val_transform(cfg),
        is_test=False,
    )
    labels = full_dataset.get_labels()
    logger.info(f"Total samples: {len(labels)}")
    
    # Stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
    )
    train_idx, val_idx = next(splitter.split(range(len(labels)), labels))
    
    logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        cfg, train_idx, val_idx, classes
    )
    
    # Create model
    model = create_model(cfg, num_classes=len(classes))
    model = model.to(device)
    
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Parameters: {count_parameters(model):,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))
    
    # Create trainer
    trainer = Trainer(model, optimizer, scheduler, cfg, device)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, mode="min")
    
    # Training loop
    best_checkpoints = []
    start_time = time.time()
    
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.info(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val LogLoss: {val_metrics['val_logloss']:.4f} | "
            f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
            f"LR: {train_metrics['lr']:.2e} | "
            f"Time: {format_time(epoch_time)}"
        )
        
        # Save checkpoint
        val_logloss = val_metrics['val_logloss']
        ckpt_name = f"epoch_{epoch:02d}_logloss_{val_logloss:.4f}.pth"
        ckpt_path = Path(cfg.output_dir) / ckpt_name
        
        trainer.save_checkpoint(str(ckpt_path), val_metrics)
        
        # Track best checkpoints
        best_checkpoints.append((val_logloss, str(ckpt_path)))
        best_checkpoints.sort(key=lambda x: x[0])
        best_checkpoints = best_checkpoints[:cfg.save_top_k]
        
        # Save best model
        if val_logloss < trainer.best_loss:
            trainer.best_loss = val_logloss
            best_path = Path(cfg.output_dir) / "best_model.pth"
            trainer.save_checkpoint(str(best_path), val_metrics)
            logger.info(f"✅ New best model saved (logloss: {val_logloss:.4f})")
        
        # Early stopping
        if early_stopping(val_logloss):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {format_time(total_time)}")
    logger.info(f"Best validation logloss: {trainer.best_loss:.4f}")
    
    # Return best checkpoint paths
    return [ckpt for _, ckpt in best_checkpoints], classes


def main():
    args = parse_args()
    
    # Load and update config
    cfg = get_config(args.config)
    cfg = update_config(cfg, args)
    
    # Create output directory
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.eval_only:
        # Load classes
        classes = np.load(Path(cfg.output_dir) / "classes.npy").tolist()
        
        # Find best checkpoints
        from utils import find_best_checkpoints
        checkpoints = find_best_checkpoints(cfg.output_dir, top_k=cfg.save_top_k)
        
        if not checkpoints:
            print("No checkpoints found!")
            return
        
        print(f"Using checkpoints: {checkpoints}")
    else:
        # Train
        checkpoints, classes = train(cfg)
    
    # Inference
    print("\n" + "=" * 50)
    print("Running inference...")
    print("=" * 50)
    
    submission_path = Path(cfg.output_dir) / "submission.csv"
    
    run_inference(
        cfg=cfg,
        checkpoint_paths=checkpoints,
        classes=classes,
        output_path=str(submission_path),
        sample_submission_path=cfg.sample_submission if Path(cfg.sample_submission).exists() else None,
    )
    
    print(f"\n✅ Done! Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
