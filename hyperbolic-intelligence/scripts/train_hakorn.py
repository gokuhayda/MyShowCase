#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright © 2026 Éric Gustavo Reis de Sena. All Rights Reserved.

"""
H-AKORN Training Script
=======================

Training script for H-AKORN Transformer with:
1. Dataset loading (using existing tokenizers)
2. Training loop with H-AKORN-specific logging
3. Phase dynamics monitoring
4. Model checkpointing
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from hakorn import (
    HAKORNTransformer,
    HAKORNLoss,
)


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_lm_loss = 0.0
    total_sync_loss = 0.0
    total_variance_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        output = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True,
        )
        
        lm_loss = output['loss']
        order_parameters = output['all_order_params']
        
        # Compute H-AKORN loss
        loss_dict = criterion(lm_loss, order_parameters)
        loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_lm_loss += loss_dict['lm']
        total_sync_loss += loss_dict['sync']
        total_variance_loss += loss_dict['variance']
        num_batches += 1
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            avg_order_param = torch.stack(order_parameters).mean().item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lm': f"{loss_dict['lm']:.4f}",
                'sync': f"{loss_dict['sync']:.4f}",
                'order': f"{avg_order_param:.4f}",
            })
    
    return {
        'total': total_loss / num_batches,
        'lm': total_lm_loss / num_batches,
        'sync': total_sync_loss / num_batches,
        'variance': total_variance_loss / num_batches,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_lm_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            output = model(
                input_ids=input_ids,
                labels=labels,
                return_dict=True,
            )
            
            lm_loss = output['loss']
            order_parameters = output['all_order_params']
            
            # Compute loss
            loss_dict = criterion(lm_loss, order_parameters)
            
            total_loss += loss_dict['total'].item()
            total_lm_loss += loss_dict['lm']
            num_batches += 1
    
    return {
        'total': total_loss / num_batches,
        'lm': total_lm_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train H-AKORN Transformer")
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--curvature', type=float, default=-1.0)
    parser.add_argument('--coupling_strength', type=float, default=1.0)
    
    # Loss arguments
    parser.add_argument('--lambda_sync', type=float, default=0.1)
    parser.add_argument('--lambda_variance', type=float, default=0.05)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_length', type=int, default=128)
    
    # Other arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating H-AKORN Transformer...")
    model = HAKORNTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        curvature=args.curvature,
        coupling_strength=args.coupling_strength,
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create loss criterion
    criterion = HAKORNLoss(
        lambda_sync=args.lambda_sync,
        lambda_variance=args.lambda_variance,
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
    )
    
    # Create dummy dataset (replace with real data)
    print("Creating dataset...")
    dummy_texts = [f"This is sample text {i}" for i in range(1000)]
    
    # Simple tokenizer (replace with real tokenizer)
    class DummyTokenizer:
        def encode(self, text):
            # Simple char-level tokenization
            return [ord(c) % args.vocab_size for c in text]
    
    tokenizer = DummyTokenizer()
    
    train_dataset = SimpleTextDataset(
        dummy_texts[:800],
        tokenizer,
        max_length=args.max_length,
    )
    
    val_dataset = SimpleTextDataset(
        dummy_texts[800:],
        tokenizer,
        max_length=args.max_length,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.log_interval
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['total']:.4f} (LM: {train_metrics['lm']:.4f}, Sync: {train_metrics['sync']:.4f})")
        print(f"  Val Loss: {val_metrics['total']:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_model_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"  Best model saved: {best_model_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
