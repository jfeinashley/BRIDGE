"""
VLM Trainer with curriculum stages (A, B, C)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
from pathlib import Path
from typing import Optional, Dict
from .losses import VLMLosses


class VLMTrainer:
    """Trainer for Cross-Modal VLM with curriculum learning"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training configuration
        self.training_config = config.get('training', {})
        self.stage = self.training_config.get('stage', 'A')
        self.num_epochs = self.training_config.get('num_epochs', 30)
        self.gradient_clip_norm = self.training_config.get('gradient_clip_norm', 1.0)
        self.gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)
        
        # Loss calculator
        self.loss_fn = VLMLosses(self.training_config)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = self.training_config.get('mixed_precision', 'fp16') == 'fp16'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(self.training_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = self.training_config.get('save_every_n_epochs', 1)
        self.save_total_limit = self.training_config.get('save_total_limit', 3)
        
        # Logging
        self.log_every_n_steps = self.training_config.get('log_every_n_steps', 50)
        tensorboard_dir = self.training_config.get('tensorboard_dir', 'logs/tensorboard')
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with separate learning rates for different parameters"""
        
        base_lr = self.training_config.get('base_lr', 1e-4)
        cross_attention_lr_multiplier = self.training_config.get('cross_attention_lr_multiplier', 10.0)
        weight_decay = self.training_config.get('weight_decay', 0.01)
        
        # Separate parameters into groups
        encoder_params = []
        cross_attention_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'interaction_layer' in name or 'pool_proj' in name:
                cross_attention_params.append(param)
            elif 'vision_encoder' in name or 'text_encoder' in name:
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {
                'params': encoder_params,
                'lr': base_lr,
                'weight_decay': weight_decay,
            },
            {
                'params': cross_attention_params,
                'lr': base_lr * cross_attention_lr_multiplier,
                'weight_decay': weight_decay,
            },
            {
                'params': other_params,
                'lr': base_lr * cross_attention_lr_multiplier,
                'weight_decay': weight_decay,
            },
        ]
        
        optimizer = AdamW(param_groups)
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine decay"""
        
        warmup_ratio = self.training_config.get('warmup_ratio', 0.05)
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7,
        )
        
        # Sequential scheduler
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        
        return scheduler
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        all_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                # Model forward
                model_outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_attention_weights=self.loss_fn.loss_weights.get('cyc', 0) > 0,
                )
                
                # Compute MLM logits if needed
                if self.stage in ['B', 'C'] and self.loss_fn.loss_weights.get('mlm', 0) > 0:
                    mlm_logits = self.model.compute_mlm_logits(model_outputs['text_hidden'])
                    model_outputs['mlm_logits'] = mlm_logits
                
                # Compute MIM predictions if needed
                if self.stage in ['B', 'C'] and self.loss_fn.loss_weights.get('mim', 0) > 0:
                    mim_predictions = self.model.compute_mim_predictions(model_outputs['vision_hidden'])
                    model_outputs['mim_predictions'] = mim_predictions
                    
                    # Use vision encoder output as target (self-distillation)
                    with torch.no_grad():
                        mim_targets = self.model.vision_encoder(batch['pixel_values'])
                else:
                    mim_targets = None
                
                # Prepare ITM if enabled
                itm_labels = None
                if self.loss_fn.loss_weights.get('itm', 0) > 0:
                    # For ITM with semi-hard negatives, we need to pass the model
                    # The loss function will handle the semi-hard negative mining
                    itm_labels = 'use_hard_negatives'  # Signal to use hard negatives
                
                # Compute total loss
                loss, metrics = self.loss_fn.compute_total_loss(
                    model_outputs=model_outputs,
                    mlm_labels=batch.get('mlm_labels'),
                    mim_targets=mim_targets,
                    mim_mask=batch.get('mim_mask'),
                    itm_labels=itm_labels,
                    stage=self.stage,
                    model=self.model,  # Pass model for ITM
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = 0.0
                all_metrics[k] += v
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
            
            # Log to tensorboard
            if self.global_step % self.log_every_n_steps == 0:
                for k, v in metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # Average metrics over epoch
        num_batches = len(self.train_loader)
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
        avg_metrics['loss'] = total_loss / num_batches
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate on validation set"""
        
        self.model.eval()
        total_loss = 0.0
        all_metrics = {}
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Val]")
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            model_outputs = self.model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],  # Use input_ids (validation doesn't mask)
                attention_mask=batch['attention_mask'],
                return_attention_weights=False,
            )
            
            # For validation, only compute ITC loss
            loss, metrics = self.loss_fn.compute_itc_loss(
                model_outputs['vision_embeds'],
                model_outputs['text_embeds'],
                model_outputs['logit_scale'],
            )
            
            # Accumulate metrics
            total_loss += loss.item()
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = 0.0
                all_metrics[k] += v
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Average metrics
        num_batches = len(self.val_loader)
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
        avg_metrics['loss'] = total_loss / num_batches
        
        return avg_metrics
    
    def save_checkpoint(self, is_best: bool = False, **extra_info):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            **extra_info,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Remove old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: x.stat().st_mtime,
        )
        
        # Keep only the most recent N checkpoints
        if len(checkpoints) > self.save_total_limit:
            for checkpoint in checkpoints[:-self.save_total_limit]:
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        
        print(f"\n{'='*50}")
        print(f"Starting training - Stage {self.stage}")
        print(f"{'='*50}\n")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch + 1} Train Metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"\nEpoch {epoch + 1} Val Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Log to tensorboard
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % self.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(
                    is_best=is_best,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
            
            print()
        
        print("\nTraining completed!")
        self.writer.close()

