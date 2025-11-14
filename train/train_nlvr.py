"""
Stage C: Task Fine-tuning - Natural Language Visual Reasoning (NLVR²)
Train on NLVR² task which requires reasoning over pairs of images with text.
"""

import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vlm_model import CrossModalVLM
from src.data import create_dataset, create_sampler, create_loader
from src.training import utils
from src.training.utils import warmup_lr_schedule, cosine_lr_schedule


class NLVRHead(nn.Module):
    """NLVR classification head for reasoning over image pairs"""
    def __init__(self, hidden_dim, num_classes=2):
        super().__init__()
        # Process concatenated features from two image-text pairs
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # Double size for two images
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, features_img1, features_img2):
        """
        Args:
            features_img1: Cross-modal features for image1-text pair
            features_img2: Cross-modal features for image2-text pair
        Returns:
            logits: Classification logits (batch, num_classes)
        """
        combined_features = torch.cat([features_img1, features_img2], dim=-1)
        return self.classifier(combined_features)


def get_cross_modal_features(model, pixel_values, input_ids, attention_mask, pooling_type='cls'):
    """Get cross-modal features after interaction layers"""
    # Forward pass through base model
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_attention_weights=False
    )
    
    # Get final hidden states after cross-modal interaction
    vision_hidden = outputs['vision_hidden']  # (batch, n_vision, vision_dim)
    text_hidden = outputs['text_hidden']      # (batch, n_text, text_dim)
    
    # Pool features
    if pooling_type == 'cls':
        vision_pooled = vision_hidden[:, 0, :]  # CLS token
        text_pooled = text_hidden[:, 0, :]      # CLS token
    else:
        # Mean pooling with mask
        vision_pooled = vision_hidden.mean(dim=1)
        text_pooled = text_hidden.mean(dim=1) if attention_mask is None else \
                       (text_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    
    # Concatenate vision and text features
    cross_modal_features = torch.cat([vision_pooled, text_pooled], dim=-1)
    
    return cross_modal_features


def train_one_epoch(model, nlvr_head, data_loader, optimizer, epoch, device, config):
    """Train for one epoch on NLVR² task"""
    model.train()
    nlvr_head.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = f'NLVR Training - Epoch: [{epoch}]'
    print_freq = 50
    
    data_loader.sampler.set_epoch(epoch)
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # NLVR has two images and one text
        image1 = batch['image1'].to(device, non_blocking=True)
        image2 = batch['image2'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)  # Binary: True/False
        
        # Warmup learning rate
        step = epoch * len(data_loader) + i
        total_steps = config['max_epoch'] * len(data_loader)
        warmup_steps = int(config.get('warmup_ratio', 0.05) * total_steps)
        
        if step < warmup_steps:
            warmup_lr_schedule(optimizer, step, warmup_steps, 0, config['init_lr'])
        
        optimizer.zero_grad()
        
        # Get cross-modal features for each image-text pair
        features_img1 = get_cross_modal_features(
            model, image1, input_ids, attention_mask, 
            pooling_type=config.get('nlvr_pooling', 'cls')
        )
        
        features_img2 = get_cross_modal_features(
            model, image2, input_ids, attention_mask,
            pooling_type=config.get('nlvr_pooling', 'cls')
        )
        
        # Get NLVR predictions
        logits = nlvr_head(features_img1, features_img2)
        
        # Compute loss (cross-entropy for binary classification)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        
        # Gradient clipping
        if config.get('grad_clip', 1.0) > 0:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(nlvr_head.parameters()), 
                config['grad_clip']
            )
        
        optimizer.step()
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        acc = (predictions == labels).float().mean()
        
        # Log metrics
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, nlvr_head, data_loader, device, config):
    """Evaluate on NLVR validation set"""
    model.eval()
    nlvr_head.eval()
    
    correct = 0
    total = 0
    
    for batch in data_loader:
        image1 = batch['image1'].to(device, non_blocking=True)
        image2 = batch['image2'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Get cross-modal features
        features_img1 = get_cross_modal_features(
            model, image1, input_ids, attention_mask,
            pooling_type=config.get('nlvr_pooling', 'cls')
        )
        
        features_img2 = get_cross_modal_features(
            model, image2, input_ids, attention_mask,
            pooling_type=config.get('nlvr_pooling', 'cls')
        )
        
        # Get predictions
        logits = nlvr_head(features_img1, features_img2)
        predictions = logits.argmax(dim=-1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    
    return accuracy


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    
    # Fix seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # Create datasets
    print("Creating NLVR² datasets")
    train_dataset = create_dataset('nlvr_train', config)
    val_dataset = create_dataset('nlvr_val', config)
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(val_dataset)}')
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    train_sampler = create_sampler([train_dataset], [True], num_tasks, global_rank)[0]
    val_sampler = create_sampler([val_dataset], [False], num_tasks, global_rank)[0]
    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    
    # Create model
    print("Creating model")
    model = CrossModalVLM(
        vision_config=config['vision_config'],
        text_config=config['text_config'],
        cross_modal_config=config['cross_modal_config'],
        pooling_config=config['pooling_config'],
    )
    
    # Create NLVR head
    # Concatenating vision and text dims for each image
    nlvr_input_dim = config['vision_config']['hidden_dim'] + config['text_config']['hidden_dim']
    nlvr_head = NLVRHead(nlvr_input_dim, num_classes=2)  # Binary classification
    
    # Load pretrained checkpoint
    if args.pretrained:
        print(f"Loading pretrained checkpoint from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Pretrained checkpoint loaded")
    
    # Optionally freeze base model
    if config.get('freeze_base_model', False):
        print("Freezing base model parameters")
        for param in model.parameters():
            param.requires_grad = False
        # Only train NLVR head
        trainable_params = list(nlvr_head.parameters())
    else:
        # Fine-tune entire model
        trainable_params = list(model.parameters()) + list(nlvr_head.parameters())
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad)}")
    
    model = model.to(device)
    nlvr_head = nlvr_head.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['init_lr'],
        weight_decay=config['weight_decay']
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_accuracy = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        nlvr_head.load_state_dict(checkpoint['nlvr_head'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_accuracy' in checkpoint:
            best_accuracy = checkpoint['best_accuracy']
        print(f'Resumed from checkpoint: {args.checkpoint}')
    
    # Setup distributed training
    model_without_ddp = model
    nlvr_head_without_ddp = nlvr_head
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        nlvr_head = torch.nn.parallel.DistributedDataParallel(nlvr_head, device_ids=[args.gpu])
        model_without_ddp = model.module
        nlvr_head_without_ddp = nlvr_head.module
    
    # Evaluation only mode
    if args.evaluate:
        print("Evaluation only mode")
        accuracy = evaluate(model, nlvr_head, val_loader, device, config)
        print(f"NLVR² Validation Accuracy: {accuracy:.4f}")
        return
    
    # Training loop
    print("Start NLVR² training")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['max_epoch']):
        # Cosine learning rate schedule
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        
        train_stats = train_one_epoch(model, nlvr_head, train_loader, optimizer, epoch, device, config)
        
        # Evaluate
        accuracy = evaluate(model, nlvr_head, val_loader, device, config)
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        if utils.is_main_process():
            # Save checkpoint
            is_best = accuracy > best_accuracy
            best_accuracy = max(accuracy, best_accuracy)
            
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'nlvr_head': nlvr_head_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_accuracy': best_accuracy,
            }
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth')
            torch.save(save_obj, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                torch.save(save_obj, best_path)
                print(f"New best accuracy: {best_accuracy:.4f}")
            
            # Log stats
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'val_accuracy': f"{accuracy:.4f}",
                'best_accuracy': f"{best_accuracy:.4f}",
                'epoch': epoch,
            }
            
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.distributed:
            dist.barrier()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'NLVR² training time: {total_time_str}')
    print(f'Best Accuracy: {best_accuracy:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/nlvr.yaml')
    parser.add_argument('--output_dir', default='output/NLVR')
    parser.add_argument('--pretrained', required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--checkpoint', default='', help='Resume from checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config, 'r'))
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    main(args, config)