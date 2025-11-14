"""
Stage C: Task Fine-tuning - Vision Question Answering (VQA)
Attach classifier head on pooled cross-modal features for VQA tasks.
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
from src.evaluation.vqa_metric import VQAMetric


class VQAHead(nn.Module):
    """VQA classification head on top of cross-modal features"""
    def __init__(self, hidden_dim, num_answers):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers)
        )
        
    def forward(self, pooled_features):
        return self.classifier(pooled_features)


def train_one_epoch(model, vqa_head, data_loader, optimizer, epoch, device, config):
    """Train for one epoch on VQA task"""
    model.train()
    vqa_head.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = f'VQA Training - Epoch: [{epoch}]'
    print_freq = 50
    
    data_loader.sampler.set_epoch(epoch)
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)  # Question tokens
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        answers = batch['answers'].to(device, non_blocking=True)  # Multi-hot answer labels
        
        # Warmup learning rate
        step = epoch * len(data_loader) + i
        total_steps = config['max_epoch'] * len(data_loader)
        warmup_steps = int(config.get('warmup_ratio', 0.05) * total_steps)
        
        if step < warmup_steps:
            warmup_lr_schedule(optimizer, step, warmup_steps, 0, config['init_lr'])
        
        optimizer.zero_grad()
        
        # Forward pass through base model
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention_weights=False
        )
        
        # Get cross-modal pooled features
        # Use both vision and text final hidden states after cross-modal interaction
        vision_hidden = outputs['vision_hidden']  # (batch, n_vision, vision_dim)
        text_hidden = outputs['text_hidden']      # (batch, n_text, text_dim)
        
        # Pool features (use CLS token or mean pooling)
        if config.get('vqa_pooling', 'cls') == 'cls':
            vision_pooled = vision_hidden[:, 0, :]  # CLS token
            text_pooled = text_hidden[:, 0, :]      # CLS token
        else:
            # Mean pooling
            vision_pooled = vision_hidden.mean(dim=1)
            text_pooled = text_hidden.mean(dim=1)
        
        # Concatenate vision and text features
        cross_modal_features = torch.cat([vision_pooled, text_pooled], dim=-1)
        
        # Get VQA predictions
        logits = vqa_head(cross_modal_features)
        
        # Compute loss (binary cross entropy for multi-label)
        if config.get('vqa_loss', 'bce') == 'bce':
            loss = F.binary_cross_entropy_with_logits(logits, answers.float())
        else:
            # Soft target cross entropy
            loss = -(answers * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        
        loss.backward()
        
        # Gradient clipping
        if config.get('grad_clip', 1.0) > 0:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(vqa_head.parameters()), 
                config['grad_clip']
            )
        
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.sigmoid(logits) if config.get('vqa_loss', 'bce') == 'bce' else F.softmax(logits, dim=-1)
        pred_labels = predictions.argmax(dim=-1)
        answer_labels = answers.argmax(dim=-1)
        acc = (pred_labels == answer_labels).float().mean()
        
        # Log metrics
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, vqa_head, data_loader, device, config):
    """Evaluate on VQA validation set"""
    model.eval()
    vqa_head.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    vqa_metric = VQAMetric()
    
    header = 'VQA Evaluation:'
    print_freq = 50
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        answers = batch['answers'].to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention_weights=False
        )
        
        # Get pooled features
        vision_hidden = outputs['vision_hidden']
        text_hidden = outputs['text_hidden']
        
        if config.get('vqa_pooling', 'cls') == 'cls':
            vision_pooled = vision_hidden[:, 0, :]
            text_pooled = text_hidden[:, 0, :]
        else:
            vision_pooled = vision_hidden.mean(dim=1)
            text_pooled = text_hidden.mean(dim=1)
        
        cross_modal_features = torch.cat([vision_pooled, text_pooled], dim=-1)
        
        # Get predictions
        logits = vqa_head(cross_modal_features)
        
        # Update metrics
        predictions = torch.sigmoid(logits) if config.get('vqa_loss', 'bce') == 'bce' else F.softmax(logits, dim=-1)
        vqa_metric.update(predictions, answers, batch.get('question_ids'))
    
    # Compute final metrics
    vqa_score = vqa_metric.compute()
    
    return vqa_score


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
    print("Creating VQA datasets")
    train_dataset = create_dataset('vqa_train', config)
    val_dataset = create_dataset('vqa_val', config)
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(val_dataset)}')
    
    # Get number of answer classes
    num_answers = train_dataset.num_answers
    print(f'Number of answer classes: {num_answers}')
    
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
    
    # Create VQA head
    # Concatenating vision and text dims
    vqa_input_dim = config['vision_config']['hidden_dim'] + config['text_config']['hidden_dim']
    vqa_head = VQAHead(vqa_input_dim, num_answers)
    
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
        # Only train VQA head
        trainable_params = list(vqa_head.parameters())
    else:
        # Fine-tune entire model
        trainable_params = list(model.parameters()) + list(vqa_head.parameters())
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad)}")
    
    model = model.to(device)
    vqa_head = vqa_head.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['init_lr'],
        weight_decay=config['weight_decay']
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_vqa_score = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        vqa_head.load_state_dict(checkpoint['vqa_head'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_vqa_score' in checkpoint:
            best_vqa_score = checkpoint['best_vqa_score']
        print(f'Resumed from checkpoint: {args.checkpoint}')
    
    # Setup distributed training
    model_without_ddp = model
    vqa_head_without_ddp = vqa_head
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        vqa_head = torch.nn.parallel.DistributedDataParallel(vqa_head, device_ids=[args.gpu])
        model_without_ddp = model.module
        vqa_head_without_ddp = vqa_head.module
    
    # Training loop
    print("Start VQA training")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['max_epoch']):
        # Cosine learning rate schedule
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        
        train_stats = train_one_epoch(model, vqa_head, train_loader, optimizer, epoch, device, config)
        
        # Evaluate
        vqa_score = evaluate(model, vqa_head, val_loader, device, config)
        print(f"VQA Score: {vqa_score:.4f}")
        
        if utils.is_main_process():
            # Save checkpoint
            is_best = vqa_score > best_vqa_score
            best_vqa_score = max(vqa_score, best_vqa_score)
            
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'vqa_head': vqa_head_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_vqa_score': best_vqa_score,
            }
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth')
            torch.save(save_obj, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                torch.save(save_obj, best_path)
                print(f"New best VQA score: {best_vqa_score:.4f}")
            
            # Log stats
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'vqa_score': f"{vqa_score:.4f}",
                'best_vqa_score': f"{best_vqa_score:.4f}",
                'epoch': epoch,
            }
            
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.distributed:
            dist.barrier()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'VQA training time: {total_time_str}')
    print(f'Best VQA score: {best_vqa_score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml')
    parser.add_argument('--output_dir', default='output/VQA')
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
    
    if args.evaluate:
        # Evaluation only mode
        pass  # TODO: Implement evaluate-only mode
    else:
        main(args, config)