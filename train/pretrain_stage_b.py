"""
Stage B: Align - Unfreeze top K blocks of each encoder and train with full loss mix
Includes MLM, MIM, and cycle-consistency losses.
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
from src.training.losses import (
    DualContrastiveLoss, 
    ITMLoss, 
    MLMLoss,
    MIMLoss,
    CycleConsistencyLoss
)
from src.training import utils
from src.training.utils import warmup_lr_schedule, cosine_lr_schedule


def train_one_epoch(model, data_loader, optimizer, epoch, device, config):
    """Train for one epoch in Stage B: align with full objectives"""
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cyc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_text', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_vision', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = f'Stage B - Epoch: [{epoch}]'
    print_freq = 50
    
    # Loss functions
    contrastive_loss_fn = DualContrastiveLoss(temperature=config.get('temperature', 0.07))
    itm_loss_fn = ITMLoss(
        margin_min=config.get('itm_margin_min', 0.2), 
        margin_max=config.get('itm_margin_max', 0.5),
        embed_dim=config['pooling_config']['projection_dim']
    )
    mlm_loss_fn = MLMLoss(vocab_size=model.module.text_encoder.vocab_size if hasattr(model, 'module') else model.text_encoder.vocab_size)
    mim_loss_fn = MIMLoss(loss_type=config.get('mim_type', 'mse'))
    cycle_loss_fn = CycleConsistencyLoss()
    
    data_loader.sampler.set_epoch(epoch)
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        # Get MLM labels if available
        mlm_labels = batch.get('mlm_labels', None)
        if mlm_labels is not None:
            mlm_labels = mlm_labels.to(device, non_blocking=True)
        
        # Get MIM labels/mask if available
        mim_labels = batch.get('mim_labels', None)
        mim_mask = batch.get('mim_mask', None)
        if mim_labels is not None:
            mim_labels = mim_labels.to(device, non_blocking=True)
        if mim_mask is not None:
            mim_mask = mim_mask.to(device, non_blocking=True)
        
        # Warmup learning rate
        step = epoch * len(data_loader) + i
        total_steps = config['max_epoch'] * len(data_loader)
        warmup_steps = int(config['warmup_ratio'] * total_steps)
        
        if step < warmup_steps:
            warmup_lr_schedule(optimizer, step, warmup_steps, 0, config['init_lr'])
        
        optimizer.zero_grad()
        
        # Forward pass with attention weights for cycle loss
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mlm_labels=mlm_labels,
            mim_labels=mim_labels,
            mim_mask=mim_mask,
            return_attention_weights=config.get('use_cycle_loss', True)
        )
        
        # Get embeddings for contrastive loss
        vision_embeds_cross = outputs['vision_embeds']  # Cross-modal aware
        text_embeds_cross = outputs['text_embeds']      # Cross-modal aware
        vision_embeds_uni = outputs['vision_embeds_unimodal']  # Unimodal
        text_embeds_uni = outputs['text_embeds_unimodal']      # Unimodal
        
        # Compute losses
        # 1. Dual ITC loss
        loss_itc_cross = contrastive_loss_fn(vision_embeds_cross, text_embeds_cross)
        loss_itc_uni = contrastive_loss_fn(vision_embeds_uni, text_embeds_uni)
        alpha = model.module.get_itc_alpha() if hasattr(model, 'module') else model.get_itc_alpha()
        loss_itc = alpha * loss_itc_cross + (1 - alpha) * loss_itc_uni
        
        # 2. MLM loss (cross-aware)
        loss_mlm = torch.tensor(0.0, device=device)
        if mlm_labels is not None and 'mlm_logits' in outputs:
            loss_mlm = mlm_loss_fn(outputs['mlm_logits'], mlm_labels)
        
        # 3. MIM loss (optional)
        loss_mim = torch.tensor(0.0, device=device)
        if mim_labels is not None and 'mim_predictions' in outputs:
            loss_mim = mim_loss_fn(outputs['mim_predictions'], mim_labels, mim_mask)
        
        # 4. ITM loss with semi-hard negatives
        loss_itm = itm_loss_fn(
            vision_embeds_cross, 
            text_embeds_cross,
            vision_embeds_uni,
            text_embeds_uni
        )
        
        # 5. Cycle consistency loss (optional)
        loss_cyc = torch.tensor(0.0, device=device)
        if config.get('use_cycle_loss', True) and 'aux_outputs' in outputs:
            loss_cyc = cycle_loss_fn(outputs['aux_outputs'])
        
        # Total loss with weights from config
        loss = (config['lambda_itc'] * loss_itc + 
                config['lambda_mlm'] * loss_mlm +
                config['lambda_mim'] * loss_mim +
                config['lambda_itm'] * loss_itm +
                config.get('lambda_cyc', 0.1) * loss_cyc)
        
        loss.backward()
        
        # Gradient clipping
        if config['grad_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        # Log metrics
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_mim=loss_mim.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_cyc=loss_cyc.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # Log gate values
        if 'aux_outputs' in outputs and len(outputs['aux_outputs']) > 0:
            gate_text = np.mean([aux['gate_text'] for aux in outputs['aux_outputs']])
            gate_vision = np.mean([aux['gate_vision'] for aux in outputs['aux_outputs']])
            metric_logger.update(gate_text=gate_text)
            metric_logger.update(gate_vision=gate_vision)
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    
    # Fix seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # Create dataset
    print("Creating dataset")
    train_dataset = create_dataset('pretrain_mlm_mim', config)  # Dataset with MLM/MIM preprocessing
    print(f'Number of training samples: {len(train_dataset)}')
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler = create_sampler([train_dataset], [True], num_tasks, global_rank)[0]
    
    data_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    # Create model
    print("Creating model")
    model = CrossModalVLM(
        vision_config=config['vision_config'],
        text_config=config['text_config'],
        cross_modal_config=config['cross_modal_config'],
        pooling_config=config['pooling_config'],
    )
    
    # Load Stage A checkpoint
    if args.stage_a_checkpoint:
        print(f"Loading Stage A checkpoint from {args.stage_a_checkpoint}")
        checkpoint = torch.load(args.stage_a_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Stage A checkpoint loaded")
    
    # STAGE B: Unfreeze top K blocks of each encoder
    num_unfreeze_blocks = config.get('num_unfreeze_blocks', 4)
    print(f"Stage B: Unfreezing top {num_unfreeze_blocks} blocks of each encoder")
    
    # Unfreeze top layers of vision encoder
    model.vision_encoder.unfreeze_top_layers(num_unfreeze_blocks)
    
    # Unfreeze top layers of text encoder
    model.text_encoder.unfreeze_top_layers(num_unfreeze_blocks)
    
    # Also ensure interaction layers and heads are trainable
    for module_name, module in model.named_modules():
        if any(key in module_name for key in ['interaction_layers', 'pool_proj', 'mlm_head', 'mim_head', 
                                              'itm_head', 'itc_alpha', 'logit_scale']):
            for param in module.parameters():
                param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    model = model.to(device)
    
    # Create optimizer with different LR for different parts
    encoder_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                new_params.append(param)
    
    optimizer = torch.optim.AdamW(
        [
            {'params': encoder_params, 'lr': config['init_lr']},
            {'params': new_params, 'lr': config['init_lr'] * config['lr_multiplier']},
        ],
        weight_decay=config['weight_decay']
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from checkpoint: {args.checkpoint}')
    
    # Setup distributed training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    # Training loop
    print("Start training Stage B: Align")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['max_epoch']):
        # Cosine learning rate schedule
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        
        train_stats = train_one_epoch(model, data_loader, optimizer, epoch, device, config)
        
        if utils.is_main_process():
            # Save checkpoint
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'stage': 'B',
            }
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_stage_b_{epoch:02d}.pth')
            torch.save(save_obj, checkpoint_path)
            
            # Log stats
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'stage': 'B',
            }
            
            with open(os.path.join(args.output_dir, "log_stage_b.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.distributed:
            dist.barrier()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Stage B training time: {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain_stage_b.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain_StageB')
    parser.add_argument('--stage_a_checkpoint', required=True, help='Path to Stage A checkpoint')
    parser.add_argument('--checkpoint', default='', help='Resume from checkpoint')
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
