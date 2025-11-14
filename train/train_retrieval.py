"""
Stage C: Task Fine-tuning - Cross-Modal Retrieval
Fine-tune for image-text and text-image retrieval using dual embeddings.
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
from data_utils import create_dataset, create_sampler, create_loader
from src.training.losses import DualContrastiveLoss, ITMLoss
from src.training import utils
from src.training.utils import warmup_lr_schedule, cosine_lr_schedule
from src.evaluation.retrieval_metric import compute_retrieval_metrics
from transformers import AutoTokenizer


def create_collate_fn(tokenizer, max_length=77):
    """Create a collate function for COCO dataset"""
    def collate_fn(batch):
        """
        Collate function to process batch from COCO dataset.
        COCO dataset returns (image, caption, img_id)
        """
        images, captions, img_ids = zip(*batch)
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        # Tokenize captions
        text_inputs = tokenizer(
            list(captions),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': images,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'image_id': list(img_ids),
            'text_id': list(img_ids),  # For retrieval, text_id = image_id
        }
    
    return collate_fn


def train_one_epoch(model, data_loader, optimizer, epoch, device, config):
    """Train for one epoch on retrieval task"""
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('itc_alpha', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = f'Retrieval Training - Epoch: [{epoch}]'
    print_freq = 50
    
    # Loss functions
    contrastive_loss_fn = DualContrastiveLoss(temperature=config.get('temperature', 0.07))
    itm_loss_fn = ITMLoss(
        margin_min=config.get('itm_margin_min', 0.2), 
        margin_max=config.get('itm_margin_max', 0.5),
        embed_dim=config['pooling_config']['projection_dim']
    )
    
    data_loader.sampler.set_epoch(epoch)
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        # Warmup learning rate
        step = epoch * len(data_loader) + i
        total_steps = config['max_epoch'] * len(data_loader)
        warmup_steps = int(config.get('warmup_ratio', 0.05) * total_steps)
        
        if step < warmup_steps:
            warmup_lr_schedule(optimizer, step, warmup_steps, 0, config['init_lr'])
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention_weights=False
        )
        
        # Get dual embeddings
        # Model returns: vision_embeds (cross-modal), vision_embeds_unimodal (uni)
        vision_embeds_cross = outputs['vision_embeds']  # Cross-modal aware
        text_embeds_cross = outputs['text_embeds']      # Cross-modal aware
        vision_embeds_uni = outputs['vision_embeds_unimodal']  # Unimodal for fast retrieval
        text_embeds_uni = outputs['text_embeds_unimodal']      # Unimodal for fast retrieval
        
        # Compute losses
        # 1. Dual contrastive loss
        loss_itc_cross = contrastive_loss_fn(vision_embeds_cross, text_embeds_cross)
        loss_itc_uni = contrastive_loss_fn(vision_embeds_uni, text_embeds_uni)
        
        # Get alpha weighting
        alpha = model.get_itc_alpha() if hasattr(model, 'get_itc_alpha') else model.module.get_itc_alpha()
        loss_itc = alpha * loss_itc_cross + (1 - alpha) * loss_itc_uni
        
        # 2. ITM loss with hard negatives (optional)
        loss_itm = torch.tensor(0.0, device=device)
        if config.get('use_itm', True):
            loss_itm = itm_loss_fn(
                vision_embeds_cross, 
                text_embeds_cross,
                vision_embeds_uni,
                text_embeds_uni
            )
        
        # Total loss
        loss = config['lambda_itc'] * loss_itc
        if config.get('use_itm', True):
            loss = loss + config['lambda_itm'] * loss_itm
        
        loss.backward()
        
        # Gradient clipping
        if config.get('grad_clip', 1.0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        # Log metrics
        metric_logger.update(loss_itc=loss_itc.item())
        if config.get('use_itm', True):
            metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(itc_alpha=alpha.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_retrieval(model, data_loader, device, config):
    """Evaluate retrieval performance"""
    model.eval()
    
    print("Computing features for retrieval evaluation...")
    
    # Storage for embeddings
    vision_embeds_list = []
    text_embeds_list = []
    vision_embeds_uni_list = []
    text_embeds_uni_list = []
    image_ids = []
    text_ids = []
    
    for batch in data_loader:
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        # Get embeddings
        if config.get('eval_cross_modal', True):
            # Full cross-modal embeddings (slower but more accurate)
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attention_weights=False
            )
            vision_embeds_list.append(outputs['vision_embeds'].cpu())
            text_embeds_list.append(outputs['text_embeds'].cpu())
        
        # Always compute unimodal embeddings for fast retrieval
        vision_embeds_uni = model.module.encode_vision_only(pixel_values) if hasattr(model, 'module') else model.encode_vision_only(pixel_values)
        text_embeds_uni = model.module.encode_text_only(input_ids, attention_mask) if hasattr(model, 'module') else model.encode_text_only(input_ids, attention_mask)
        
        vision_embeds_uni_list.append(vision_embeds_uni.cpu())
        text_embeds_uni_list.append(text_embeds_uni.cpu())
        
        image_ids.extend(batch['image_id'])
        text_ids.extend(batch['text_id'])
    
    # Concatenate all embeddings
    vision_embeds_uni = torch.cat(vision_embeds_uni_list, dim=0)
    text_embeds_uni = torch.cat(text_embeds_uni_list, dim=0)
    
    if config.get('eval_cross_modal', True):
        vision_embeds = torch.cat(vision_embeds_list, dim=0)
        text_embeds = torch.cat(text_embeds_list, dim=0)
    else:
        vision_embeds = vision_embeds_uni
        text_embeds = text_embeds_uni
    
    # Compute retrieval metrics
    print("Computing retrieval metrics...")
    
    # For unimodal (fast bi-encoder retrieval)
    metrics_uni = compute_retrieval_metrics(
        vision_embeds_uni, 
        text_embeds_uni,
        image_ids,
        text_ids,
        k_values=[1, 5, 10]
    )
    
    results = {
        'i2t_r1_uni': metrics_uni['i2t_r1'],
        'i2t_r5_uni': metrics_uni['i2t_r5'],
        'i2t_r10_uni': metrics_uni['i2t_r10'],
        't2i_r1_uni': metrics_uni['t2i_r1'],
        't2i_r5_uni': metrics_uni['t2i_r5'],
        't2i_r10_uni': metrics_uni['t2i_r10'],
    }
    
    # For cross-modal (if evaluated)
    if config.get('eval_cross_modal', True):
        metrics_cross = compute_retrieval_metrics(
            vision_embeds, 
            text_embeds,
            image_ids,
            text_ids,
            k_values=[1, 5, 10]
        )
        
        results.update({
            'i2t_r1_cross': metrics_cross['i2t_r1'],
            'i2t_r5_cross': metrics_cross['i2t_r5'],
            'i2t_r10_cross': metrics_cross['i2t_r10'],
            't2i_r1_cross': metrics_cross['t2i_r1'],
            't2i_r5_cross': metrics_cross['t2i_r5'],
            't2i_r10_cross': metrics_cross['t2i_r10'],
        })
    
    # Compute mean recall
    r1_uni = (results['i2t_r1_uni'] + results['t2i_r1_uni']) / 2
    r5_uni = (results['i2t_r5_uni'] + results['t2i_r5_uni']) / 2
    r10_uni = (results['i2t_r10_uni'] + results['t2i_r10_uni']) / 2
    results['mean_r1_uni'] = r1_uni
    results['mean_r5_uni'] = r5_uni
    results['mean_r10_uni'] = r10_uni
    
    if config.get('eval_cross_modal', True):
        r1_cross = (results['i2t_r1_cross'] + results['t2i_r1_cross']) / 2
        r5_cross = (results['i2t_r5_cross'] + results['t2i_r5_cross']) / 2
        r10_cross = (results['i2t_r10_cross'] + results['t2i_r10_cross']) / 2
        results['mean_r1_cross'] = r1_cross
        results['mean_r5_cross'] = r5_cross
        results['mean_r10_cross'] = r10_cross
    
    return results


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
    print("Creating retrieval datasets")
    # Create dataset based on config
    if config.get('dataset') == 'retrieval_coco':
        train_dataset, val_dataset, test_dataset = create_dataset('retrieval_coco', config)
    else:
        # Fallback for other dataset types
        train_dataset = create_dataset(config.get('dataset', 'retrieval_train'), config)
        val_dataset = create_dataset(config.get('dataset_val', 'retrieval_val'), config)
        test_dataset = None
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(val_dataset)}')
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    train_sampler = create_sampler([train_dataset], [True], num_tasks, global_rank)[0]
    val_sampler = create_sampler([val_dataset], [False], num_tasks, global_rank)[0]
    
    # Create tokenizer for text processing
    tokenizer = AutoTokenizer.from_pretrained(config['text_config']['model_name'])
    collate_fn = create_collate_fn(tokenizer, max_length=config.get('max_words', 77))
    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=config['eval_batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    # Create model
    print("Creating model")
    model = CrossModalVLM(
        vision_config=config['vision_config'],
        text_config=config['text_config'],
        cross_modal_config=config['cross_modal_config'],
        pooling_config=config['pooling_config'],
    )
    
    # Load pretrained checkpoint
    if args.pretrained:
        print(f"Loading pretrained checkpoint from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Pretrained checkpoint loaded")
    
    # Fine-tuning strategy
    if config.get('freeze_encoders', False):
        print("Freezing encoder parameters")
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['init_lr'],
        weight_decay=config['weight_decay']
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_r1 = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_r1' in checkpoint:
            best_r1 = checkpoint['best_r1']
        print(f'Resumed from checkpoint: {args.checkpoint}')
    
    # Setup distributed training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True  # Some heads (MLM, MIM, ITM) may not be used in retrieval
        )
        model_without_ddp = model.module
    
    # Evaluation only mode
    if args.evaluate:
        print("Evaluation only mode")
        results = evaluate_retrieval(model, val_loader, device, config)
        
        print("\n=== Retrieval Results ===")
        if config.get('eval_cross_modal', True):
            print("\nCross-modal (accurate):")
            print(f"Image-to-Text: R@1={results['i2t_r1_cross']:.2f}, R@5={results['i2t_r5_cross']:.2f}, R@10={results['i2t_r10_cross']:.2f}")
            print(f"Text-to-Image: R@1={results['t2i_r1_cross']:.2f}, R@5={results['t2i_r5_cross']:.2f}, R@10={results['t2i_r10_cross']:.2f}")
            print(f"Mean: R@1={results['mean_r1_cross']:.2f}, R@5={results['mean_r5_cross']:.2f}, R@10={results['mean_r10_cross']:.2f}")
        
        print("\nUnimodal (fast):")
        print(f"Image-to-Text: R@1={results['i2t_r1_uni']:.2f}, R@5={results['i2t_r5_uni']:.2f}, R@10={results['i2t_r10_uni']:.2f}")
        print(f"Text-to-Image: R@1={results['t2i_r1_uni']:.2f}, R@5={results['t2i_r5_uni']:.2f}, R@10={results['t2i_r10_uni']:.2f}")
        print(f"Mean: R@1={results['mean_r1_uni']:.2f}, R@5={results['mean_r5_uni']:.2f}, R@10={results['mean_r10_uni']:.2f}")
        return
    
    # Training loop
    print("Start retrieval training")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['max_epoch']):
        # Cosine learning rate schedule
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        
        train_stats = train_one_epoch(model, train_loader, optimizer, epoch, device, config)
        
        # Evaluate
        if (epoch + 1) % config.get('eval_freq', 1) == 0:
            results = evaluate_retrieval(model, val_loader, device, config)
            
            # Use unimodal R@1 as the main metric
            current_r1 = results['mean_r1_uni']
            
            print(f"\nEpoch {epoch} - Retrieval Results:")
            print(f"Unimodal Mean R@1={results['mean_r1_uni']:.2f}, R@5={results['mean_r5_uni']:.2f}, R@10={results['mean_r10_uni']:.2f}")
            if config.get('eval_cross_modal', True):
                print(f"Cross-modal Mean R@1={results['mean_r1_cross']:.2f}, R@5={results['mean_r5_cross']:.2f}, R@10={results['mean_r10_cross']:.2f}")
        else:
            current_r1 = 0
            results = {}
        
        if utils.is_main_process():
            # Save checkpoint
            is_best = current_r1 > best_r1
            best_r1 = max(current_r1, best_r1)
            
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_r1': best_r1,
            }
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth')
            torch.save(save_obj, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                torch.save(save_obj, best_path)
                print(f"New best Mean R@1: {best_r1:.2f}")
            
            # Log stats
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **results,
                'epoch': epoch,
            }
            
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.distributed:
            dist.barrier()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Retrieval training time: {total_time_str}')
    print(f'Best Mean R@1: {best_r1:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval')
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