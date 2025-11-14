"""
Experiment script for COCO Retrieval with detailed parameter analysis and progress tracking
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
from tqdm import tqdm
from collections import OrderedDict

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
        Train dataset returns (image, caption, img_id)
        Eval dataset returns (image, index)
        """
        # Check if this is training data (3 values) or eval data (2 values)
        if len(batch[0]) == 3:
            # Training data: (image, caption, img_id)
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
                'indices': list(img_ids),  # Track original indices
            }
        else:
            # Evaluation data: (image, index)
            images, indices = zip(*batch)
            
            # Stack images
            images = torch.stack(images, dim=0)
            
            # For evaluation, we don't have captions in the batch
            # Create dummy text inputs (will not be used in evaluation)
            dummy_text = [""] * len(images)
            text_inputs = tokenizer(
                dummy_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            return {
                'pixel_values': images,
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask'],
                'image_id': list(indices),
                'text_id': list(indices),
                'indices': list(indices),  # Track original dataset indices
            }
    
    return collate_fn


def configure_encoder_freezing(model, stage_config, stage_name):
    """
    Configure encoder freezing based on stage configuration.
    Implements selective unfreezing of top encoder blocks.
    """
    # First, freeze or unfreeze entire encoders
    if stage_config.get('freeze_vision_encoder', False):
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.vision_encoder.parameters():
            param.requires_grad = True
    
    if stage_config.get('freeze_text_encoder', False):
        for param in model.text_encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.text_encoder.parameters():
            param.requires_grad = True
    
    # Selectively unfreeze top blocks if specified
    unfreeze_top_vision = stage_config.get('unfreeze_top_vision_blocks', 0)
    unfreeze_top_text = stage_config.get('unfreeze_top_text_blocks', 0)
    
    # Vision encoder - unfreeze top K blocks
    if unfreeze_top_vision > 0 and stage_config.get('freeze_vision_encoder', False):
        # For ViT models, blocks are in encoder.layer
        if hasattr(model.vision_encoder, 'encoder') and hasattr(model.vision_encoder.encoder, 'layer'):
            total_blocks = len(model.vision_encoder.encoder.layer)
            start_idx = max(0, total_blocks - unfreeze_top_vision)
            for idx in range(start_idx, total_blocks):
                for param in model.vision_encoder.encoder.layer[idx].parameters():
                    param.requires_grad = True
            print(f"  Unfroze top {unfreeze_top_vision} vision blocks (blocks {start_idx} to {total_blocks-1})")
    elif unfreeze_top_vision == -1:  # -1 means unfreeze all
        for param in model.vision_encoder.parameters():
            param.requires_grad = True
    
    # Text encoder - unfreeze top K blocks
    if unfreeze_top_text > 0 and stage_config.get('freeze_text_encoder', False):
        # For BERT models, blocks are in encoder.layer
        if hasattr(model.text_encoder, 'encoder') and hasattr(model.text_encoder.encoder, 'layer'):
            total_blocks = len(model.text_encoder.encoder.layer)
            start_idx = max(0, total_blocks - unfreeze_top_text)
            for idx in range(start_idx, total_blocks):
                for param in model.text_encoder.encoder.layer[idx].parameters():
                    param.requires_grad = True
            print(f"  Unfroze top {unfreeze_top_text} text blocks (blocks {start_idx} to {total_blocks-1})")
    elif unfreeze_top_text == -1:  # -1 means unfreeze all
        for param in model.text_encoder.parameters():
            param.requires_grad = True
    
    # Interaction layers and pooling heads are always trainable
    for param in model.interaction_layers.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    vision_trainable = sum(p.numel() for p in model.vision_encoder.parameters() if p.requires_grad)
    text_trainable = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    interaction_trainable = sum(p.numel() for p in model.interaction_layers.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Stage {stage_name} - Trainable parameters:")
    print(f"    Vision Encoder: {vision_trainable:,}")
    print(f"    Text Encoder: {text_trainable:,}")
    print(f"    Interaction Layers: {interaction_trainable:,}")
    print(f"    Total Trainable: {total_trainable:,}")
    
    return total_trainable


def analyze_model_parameters(model):
    """Analyze and display model parameter counts"""
    
    print("\n" + "="*80)
    print("MODEL PARAMETER ANALYSIS")
    print("="*80)
    
    # Count parameters by component
    param_counts = OrderedDict()
    
    # Vision encoder parameters
    vision_params = sum(p.numel() for p in model.vision_encoder.parameters())
    vision_trainable = sum(p.numel() for p in model.vision_encoder.parameters() if p.requires_grad)
    param_counts['Vision Encoder'] = {'total': vision_params, 'trainable': vision_trainable}
    
    # Text encoder parameters
    text_params = sum(p.numel() for p in model.text_encoder.parameters())
    text_trainable = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    param_counts['Text Encoder'] = {'total': text_params, 'trainable': text_trainable}
    
    # Cross-modal interaction layers (our method's overhead)
    interaction_params = sum(p.numel() for p in model.interaction_layers.parameters())
    interaction_trainable = sum(p.numel() for p in model.interaction_layers.parameters() if p.requires_grad)
    param_counts['Cross-Modal Interaction'] = {'total': interaction_params, 'trainable': interaction_trainable}
    
    # Projection heads
    proj_params = 0
    proj_trainable = 0
    for name, module in model.named_modules():
        if 'pool_proj' in name:
            proj_params += sum(p.numel() for p in module.parameters())
            proj_trainable += sum(p.numel() for p in module.parameters() if p.requires_grad)
    param_counts['Projection Heads'] = {'total': proj_params, 'trainable': proj_trainable}
    
    # Other parameters (gates, temperatures, etc.)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    other_params = total_params - vision_params - text_params - interaction_params - proj_params
    other_trainable = total_trainable - vision_trainable - text_trainable - interaction_trainable - proj_trainable
    param_counts['Other (gates, temps, etc.)'] = {'total': other_params, 'trainable': other_trainable}
    
    # Display parameter counts
    print("\nParameter Count by Component:")
    print("-" * 60)
    for component, counts in param_counts.items():
        print(f"{component:30s} | Total: {counts['total']:,} | Trainable: {counts['trainable']:,}")
    
    print("-" * 60)
    print(f"{'TOTAL':30s} | Total: {total_params:,} | Trainable: {total_trainable:,}")
    
    # Calculate overhead
    base_params = vision_params + text_params
    overhead_params = interaction_params + proj_params + other_params
    overhead_percent = (overhead_params / base_params) * 100
    
    print("\n" + "="*80)
    print("METHOD OVERHEAD ANALYSIS")
    print("="*80)
    print(f"Base Model Parameters (Vision + Text): {base_params:,}")
    print(f"Method Overhead Parameters: {overhead_params:,}")
    print(f"Overhead Percentage: {overhead_percent:.2f}%")
    print("="*80 + "\n")
    
    return {
        'total_params': total_params,
        'trainable_params': total_trainable,
        'base_params': base_params,
        'overhead_params': overhead_params,
        'overhead_percent': overhead_percent
    }


def train_one_epoch_with_progress(model, data_loader, optimizer, epoch, device, config, itm_loss_fn, stage_config=None):
    """Train for one epoch with tqdm progress bar"""
    model.train()
    
    # Use stage-specific config if provided, otherwise fall back to main config
    effective_config = stage_config if stage_config is not None else config
    
    # Loss functions
    contrastive_loss_fn = DualContrastiveLoss(temperature=effective_config.get('temperature', 0.07))
    
    # Metrics tracking
    loss_meter = utils.AverageMeter()
    loss_itc_meter = utils.AverageMeter()
    loss_itm_meter = utils.AverageMeter()
    
    data_loader.sampler.set_epoch(epoch)
    
    # Progress bar (only on main process)
    if utils.is_main_process():
        pbar = tqdm(data_loader, desc=f'Epoch {epoch}', ncols=120)
    else:
        pbar = data_loader
    
    for i, batch in enumerate(pbar):
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
        loss_itc_cross = contrastive_loss_fn(vision_embeds_cross, text_embeds_cross)
        loss_itc_uni = contrastive_loss_fn(vision_embeds_uni, text_embeds_uni)
        
        alpha = model.get_itc_alpha() if hasattr(model, 'get_itc_alpha') else model.module.get_itc_alpha()
        loss_itc = alpha * loss_itc_cross + (1 - alpha) * loss_itc_uni
        
        loss_itm = torch.tensor(0.0, device=device)
        if effective_config.get('use_itm', True) and effective_config.get('lambda_itm', 0.0) > 0:
            loss_itm = itm_loss_fn(
                vision_embeds_cross, 
                text_embeds_cross,
                vision_embeds_uni,
                text_embeds_uni
            )
        
        # Use stage-specific loss weights
        loss = effective_config.get('lambda_itc', config.get('lambda_itc', 1.0)) * loss_itc
        if effective_config.get('use_itm', True) and effective_config.get('lambda_itm', 0.0) > 0:
            loss = loss + effective_config.get('lambda_itm', config.get('lambda_itm', 1.0)) * loss_itm
        
        loss.backward()
        
        if config.get('grad_clip', 1.0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), pixel_values.size(0))
        loss_itc_meter.update(loss_itc.item(), pixel_values.size(0))
        if effective_config.get('use_itm', True) and effective_config.get('lambda_itm', 0.0) > 0:
            loss_itm_meter.update(loss_itm.item(), pixel_values.size(0))
        
        # Get learning rate (needed for all processes)
        lr = optimizer.param_groups[0]["lr"]
        
        # Update progress bar with current and average loss (only on main process)
        if utils.is_main_process():
            pbar.set_postfix(OrderedDict([
                ('loss', f'{loss.item():.4f}'),
                ('loss_avg', f'{loss_meter.avg:.4f}'),
                ('itc', f'{loss_itc.item():.4f}'),
                ('itm', f'{loss_itm.item():.4f}' if effective_config.get('use_itm', True) and effective_config.get('lambda_itm', 0.0) > 0 else '0.0000'),
                ('lr', f'{lr:.6f}'),
                ('α', f'{alpha.item():.3f}')
            ]))
    
    return {
        'loss': loss_meter.avg,
        'loss_itc': loss_itc_meter.avg,
        'loss_itm': loss_itm_meter.avg,
        'lr': lr,
        'alpha': alpha.item()
    }


@torch.no_grad()
def evaluate_retrieval_with_progress(model, data_loader, device, config):
    """Evaluate retrieval with progress bar"""
    model.eval()
    
    # Get the dataset to access text data
    dataset = data_loader.dataset
    
    # Check if this is a retrieval eval dataset with text attribute
    has_text_data = hasattr(dataset, 'text')
    
    if not has_text_data:
        print("Warning: Dataset doesn't have text attribute, skipping retrieval evaluation")
        return {}
    
    # Get tokenizer for text processing
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['text_config']['model_name'])
    
    # Storage for embeddings and indices
    vision_embeds_uni_list = []
    image_indices_list = []  # Track which indices we processed
    
    # Extract vision features from images
    if utils.is_main_process():
        pbar = tqdm(data_loader, desc='Extracting image features', ncols=120)
    else:
        pbar = data_loader
    
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        batch_indices = batch['indices']  # Get the original dataset indices
        
        # Get vision embeddings (unimodal)
        vision_embeds_uni = model.module.encode_vision_only(pixel_values) if hasattr(model, 'module') else model.encode_vision_only(pixel_values)
        vision_embeds_uni_list.append(vision_embeds_uni)  # Keep on device for now
        image_indices_list.extend(batch_indices)  # Track indices
    
    # Concatenate all vision embeddings (still on device)
    vision_embeds_uni = torch.cat(vision_embeds_uni_list, dim=0)
    image_indices = torch.tensor(image_indices_list, device=device)
    
    # For DDP, gather vision embeddings and indices from all processes
    if utils.get_world_size() > 1:
        # Gather embeddings
        vision_embeds_all = [torch.zeros_like(vision_embeds_uni) for _ in range(utils.get_world_size())]
        torch.distributed.all_gather(vision_embeds_all, vision_embeds_uni)
        
        # Gather indices
        indices_all = [torch.zeros_like(image_indices) for _ in range(utils.get_world_size())]
        torch.distributed.all_gather(indices_all, image_indices)
        
        # Concatenate gathered data
        vision_embeds_uni = torch.cat(vision_embeds_all, dim=0)
        image_indices = torch.cat(indices_all, dim=0)
        
        # Sort by indices to restore original order
        sorted_indices = torch.argsort(image_indices)
        vision_embeds_uni = vision_embeds_uni[sorted_indices]
        image_indices = image_indices[sorted_indices]
    
    # Move to CPU after gathering and sorting
    vision_embeds_uni = vision_embeds_uni.cpu()
    image_indices = image_indices.cpu()
    
    # Sanity check: ensure we have all expected indices
    if utils.is_main_process():
        expected_indices = list(range(len(dataset)))
        actual_indices = sorted(image_indices.tolist())
        if actual_indices != expected_indices:
            print(f"WARNING: Index mismatch! Expected {len(expected_indices)} indices, got {len(actual_indices)}")
            print(f"Missing indices: {set(expected_indices) - set(actual_indices)}")
        else:
            print(f"✓ Successfully gathered all {len(dataset)} image embeddings in correct order")
    
    # Now process all texts from the dataset
    text_embeds_uni_list = []
    batch_size = config.get('batch_size', 32)
    all_texts = dataset.text
    
    if utils.is_main_process():
        print(f"Processing {len(all_texts)} text captions...")
    
    # Process texts in batches
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:min(i+batch_size, len(all_texts))]
        
        # Tokenize batch
        text_inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=config.get('max_words', 77),
            return_tensors='pt'
        )
        
        input_ids = text_inputs['input_ids'].to(device, non_blocking=True)
        attention_mask = text_inputs['attention_mask'].to(device, non_blocking=True)
        
        # Get text embeddings
        text_embeds_uni = model.module.encode_text_only(input_ids, attention_mask) if hasattr(model, 'module') else model.encode_text_only(input_ids, attention_mask)
        text_embeds_uni_list.append(text_embeds_uni)  # Keep on device for now
    
    # Concatenate all text embeddings (still on device)
    text_embeds_uni = torch.cat(text_embeds_uni_list, dim=0)
    
    # Move to CPU after concatenation
    text_embeds_uni = text_embeds_uni.cpu()
    
    # Only compute metrics on main process
    if utils.is_main_process():
        print("Computing retrieval metrics...")
        
        # Normalize embeddings
        vision_embeds_uni = F.normalize(vision_embeds_uni, dim=-1)
        text_embeds_uni = F.normalize(text_embeds_uni, dim=-1)
        
        # Compute similarity matrix
        sims_matrix = vision_embeds_uni @ text_embeds_uni.t()
        
        # Get ground truth mappings from dataset
        img2txt = dataset.img2txt if hasattr(dataset, 'img2txt') else {}
        txt2img = dataset.txt2img if hasattr(dataset, 'txt2img') else {}
        
        # Compute retrieval metrics
        scores = compute_retrieval_scores(sims_matrix.cpu().numpy(), img2txt, txt2img)
    else:
        # Non-main processes return dummy scores
        scores = {
            'i2t_r1': 0, 'i2t_r5': 0, 'i2t_r10': 0,
            't2i_r1': 0, 't2i_r5': 0, 't2i_r10': 0
        }
    
    results = {
        'i2t_r1': scores['i2t_r1'],
        'i2t_r5': scores['i2t_r5'],
        'i2t_r10': scores['i2t_r10'],
        't2i_r1': scores['t2i_r1'],
        't2i_r5': scores['t2i_r5'],
        't2i_r10': scores['t2i_r10'],
        'mean_r1': (scores['i2t_r1'] + scores['t2i_r1']) / 2,
        'mean_r5': (scores['i2t_r5'] + scores['t2i_r5']) / 2,
        'mean_r10': (scores['i2t_r10'] + scores['t2i_r10']) / 2,
    }
    
    return results


def compute_retrieval_scores(sims_matrix, img2txt, txt2img):
    """Compute retrieval scores from similarity matrix"""
    # Image->Text
    ranks = np.zeros(sims_matrix.shape[0])
    for index, score in enumerate(sims_matrix):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        if index in img2txt:
            for i in img2txt[index]:
                matches = np.where(inds == i)[0]
                if len(matches) > 0:
                    tmp = matches[0]
                    if tmp < rank:
                        rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks) if len(ranks) > 0 else 0
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks) if len(ranks) > 0 else 0
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks) if len(ranks) > 0 else 0
    
    # Text->Image 
    ranks = np.zeros(sims_matrix.shape[1])
    for index, score in enumerate(sims_matrix.T):
        inds = np.argsort(score)[::-1]
        if index in txt2img:
            matches = np.where(inds == txt2img[index])[0]
            if len(matches) > 0:
                ranks[index] = matches[0]
            else:
                ranks[index] = 1e20
        else:
            ranks[index] = 1e20
        
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks) if len(ranks) > 0 else 0
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks) if len(ranks) > 0 else 0
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks) if len(ranks) > 0 else 0        

    return {
        'i2t_r1': tr1,
        'i2t_r5': tr5,
        'i2t_r10': tr10,
        't2i_r1': ir1,
        't2i_r5': ir5,
        't2i_r10': ir10,
    }


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    
    # Fix seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # Create datasets
    if utils.is_main_process():
        print("\n" + "="*80)
        print("DATASET CREATION")
        print("="*80)
    
    if config.get('dataset') == 'retrieval_coco':
        train_dataset, val_dataset, test_dataset = create_dataset('retrieval_coco', config)
    else:
        train_dataset = create_dataset(config.get('dataset', 'retrieval_train'), config)
        val_dataset = create_dataset(config.get('dataset_val', 'retrieval_val'), config)
        test_dataset = None
    
    if utils.is_main_process():
        print(f'Training samples: {len(train_dataset):,}')
        print(f'Validation samples: {len(val_dataset):,}')
        if test_dataset:
            print(f'Test samples: {len(test_dataset):,}')
    
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
    if utils.is_main_process():
        print("\n" + "="*80)
        print("MODEL CREATION")
        print("="*80)
    
    model = CrossModalVLM(
        vision_config=config['vision_config'],
        text_config=config['text_config'],
        cross_modal_config=config['cross_modal_config'],
        pooling_config=config['pooling_config'],
    )
    
    # Analyze parameters BEFORE loading pretrained weights (only on main process)
    if utils.is_main_process():
        param_analysis = analyze_model_parameters(model)
    else:
        param_analysis = {}
    
    # Load pretrained checkpoint
    if args.pretrained:
        if utils.is_main_process():
            print(f"\nLoading pretrained checkpoint: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if utils.is_main_process():
            print("✓ Pretrained checkpoint loaded")
    
    # Fine-tuning strategy
    if config.get('freeze_encoders', False):
        if utils.is_main_process():
            print("\nFreezing encoder parameters")
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        
        # Re-analyze after freezing
        if utils.is_main_process():
            print("\nParameter counts after freezing encoders:")
            analyze_model_parameters(model)
    
    model = model.to(device)
    
    # Create ITM loss function and move to device (it has learnable parameters)
    itm_loss_fn = ITMLoss(
        margin_min=config.get('itm_margin_min', 0.2), 
        margin_max=config.get('itm_margin_max', 0.5),
        embed_dim=config['pooling_config']['projection_dim']
    ).to(device)
    
    # Create optimizer (include ITM head parameters if ITM is enabled)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.get('use_itm', True):
        trainable_params += list(itm_loss_fn.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['init_lr'],
        weight_decay=config['weight_decay']
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_r1 = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'itm_loss_state' in checkpoint and config.get('use_itm', True):
            itm_loss_fn.load_state_dict(checkpoint['itm_loss_state'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_r1' in checkpoint:
            best_r1 = checkpoint['best_r1']
        if utils.is_main_process():
            print(f'✓ Resumed from checkpoint: {args.checkpoint}')
    
    # Setup distributed training
    model_without_ddp = model
    itm_loss_fn_without_ddp = itm_loss_fn
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True  # Some heads (MLM, MIM, ITM) may not be used in retrieval
        )
        model_without_ddp = model.module
        
        # Also wrap ITM loss in DDP if using it (it has learnable parameters)
        if config.get('use_itm', True):
            itm_loss_fn = torch.nn.parallel.DistributedDataParallel(
                itm_loss_fn,
                device_ids=[args.gpu]
            )
            itm_loss_fn_without_ddp = itm_loss_fn.module
    
    # Training loop
    if utils.is_main_process():
        print("\n" + "="*80)
        print("TRAINING START")
        print("="*80)
    start_time = time.time()
    
    # Get save frequency
    save_freq = config.get('save_freq', args.save_freq)
    if utils.is_main_process():
        print(f"Model will be saved every {save_freq} epochs")
    
    # Check if multi-stage training is enabled
    enable_multistage = config.get('training_stages', {}).get('enable_multistage', False)
    
    if enable_multistage:
        # Multi-stage training
        if utils.is_main_process():
            print("\n" + "="*80)
            print("MULTI-STAGE TRAINING ENABLED")
            print("="*80)
        
        stages = ['stage_a', 'stage_b', 'stage_c']
        stage_names = ['A (Stabilize)', 'B (Align)', 'C (Task Tune)']
        global_epoch = start_epoch
        
        for stage_idx, (stage_key, stage_name) in enumerate(zip(stages, stage_names)):
            stage_config = config.get('training_stages', {}).get(stage_key, {})
            stage_epochs = stage_config.get('epochs', 0)
            
            if stage_epochs <= 0:
                if utils.is_main_process():
                    print(f"\nSkipping Stage {stage_name} (0 epochs)")
                continue
            
            if utils.is_main_process():
                print("\n" + "="*80)
                print(f"STAGE {stage_name.split()[0]}: {stage_name}")
                print("="*80)
                print(f"Training for {stage_epochs} epochs")
                print(f"Learning rate: {stage_config.get('init_lr', config['init_lr'])}")
                print(f"Loss weights: ITC={stage_config.get('lambda_itc', 1.0)}, ITM={stage_config.get('lambda_itm', 1.0)}")
            
            # Configure encoder freezing for this stage
            if utils.is_main_process():
                print("\nConfiguring model for this stage:")
            configure_encoder_freezing(model_without_ddp, stage_config, stage_name)
            
            # Update optimizer with stage-specific learning rate
            stage_lr = stage_config.get('init_lr', config['init_lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = stage_lr
            
            # Merge stage config with main config for this stage
            stage_effective_config = {**config}
            stage_effective_config.update(stage_config)
            stage_effective_config['max_epoch'] = stage_epochs
            stage_effective_config['init_lr'] = stage_lr
            
            # Train for this stage
            for epoch_in_stage in range(stage_epochs):
                # Cosine learning rate schedule within this stage
                cosine_lr_schedule(optimizer, epoch_in_stage, stage_epochs, stage_lr, config['min_lr'])
                
                # Train
                if utils.is_main_process():
                    print(f"\n[Stage {stage_name.split()[0]} | Epoch {epoch_in_stage+1}/{stage_epochs} | Global Epoch {global_epoch+1}]")
                
                train_stats = train_one_epoch_with_progress(
                    model, train_loader, optimizer, epoch_in_stage, device, 
                    config, itm_loss_fn, stage_config=stage_config
                )
                
                # Evaluate every epoch
                if utils.is_main_process():
                    print("\nEvaluating...")
                val_results = evaluate_retrieval_with_progress(model, val_loader, device, config)
                
                # Print results (only on main process)
                if utils.is_main_process():
                    print("\n" + "-"*80)
                    print(f"STAGE {stage_name.split()[0]} | EPOCH {epoch_in_stage+1} RESULTS:")
                    print("-"*80)
                    print(f"Training Loss: {train_stats['loss']:.4f} (ITC: {train_stats['loss_itc']:.4f}, ITM: {train_stats['loss_itm']:.4f})")
                    print(f"Learning Rate: {train_stats['lr']:.6f}, Alpha: {train_stats['alpha']:.3f}")
                    print("\nValidation Recall (Unimodal/Fast):")
                    print(f"  Image→Text: R@1={val_results['i2t_r1']:.2f}%, R@5={val_results['i2t_r5']:.2f}%, R@10={val_results['i2t_r10']:.2f}%")
                    print(f"  Text→Image: R@1={val_results['t2i_r1']:.2f}%, R@5={val_results['t2i_r5']:.2f}%, R@10={val_results['t2i_r10']:.2f}%")
                    print(f"  Mean: R@1={val_results['mean_r1']:.2f}%, R@5={val_results['mean_r5']:.2f}%, R@10={val_results['mean_r10']:.2f}%")
                    
                    if 'mean_r1_cross' in val_results:
                        print("\nValidation Recall (Cross-modal/Accurate):")
                        print(f"  Mean: R@1={val_results['mean_r1_cross']:.2f}%, R@5={val_results['mean_r5_cross']:.2f}%, R@10={val_results['mean_r10_cross']:.2f}%")
                    
                    print("-"*80)
                
                current_r1 = val_results['mean_r1']
                
                if utils.is_main_process():
                    # Save checkpoint every N epochs
                    if (global_epoch + 1) % save_freq == 0:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'itm_loss_state': itm_loss_fn_without_ddp.state_dict() if config.get('use_itm', True) else None,
                            'optimizer': optimizer.state_dict(),
                            'config': config,
                            'epoch': global_epoch,
                            'stage': stage_name,
                            'best_r1': best_r1,
                            'param_analysis': param_analysis,
                            'val_results': val_results,
                            'train_stats': train_stats
                        }
                        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{global_epoch+1:03d}_stage_{stage_key}.pth')
                        torch.save(save_obj, checkpoint_path)
                        print(f"✓ Checkpoint saved: {checkpoint_path}")
                    
                    # Save best model
                    is_best = current_r1 > best_r1
                    if is_best:
                        best_r1 = current_r1
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'itm_loss_state': itm_loss_fn_without_ddp.state_dict() if config.get('use_itm', True) else None,
                            'optimizer': optimizer.state_dict(),
                            'config': config,
                            'epoch': global_epoch,
                            'stage': stage_name,
                            'best_r1': best_r1,
                            'param_analysis': param_analysis,
                            'val_results': val_results,
                            'train_stats': train_stats
                        }
                        best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                        torch.save(save_obj, best_path)
                        print(f"★ New best model! Mean R@1: {best_r1:.2f}%")
                    
                    # Log stats
                    log_stats = {
                        'epoch': global_epoch + 1,
                        'stage': stage_name,
                        **train_stats,
                        **val_results,
                        'best_r1': best_r1
                    }
                    
                    with open(os.path.join(args.output_dir, "training_log.json"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                
                global_epoch += 1
                
                if args.distributed:
                    dist.barrier()
    else:
        # Single-stage training (original behavior)
        if utils.is_main_process():
            print("\nSingle-stage training mode")
        
        for epoch in range(start_epoch, config['max_epoch']):
            # Cosine learning rate schedule
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            # Train
            if utils.is_main_process():
                print(f"\n[Epoch {epoch+1}/{config['max_epoch']}]")
            train_stats = train_one_epoch_with_progress(model, train_loader, optimizer, epoch, device, config, itm_loss_fn)
            
            # Evaluate every epoch
            if utils.is_main_process():
                print("\nEvaluating...")
            val_results = evaluate_retrieval_with_progress(model, val_loader, device, config)
            
            # Print results (only on main process)
            if utils.is_main_process():
                print("\n" + "-"*80)
                print(f"EPOCH {epoch+1} RESULTS:")
                print("-"*80)
                print(f"Training Loss: {train_stats['loss']:.4f} (ITC: {train_stats['loss_itc']:.4f}, ITM: {train_stats['loss_itm']:.4f})")
                print(f"Learning Rate: {train_stats['lr']:.6f}, Alpha: {train_stats['alpha']:.3f}")
                print("\nValidation Recall (Unimodal/Fast):")
                print(f"  Image→Text: R@1={val_results['i2t_r1']:.2f}%, R@5={val_results['i2t_r5']:.2f}%, R@10={val_results['i2t_r10']:.2f}%")
                print(f"  Text→Image: R@1={val_results['t2i_r1']:.2f}%, R@5={val_results['t2i_r5']:.2f}%, R@10={val_results['t2i_r10']:.2f}%")
                print(f"  Mean: R@1={val_results['mean_r1']:.2f}%, R@5={val_results['mean_r5']:.2f}%, R@10={val_results['mean_r10']:.2f}%")
                
                if 'mean_r1_cross' in val_results:
                    print("\nValidation Recall (Cross-modal/Accurate):")
                    print(f"  Mean: R@1={val_results['mean_r1_cross']:.2f}%, R@5={val_results['mean_r5_cross']:.2f}%, R@10={val_results['mean_r10_cross']:.2f}%")
                
                print("-"*80)
            
            current_r1 = val_results['mean_r1']
            
            if utils.is_main_process():
                # Save checkpoint every N epochs
                if (epoch + 1) % save_freq == 0:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'itm_loss_state': itm_loss_fn_without_ddp.state_dict() if config.get('use_itm', True) else None,
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'best_r1': best_r1,
                        'param_analysis': param_analysis,
                        'val_results': val_results,
                        'train_stats': train_stats
                    }
                    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
                    torch.save(save_obj, checkpoint_path)
                    print(f"✓ Checkpoint saved: {checkpoint_path}")
                
                # Save best model
                is_best = current_r1 > best_r1
                if is_best:
                    best_r1 = current_r1
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'itm_loss_state': itm_loss_fn_without_ddp.state_dict() if config.get('use_itm', True) else None,
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'best_r1': best_r1,
                        'param_analysis': param_analysis,
                        'val_results': val_results,
                        'train_stats': train_stats
                    }
                    best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                    torch.save(save_obj, best_path)
                    print(f"★ New best model! Mean R@1: {best_r1:.2f}%")
                
                # Log stats
                log_stats = {
                    'epoch': epoch + 1,
                    **train_stats,
                    **val_results,
                    'best_r1': best_r1
                }
                
                with open(os.path.join(args.output_dir, "training_log.json"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
            if args.distributed:
                dist.barrier()
    
    # Training complete
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    if utils.is_main_process():
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total training time: {total_time_str}")
        print(f"Best Mean R@1: {best_r1:.2f}%")
        print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/coco_base.yaml')
    parser.add_argument('--output_dir', default='output/COCO_Retrieval_Experiment')
    parser.add_argument('--pretrained', default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--checkpoint', default='', help='Resume from checkpoint')
    parser.add_argument('--save_freq', default=5, type=int, help='Save checkpoint every N epochs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config, 'r'))
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    main(args, config)
