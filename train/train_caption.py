"""
Stage C: Task Fine-tuning - Caption Generation
Attach a causal decoder for autoregressive caption generation.
The encoders remain non-causal as per method.tex.
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
from src.models.caption_decoder import CaptionDecoder
from src.data import create_dataset, create_sampler, create_loader
from src.training import utils
from src.training.utils import warmup_lr_schedule, cosine_lr_schedule
from src.evaluation.caption_metrics import compute_caption_metrics


def train_one_epoch(model, decoder, data_loader, optimizer, epoch, device, config):
    """Train for one epoch on caption generation task"""
    model.train()
    decoder.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('ppl', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = f'Caption Training - Epoch: [{epoch}]'
    print_freq = 50
    
    data_loader.sampler.set_epoch(epoch)
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        caption_ids = batch['caption_ids'].to(device, non_blocking=True)  # (batch, seq_len)
        caption_mask = batch['caption_mask'].to(device, non_blocking=True)
        
        # Warmup learning rate
        step = epoch * len(data_loader) + i
        total_steps = config['max_epoch'] * len(data_loader)
        warmup_steps = int(config.get('warmup_ratio', 0.05) * total_steps)
        
        if step < warmup_steps:
            warmup_lr_schedule(optimizer, step, warmup_steps, 0, config['init_lr'])
        
        optimizer.zero_grad()
        
        # Get vision features from encoder (frozen or fine-tuned)
        with torch.set_grad_enabled(not config.get('freeze_vision_encoder', True)):
            # Forward pass through vision encoder only
            vision_hidden = model.vision_encoder(pixel_values)  # (batch, n_vision, vision_dim)
            
            # Optional: pass through cross-modal layers for better features
            if config.get('use_cross_modal_features', True):
                # Create dummy text input for cross-modal processing
                batch_size = pixel_values.shape[0]
                dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                dummy_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                
                text_hidden = model.text_encoder(dummy_input_ids, dummy_mask)
                
                # Pass through interaction layers
                for layer_idx, interaction_layer in enumerate(model.interaction_layers):
                    vision_hidden, text_hidden, aux = interaction_layer(
                        vision_hidden, 
                        text_hidden,
                        vision_mask=None,
                        text_mask=dummy_mask,
                        return_attention_weights=False
                    )
        
        # Generate captions autoregressively
        # Shift right for decoder input (add BOS, remove last token)
        decoder_input_ids = caption_ids[:, :-1]  # Remove EOS for input
        decoder_labels = caption_ids[:, 1:]      # Remove BOS for labels
        decoder_mask = caption_mask[:, 1:]       # Adjust mask
        
        # Forward through caption decoder
        logits = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=vision_hidden,
            encoder_attention_mask=None,  # All vision tokens are valid
        )
        
        # Compute loss (cross-entropy)
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            decoder_labels.reshape(-1),
            ignore_index=-100,  # Padding token
            reduction='none'
        )
        
        # Mask out padding tokens
        loss = loss * decoder_mask.reshape(-1)
        loss = loss.sum() / decoder_mask.sum()
        
        # Calculate perplexity
        ppl = torch.exp(loss)
        
        loss.backward()
        
        # Gradient clipping
        if config.get('grad_clip', 1.0) > 0:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(decoder.parameters()), 
                config['grad_clip']
            )
        
        optimizer.step()
        
        # Log metrics
        metric_logger.update(loss=loss.item())
        metric_logger.update(ppl=ppl.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_caption(model, decoder, data_loader, tokenizer, device, config):
    """Evaluate caption generation"""
    model.eval()
    decoder.eval()
    
    predictions = []
    references = []
    
    for batch in data_loader:
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        captions = batch['captions']  # List of string captions
        image_ids = batch['image_ids']
        
        # Get vision features
        vision_hidden = model.vision_encoder(pixel_values)
        
        # Optional: use cross-modal features
        if config.get('use_cross_modal_features', True):
            batch_size = pixel_values.shape[0]
            dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            dummy_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            text_hidden = model.text_encoder(dummy_input_ids, dummy_mask)
            
            for interaction_layer in model.interaction_layers:
                vision_hidden, text_hidden, _ = interaction_layer(
                    vision_hidden, 
                    text_hidden,
                    vision_mask=None,
                    text_mask=dummy_mask,
                    return_attention_weights=False
                )
        
        # Generate captions
        generated_ids = decoder.generate(
            encoder_hidden_states=vision_hidden,
            max_length=config.get('max_caption_length', 30),
            num_beams=config.get('num_beams', 3),
            temperature=config.get('temperature', 1.0),
            top_k=config.get('top_k', 50),
            top_p=config.get('top_p', 0.95),
            repetition_penalty=config.get('repetition_penalty', 1.0),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Decode generated captions
        generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Collect predictions and references
        for pred, ref, img_id in zip(generated_captions, captions, image_ids):
            predictions.append({'image_id': img_id, 'caption': pred})
            references.append({'image_id': img_id, 'caption': ref})
    
    # Compute metrics (BLEU, METEOR, CIDEr, etc.)
    metrics = compute_caption_metrics(predictions, references)
    
    return metrics


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
    print("Creating captioning datasets")
    train_dataset = create_dataset('caption_train', config)
    val_dataset = create_dataset('caption_val', config)
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(val_dataset)}')
    
    # Get tokenizer from dataset
    tokenizer = train_dataset.tokenizer
    vocab_size = len(tokenizer)
    
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
        batch_size=config['eval_batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    
    # Create model
    print("Creating model and decoder")
    model = CrossModalVLM(
        vision_config=config['vision_config'],
        text_config=config['text_config'],
        cross_modal_config=config['cross_modal_config'],
        pooling_config=config['pooling_config'],
    )
    
    # Create caption decoder (separate causal decoder as per method.tex)
    decoder = CaptionDecoder(
        vocab_size=vocab_size,
        embed_dim=config['decoder_embed_dim'],
        num_layers=config['decoder_layers'],
        num_heads=config['decoder_heads'],
        ff_dim=config['decoder_ff_dim'],
        dropout=config.get('decoder_dropout', 0.1),
        max_seq_length=config.get('max_caption_length', 128),
        encoder_hidden_dim=config['vision_config']['hidden_dim'],
    )
    
    # Load pretrained checkpoint
    if args.pretrained:
        print(f"Loading pretrained checkpoint from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Pretrained checkpoint loaded")
    
    # Freeze encoders (as per method.tex - encoders remain non-causal)
    if config.get('freeze_vision_encoder', True):
        print("Freezing vision encoder")
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    
    if config.get('freeze_text_encoder', True):
        print("Freezing text encoder")
        for param in model.text_encoder.parameters():
            param.requires_grad = False
    
    if config.get('freeze_interaction_layers', True):
        print("Freezing interaction layers")
        for param in model.interaction_layers.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    model_params = [p for p in model.parameters() if p.requires_grad]
    decoder_params = list(decoder.parameters())
    trainable_params = model_params + decoder_params
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    model = model.to(device)
    decoder = decoder.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [
            {'params': model_params, 'lr': config['init_lr'] * config.get('encoder_lr_scale', 0.1)},
            {'params': decoder_params, 'lr': config['init_lr']},
        ],
        weight_decay=config['weight_decay']
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_cider = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        decoder.load_state_dict(checkpoint['decoder'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_cider' in checkpoint:
            best_cider = checkpoint['best_cider']
        print(f'Resumed from checkpoint: {args.checkpoint}')
    
    # Setup distributed training
    model_without_ddp = model
    decoder_without_ddp = decoder
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[args.gpu])
        model_without_ddp = model.module
        decoder_without_ddp = decoder.module
    
    # Evaluation only mode
    if args.evaluate:
        print("Evaluation only mode")
        metrics = evaluate_caption(model, decoder, val_loader, tokenizer, device, config)
        
        print("\n=== Caption Generation Results ===")
        print(f"BLEU-1: {metrics.get('bleu1', 0):.2f}")
        print(f"BLEU-2: {metrics.get('bleu2', 0):.2f}")
        print(f"BLEU-3: {metrics.get('bleu3', 0):.2f}")
        print(f"BLEU-4: {metrics.get('bleu4', 0):.2f}")
        print(f"METEOR: {metrics.get('meteor', 0):.2f}")
        print(f"CIDEr: {metrics.get('cider', 0):.2f}")
        print(f"ROUGE-L: {metrics.get('rouge_l', 0):.2f}")
        return
    
    # Training loop
    print("Start caption generation training")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['max_epoch']):
        # Cosine learning rate schedule
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        
        train_stats = train_one_epoch(model, decoder, train_loader, optimizer, epoch, device, config)
        
        # Evaluate
        if (epoch + 1) % config.get('eval_freq', 1) == 0:
            metrics = evaluate_caption(model, decoder, val_loader, tokenizer, device, config)
            
            current_cider = metrics.get('cider', 0)
            
            print(f"\nEpoch {epoch} - Caption Metrics:")
            print(f"BLEU-4: {metrics.get('bleu4', 0):.2f}, METEOR: {metrics.get('meteor', 0):.2f}, CIDEr: {current_cider:.2f}")
        else:
            current_cider = 0
            metrics = {}
        
        if utils.is_main_process():
            # Save checkpoint
            is_best = current_cider > best_cider
            best_cider = max(current_cider, best_cider)
            
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'decoder': decoder_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_cider': best_cider,
            }
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth')
            torch.save(save_obj, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                torch.save(save_obj, best_path)
                print(f"New best CIDEr: {best_cider:.2f}")
            
            # Log stats
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **metrics,
                'epoch': epoch,
            }
            
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.distributed:
            dist.barrier()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Caption training time: {total_time_str}')
    print(f'Best CIDEr: {best_cider:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption.yaml')
    parser.add_argument('--output_dir', default='output/Caption')
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