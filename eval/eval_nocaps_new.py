"""
Evaluation script for NoCaps (Novel Object Captioning at Scale)
Evaluates caption generation on the NoCaps benchmark.
"""

import argparse
import os
import yaml
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vlm_model import CrossModalVLM
from src.models.caption_decoder import CaptionDecoder
from src.data import create_dataset, create_sampler
from src.data.utils import save_result
from src.training import utils


@torch.no_grad()
def evaluate(model, decoder, data_loader, tokenizer, device, config):
    """Evaluate caption generation on NoCaps"""
    model.eval()
    decoder.eval()
    
    results = []
    
    print("Generating captions for NoCaps evaluation...")
    
    for batch in tqdm(data_loader, desc="Generating"):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        image_ids = batch['image_id']
        
        batch_size = pixel_values.shape[0]
        
        # Get vision features
        vision_hidden = model.vision_encoder(pixel_values)
        
        # Optional: use cross-modal features for better caption generation
        if config.get('use_cross_modal_features', True):
            # Create dummy text input for cross-modal processing
            dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            dummy_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            text_hidden = model.text_encoder(dummy_input_ids, dummy_mask)
            
            # Pass through interaction layers
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
            max_length=config.get('max_length', 30),
            min_length=config.get('min_length', 5),
            num_beams=config.get('num_beams', 3),
            temperature=config.get('temperature', 1.0),
            top_k=config.get('top_k', 50),
            top_p=config.get('top_p', 0.95),
            repetition_penalty=config.get('repetition_penalty', 1.1),
            length_penalty=config.get('length_penalty', 1.0),
            no_repeat_ngram_size=config.get('no_repeat_ngram_size', 3),
            early_stopping=config.get('early_stopping', True),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Decode generated captions
        captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Collect results
        for caption, img_id in zip(captions, image_ids):
            # Clean up caption
            caption = caption.strip()
            if not caption:
                caption = "a picture"  # Fallback caption
            
            results.append({
                "image_id": img_id.item() if torch.is_tensor(img_id) else img_id,
                "caption": caption
            })
    
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
    print("Creating NoCaps datasets")
    val_dataset, test_dataset = create_dataset('nocaps', config)
    
    # Get tokenizer
    tokenizer = val_dataset.tokenizer if hasattr(val_dataset, 'tokenizer') else None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['text_config']['model_name'])
    
    vocab_size = len(tokenizer)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        val_sampler = create_sampler([val_dataset], [False], num_tasks, global_rank)[0]
        test_sampler = create_sampler([test_dataset], [False], num_tasks, global_rank)[0]
    else:
        val_sampler = None
        test_sampler = None
    
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=config['batch_size'],
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
    
    # Create caption decoder
    decoder = CaptionDecoder(
        vocab_size=vocab_size,
        embed_dim=config.get('decoder_embed_dim', 512),
        num_layers=config.get('decoder_layers', 6),
        num_heads=config.get('decoder_heads', 8),
        ff_dim=config.get('decoder_ff_dim', 2048),
        dropout=config.get('decoder_dropout', 0.1),
        max_seq_length=config.get('max_length', 128),
        encoder_hidden_dim=config['vision_config']['hidden_dim'],
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Load model state
    model_state = checkpoint.get('model', checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    if missing_keys:
        print(f"Missing keys in model: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in model: {unexpected_keys}")
    
    # Load decoder state
    if 'decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder'])
        print("Decoder loaded from checkpoint")
    elif 'caption_decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['caption_decoder'])
        print("Caption decoder loaded from checkpoint")
    else:
        print("Warning: Decoder not found in checkpoint, using random initialization")
    
    model = model.to(device)
    decoder = decoder.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[args.gpu])
        model = model.module
        decoder = decoder.module
    
    # Evaluate on validation set
    print("\nEvaluating on NoCaps validation set...")
    val_results = evaluate(model, decoder, val_loader, tokenizer, device, config)
    
    # Evaluate on test set
    print("\nEvaluating on NoCaps test set...")
    test_results = evaluate(model, decoder, test_loader, tokenizer, device, config)
    
    # Save results
    if utils.is_main_process():
        # Save validation results
        val_result_file = save_result(
            val_results, 
            args.result_dir, 
            'val', 
            remove_duplicate='image_id'
        )
        print(f"Validation results saved to {val_result_file}")
        
        # Save test results
        test_result_file = save_result(
            test_results, 
            args.result_dir, 
            'test', 
            remove_duplicate='image_id'
        )
        print(f"Test results saved to {test_result_file}")
        
        # Print sample results
        print("\n" + "="*50)
        print("SAMPLE GENERATED CAPTIONS")
        print("="*50)
        
        num_samples = min(10, len(val_results))
        print(f"\nValidation samples (showing {num_samples}):")
        for i in range(num_samples):
            result = val_results[i]
            print(f"  Image {result['image_id']}: {result['caption']}")
        
        print("="*50)
        
        # Save all results in a single JSON for convenience
        all_results = {
            'validation': val_results,
            'test': test_results,
            'config': config
        }
        
        all_results_file = os.path.join(args.output_dir, 'nocaps_all_results.json')
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {all_results_file}")
        
        print("\nSubmission files are ready for evaluation on the NoCaps server:")
        print(f"  - Validation: {val_result_file}")
        print(f"  - Test: {test_result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/nocaps.yaml')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='output/NoCaps')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config, 'r'))
    
    args.result_dir = os.path.join(args.output_dir, 'result')
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    main(args, config)
