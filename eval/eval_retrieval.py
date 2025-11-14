"""
Evaluation script for Cross-Modal Retrieval
Evaluates both image-to-text and text-to-image retrieval using dual embeddings.
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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vlm_model import CrossModalVLM
from src.data import create_dataset, create_sampler
from src.training import utils


@torch.no_grad()
def compute_similarity_matrix(vision_embeds, text_embeds, chunk_size=256):
    """
    Compute similarity matrix between vision and text embeddings.
    Uses chunking for memory efficiency.
    """
    device = vision_embeds.device
    n_images = vision_embeds.shape[0]
    n_texts = text_embeds.shape[0]
    
    similarity_matrix = torch.zeros(n_images, n_texts, device=device)
    
    # Process in chunks to avoid OOM
    for i in range(0, n_images, chunk_size):
        end_i = min(i + chunk_size, n_images)
        vision_chunk = vision_embeds[i:end_i]
        
        for j in range(0, n_texts, chunk_size):
            end_j = min(j + chunk_size, n_texts)
            text_chunk = text_embeds[j:end_j]
            
            # Compute cosine similarity
            sim = torch.matmul(vision_chunk, text_chunk.t())
            similarity_matrix[i:end_i, j:end_j] = sim
    
    return similarity_matrix


def compute_retrieval_metrics(similarity_matrix, k_values=[1, 5, 10]):
    """
    Compute retrieval metrics (Recall@K) from similarity matrix.
    Assumes diagonal correspondence (i-th image matches i-th text).
    """
    n_samples = similarity_matrix.shape[0]
    device = similarity_matrix.device
    
    # Image-to-Text retrieval
    i2t_ranks = []
    for i in range(n_samples):
        sim_scores = similarity_matrix[i]
        # Get rank of the correct match (diagonal element)
        rank = (sim_scores > sim_scores[i]).sum().item() + 1
        i2t_ranks.append(rank)
    
    i2t_ranks = torch.tensor(i2t_ranks, device=device)
    
    # Text-to-Image retrieval
    t2i_ranks = []
    similarity_matrix_t = similarity_matrix.t()
    for i in range(n_samples):
        sim_scores = similarity_matrix_t[i]
        # Get rank of the correct match (diagonal element)
        rank = (sim_scores > sim_scores[i]).sum().item() + 1
        t2i_ranks.append(rank)
    
    t2i_ranks = torch.tensor(t2i_ranks, device=device)
    
    # Compute Recall@K metrics
    metrics = {}
    for k in k_values:
        metrics[f'i2t_r{k}'] = (i2t_ranks <= k).float().mean().item() * 100
        metrics[f't2i_r{k}'] = (t2i_ranks <= k).float().mean().item() * 100
    
    # Mean recall
    metrics['i2t_mean_r'] = i2t_ranks.float().mean().item()
    metrics['t2i_mean_r'] = t2i_ranks.float().mean().item()
    
    # Median rank
    metrics['i2t_median_r'] = i2t_ranks.float().median().item()
    metrics['t2i_median_r'] = t2i_ranks.float().median().item()
    
    return metrics


@torch.no_grad()
def extract_features(model, data_loader, device, use_cross_modal=False):
    """
    Extract vision and text features from the dataset.
    
    Args:
        model: The VLM model
        data_loader: DataLoader for the dataset
        device: Device to use
        use_cross_modal: If True, extract cross-modal features (slower but more accurate)
                        If False, extract unimodal features (faster)
    """
    vision_embeds_list = []
    text_embeds_list = []
    
    print(f"Extracting {'cross-modal' if use_cross_modal else 'unimodal'} features...")
    
    for batch in tqdm(data_loader, desc="Extracting features"):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        if use_cross_modal:
            # Extract cross-modal features (requires full forward pass)
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attention_weights=False
            )
            vision_embeds = outputs['vision_embeds_cross']
            text_embeds = outputs['text_embeds_cross']
        else:
            # Extract unimodal features (faster, independent encoding)
            vision_embeds = model.encode_vision_only(pixel_values)
            text_embeds = model.encode_text_only(input_ids, attention_mask)
        
        vision_embeds_list.append(vision_embeds.cpu())
        text_embeds_list.append(text_embeds.cpu())
    
    # Concatenate all features
    vision_embeds = torch.cat(vision_embeds_list, dim=0)
    text_embeds = torch.cat(text_embeds_list, dim=0)
    
    return vision_embeds, text_embeds


def evaluate_retrieval(model, data_loader, device, config):
    """Main evaluation function for retrieval"""
    model.eval()
    
    # Extract features for both unimodal and cross-modal if requested
    results = {}
    
    # Always evaluate unimodal (fast bi-encoder)
    print("\n=== Evaluating Unimodal (Fast Bi-Encoder) Retrieval ===")
    vision_embeds_uni, text_embeds_uni = extract_features(
        model, data_loader, device, use_cross_modal=False
    )
    
    # Move to GPU for similarity computation
    vision_embeds_uni = vision_embeds_uni.to(device)
    text_embeds_uni = text_embeds_uni.to(device)
    
    # Compute similarity and metrics
    similarity_uni = compute_similarity_matrix(vision_embeds_uni, text_embeds_uni)
    metrics_uni = compute_retrieval_metrics(similarity_uni, k_values=[1, 5, 10])
    
    # Add prefix to metrics
    for key, value in metrics_uni.items():
        results[f'uni_{key}'] = value
    
    # Optionally evaluate cross-modal
    if config.get('eval_cross_modal', True):
        print("\n=== Evaluating Cross-Modal (Accurate) Retrieval ===")
        vision_embeds_cross, text_embeds_cross = extract_features(
            model, data_loader, device, use_cross_modal=True
        )
        
        # Move to GPU for similarity computation
        vision_embeds_cross = vision_embeds_cross.to(device)
        text_embeds_cross = text_embeds_cross.to(device)
        
        # Compute similarity and metrics
        similarity_cross = compute_similarity_matrix(vision_embeds_cross, text_embeds_cross)
        metrics_cross = compute_retrieval_metrics(similarity_cross, k_values=[1, 5, 10])
        
        # Add prefix to metrics
        for key, value in metrics_cross.items():
            results[f'cross_{key}'] = value
    
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
    
    # Create dataset
    print("Creating retrieval evaluation dataset")
    dataset = create_dataset(config['dataset'], config)
    print(f'Number of samples: {len(dataset)}')
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler = create_sampler([dataset], [False], num_tasks, global_rank)[0]
    else:
        sampler = None
    
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
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
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    print("Checkpoint loaded")
    
    model = model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = model.module
    
    # Evaluate
    results = evaluate_retrieval(model, data_loader, device, config)
    
    # Print results
    print("\n" + "="*50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*50)
    
    if config.get('eval_cross_modal', True):
        print("\nCross-Modal (Accurate) Results:")
        print(f"  Image-to-Text:")
        print(f"    R@1: {results['cross_i2t_r1']:.2f}%")
        print(f"    R@5: {results['cross_i2t_r5']:.2f}%")
        print(f"    R@10: {results['cross_i2t_r10']:.2f}%")
        print(f"    Mean Rank: {results['cross_i2t_mean_r']:.2f}")
        print(f"    Median Rank: {results['cross_i2t_median_r']:.2f}")
        
        print(f"  Text-to-Image:")
        print(f"    R@1: {results['cross_t2i_r1']:.2f}%")
        print(f"    R@5: {results['cross_t2i_r5']:.2f}%")
        print(f"    R@10: {results['cross_t2i_r10']:.2f}%")
        print(f"    Mean Rank: {results['cross_t2i_mean_r']:.2f}")
        print(f"    Median Rank: {results['cross_t2i_median_r']:.2f}")
    
    print("\nUnimodal (Fast Bi-Encoder) Results:")
    print(f"  Image-to-Text:")
    print(f"    R@1: {results['uni_i2t_r1']:.2f}%")
    print(f"    R@5: {results['uni_i2t_r5']:.2f}%")
    print(f"    R@10: {results['uni_i2t_r10']:.2f}%")
    print(f"    Mean Rank: {results['uni_i2t_mean_r']:.2f}")
    print(f"    Median Rank: {results['uni_i2t_median_r']:.2f}")
    
    print(f"  Text-to-Image:")
    print(f"    R@1: {results['uni_t2i_r1']:.2f}%")
    print(f"    R@5: {results['uni_t2i_r5']:.2f}%")
    print(f"    R@10: {results['uni_t2i_r10']:.2f}%")
    print(f"    Mean Rank: {results['uni_t2i_mean_r']:.2f}")
    print(f"    Median Rank: {results['uni_t2i_median_r']:.2f}")
    
    print("="*50)
    
    # Save results
    if utils.is_main_process():
        result_file = os.path.join(args.output_dir, 'retrieval_results.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/eval_retrieval.yaml')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='output/eval_retrieval')
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
