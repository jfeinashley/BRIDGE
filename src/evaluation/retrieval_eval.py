"""
Proper retrieval evaluation for COCO Karpathy split.

This module handles the correct computation of retrieval metrics:
- Image-to-Text retrieval (i2t): Find correct captions for each image
- Text-to-Image retrieval (t2i): Find correct image for each caption
- Handles multiple captions per image (5 for COCO)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    ground_truth: Dict[str, List],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics given similarity matrix and ground truth.
    
    Args:
        similarity_matrix: (n_images, n_captions) similarity scores
        ground_truth: Dict with 'i2t' and 't2i' mappings
        k_values: List of K values for Recall@K
    
    Returns:
        Dictionary with retrieval metrics
    """
    n_images = similarity_matrix.shape[0]
    n_captions = similarity_matrix.shape[1]
    
    metrics = {}
    
    # Image-to-Text Retrieval (i2t)
    # For each image, find its matching captions
    i2t_ranks = []
    for img_idx in range(n_images):
        # Get similarity scores for this image with all captions
        scores = similarity_matrix[img_idx]
        
        # Get indices of captions sorted by similarity (descending)
        _, sorted_indices = scores.sort(descending=True)
        
        # Find the rank of the first correct caption
        correct_captions = ground_truth['i2t'][img_idx]
        min_rank = n_captions + 1  # Initialize to worse than worst possible
        
        for rank, caption_idx in enumerate(sorted_indices.cpu().numpy()):
            if caption_idx in correct_captions:
                min_rank = min(min_rank, rank + 1)  # rank is 0-indexed
                break  # Found the first correct caption
        
        i2t_ranks.append(min_rank)
    
    # Calculate i2t Recall@K
    i2t_ranks = np.array(i2t_ranks)
    for k in k_values:
        recall_k = (i2t_ranks <= k).astype(float).mean() * 100
        metrics[f'i2t_r{k}'] = recall_k
    
    # Text-to-Image Retrieval (t2i)
    # For each caption, find its matching image
    t2i_ranks = []
    for caption_idx in range(n_captions):
        # Get similarity scores for this caption with all images
        scores = similarity_matrix[:, caption_idx]
        
        # Get indices of images sorted by similarity (descending)
        _, sorted_indices = scores.sort(descending=True)
        
        # Find the rank of the correct image
        correct_image = ground_truth['t2i'][caption_idx]
        
        for rank, img_idx in enumerate(sorted_indices.cpu().numpy()):
            if img_idx == correct_image:
                t2i_ranks.append(rank + 1)  # rank is 0-indexed
                break
    
    # Calculate t2i Recall@K
    t2i_ranks = np.array(t2i_ranks)
    for k in k_values:
        recall_k = (t2i_ranks <= k).astype(float).mean() * 100
        metrics[f't2i_r{k}'] = recall_k
    
    # Calculate rSum (sum of all recall values)
    rsum = sum([metrics[f'i2t_r{k}'] for k in k_values] + 
               [metrics[f't2i_r{k}'] for k in k_values])
    metrics['rsum'] = rsum
    
    # Calculate mean rank and median rank
    metrics['i2t_mean_rank'] = i2t_ranks.mean()
    metrics['i2t_median_rank'] = np.median(i2t_ranks)
    metrics['t2i_mean_rank'] = t2i_ranks.mean()
    metrics['t2i_median_rank'] = np.median(t2i_ranks)
    
    return metrics


@torch.no_grad()
def evaluate_retrieval(
    model,
    eval_dataset,
    batch_size: int = 32,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate retrieval performance on the full validation/test set.
    
    Args:
        model: The VLM model
        eval_dataset: COCOKarpathyEvalDataset instance
        batch_size: Batch size for processing
        device: Device to run on
        verbose: Whether to show progress bars
    
    Returns:
        Dictionary with retrieval metrics
    """
    model.eval()
    
    # Get all images and process in batches
    all_images = eval_dataset.get_all_images()
    n_images = len(all_images)
    
    vision_embeds_list = []
    if verbose:
        print(f"Processing {n_images} images...")
    
    for i in tqdm(range(0, n_images, batch_size), disable=not verbose):
        batch_images = all_images[i:min(i + batch_size, n_images)]
        pixel_values = torch.stack(batch_images).to(device)
        
        # Get vision embeddings using encode_vision_only (no cross-modal interaction needed)
        # This is more efficient and avoids shape mismatch issues
        if hasattr(model, 'encode_vision_only'):
            vision_embeds = model.encode_vision_only(pixel_values)
        else:
            # Fallback to unimodal encoding if available
            vision_embeds = model.encode_vision_unimodal(pixel_values)
        
        vision_embeds_list.append(vision_embeds.cpu())
    
    vision_embeds = torch.cat(vision_embeds_list, dim=0)
    # Already normalized in encode_vision_only, but ensure normalization
    vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
    
    # Get all captions and process in batches
    all_captions = eval_dataset.get_all_captions()
    n_captions = len(all_captions)
    
    text_embeds_list = []
    if verbose:
        print(f"Processing {n_captions} captions...")
    
    for i in tqdm(range(0, n_captions, batch_size), disable=not verbose):
        batch_captions = all_captions[i:min(i + batch_size, n_captions)]
        
        # Stack input_ids and attention_masks
        input_ids = torch.stack([c['input_ids'] for c in batch_captions]).to(device)
        attention_mask = torch.stack([c['attention_mask'] for c in batch_captions]).to(device)
        
        # Get text embeddings using encode_text_only (no cross-modal interaction needed)
        if hasattr(model, 'encode_text_only'):
            text_embeds = model.encode_text_only(input_ids, attention_mask)
        else:
            # Fallback to unimodal encoding if available
            text_embeds = model.encode_text_unimodal(input_ids, attention_mask)
        
        text_embeds_list.append(text_embeds.cpu())
    
    text_embeds = torch.cat(text_embeds_list, dim=0)
    # Already normalized in encode_text_only, but ensure normalization
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    # Compute similarity matrix
    if verbose:
        print("Computing similarity matrix...")
    similarity = torch.matmul(vision_embeds, text_embeds.t())
    
    # Get ground truth mappings
    ground_truth = eval_dataset.get_retrieval_groundtruth()
    
    # Compute retrieval metrics
    if verbose:
        print("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(similarity, ground_truth)
    
    return metrics


@torch.no_grad()
def evaluate_retrieval_ddp(
    model,
    eval_dataset,
    batch_size: int = 32,
    device: str = 'cuda',
    rank: int = 0,
    world_size: int = 1
) -> Dict[str, float]:
    """
    DDP version of retrieval evaluation.
    Only rank 0 computes and returns metrics.
    
    This is a simplified version - for production, you'd want to distribute
    the embedding computation across GPUs and gather results.
    """
    if rank != 0:
        # Only rank 0 does evaluation for simplicity
        return {}
    
    # Unwrap DDP model if needed
    if hasattr(model, 'module'):
        model = model.module
    
    return evaluate_retrieval(
        model, 
        eval_dataset, 
        batch_size, 
        device,
        verbose=True
    )
