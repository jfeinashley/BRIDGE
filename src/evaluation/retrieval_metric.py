"""
Retrieval metrics computation for training
Simplified version that works with embeddings directly
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


def compute_retrieval_metrics(
    vision_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_ids: Optional[List] = None,
    text_ids: Optional[List] = None,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics from vision and text embeddings.
    Assumes diagonal correspondence: i-th image matches i-th text.
    
    Args:
        vision_embeds: (N, dim) - normalized vision embeddings
        text_embeds: (N, dim) - normalized text embeddings
        image_ids: Optional list of image IDs
        text_ids: Optional list of text IDs
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary with retrieval metrics
    """
    # Ensure embeddings are on CPU and normalized
    if vision_embeds.device != torch.device('cpu'):
        vision_embeds = vision_embeds.cpu()
    if text_embeds.device != torch.device('cpu'):
        text_embeds = text_embeds.cpu()
    
    vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(vision_embeds, text_embeds.t())  # (N, N)
    n_samples = similarity.shape[0]
    
    metrics = {}
    
    # Image-to-Text Retrieval (i2t)
    # For each image, find the rank of its corresponding text
    i2t_ranks = []
    for i in range(n_samples):
        # Get similarity scores for this image with all texts
        scores = similarity[i]
        
        # Sort in descending order
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Find the rank of the correct text (diagonal element = index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    
    # Calculate i2t Recall@K
    for k in k_values:
        recall_k = (i2t_ranks <= k).astype(float).mean() * 100
        metrics[f'i2t_r{k}'] = recall_k
    
    # Text-to-Image Retrieval (t2i)
    # For each text, find the rank of its corresponding image
    t2i_ranks = []
    for i in range(n_samples):
        # Get similarity scores for this text with all images
        scores = similarity[:, i]
        
        # Sort in descending order
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Find the rank of the correct image (diagonal element = index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)
    
    t2i_ranks = np.array(t2i_ranks)
    
    # Calculate t2i Recall@K
    for k in k_values:
        recall_k = (t2i_ranks <= k).astype(float).mean() * 100
        metrics[f't2i_r{k}'] = recall_k
    
    # Calculate additional metrics
    metrics['i2t_mean_rank'] = float(i2t_ranks.mean())
    metrics['i2t_median_rank'] = float(np.median(i2t_ranks))
    metrics['t2i_mean_rank'] = float(t2i_ranks.mean())
    metrics['t2i_median_rank'] = float(np.median(t2i_ranks))
    
    # Calculate rSum (sum of all recall@K values)
    rsum = sum([metrics[f'i2t_r{k}'] for k in k_values] + 
               [metrics[f't2i_r{k}'] for k in k_values])
    metrics['rsum'] = rsum
    
    return metrics


def compute_retrieval_metrics_with_multiple_captions(
    vision_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_to_captions: Dict[int, List[int]],
    caption_to_image: Dict[int, int],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics when each image has multiple captions (e.g., COCO).
    
    Args:
        vision_embeds: (N_images, dim) - normalized vision embeddings
        text_embeds: (N_captions, dim) - normalized text embeddings
        image_to_captions: Dict mapping image_idx -> list of caption_indices
        caption_to_image: Dict mapping caption_idx -> image_idx
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary with retrieval metrics
    """
    # Ensure embeddings are normalized
    vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(vision_embeds, text_embeds.t())  # (N_images, N_captions)
    n_images = similarity.shape[0]
    n_captions = similarity.shape[1]
    
    metrics = {}
    
    # Image-to-Text Retrieval (i2t)
    i2t_ranks = []
    for img_idx in range(n_images):
        # Get similarity scores for this image with all captions
        scores = similarity[img_idx]
        
        # Sort in descending order
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Find the rank of the first correct caption
        correct_captions = image_to_captions[img_idx]
        min_rank = n_captions + 1
        
        for rank, caption_idx in enumerate(sorted_indices):
            if caption_idx.item() in correct_captions:
                min_rank = rank + 1
                break
        
        i2t_ranks.append(min_rank)
    
    i2t_ranks = np.array(i2t_ranks)
    
    # Calculate i2t Recall@K
    for k in k_values:
        recall_k = (i2t_ranks <= k).astype(float).mean() * 100
        metrics[f'i2t_r{k}'] = recall_k
    
    # Text-to-Image Retrieval (t2i)
    t2i_ranks = []
    for caption_idx in range(n_captions):
        # Get similarity scores for this caption with all images
        scores = similarity[:, caption_idx]
        
        # Sort in descending order
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Find the rank of the correct image
        correct_image = caption_to_image[caption_idx]
        
        rank = (sorted_indices == correct_image).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)
    
    t2i_ranks = np.array(t2i_ranks)
    
    # Calculate t2i Recall@K
    for k in k_values:
        recall_k = (t2i_ranks <= k).astype(float).mean() * 100
        metrics[f't2i_r{k}'] = recall_k
    
    # Calculate additional metrics
    metrics['i2t_mean_rank'] = float(i2t_ranks.mean())
    metrics['i2t_median_rank'] = float(np.median(i2t_ranks))
    metrics['t2i_mean_rank'] = float(t2i_ranks.mean())
    metrics['t2i_median_rank'] = float(np.median(t2i_ranks))
    
    # Calculate rSum
    rsum = sum([metrics[f'i2t_r{k}'] for k in k_values] + 
               [metrics[f't2i_r{k}'] for k in k_values])
    metrics['rsum'] = rsum
    
    return metrics

