"""
Optimizer and scheduler utilities
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math


def create_optimizer_and_scheduler(model, training_config, num_training_steps_per_epoch):
    """
    Create optimizer and learning rate scheduler
    
    Args:
        model: The model to optimize
        training_config: Dictionary with training configuration
        num_training_steps_per_epoch: Number of training steps per epoch
    
    Returns:
        optimizer, scheduler
    """
    # Get config values
    learning_rate = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0.01)
    warmup_steps = training_config.get('warmup_steps', 1000)
    num_epochs = training_config['num_epochs']
    grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
    
    # Total training steps
    total_steps = num_training_steps_per_epoch * num_epochs // grad_accum_steps
    
    # Separate parameters into groups
    # No weight decay for bias and layer norm parameters
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'layer_norm.weight', 'layer_norm.bias']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        },
    ]
    
    # Create optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Create learning rate scheduler with linear warmup and cosine decay
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def create_optimizer(model, config):
    """
    Simple optimizer creation (backward compatibility)
    
    Args:
        model: The model to optimize
        config: Dictionary with optimizer configuration
    
    Returns:
        optimizer
    """
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    
    # Separate parameters into groups
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    return optimizer





