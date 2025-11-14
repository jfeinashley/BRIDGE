#!/usr/bin/env python3
"""
Parameter Analysis Script

Analyzes the parameter count of a VLM model for a given config file.
Reports parameters for each encoder, total counts, overhead, and each component.

Usage:
    python scripts/analyze_params.py --config configs/coco_base.yaml
    python scripts/analyze_params.py --config configs/coco_large.yaml --device cpu
"""

import argparse
import sys
import os
from pathlib import Path
from collections import OrderedDict
import yaml
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vlm_model import CrossModalVLM


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(module):
    """Count total and trainable parameters in a module"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_number(num):
    """Format number with comma separators"""
    return f"{num:,}"


def format_millions(num):
    """Format number in millions"""
    return f"{num/1e6:.2f}M"


def analyze_model_parameters(model, verbose=True):
    """
    Analyze and display model parameter counts
    
    Args:
        model: The CrossModalVLM model
        verbose: Whether to print detailed information
        
    Returns:
        dict: Dictionary containing parameter analysis
    """
    
    if verbose:
        print("\n" + "="*80)
        print("MODEL PARAMETER ANALYSIS")
        print("="*80)
    
    # Count parameters by component
    param_counts = OrderedDict()
    
    # Vision encoder parameters
    vision_total, vision_trainable = count_parameters(model.vision_encoder)
    param_counts['Vision Encoder'] = {
        'total': vision_total, 
        'trainable': vision_trainable
    }
    
    # Text encoder parameters
    text_total, text_trainable = count_parameters(model.text_encoder)
    param_counts['Text Encoder'] = {
        'total': text_total, 
        'trainable': text_trainable
    }
    
    # Cross-modal interaction layers (our method's overhead)
    interaction_total, interaction_trainable = count_parameters(model.interaction_layers)
    param_counts['Cross-Modal Interaction'] = {
        'total': interaction_total, 
        'trainable': interaction_trainable
    }
    
    # Analyze interaction layers in detail
    if verbose and len(model.interaction_layers) > 0:
        print("\nCross-Modal Interaction Layer Details:")
        print("-" * 60)
        for i, layer in enumerate(model.interaction_layers):
            layer_total, layer_trainable = count_parameters(layer)
            print(f"  Layer {i}: {format_number(layer_total)} parameters "
                  f"({format_millions(layer_total)})")
        print(f"  Total: {format_number(interaction_total)} parameters "
              f"({format_millions(interaction_total)})")
    
    # Vision projection heads
    vision_proj_total, vision_proj_trainable = 0, 0
    if hasattr(model, 'vision_pool_proj_cross'):
        t, tr = count_parameters(model.vision_pool_proj_cross)
        vision_proj_total += t
        vision_proj_trainable += tr
    if hasattr(model, 'vision_pool_proj_self'):
        t, tr = count_parameters(model.vision_pool_proj_self)
        vision_proj_total += t
        vision_proj_trainable += tr
    param_counts['Vision Projection Heads'] = {
        'total': vision_proj_total, 
        'trainable': vision_proj_trainable
    }
    
    # Text projection heads
    text_proj_total, text_proj_trainable = 0, 0
    if hasattr(model, 'text_pool_proj_cross'):
        t, tr = count_parameters(model.text_pool_proj_cross)
        text_proj_total += t
        text_proj_trainable += tr
    if hasattr(model, 'text_pool_proj_self'):
        t, tr = count_parameters(model.text_pool_proj_self)
        text_proj_total += t
        text_proj_trainable += tr
    param_counts['Text Projection Heads'] = {
        'total': text_proj_total, 
        'trainable': text_proj_trainable
    }
    
    # Calculate total
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Other parameters (gates, temperatures, positional biases, etc.)
    other_params = total_params - vision_total - text_total - interaction_total - vision_proj_total - text_proj_total
    other_trainable = total_trainable - vision_trainable - text_trainable - interaction_trainable - vision_proj_trainable - text_proj_trainable
    param_counts['Other (gates, temps, biases)'] = {
        'total': other_params, 
        'trainable': other_trainable
    }
    
    # Display parameter counts
    if verbose:
        print("\n" + "="*80)
        print("PARAMETER COUNT BY COMPONENT")
        print("="*80)
        print(f"{'Component':<35} | {'Total':>15} | {'Trainable':>15} | {'Million':>10}")
        print("-" * 80)
        
        for component, counts in param_counts.items():
            print(f"{component:<35} | {format_number(counts['total']):>15} | "
                  f"{format_number(counts['trainable']):>15} | "
                  f"{format_millions(counts['total']):>10}")
        
        print("-" * 80)
        print(f"{'TOTAL':<35} | {format_number(total_params):>15} | "
              f"{format_number(total_trainable):>15} | "
              f"{format_millions(total_params):>10}")
        print("="*80)
    
    # Calculate overhead
    base_params = vision_total + text_total
    overhead_params = interaction_total + vision_proj_total + text_proj_total + other_params
    overhead_percent = (overhead_params / base_params) * 100 if base_params > 0 else 0
    
    if verbose:
        print("\n" + "="*80)
        print("METHOD OVERHEAD ANALYSIS")
        print("="*80)
        print(f"Base Model Parameters (Vision + Text Encoders):")
        print(f"  Total: {format_number(base_params)} ({format_millions(base_params)})")
        print()
        print(f"Method Overhead Parameters (Interaction + Projections + Other):")
        print(f"  Cross-Modal Interaction: {format_number(interaction_total)} ({format_millions(interaction_total)})")
        print(f"  Vision Projections:      {format_number(vision_proj_total)} ({format_millions(vision_proj_total)})")
        print(f"  Text Projections:        {format_number(text_proj_total)} ({format_millions(text_proj_total)})")
        print(f"  Other Components:        {format_number(other_params)} ({format_millions(other_params)})")
        print(f"  ----------------------------------------")
        print(f"  Total Overhead:          {format_number(overhead_params)} ({format_millions(overhead_params)})")
        print()
        print(f"Overhead Percentage: {overhead_percent:.2f}%")
        print(f"  (Method adds {overhead_percent:.2f}% parameters on top of base encoders)")
        print("="*80)
    
    # Return analysis dictionary
    return {
        'total_params': total_params,
        'trainable_params': total_trainable,
        'vision_encoder_params': vision_total,
        'text_encoder_params': text_total,
        'interaction_params': interaction_total,
        'vision_projection_params': vision_proj_total,
        'text_projection_params': text_proj_total,
        'other_params': other_params,
        'base_params': base_params,
        'overhead_params': overhead_params,
        'overhead_percent': overhead_percent,
        'param_counts': param_counts,
    }


def print_model_architecture(model):
    """Print model architecture summary"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    print(f"Vision Encoder: {model.vision_encoder.__class__.__name__}")
    if hasattr(model.vision_encoder, 'vit'):
        print(f"  Base Model: {model.vision_encoder.vit.__class__.__name__}")
        print(f"  Model Name: {model.vision_encoder.model_name}")
    print(f"  Hidden Dim: {model.vision_dim}")
    print()
    print(f"Text Encoder: {model.text_encoder.__class__.__name__}")
    if hasattr(model.text_encoder, 'bert'):
        print(f"  Base Model: {model.text_encoder.bert.__class__.__name__}")
        print(f"  Model Name: {model.text_encoder.model_name}")
    print(f"  Hidden Dim: {model.text_dim}")
    print()
    print(f"Cross-Modal Interaction:")
    print(f"  Number of Layers: {model.num_layers}")
    print(f"  Shared Dim: {model.shared_dim}")
    print()
    print(f"Pooling Strategy: {model.pooling_type}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze VLM model parameters from config')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/coco_base.yaml)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to load model on (default: cuda)')
    parser.add_argument('--no-architecture', action='store_true',
                        help='Skip printing architecture summary')
    parser.add_argument('--save', type=str, default=None,
                        help='Save analysis to JSON file')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print("="*80)
    print("VLM MODEL PARAMETER ANALYSIS")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load config
    print("\nLoading configuration...")
    config = load_config(args.config)
    print("✓ Configuration loaded")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("\nWarning: CUDA not available, using CPU instead")
        device = 'cpu'
    else:
        device = args.device
    
    # Create model
    print("\nCreating model...")
    try:
        model = CrossModalVLM(
            vision_config=config['vision_config'],
            text_config=config['text_config'],
            cross_modal_config=config['cross_modal_config'],
            pooling_config=config['pooling_config'],
        )
        print("✓ Model created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Move model to device (not strictly necessary for parameter counting, but matches training)
    print(f"\nMoving model to {device}...")
    try:
        model = model.to(device)
        print(f"✓ Model moved to {device}")
    except Exception as e:
        print(f"Warning: Could not move model to {device}: {e}")
        print("Continuing with CPU...")
    
    # Print architecture
    if not args.no_architecture:
        print_model_architecture(model)
    
    # Analyze parameters
    analysis = analyze_model_parameters(model, verbose=True)
    
    # Save analysis if requested
    if args.save:
        import json
        # Convert to JSON-serializable format
        save_data = {
            'config_path': args.config,
            'total_params': analysis['total_params'],
            'trainable_params': analysis['trainable_params'],
            'vision_encoder_params': analysis['vision_encoder_params'],
            'text_encoder_params': analysis['text_encoder_params'],
            'interaction_params': analysis['interaction_params'],
            'vision_projection_params': analysis['vision_projection_params'],
            'text_projection_params': analysis['text_projection_params'],
            'other_params': analysis['other_params'],
            'base_params': analysis['base_params'],
            'overhead_params': analysis['overhead_params'],
            'overhead_percent': analysis['overhead_percent'],
        }
        
        with open(args.save, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\n✓ Analysis saved to: {args.save}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

