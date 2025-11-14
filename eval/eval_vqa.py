"""
Evaluation script for Vision Question Answering (VQA)
Evaluates VQA performance on standard benchmarks.
"""

import argparse
import os
import yaml
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vlm_model import CrossModalVLM
from src.data import create_dataset, create_sampler
from src.training import utils


class VQAHead(nn.Module):
    """VQA classification head on top of cross-modal features"""
    def __init__(self, hidden_dim, num_answers):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers)
        )
        
    def forward(self, pooled_features):
        return self.classifier(pooled_features)


@torch.no_grad()
def evaluate_vqa(model, vqa_head, data_loader, device, config):
    """Evaluate VQA performance"""
    model.eval()
    vqa_head.eval()
    
    all_predictions = []
    all_question_ids = []
    all_answers = []
    
    print("Running VQA evaluation...")
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        # Get cross-modal features
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention_weights=False
        )
        
        # Get pooled features
        vision_hidden = outputs['vision_hidden']
        text_hidden = outputs['text_hidden']
        
        if config.get('vqa_pooling', 'cls') == 'cls':
            vision_pooled = vision_hidden[:, 0, :]
            text_pooled = text_hidden[:, 0, :]
        else:
            vision_pooled = vision_hidden.mean(dim=1)
            text_pooled = text_hidden.mean(dim=1)
        
        cross_modal_features = torch.cat([vision_pooled, text_pooled], dim=-1)
        
        # Get VQA predictions
        logits = vqa_head(cross_modal_features)
        
        # Convert to probabilities
        if config.get('vqa_loss', 'bce') == 'bce':
            predictions = torch.sigmoid(logits)
        else:
            predictions = F.softmax(logits, dim=-1)
        
        all_predictions.append(predictions.cpu())
        
        if 'question_id' in batch:
            all_question_ids.extend(batch['question_id'])
        
        if 'answers' in batch:
            all_answers.append(batch['answers'].cpu())
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    
    results = {}
    
    # If we have ground truth answers, compute accuracy
    if all_answers:
        all_answers = torch.cat(all_answers, dim=0)
        
        # Get top predictions
        top1_pred = all_predictions.argmax(dim=-1)
        top5_pred = all_predictions.topk(5, dim=-1)[1]
        
        # Compute accuracies
        if all_answers.dim() == 2:  # Multi-hot labels
            # Find the highest scoring answer among the ground truth
            answer_labels = all_answers.argmax(dim=-1)
        else:
            answer_labels = all_answers
        
        top1_acc = (top1_pred == answer_labels).float().mean().item()
        
        # Top-5 accuracy
        top5_acc = 0
        for i in range(len(answer_labels)):
            if answer_labels[i] in top5_pred[i]:
                top5_acc += 1
        top5_acc = top5_acc / len(answer_labels)
        
        results['top1_accuracy'] = top1_acc * 100
        results['top5_accuracy'] = top5_acc * 100
        
        # Per question type accuracy if available
        if 'question_types' in batch:
            question_types = batch['question_types']
            type_accuracies = defaultdict(list)
            
            for i, qtype in enumerate(question_types):
                correct = (top1_pred[i] == answer_labels[i]).item()
                type_accuracies[qtype].append(correct)
            
            for qtype, acc_list in type_accuracies.items():
                results[f'acc_{qtype}'] = np.mean(acc_list) * 100
    
    # Create submission format if question IDs are available
    if all_question_ids:
        submission = []
        answer_vocab = data_loader.dataset.answer_vocab if hasattr(data_loader.dataset, 'answer_vocab') else None
        
        for i, qid in enumerate(all_question_ids):
            pred_idx = all_predictions[i].argmax().item()
            
            # Convert prediction index to answer string if vocab is available
            if answer_vocab:
                pred_answer = answer_vocab[pred_idx]
            else:
                pred_answer = str(pred_idx)
            
            submission.append({
                'question_id': qid,
                'answer': pred_answer
            })
        
        results['submission'] = submission
    
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
    print("Creating VQA evaluation dataset")
    dataset = create_dataset(config['dataset'], config)
    print(f'Number of samples: {len(dataset)}')
    
    # Get number of answer classes
    num_answers = dataset.num_answers if hasattr(dataset, 'num_answers') else config.get('num_answers', 3129)
    print(f'Number of answer classes: {num_answers}')
    
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
    
    # Create VQA head
    vqa_input_dim = config['vision_config']['hidden_dim'] + config['text_config']['hidden_dim']
    vqa_head = VQAHead(vqa_input_dim, num_answers)
    
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
    
    # Load VQA head state
    if 'vqa_head' in checkpoint:
        vqa_head.load_state_dict(checkpoint['vqa_head'])
        print("VQA head loaded from checkpoint")
    else:
        print("Warning: VQA head not found in checkpoint, using random initialization")
    
    model = model.to(device)
    vqa_head = vqa_head.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        vqa_head = torch.nn.parallel.DistributedDataParallel(vqa_head, device_ids=[args.gpu])
        model = model.module
        vqa_head = vqa_head.module
    
    # Evaluate
    results = evaluate_vqa(model, vqa_head, data_loader, device, config)
    
    # Print results
    print("\n" + "="*50)
    print("VQA EVALUATION RESULTS")
    print("="*50)
    
    if 'top1_accuracy' in results:
        print(f"\nOverall Accuracy:")
        print(f"  Top-1: {results['top1_accuracy']:.2f}%")
        print(f"  Top-5: {results['top5_accuracy']:.2f}%")
        
        # Print per-type accuracies if available
        type_accs = {k: v for k, v in results.items() if k.startswith('acc_') and k != 'accuracy'}
        if type_accs:
            print(f"\nPer Question Type Accuracy:")
            for qtype, acc in sorted(type_accs.items()):
                qtype_name = qtype.replace('acc_', '')
                print(f"  {qtype_name}: {acc:.2f}%")
    
    print("="*50)
    
    # Save results
    if utils.is_main_process():
        # Save metrics
        metrics_file = os.path.join(args.output_dir, 'vqa_metrics.json')
        metrics = {k: v for k, v in results.items() if k != 'submission'}
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")
        
        # Save submission if available
        if 'submission' in results:
            submission_file = os.path.join(args.output_dir, 'vqa_submission.json')
            with open(submission_file, 'w') as f:
                json.dump(results['submission'], f)
            print(f"Submission saved to {submission_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/eval_vqa.yaml')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='output/eval_vqa')
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
