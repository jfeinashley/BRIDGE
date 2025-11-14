"""
All loss functions for VLM training:
1. ITC (Image-Text Contrastive)
2. MLM (Masked Language Modeling)
3. MIM (Masked Image Modeling)
4. ITM (Image-Text Matching)
5. Cycle-consistency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class VLMLosses:
    """Container for all VLM training losses"""
    
    def __init__(self, config: dict):
        self.config = config
        self.loss_weights = config.get('loss_weights', {})
        
        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse_loss = nn.MSELoss()
        
    def compute_dual_itc_loss(
        self,
        vision_embeds_cross: torch.Tensor,
        text_embeds_cross: torch.Tensor,
        vision_embeds_uni: torch.Tensor,
        text_embeds_uni: torch.Tensor,
        logit_scale: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute dual image-text contrastive loss (InfoNCE) as per method.tex:
        loss = alpha * cross-modal + (1-alpha) * unimodal
        
        Args:
            vision_embeds_cross: (batch, projection_dim) - cross-modal aware embeddings
            text_embeds_cross: (batch, projection_dim) - cross-modal aware embeddings
            vision_embeds_uni: (batch, projection_dim) - unimodal embeddings
            text_embeds_uni: (batch, projection_dim) - unimodal embeddings
            logit_scale: scalar temperature^{-1}
            alpha: learnable weight in [0, 1]
            
        Returns:
            loss: scalar
            metrics: dict with individual losses
        """
        batch_size = vision_embeds_cross.shape[0]
        device = vision_embeds_cross.device
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # Cross-modal contrastive loss
        # S_ij = <text_i, vision_j>
        logits_cross_t2v = logit_scale * (text_embeds_cross @ vision_embeds_cross.t())
        logits_cross_v2t = logits_cross_t2v.t()
        
        loss_cross_t2v = F.cross_entropy(logits_cross_t2v, labels)
        loss_cross_v2t = F.cross_entropy(logits_cross_v2t, labels)
        loss_itc_cross = (loss_cross_t2v + loss_cross_v2t) / 2.0
        
        # Unimodal contrastive loss
        logits_uni_t2v = logit_scale * (text_embeds_uni @ vision_embeds_uni.t())
        logits_uni_v2t = logits_uni_t2v.t()
        
        loss_uni_t2v = F.cross_entropy(logits_uni_t2v, labels)
        loss_uni_v2t = F.cross_entropy(logits_uni_v2t, labels)
        loss_itc_uni = (loss_uni_t2v + loss_uni_v2t) / 2.0
        
        # Combined dual loss
        loss_itc = alpha * loss_itc_cross + (1 - alpha) * loss_itc_uni
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            # Cross-modal accuracy
            pred_cross_t2v = logits_cross_t2v.argmax(dim=-1)
            acc_cross_t2v = (pred_cross_t2v == labels).float().mean()
            pred_cross_v2t = logits_cross_v2t.argmax(dim=-1)
            acc_cross_v2t = (pred_cross_v2t == labels).float().mean()
            
            # Unimodal accuracy
            pred_uni_t2v = logits_uni_t2v.argmax(dim=-1)
            acc_uni_t2v = (pred_uni_t2v == labels).float().mean()
            pred_uni_v2t = logits_uni_v2t.argmax(dim=-1)
            acc_uni_v2t = (pred_uni_v2t == labels).float().mean()
        
        metrics = {
            'loss_cross_t2v': loss_cross_t2v.item(),
            'loss_cross_v2t': loss_cross_v2t.item(),
            'loss_uni_t2v': loss_uni_t2v.item(),
            'loss_uni_v2t': loss_uni_v2t.item(),
            'acc_cross_t2v': acc_cross_t2v.item(),
            'acc_cross_v2t': acc_cross_v2t.item(),
            'acc_uni_t2v': acc_uni_t2v.item(),
            'acc_uni_v2t': acc_uni_v2t.item(),
            'alpha': alpha.item(),
        }
        
        return loss_itc, metrics
    
    def compute_itc_loss(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute simple symmetric image-text contrastive loss (for backward compatibility).
        """
        batch_size = vision_embeds.shape[0]
        device = vision_embeds.device
        
        logits_per_text = logit_scale * (text_embeds @ vision_embeds.t())
        logits_per_vision = logits_per_text.t()
        
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        loss_t2v = F.cross_entropy(logits_per_text, labels)
        loss_v2t = F.cross_entropy(logits_per_vision, labels)
        loss_itc = (loss_t2v + loss_v2t) / 2.0
        
        with torch.no_grad():
            pred_t2v = logits_per_text.argmax(dim=-1)
            acc_t2v = (pred_t2v == labels).float().mean()
            pred_v2t = logits_per_vision.argmax(dim=-1)
            acc_v2t = (pred_v2t == labels).float().mean()
        
        metrics = {
            'loss_t2v': loss_t2v.item(),
            'loss_v2t': loss_v2t.item(),
            'acc_t2v': acc_t2v.item(),
            'acc_v2t': acc_v2t.item(),
        }
        
        return loss_itc, metrics
    
    def compute_mlm_loss(
        self,
        mlm_logits: torch.Tensor,
        mlm_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute masked language modeling loss.
        
        Args:
            mlm_logits: (batch, seq_len, vocab_size)
            mlm_labels: (batch, seq_len) - with -100 for non-masked tokens
            
        Returns:
            loss: scalar
            metrics: dict
        """
        # Flatten for cross-entropy
        mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
        mlm_labels_flat = mlm_labels.view(-1)
        
        loss_mlm = self.ce_loss(mlm_logits_flat, mlm_labels_flat)
        
        # Compute accuracy on masked tokens
        with torch.no_grad():
            mask = mlm_labels_flat != -100
            if mask.sum() > 0:
                pred = mlm_logits_flat.argmax(dim=-1)
                acc = (pred[mask] == mlm_labels_flat[mask]).float().mean()
            else:
                acc = torch.tensor(0.0)
        
        metrics = {
            'mlm_acc': acc.item() if isinstance(acc, torch.Tensor) else acc,
        }
        
        return loss_mlm, metrics
    
    def compute_mim_loss(
        self,
        mim_predictions: torch.Tensor,
        mim_targets: torch.Tensor,
        mim_mask: torch.Tensor,
        loss_type: str = 'mse',
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute masked image modeling loss.
        
        Args:
            mim_predictions: (batch, num_patches, vision_dim)
            mim_targets: (batch, num_patches, vision_dim) - target features
            mim_mask: (batch, num_patches) - 1 for masked patches
            loss_type: 'mse' or 'ce' (cross-entropy with discrete codes)
            
        Returns:
            loss: scalar
            metrics: dict
        """
        if loss_type == 'mse':
            # MSE loss only on masked patches
            # Expand mask for broadcasting
            mask_expanded = mim_mask.unsqueeze(-1).float()
            
            # Compute MSE
            squared_diff = (mim_predictions - mim_targets) ** 2
            masked_loss = (squared_diff * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
            
            loss_mim = masked_loss
            
            metrics = {}
        else:
            # Cross-entropy with discrete codes (not implemented yet)
            raise NotImplementedError("MIM with discrete codes not implemented")
        
        return loss_mim, metrics
    
    def compute_itm_loss_with_hard_negatives(
        self,
        model,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        m_min: float = 0.1,
        m_max: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute image-text matching loss with semi-hard negatives (as per method.tex).
        
        Args:
            model: The VLM model (needed to compute ITM logits)
            vision_embeds: (batch, projection_dim) - normalized embeddings
            text_embeds: (batch, projection_dim) - normalized embeddings
            pixel_values: (batch, 3, H, W) - original images
            input_ids: (batch, seq_len) - original text input ids
            attention_mask: (batch, seq_len) - text attention mask
            m_min: Minimum margin for semi-hard negatives
            m_max: Maximum margin for semi-hard negatives
            
        Returns:
            loss: scalar
            metrics: dict
        """
        batch_size = vision_embeds.shape[0]
        device = vision_embeds.device
        
        # Compute similarity matrix
        similarity = text_embeds @ vision_embeds.t()  # (batch, batch)
        
        # Get diagonal (positive pairs)
        pos_sim = similarity.diagonal()
        
        # Find semi-hard negatives for each sample
        negative_indices = []
        for i in range(batch_size):
            # Find candidates in the semi-hard band
            sim_i = similarity[i]
            pos_sim_i = pos_sim[i]
            
            # Mask for semi-hard negatives: S_{i,i} - m_max < S_{i,j} < S_{i,i} - m_min
            mask = (sim_i > pos_sim_i - m_max) & (sim_i < pos_sim_i - m_min)
            mask[i] = False  # Exclude self
            
            candidates = mask.nonzero(as_tuple=True)[0]
            
            if len(candidates) > 0:
                # Randomly select one semi-hard negative
                idx = torch.randint(len(candidates), (1,), device=device)
                neg_idx = candidates[idx].item()
            else:
                # Fallback: select hardest negative (excluding self)
                sim_i_masked = sim_i.clone()
                sim_i_masked[i] = -float('inf')
                neg_idx = sim_i_masked.argmax().item()
            
            negative_indices.append(neg_idx)
        
        # Prepare positive and negative pairs
        positive_logits = []
        negative_logits = []
        
        # Compute ITM logits for positive pairs
        pos_itm_logits = model.compute_itm_logits(vision_embeds, text_embeds)
        positive_logits = pos_itm_logits[:, 1]  # Select "match" score
        
        # Compute ITM logits for negative pairs
        neg_vision_embeds = vision_embeds[negative_indices]
        neg_itm_logits = model.compute_itm_logits(neg_vision_embeds, text_embeds)
        negative_logits = neg_itm_logits[:, 1]  # Select "match" score
        
        # Binary cross-entropy loss
        pos_loss = -torch.log(torch.sigmoid(positive_logits) + 1e-8).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(negative_logits) + 1e-8).mean()
        loss_itm = (pos_loss + neg_loss) / 2.0
        
        # Compute accuracy
        with torch.no_grad():
            pos_pred = (positive_logits > 0).float()
            neg_pred = (negative_logits <= 0).float()
            acc = (pos_pred.sum() + neg_pred.sum()) / (2 * batch_size)
        
        metrics = {
            'itm_acc': acc.item(),
            'itm_pos_loss': pos_loss.item(),
            'itm_neg_loss': neg_loss.item(),
        }
        
        return loss_itm, metrics
    
    def compute_itm_loss(
        self,
        itm_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Simple ITM loss (for backward compatibility).
        """
        loss_itm = F.cross_entropy(itm_logits, labels)
        
        with torch.no_grad():
            pred = itm_logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()
        
        metrics = {
            'itm_acc': acc.item(),
        }
        
        return loss_itm, metrics
    
    def compute_cycle_consistency_loss(
        self,
        attn_weights_text_list: list,
        attn_weights_vision_list: list,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute cycle-consistency loss over cross-attention weights.
        
        Encourages P_{t←v} @ P_{v←t} ≈ I and P_{v←t} @ P_{t←v} ≈ I
        
        Args:
            attn_weights_text_list: list of attention weights (batch, heads, n_text, n_vision)
            attn_weights_vision_list: list of attention weights (batch, heads, n_vision, n_text)
            
        Returns:
            loss: scalar
            metrics: dict
        """
        if not attn_weights_text_list or not attn_weights_vision_list:
            return torch.tensor(0.0), {}
        
        total_loss = 0.0
        num_layers = len(attn_weights_text_list)
        
        for attn_t, attn_v in zip(attn_weights_text_list, attn_weights_vision_list):
            # Average over heads: (batch, n_text, n_vision)
            P_t_from_v = attn_t.mean(dim=1)
            P_v_from_t = attn_v.mean(dim=1)
            
            # Compute cycle: (batch, n_text, n_text)
            C_text = torch.bmm(P_t_from_v, P_v_from_t)
            # Compute cycle: (batch, n_vision, n_vision)
            C_vision = torch.bmm(P_v_from_t, P_t_from_v)
            
            # Encourage diagonal elements to be 1 (identity)
            n_text = C_text.shape[1]
            n_vision = C_vision.shape[1]
            
            # Negative log of diagonal elements (want them close to 1)
            # Clamp diagonal values to avoid log(0) or log(>1)
            text_diag = C_text.diagonal(dim1=1, dim2=2).clamp(min=1e-8, max=1.0 - 1e-8)
            vision_diag = C_vision.diagonal(dim1=1, dim2=2).clamp(min=1e-8, max=1.0 - 1e-8)
            
            loss_text = -torch.log(text_diag).mean()
            loss_vision = -torch.log(vision_diag).mean()
            
            total_loss += (loss_text + loss_vision) / 2.0
        
        loss_cyc = total_loss / num_layers
        
        metrics = {}
        
        return loss_cyc, metrics
    
    def compute_total_loss(
        self,
        model_outputs: Dict,
        mlm_labels: Optional[torch.Tensor] = None,
        mim_targets: Optional[torch.Tensor] = None,
        mim_mask: Optional[torch.Tensor] = None,
        itm_labels: Optional = None,
        stage: str = 'A',
        model: Optional = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted total loss based on training stage.
        
        Args:
            model_outputs: dict from model forward pass
            mlm_labels: labels for MLM
            mim_targets: targets for MIM
            mim_mask: mask for MIM
            itm_labels: labels for ITM
            stage: training stage ('A', 'B', or 'C')
            
        Returns:
            total_loss: scalar
            all_metrics: dict with all losses and metrics
        """
        all_metrics = {}
        total_loss = 0.0
        
        # (1) Dual ITC loss - always computed (as per method.tex)
        if 'vision_embeds' in model_outputs and 'text_embeds' in model_outputs:
            # Check if we have both cross-modal and unimodal embeddings
            if 'vision_embeds_unimodal' in model_outputs and 'text_embeds_unimodal' in model_outputs:
                # Use dual ITC loss with alpha weighting
                alpha = model_outputs.get('alpha', torch.tensor(0.5))  # Get alpha from model
                loss_itc, metrics_itc = self.compute_dual_itc_loss(
                    model_outputs['vision_embeds'],
                    model_outputs['text_embeds'],
                    model_outputs['vision_embeds_unimodal'],
                    model_outputs['text_embeds_unimodal'],
                    model_outputs['logit_scale'],
                    alpha,
                )
            else:
                # Fallback to simple ITC loss
                loss_itc, metrics_itc = self.compute_itc_loss(
                    model_outputs['vision_embeds'],
                    model_outputs['text_embeds'],
                    model_outputs['logit_scale'],
                )
            weight = self.loss_weights.get('itc', 1.0)
            total_loss += weight * loss_itc
            all_metrics['loss_itc'] = loss_itc.item()
            all_metrics.update({f'itc_{k}': v for k, v in metrics_itc.items()})
        
        # (2) MLM loss - Stage B+
        if stage in ['B', 'C'] and mlm_labels is not None:
            mlm_logits = model_outputs.get('mlm_logits')
            if mlm_logits is not None:
                loss_mlm, metrics_mlm = self.compute_mlm_loss(mlm_logits, mlm_labels)
                weight = self.loss_weights.get('mlm', 0.5)
                total_loss += weight * loss_mlm
                all_metrics['loss_mlm'] = loss_mlm.item()
                all_metrics.update({f'mlm_{k}': v for k, v in metrics_mlm.items()})
        
        # (3) MIM loss - Stage B+ (optional)
        if stage in ['B', 'C'] and mim_targets is not None and mim_mask is not None:
            mim_predictions = model_outputs.get('mim_predictions')
            if mim_predictions is not None:
                loss_mim, metrics_mim = self.compute_mim_loss(
                    mim_predictions, mim_targets, mim_mask
                )
                weight = self.loss_weights.get('mim', 0.5)
                total_loss += weight * loss_mim
                all_metrics['loss_mim'] = loss_mim.item()
                all_metrics.update({f'mim_{k}': v for k, v in metrics_mim.items()})
        
        # (4) ITM loss - optional, all stages
        if itm_labels is not None:
            if itm_labels == 'use_hard_negatives' and model is not None:
                # Use ITM with semi-hard negatives (as per method.tex)
                vision_embeds = model_outputs.get('vision_embeds')
                text_embeds = model_outputs.get('text_embeds')
                if vision_embeds is not None and text_embeds is not None:
                    loss_itm, metrics_itm = self.compute_itm_loss_with_hard_negatives(
                        model=model,
                        vision_embeds=vision_embeds,
                        text_embeds=text_embeds,
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    weight = self.loss_weights.get('itm', 0.3)
                    total_loss += weight * loss_itm
                    all_metrics['loss_itm'] = loss_itm.item()
                    all_metrics.update({f'itm_{k}': v for k, v in metrics_itm.items()})
            else:
                # Simple ITM loss (backward compatibility)
                itm_logits = model_outputs.get('itm_logits')
                if itm_logits is not None:
                    loss_itm, metrics_itm = self.compute_itm_loss(itm_logits, itm_labels)
                    weight = self.loss_weights.get('itm', 0.3)
                    total_loss += weight * loss_itm
                    all_metrics['loss_itm'] = loss_itm.item()
                    all_metrics.update({f'itm_{k}': v for k, v in metrics_itm.items()})
        
        # (5) Cycle-consistency loss - optional
        if 'attn_weights_text' in model_outputs and 'attn_weights_vision' in model_outputs:
            loss_cyc, metrics_cyc = self.compute_cycle_consistency_loss(
                model_outputs['attn_weights_text'],
                model_outputs['attn_weights_vision'],
            )
            weight = self.loss_weights.get('cyc', 0.1)
            total_loss += weight * loss_cyc
            all_metrics['loss_cyc'] = loss_cyc.item()
            all_metrics.update({f'cyc_{k}': v for k, v in metrics_cyc.items()})
        
        all_metrics['loss_total'] = total_loss.item()
        
        return total_loss, all_metrics


# Simplified wrapper classes for ease of use in training scripts
class DualContrastiveLoss(nn.Module):
    """
    Dual Image-Text Contrastive Loss (InfoNCE)
    Combines cross-modal and unimodal contrastive losses as per method.tex Eq (9)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, vision_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss
        
        Args:
            vision_embeds: (batch, dim) - normalized embeddings
            text_embeds: (batch, dim) - normalized embeddings
            
        Returns:
            loss: scalar
        """
        batch_size = vision_embeds.shape[0]
        device = vision_embeds.device
        
        # Compute similarity matrix with temperature
        logits_per_text = (text_embeds @ vision_embeds.t()) / self.temperature
        logits_per_vision = logits_per_text.t()
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # Symmetric loss
        loss_t2v = F.cross_entropy(logits_per_text, labels)
        loss_v2t = F.cross_entropy(logits_per_vision, labels)
        
        return (loss_t2v + loss_v2t) / 2.0


class ITMLoss(nn.Module):
    """
    Image-Text Matching Loss with semi-hard negatives (method.tex Section 4, Eq. 178-180)
    
    Implements:
    1. Semi-hard negative mining from band: S_{i,i} - m_max < S_{i,j} < S_{i,i} - m_min
    2. MLP on [p_t; p_v; <p_t, p_v>] for matching score
    3. Binary cross-entropy: -1/(2B) Σ (log σ(r_{i,i}) + log(1 - σ(r_{i,j̃})))
    """
    def __init__(self, margin_min: float = 0.2, margin_max: float = 0.5, embed_dim: int = 256):
        super().__init__()
        self.margin_min = margin_min
        self.margin_max = margin_max
        
        # Small MLP for ITM as per method.tex
        # Input: [p_t; p_v; <p_t, p_v>] = concat of dim, dim, 1 = 2*dim + 1
        self.itm_head = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),  # Single logit for matching score
        )
        
    def forward(
        self, 
        vision_embeds_cross: torch.Tensor,
        text_embeds_cross: torch.Tensor,
        vision_embeds_uni: torch.Tensor,
        text_embeds_uni: torch.Tensor,
    ) -> torch.Tensor:
        """
        ITM loss with semi-hard negatives as per method.tex
        
        Args:
            vision_embeds_cross: (batch, dim) - cross-modal vision embeddings (for ITM head)
            text_embeds_cross: (batch, dim) - cross-modal text embeddings (for ITM head)
            vision_embeds_uni: (batch, dim) - unimodal vision embeddings (for negative mining)
            text_embeds_uni: (batch, dim) - unimodal text embeddings (for negative mining)
            
        Returns:
            loss: scalar
        """
        batch_size = vision_embeds_cross.shape[0]
        device = vision_embeds_cross.device
        
        # Use unimodal similarity to find semi-hard negatives (as per method.tex)
        S = text_embeds_uni @ vision_embeds_uni.t()  # Similarity matrix
        S_diag = S.diagonal()  # S_{i,i} = similarity of positive pairs
        
        # Sample semi-hard negatives from the band for each sample
        negative_indices = []
        for i in range(batch_size):
            # Find candidates in semi-hard band: S_{i,i} - m_max < S_{i,j} < S_{i,i} - m_min
            mask = (S[i] > S_diag[i] - self.margin_max) & (S[i] < S_diag[i] - self.margin_min)
            mask[i] = False  # Exclude self
            
            candidates = mask.nonzero(as_tuple=True)[0]
            
            if len(candidates) > 0:
                # Randomly select one semi-hard negative
                idx = torch.randint(0, len(candidates), (1,), device=device)
                neg_idx = candidates[idx].item()
            else:
                # Fallback: select hardest negative (highest similarity, excluding self)
                S_i = S[i].clone()
                S_i[i] = -float('inf')
                neg_idx = S_i.argmax().item()
            
            negative_indices.append(neg_idx)
        
        # Compute ITM logits for positive pairs: MLP([p_t; p_v; <p_t, p_v>])
        dot_product_pos = (vision_embeds_cross * text_embeds_cross).sum(dim=-1, keepdim=True)
        concat_pos = torch.cat([text_embeds_cross, vision_embeds_cross, dot_product_pos], dim=-1)
        logits_pos = self.itm_head(concat_pos).squeeze(-1)  # (batch,)
        
        # Compute ITM logits for negative pairs
        vision_embeds_neg = vision_embeds_cross[negative_indices]
        dot_product_neg = (vision_embeds_neg * text_embeds_cross).sum(dim=-1, keepdim=True)
        concat_neg = torch.cat([text_embeds_cross, vision_embeds_neg, dot_product_neg], dim=-1)
        logits_neg = self.itm_head(concat_neg).squeeze(-1)  # (batch,)
        
        # Binary cross-entropy loss as per method.tex Eq. 180
        # L_itm = -1/(2B) Σ (log σ(r_{i,i}) + log(1 - σ(r_{i,j̃})))
        pos_loss = -torch.log(torch.sigmoid(logits_pos) + 1e-8).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(logits_neg) + 1e-8).mean()
        
        loss_itm = (pos_loss + neg_loss) / 2.0
        
        return loss_itm


class MLMLoss(nn.Module):
    """Masked Language Modeling Loss"""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len) with -100 for non-masked tokens
        """
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)
        return self.ce_loss(logits_flat, labels_flat)


class MIMLoss(nn.Module):
    """Masked Image Modeling Loss"""
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (batch, num_patches, dim)
            targets: (batch, num_patches, dim)
            mask: (batch, num_patches) - 1 for masked patches
        """
        if self.loss_type == 'mse':
            # MSE loss only on masked patches
            mask_expanded = mask.unsqueeze(-1).float()
            squared_diff = (predictions - targets) ** 2
            masked_loss = (squared_diff * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
            return masked_loss
        else:
            raise NotImplementedError(f"MIM loss type '{self.loss_type}' not implemented")


class CycleConsistencyLoss(nn.Module):
    """
    Cycle-Consistency Loss for cross-attention (method.tex Section 5)
    Encourages P_{t←v} @ P_{v←t} ≈ I and P_{v←t} @ P_{t←v} ≈ I
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, aux_outputs: list) -> torch.Tensor:
        """
        Args:
            aux_outputs: list of dicts from interaction layers, each containing:
                - 'attn_weights_text': (batch, heads, n_text, n_vision)
                - 'attn_weights_vision': (batch, heads, n_vision, n_text)
        """
        if not aux_outputs:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        num_layers = 0
        
        for aux in aux_outputs:
            if 'attn_weights_text' not in aux or 'attn_weights_vision' not in aux:
                continue
                
            attn_t = aux['attn_weights_text']  # (batch, heads, n_text, n_vision)
            attn_v = aux['attn_weights_vision']  # (batch, heads, n_vision, n_text)
            
            # Average over heads
            P_t_from_v = attn_t.mean(dim=1)  # (batch, n_text, n_vision)
            P_v_from_t = attn_v.mean(dim=1)  # (batch, n_vision, n_text)
            
            # Compute cycles
            C_text = torch.bmm(P_t_from_v, P_v_from_t)      # (batch, n_text, n_text)
            C_vision = torch.bmm(P_v_from_t, P_t_from_v)    # (batch, n_vision, n_vision)
            
            # Encourage diagonal elements to be 1
            text_diag = C_text.diagonal(dim1=1, dim2=2).clamp(min=1e-8, max=1.0)
            vision_diag = C_vision.diagonal(dim1=1, dim2=2).clamp(min=1e-8, max=1.0)
            
            loss_text = -torch.log(text_diag).mean()
            loss_vision = -torch.log(vision_diag).mean()
            
            total_loss += (loss_text + loss_vision) / 2.0
            num_layers += 1
        
        if num_layers > 0:
            return total_loss / num_layers
        else:
            return torch.tensor(0.0)