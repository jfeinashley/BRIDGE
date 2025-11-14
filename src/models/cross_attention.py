"""
Cross-Modal Interaction Layer

Implements the bidirectional cross-modal attention mechanism described in vlm_idea.md:
- PreNorm and project to shared space
- Cross-only multi-head attention (no self-attention within this block)
- Stabilization via row-wise L2 normalization, temperature scaling, and positional bias
- Gated residual back to native streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossModalInteractionLayer(nn.Module):
    """
    One interaction layer at depth ℓ that allows vision and text encoders
    to exchange information through cross-modal attention.
    
    Note: No explicit router needed - the cross-modal attention itself
    implements the "mixture of thoughts" by allowing each modality to
    attend to the other modality's representations.
    """
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        shared_dim: int,
        num_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_positional_bias: bool = True,
        init_temperature: float = 1.0,
        init_gate_bias: float = -2.2,  # logit(0.1) ≈ -2.2 as per vlm_idea.md
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.shared_dim = shared_dim
        self.num_heads = num_heads
        self.head_dim = shared_dim // num_heads
        self.use_positional_bias = use_positional_bias
        
        assert shared_dim % num_heads == 0, "shared_dim must be divisible by num_heads"
        
        # (A) PreNorm and project to shared space
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        
        self.vision_to_shared = nn.Linear(vision_dim, shared_dim)
        self.text_to_shared = nn.Linear(text_dim, shared_dim)
        
        # (B) Cross-only attention parameters
        # Text queries attend to vision keys/values
        self.Q_text = nn.Linear(shared_dim, shared_dim)
        self.K_vision_to_text = nn.Linear(shared_dim, shared_dim)
        self.V_vision_to_text = nn.Linear(shared_dim, shared_dim)
        self.O_text = nn.Linear(shared_dim, shared_dim)
        
        # Vision queries attend to text keys/values
        self.Q_vision = nn.Linear(shared_dim, shared_dim)
        self.K_text_to_vision = nn.Linear(shared_dim, shared_dim)
        self.V_text_to_vision = nn.Linear(shared_dim, shared_dim)
        self.O_vision = nn.Linear(shared_dim, shared_dim)
        
        # Learnable temperature parameters (per direction)
        # Using softplus parameterization to ensure positivity (as per method.tex)
        # Initialize such that softplus(x) = init_temperature
        init_temp_param = math.log(math.exp(init_temperature) - 1.0) if init_temperature > 0.1 else 0.0
        self.tau_text_param = nn.Parameter(torch.tensor(init_temp_param))
        self.tau_vision_param = nn.Parameter(torch.tensor(init_temp_param))
        
        # Cross-modal positional bias (learnable)
        if use_positional_bias:
            # These will be initialized based on actual sequence lengths
            self.register_buffer('B_text_from_vision', None)
            self.register_buffer('B_vision_from_text', None)
            # Learnable bias parameters (will be initialized on first forward pass)
            self.bias_text_from_vision = None
            self.bias_vision_from_text = None
        
        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # (C) Reverse-project and gated residual
        self.shared_to_text = nn.Linear(shared_dim, text_dim)
        self.shared_to_vision = nn.Linear(shared_dim, vision_dim)
        
        # Learnable gates (initialized small for gradual warmup)
        self.gate_text_bias = nn.Parameter(torch.tensor(init_gate_bias))
        self.gate_vision_bias = nn.Parameter(torch.tensor(init_gate_bias))
        
    def _init_positional_bias(self, n_vision: int, n_text: int, device: torch.device):
        """Initialize positional bias matrices on first forward pass"""
        if self.use_positional_bias and self.bias_text_from_vision is None:
            # Simple learnable bias matrices
            self.bias_text_from_vision = nn.Parameter(
                torch.zeros(n_text, n_vision, device=device)
            )
            self.bias_vision_from_text = nn.Parameter(
                torch.zeros(n_vision, n_text, device=device)
            )
    
    def _row_normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Row-wise L2 normalization for stability"""
        return F.normalize(x, p=2, dim=-1, eps=eps)
    
    def _cross_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        tau: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head cross-attention with temperature and positional bias.
        
        Args:
            Q: queries (batch, n_queries, shared_dim)
            K: keys (batch, n_keys, shared_dim)
            V: values (batch, n_keys, shared_dim)
            tau: temperature scalar
            bias: positional bias (n_queries, n_keys)
            mask: attention mask (batch, n_queries, n_keys)
            
        Returns:
            attended output (batch, n_queries, shared_dim)
            attention weights (batch, num_heads, n_queries, n_keys)
        """
        batch_size, n_q, _ = Q.shape
        n_k = K.shape[1]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Row-wise L2 normalization for stability
        Q = self._row_normalize(Q)
        K = self._row_normalize(K)
        
        # Compute attention scores with temperature
        # (batch, num_heads, n_q, n_k)
        # IMPORTANT: Because of L2 normalization, we omit the usual 1/sqrt(d_h) scaling
        # and use only the learnable temperature (as per vlm_idea.md)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / tau
        
        # Add positional bias if provided
        if bias is not None:
            scores = scores + bias.unsqueeze(0).unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, n_q, self.shared_dim)
        
        return output, attn_weights
    
    def forward(
        self,
        vision_hidden: torch.Tensor,
        text_hidden: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Forward pass of one cross-modal interaction layer.
        
        Args:
            vision_hidden: (batch, n_vision, vision_dim)
            text_hidden: (batch, n_text, text_dim)
            vision_mask: (batch, n_vision) - 1 for valid, 0 for padding
            text_mask: (batch, n_text) - 1 for valid, 0 for padding
            return_attention_weights: if True, return attention weights for analysis
            
        Returns:
            updated_vision: (batch, n_vision, vision_dim)
            updated_text: (batch, n_text, text_dim)
            aux_outputs: dictionary with auxiliary outputs (MOT loss, attention weights, etc.)
        """
        batch_size = vision_hidden.shape[0]
        n_vision = vision_hidden.shape[1]
        n_text = text_hidden.shape[1]
        
        # Initialize positional bias on first forward pass
        self._init_positional_bias(n_vision, n_text, vision_hidden.device)
        
        aux_outputs = {}
        
        # (A) PreNorm and project to shared space
        Z_vision = self.vision_to_shared(self.vision_norm(vision_hidden))
        Z_text = self.text_to_shared(self.text_norm(text_hidden))
        
        # (B) Cross-only bidirectional attention
        # Text queries attend to vision
        Q_text = self.Q_text(Z_text)
        K_vision_to_text = self.K_vision_to_text(Z_vision)
        V_vision_to_text = self.V_vision_to_text(Z_vision)
        
        tau_text = F.softplus(self.tau_text_param)
        attn_text, attn_weights_text = self._cross_attention(
            Q_text, K_vision_to_text, V_vision_to_text,
            tau=tau_text,
            bias=self.bias_text_from_vision if self.use_positional_bias else None,
            mask=None,  # Can add cross-modal masking here if needed
        )
        attn_text = self.O_text(attn_text)
        
        # Vision queries attend to text
        Q_vision = self.Q_vision(Z_vision)
        K_text_to_vision = self.K_text_to_vision(Z_text)
        V_text_to_vision = self.V_text_to_vision(Z_text)
        
        tau_vision = F.softplus(self.tau_vision_param)
        attn_vision, attn_weights_vision = self._cross_attention(
            Q_vision, K_text_to_vision, V_text_to_vision,
            tau=tau_vision,
            bias=self.bias_vision_from_text if self.use_positional_bias else None,
            mask=None,
        )
        attn_vision = self.O_vision(attn_vision)
        
        # Dropout on attention outputs
        attn_text = self.output_dropout(attn_text)
        attn_vision = self.output_dropout(attn_vision)
        
        # (C) Reverse-project and gated residual
        # Compute gates (sigmoid of learned biases)
        gate_text = torch.sigmoid(self.gate_text_bias)
        gate_vision = torch.sigmoid(self.gate_vision_bias)
        
        # Project back to native dimensions and add gated residual
        text_update = self.shared_to_text(attn_text)
        vision_update = self.shared_to_vision(attn_vision)
        
        updated_text = text_hidden + gate_text * text_update
        updated_vision = vision_hidden + gate_vision * vision_update
        
        # Store auxiliary outputs
        aux_outputs['gate_text'] = gate_text.item()
        aux_outputs['gate_vision'] = gate_vision.item()
        aux_outputs['tau_text'] = tau_text.item()
        aux_outputs['tau_vision'] = tau_vision.item()
        aux_outputs['Z_vision'] = Z_vision  # Shared space representations for pooling
        aux_outputs['Z_text'] = Z_text      # As per vlm_idea.md, pool from shared space
        
        if return_attention_weights:
            aux_outputs['attn_weights_text'] = attn_weights_text  # For cycle consistency
            aux_outputs['attn_weights_vision'] = attn_weights_vision
        
        return updated_vision, updated_text, aux_outputs

