"""
Caption Decoder for Generative Image Captioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CaptionDecoder(nn.Module):
    """
    Transformer decoder for generating captions from visual features.
    Uses causal self-attention and cross-attention to visual features.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, embed_dim)
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm like in the VLM design
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_encoding, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for caption generation.
        
        Args:
            input_ids: (batch_size, seq_len) - Token IDs
            visual_features: (batch_size, num_visual_tokens, visual_dim) - Visual features
            attention_mask: (batch_size, seq_len) - Attention mask for text
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        token_embeds = self.token_embedding(input_ids)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        text_embeds = token_embeds + pos_encoding
        
        # Create causal mask for decoder (prevent looking ahead)
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        # Create padding mask if provided
        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = (attention_mask == 0)
        
        # Decode
        decoded = self.transformer_decoder(
            tgt=text_embeds,
            memory=visual_features,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        
        return logits
    
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate captions autoregressively using greedy or sampling.
        
        Args:
            visual_features: (batch_size, num_visual_tokens, visual_dim)
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (1.0 = disabled)
        
        Returns:
            generated_ids: (batch_size, seq_len)
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # Start with BOS token
        generated_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Forward pass
            logits = self.forward(
                input_ids=generated_ids,
                visual_features=visual_features,
            )
            
            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated_ids
    
    @torch.no_grad()
    def generate_greedy(
        self,
        visual_features: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 50,
    ) -> torch.Tensor:
        """
        Generate captions using greedy decoding (faster, deterministic).
        
        Args:
            visual_features: (batch_size, num_visual_tokens, visual_dim)
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum generation length
        
        Returns:
            generated_ids: (batch_size, seq_len)
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # Start with BOS token
        generated_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Forward pass
            logits = self.forward(
                input_ids=generated_ids,
                visual_features=visual_features,
            )
            
            # Get next token (argmax)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Mark sequences that generated EOS
            done |= (next_token.squeeze(-1) == eos_token_id)
            
            # Stop if all sequences are done
            if done.all():
                break
        
        return generated_ids






