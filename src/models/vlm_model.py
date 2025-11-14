"""
Main Cross-Modal VLM Model

Combines vision and text encoders with cross-modal interaction layers
and various pooling/prediction heads for different objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from .cross_attention import CrossModalInteractionLayer
from .vision_text_encoders import VisionEncoder, TextEncoder



class CrossModalVLM(nn.Module):
    """
    Cross-Modal Vision-Language Model with MOT-inspired shared latent space.
    
    Architecture:
    1. Vision encoder (ViT) and Text encoder (BERT) process inputs independently
    2. Q interaction layers perform bidirectional cross-modal attention
    3. Multiple prediction heads for different objectives (ITC, MLM, MIM, ITM)
    """
    
    def __init__(
        self,
        vision_config: dict,
        text_config: dict,
        cross_modal_config: dict,
        pooling_config: dict,
    ):
        
        super().__init__()
        
        # Encoders
        self.vision_encoder = VisionEncoder(
            model_name=vision_config['model_name'],
            hidden_dim=vision_config['hidden_dim'],
            freeze=vision_config.get('freeze', False),
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_config['model_name'],
            hidden_dim=text_config['hidden_dim'],
            freeze=text_config.get('freeze', False),
        )
        
        # Dimensions
        self.vision_dim = vision_config['hidden_dim']
        self.text_dim = text_config['hidden_dim']
        self.shared_dim = cross_modal_config['shared_dim']
        
        # Cross-modal interaction layers
        self.num_layers = cross_modal_config['num_layers']
        self.interaction_layers = nn.ModuleList([
            CrossModalInteractionLayer(
                vision_dim=self.vision_dim,
                text_dim=self.text_dim,
                shared_dim=self.shared_dim,
                num_heads=cross_modal_config['num_heads'],
                dropout=cross_modal_config.get('dropout', 0.1),
                attention_dropout=cross_modal_config.get('attention_dropout', 0.1),
                use_positional_bias=cross_modal_config.get('use_positional_bias', True),
                init_temperature=cross_modal_config.get('init_temperature', 1.0),
                init_gate_bias=cross_modal_config.get('init_gate_bias', -2.2),
            )
            for _ in range(self.num_layers)
        ])
        
        # Pooling strategy
        self.pooling_type = pooling_config.get('type', 'mean')
        
        # Projection heads for contrastive learning
        projection_dim = pooling_config.get('projection_dim', 256)
        
        # DUAL heads for DUAL ITC loss (as in vlm_idea.md)
        # Cross-modal aware projections: W_p^v, W_p^t
        self.vision_pool_proj_cross = nn.Linear(self.shared_dim, projection_dim)
        self.text_pool_proj_cross = nn.Linear(self.shared_dim, projection_dim)
        
        # Unimodal projections for bi-encoder retrieval: W_pu^v, W_pu^t
        # These process features BEFORE interaction layers
        self.vision_pool_proj_uni = nn.Linear(self.vision_dim, projection_dim)
        self.text_pool_proj_uni = nn.Linear(self.text_dim, projection_dim)
        
        # Learnable temperature for contrastive loss (as in CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(14) as in CLIP
        
        # Learnable alpha for DUAL ITC loss weighting (Eq 9 in vlm_idea.md)
        # α * L_ITC^cross + (1-α) * L_ITC^uni
        self.itc_alpha = nn.Parameter(torch.tensor(0.5))  # Start with equal weighting
        
        # MLM head for masked language modeling
        self.mlm_head = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.LayerNorm(self.text_dim),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_encoder.get_vocab_size()),
        )
        
        # MIM head for masked image modeling (predict patch features)
        self.mim_head = nn.Sequential(
            nn.Linear(self.vision_dim, self.vision_dim),
            nn.LayerNorm(self.vision_dim),
            nn.GELU(),
            nn.Linear(self.vision_dim, self.vision_dim),  # Predict vision features
        )
        
        # ITM head for image-text matching
        self.itm_head = nn.Sequential(
            nn.Linear(projection_dim * 2 + 1, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, 2),  # Binary classification
        )
        
        # Caption generation decoder (optional, for generative tasks)
        if cross_modal_config.get('use_caption_decoder', False):
            from .caption_decoder import CaptionDecoder
            self.caption_decoder = CaptionDecoder(
                vocab_size=self.text_encoder.get_vocab_size(),
                embed_dim=self.vision_dim,
                num_layers=cross_modal_config.get('decoder_layers', 6),
                num_heads=cross_modal_config.get('decoder_heads', 8),
                max_seq_length=128,
            )
        else:
            self.caption_decoder = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for projection heads"""
        for module in [self.vision_pool_proj_cross, self.text_pool_proj_cross,
                      self.vision_pool_proj_uni, self.text_pool_proj_uni]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        
        # Initialize MLM/MIM/ITM heads
        for module in [self.mlm_head, self.mim_head, self.itm_head]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, std=0.02)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
    
    def get_itc_alpha(self) -> torch.Tensor:
        """Get the alpha parameter for DUAL ITC loss, clamped to [0, 1]"""
        return torch.sigmoid(self.itc_alpha)  # Use sigmoid to ensure [0, 1] range
    
    def pool_features(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence features based on configured strategy.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            pooled: (batch, hidden_dim)
        """
        if self.pooling_type == 'cls':
            # Use CLS token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_type == 'mean':
            # Mean pooling with masking
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask
                pooled = hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
            return pooled
        
        elif self.pooling_type == 'max':
            # Max pooling with masking
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            pooled, _ = hidden_states.max(dim=1)
            return pooled
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
    
    def encode_vision_only(
        self, 
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode images without cross-modal interaction (for efficient retrieval evaluation).
        
        Args:
            pixel_values: (batch, 3, H, W)
            
        Returns:
            vision_embeds: (batch, projection_dim) - normalized embeddings
        """
        # Encode vision
        vision_hidden = self.vision_encoder(pixel_values)  # (batch, n_vision, vision_dim)
        
        # Pool features
        vision_pooled = self.pool_features(vision_hidden)
        
        # Project using unimodal projection (no cross-modal interaction)
        vision_embeds = self.vision_pool_proj_uni(vision_pooled)
        
        # Normalize for cosine similarity
        vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
        
        return vision_embeds
    
    def encode_text_only(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text without cross-modal interaction (for efficient retrieval evaluation).
        
        Args:
            input_ids: (batch, text_len)
            attention_mask: (batch, text_len)
            
        Returns:
            text_embeds: (batch, projection_dim) - normalized embeddings
        """
        # Encode text
        text_hidden = self.text_encoder(input_ids, attention_mask)  # (batch, n_text, text_dim)
        
        # Pool features
        text_pooled = self.pool_features(text_hidden, mask=attention_mask)
        
        # Project using unimodal projection (no cross-modal interaction)
        text_embeds = self.text_pool_proj_uni(text_pooled)
        
        # Normalize for cosine similarity
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return text_embeds
    
    def encode_vision_unimodal(
        self, 
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Alias for encode_vision_only for backward compatibility"""
        return self.encode_vision_only(pixel_values)
    
    def encode_text_unimodal(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Alias for encode_text_only for backward compatibility"""
        return self.encode_text_only(input_ids, attention_mask)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        mim_labels: Optional[torch.Tensor] = None,
        mim_mask: Optional[torch.Tensor] = None,
        negative_pixel_values: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full VLM model.
        
        Args:
            pixel_values: (batch, 3, H, W)
            input_ids: (batch, text_len)
            attention_mask: (batch, text_len)
            mlm_labels: (batch, text_len) - labels for MLM, -100 for non-masked
            mim_labels: (batch, num_patches, vision_dim) - targets for MIM
            mim_mask: (batch, num_patches) - 1 for masked patches
            negative_pixel_values: (batch, 3, H, W) - for ITM hard negatives
            return_attention_weights: return attention weights for cycle loss
            
        Returns:
            Dictionary with:
                - vision_embeds: pooled vision embeddings
                - text_embeds: pooled text embeddings
                - vision_hidden: final vision hidden states
                - text_hidden: final text hidden states
                - aux_outputs: auxiliary outputs from interaction layers
        """
        batch_size = pixel_values.shape[0]
        
        # Encode vision and text
        vision_hidden = self.vision_encoder(pixel_values)  # (batch, n_vision, vision_dim)
        text_hidden = self.text_encoder(input_ids, attention_mask)  # (batch, n_text, text_dim)
        
        # Cache pre-interaction hidden states for unimodal embeddings
        vision_hidden_pre_interaction = vision_hidden.clone()
        text_hidden_pre_interaction = text_hidden.clone()
        
        # Apply cross-modal interaction layers
        all_aux_outputs = []
        all_attn_weights_text = []
        all_attn_weights_vision = []
        
        for layer in self.interaction_layers:
            vision_hidden, text_hidden, aux = layer(
                vision_hidden,
                text_hidden,
                vision_mask=None,
                text_mask=attention_mask,
                return_attention_weights=return_attention_weights,
            )
            all_aux_outputs.append(aux)
            
            if return_attention_weights:
                all_attn_weights_text.append(aux['attn_weights_text'])
                all_attn_weights_vision.append(aux['attn_weights_vision'])
        
        # Final fused states
        final_vision_hidden = vision_hidden
        final_text_hidden = text_hidden
        
        # Pool from shared space Z_v and Z_t (as per vlm_idea.md)
        # "p_v = nrm(pool(Z_v^(ℓ*)) W_p^v), p_t = nrm(pool(Z_t^(ℓ*)) W_p^t)"
        # Get shared space representations from the last interaction layer
        last_aux = all_aux_outputs[-1] if all_aux_outputs else None
        if last_aux is not None:
            Z_vision = last_aux['Z_vision']  # (batch, n_vision, shared_dim)
            Z_text = last_aux['Z_text']      # (batch, n_text, shared_dim)
        else:
            # Fallback to native streams if no interaction layers
            Z_vision = final_vision_hidden
            Z_text = final_text_hidden
        
        # Pool the shared space representations
        vision_for_pool = self.pool_features(Z_vision)
        text_for_pool = self.pool_features(Z_text, mask=attention_mask)
        
        # Project to embedding space for contrastive learning (cross-modal aware)
        vision_embeds = self.vision_pool_proj_cross(vision_for_pool)  # (batch, projection_dim)
        text_embeds = self.text_pool_proj_cross(text_for_pool)  # (batch, projection_dim)
        
        # Normalize for cosine similarity (as per vlm_idea.md: "nrm")
        vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # ALSO compute unimodal embeddings (before interaction) for bi-encoder retrieval
        # This allows the model to learn retrieval-friendly unimodal alignments
        vision_for_pool_uni = self.pool_features(vision_hidden_pre_interaction)
        text_for_pool_uni = self.pool_features(text_hidden_pre_interaction, mask=attention_mask)
        
        # Use separate unimodal projections (W_pu^v, W_pu^t)
        vision_embeds_unimodal = self.vision_pool_proj_uni(vision_for_pool_uni)
        text_embeds_unimodal = self.text_pool_proj_uni(text_for_pool_uni)
        
        vision_embeds_unimodal = F.normalize(vision_embeds_unimodal, p=2, dim=-1)
        text_embeds_unimodal = F.normalize(text_embeds_unimodal, p=2, dim=-1)
        
        # Prepare outputs
        outputs = {
            'vision_embeds': vision_embeds,  # Cross-modal aware (for understanding tasks)
            'text_embeds': text_embeds,  # Cross-modal aware (for understanding tasks)
            'vision_embeds_unimodal': vision_embeds_unimodal,  # Unimodal (for retrieval)
            'text_embeds_unimodal': text_embeds_unimodal,  # Unimodal (for retrieval)
            'vision_hidden': final_vision_hidden,
            'text_hidden': final_text_hidden,
            'logit_scale': self.logit_scale.exp(),
            'alpha': self.get_itc_alpha(),  # Alpha for dual ITC loss weighting
            'aux_outputs': all_aux_outputs,
        }
        
        if return_attention_weights:
            outputs['attn_weights_text'] = all_attn_weights_text
            outputs['attn_weights_vision'] = all_attn_weights_vision
        
        return outputs
    
    def compute_mlm_logits(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """Compute MLM prediction logits"""
        return self.mlm_head(text_hidden)
    
    def compute_mim_predictions(self, vision_hidden: torch.Tensor) -> torch.Tensor:
        """Compute MIM predictions"""
        return self.mim_head(vision_hidden)
    
    def compute_itm_logits(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Image-Text Matching logits.
        
        Args:
            vision_embeds: (batch, projection_dim)
            text_embeds: (batch, projection_dim)
            
        Returns:
            logits: (batch, 2) - [not_match, match]
        """
        # Concatenate [text_embeds; vision_embeds; dot_product]
        dot_product = (vision_embeds * text_embeds).sum(dim=-1, keepdim=True)
        concat_features = torch.cat([text_embeds, vision_embeds, dot_product], dim=-1)
        
        return self.itm_head(concat_features)
    
    def unfreeze_encoders(self, num_layers: int = 4):
        """Unfreeze top layers of both encoders (for Stage B training)"""
        self.vision_encoder.unfreeze_top_layers(num_layers)
        self.text_encoder.unfreeze_top_layers(num_layers)
    
    def get_tokenizer(self):
        """Get the text tokenizer"""
        return self.text_encoder.get_tokenizer()
    
    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 50,
        use_greedy: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate captions for images.
        
        Args:
            pixel_values: (batch, 3, H, W)
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum generation length
            use_greedy: Use greedy decoding (True) or sampling (False)
            temperature: Sampling temperature (if use_greedy=False)
        
        Returns:
            generated_ids: (batch, seq_len)
        """
        self.eval()
        
        # Encode image
        vision_hidden = self.vision_encoder(pixel_values)
        
        # Apply cross-modal layers (use vision features as memory for decoder)
        # For generation, we use vision features directly without text
        # So we'll use the vision hidden states as the visual memory
        visual_features = vision_hidden  # (batch, num_patches, vision_dim)
        
        # Generate caption using decoder
        if use_greedy:
            generated_ids = self.caption_decoder.generate_greedy(
                visual_features=visual_features,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_length=max_length,
            )
        else:
            generated_ids = self.caption_decoder.generate(
                visual_features=visual_features,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_length=max_length,
                temperature=temperature,
            )
        
        return generated_ids
    
    def compute_caption_loss(
        self,
        pixel_values: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute caption generation loss (cross-entropy).
        
        Args:
            pixel_values: (batch, 3, H, W)
            caption_ids: (batch, seq_len) - Ground truth caption tokens
            caption_mask: (batch, seq_len) - Attention mask
        
        Returns:
            loss: Scalar cross-entropy loss
        """
        batch_size, seq_len = caption_ids.shape
        
        # Encode image
        vision_hidden = self.vision_encoder(pixel_values)
        
        # Prepare decoder input (shift right: [BOS, token1, token2, ...])
        decoder_input_ids = caption_ids[:, :-1]
        decoder_attention_mask = caption_mask[:, :-1] if caption_mask is not None else None
        
        # Prepare labels (shift left: [token1, token2, ..., EOS])
        labels = caption_ids[:, 1:]
        
        # Forward through decoder
        logits = self.caption_decoder(
            decoder_input_ids,
            vision_hidden,
            decoder_attention_mask,
        )
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,  # Ignore padding tokens
        )
        
        return loss

