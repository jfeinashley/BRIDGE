"""
Vision and Text Encoders for Cross-Modal VLM

Wrappers around pretrained ViT and BERT models with optional projection layers.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, BertModel, AutoModel


class VisionEncoder(nn.Module):
    """
    Vision encoder using pretrained ViT.
    """
    
    def __init__(self, model_name: str, hidden_dim: int, freeze: bool = False):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        
        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(model_name)
        self.vit_dim = self.vit.config.hidden_size
        
        # Optional projection if dimensions don't match
        if self.vit_dim != hidden_dim:
            self.projection = nn.Linear(self.vit_dim, hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Optionally freeze the base model
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor, return_cls: bool = False):
        """
        Args:
            pixel_values: (batch, 3, H, W)
            return_cls: If True, also return CLS token embedding
            
        Returns:
            vision_hidden: (batch, num_patches, hidden_dim)
            cls_embed (optional): (batch, hidden_dim)
        """
        outputs = self.vit(pixel_values=pixel_values, return_dict=True)
        
        # Get all patch embeddings (including CLS token)
        hidden_states = outputs.last_hidden_state  # (batch, num_patches+1, vit_dim)
        
        # Project to target dimension
        hidden_states = self.projection(hidden_states)  # (batch, num_patches+1, hidden_dim)
        
        if return_cls:
            cls_embed = hidden_states[:, 0, :]  # (batch, hidden_dim)
            return hidden_states, cls_embed
        else:
            return hidden_states
    
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers: int = 4):
        """Unfreeze top N layers of the vision encoder"""
        # ViT layers are in vit.encoder.layer
        if hasattr(self.vit, 'encoder') and hasattr(self.vit.encoder, 'layer'):
            total_layers = len(self.vit.encoder.layer)
            start_layer = max(0, total_layers - num_layers)
            for i in range(start_layer, total_layers):
                for param in self.vit.encoder.layer[i].parameters():
                    param.requires_grad = True
            # Also unfreeze the projection if it exists
            if hasattr(self, 'projection'):
                for param in self.projection.parameters():
                    param.requires_grad = True


class TextEncoder(nn.Module):
    """
    Text encoder using pretrained BERT.
    """
    
    def __init__(self, model_name: str, hidden_dim: int, freeze: bool = False):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_dim = self.bert.config.hidden_size
        self.vocab_size = self.bert.config.vocab_size
        
        # Optional projection if dimensions don't match
        if self.bert_dim != hidden_dim:
            self.projection = nn.Linear(self.bert_dim, hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Optionally freeze the base model
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_cls: bool = False):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            return_cls: If True, also return CLS token embedding
            
        Returns:
            text_hidden: (batch, seq_len, hidden_dim)
            cls_embed (optional): (batch, hidden_dim)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get all token embeddings (including CLS token)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, bert_dim)
        
        # Project to target dimension
        hidden_states = self.projection(hidden_states)  # (batch, seq_len, hidden_dim)
        
        if return_cls:
            cls_embed = hidden_states[:, 0, :]  # (batch, hidden_dim)
            return hidden_states, cls_embed
        else:
            return hidden_states
    
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers: int = 4):
        """Unfreeze top N layers of the text encoder"""
        # BERT layers are in bert.encoder.layer
        if hasattr(self.bert, 'encoder') and hasattr(self.bert.encoder, 'layer'):
            total_layers = len(self.bert.encoder.layer)
            start_layer = max(0, total_layers - num_layers)
            for i in range(start_layer, total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            # Also unfreeze the projection if it exists
            if hasattr(self, 'projection'):
                for param in self.projection.parameters():
                    param.requires_grad = True
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size of the text encoder"""
        return self.vocab_size
    
    def get_tokenizer(self):
        """Get the tokenizer for this text encoder"""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_name)




