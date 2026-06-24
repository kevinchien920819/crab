import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:,:x.size(1)]
        return x

class RhythmEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)
    
    def forward(self, *features: Tensor) -> Tensor:
        """
        features:多個 [B, T] tensor
            EX: [vowel_duration, vowel_deviation, vowel_difference,consonant_duration,consonant_deviation, consonant_difference] each shape [B, T]
        """
        x = torch.stack(features, dim=-1)  # [B, T, F]
        x = self.linear(x) # [B, T, D]
        x = self.pos(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x
    