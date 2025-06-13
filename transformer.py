"""
next steps: add dropout
"""

import torch
from torch import nn
import component


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class EncoderLayer(nn.Module):
    """
    Transformer encoder lahyer using multi-headed attention

    Args:
        d_model (int): The dimensions for word embedding for this model
        h (int): Number of attention heades
        d_ff (int): The dimension for feed-forward network
        d_out (int): The embedding dimension for output
    """
    def __init__(self, d_model, h, d_ff):
        assert d_model % h == 0
        super().__init__()
        self.attention = component.MultiHeadedAttention(d_model, h, d_model // h)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = component.FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        att:torch.Tensor = self.attention(x)
        att = self.norm1(x + att)
        ff:torch.Tensor = self.feed_forward(att)
        ff = self.norm2(att + ff)
        return ff


class Encoder(nn.Module):
    """
    
    """
    def __init__(self, num_embeds:int, d_model:int, n_layer:int, h:int, d_ff:int):
        super().__init__()
        self.embedding:torch.nn.Embedding = nn.Embedding(num_embeds, d_model)
        self.layers:torch.nn.ModuleList = nn.ModuleList([EncoderLayer(d_model, h, d_ff) 
                                                         for _ in range(n_layer)])
        self.norm:torch.nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


if __name__ == "__main__":
    pass