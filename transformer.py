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
    Transformer encoder layer using multi-headed attention.

    Args:
        d_model (int): The dimension for word embedding for this model
        h (int): Number of attention heades
        d_ff (int): The dimension for feed-forward network
    """

    attention:component.MultiHeadedAttention
    norm1:torch.nn.LayerNorm
    feed_forward:component.FeedForward
    norm2:torch.nn.LayerNorm

    def __init__(self, d_model:int, h:int, d_ff:int):
        assert d_model % h == 0
        super().__init__()
        self.attention = component.MultiHeadedAttention(d_model, h, d_model // h)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = component.FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        att:torch.Tensor = self.attention(x, mask=mask)
        att = self.norm1(x + att)
        ff:torch.Tensor = self.feed_forward(att)
        ff = self.norm2(att + ff)
        return ff


class EncoderLayer1(nn.Module):
    """
    Transformer encoder layer using single-headed attention.

    Args:
        d_model (int): The dimension for word embedding for this model
        d_ff (int): The dimension for feed-forward network
    """

    attention:component.SingleHeadedAttention
    norm1:torch.nn.LayerNorm
    feed_forward:component.FeedForward
    norm2:torch.nn.LayerNorm

    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.attention = component.SingleHeadedAttention(d_model, d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = component.FeedForward(d_model, d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        att:torch.Tensor = self.attention(x, mask=mask)
        att = self.norm1(x + att)
        ff:torch.Tensor = self.feed_forward(att)
        ff = self.norm2(att + ff)
        return ff


class Decoder(nn.Module):
    """
    
    """
    def __init__(self, num_embeds:int, d_model:int, n_layer:int, h:int, d_ff:int):
        super().__init__()

    def forward(self, x:torch.Tensor):
        return x


class EncoderBasic(nn.Module):
    """
    Transformer encoder model using multi-headed attention. Each layer is identical, with same
    input and output dimensions.

    Args:
        num_embeds (int): Size of the dictionary of embeddings
        d_model (int): The dimension for word embedding for this model
        n_layer (int): The number of encoder layers
        h (int): Number of attention heades
        d_ff (int): The dimension for feed-forward network
    """

    embedding:torch.nn.Embedding
    pos_enc:component.PositionalEncodingSinusoidal
    layers:torch.nn.ModuleList
    norm:torch.nn.LayerNorm

    def __init__(self, num_embeds:int, d_model:int, n_layer:int, h:int, d_ff:int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeds, d_model)
        self.pos_enc = component.PositionalEncodingSinusoidal(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)
    

class EncoderBasic1(nn.Module):
    """
    Transformer encoder model using multi-headed attention. Each layer is identical, with same
    input and output dimensions.

    Args:
        num_embeds (int): Size of the dictionary of embeddings
        d_model (int): The dimension for word embedding for this model
        n_layer (int): The number of encoder layers
        h (int): Number of attention heades
        d_ff (int): The dimension for feed-forward network
    """
    embedding:torch.nn.Embedding
    pos_enc:component.PositionalEncodingSinusoidal
    layers:torch.nn.ModuleList
    norm:torch.nn.LayerNorm

    def __init__(self, num_embeds:int, d_model:int, n_layer:int, d_ff:int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeds, d_model)
        self.pos_enc = component.PositionalEncodingSinusoidal(d_model)
        self.layers = nn.ModuleList([EncoderLayer1(d_model, d_ff) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)