"""
Next step: add support for mask; add dropout
"""

import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int = 4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]


class SingleHeadedAttention(nn.Module):
    """
    Single-headed attention layer

    Args:
        d_model (int): The dimensions for word embedding for this model
        d_k (int): The dimension for query and key vector
        d_v (int): The dimension for value vector
    
    Notes:
        In attention masks, assign `-torch.inf` (negative infinity) to positions that should be 
        ignored (masked), and use 0 for all unmasked positions
    """
    def __init__(self, d_model:int, d_k:int, d_v:int):
        super().__init__()
        # weights for q, k, and v
        self.d_k = d_k
        self.wq:torch.nn.Linear = nn.Linear(d_model, d_k)
        self.wk:torch.nn.Linear = nn.Linear(d_model, d_k)
        self.wv:torch.nn.Linear = nn.Linear(d_model, d_v)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        # calculate q, k and v projections
        q:torch.Tensor = self.wq(x)
        k:torch.Tensor = self.wk(x)
        v:torch.Tensor = self.wv(x)
        # calculate attention score
        scores:torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # apply mask
        if mask is not None:
            scores += mask
        # combine attention score with value projection
        result:torch.Tensor = torch.matmul(torch.softmax(scores, -1), v)
        return result


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention layer. Output dimension will be `d_v * h`

    Args:
        d_model (int): The dimensions for word embedding for this model
        h (int): Number of attention heades
        d_v (int): The dimension for value vector in each attention head
    
    Note:
        `d_model` must be divisible by `h`\n
        In attention masks, assign `-torch.inf` (negative infinity) to positions that should be 
        ignored (masked), and use 0 for all unmasked positions
    """
    def __init__(self, d_model:int, h:int, d_v:int):
        super().__init__()
        # make sure `d_model` is divisible by `h`
        assert d_model % h == 0
        self.h:int = h
        self.d_k:int = d_model // h
        self.d_v:int = d_v
        # weights for q, k, and v
        self.wq:torch.nn.Linear = nn.Linear(d_model, self.d_k * h)
        self.wk:torch.nn.Linear = nn.Linear(d_model, self.d_k * h)
        self.wv:torch.nn.Linear = nn.Linear(d_model, self.d_v * h)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        # get dimensions of input
        batch_size, d_context, d_model = x.shape
        # calculate q, k and v projections and reshape to multiple heads
        q:torch.Tensor = self.wq(x).view(batch_size, d_context, self.h, self.d_k)
        k:torch.Tensor = self.wk(x).view(batch_size, d_context, self.h, self.d_k)
        v:torch.Tensor = self.wv(x).view(batch_size, d_context, self.h, self.d_v)
        # transpose q, k, and v to make attention head to be the second-major axis
        q = q.transpose(1, 2) # (batch_size, self.h, d_context, self.d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) # (batch_size, self.h, d_context, self.d_v)
        # calculate attention scores (batch_size, self.h, d_context, d_context)
        scores:torch.Tensor = torch.matmul(q, k.transopose(-2, -1)) / math.sqrt(self.d_k)
        # apply attention mask
        if mask is not None:
            scores += mask.unsqueeze(1)
        # combine attention score with value projection
        result:torch.Tensor = torch.matmul(torch.softmax(scores, -1), v)
        return result
        

class FeedForward(nn.Module):
    """
    Multi-headed attention layer

    Args:
        d_in (int): The embedding dimensions for input
        d_ff (int): The dimension for feed-forward network
        d_out (int): The embedding dimension for output
    """
    def __init__(self, d_in:int, d_ff:int, d_out:int):
        super().__init__()
        self.layer1:torch.nn.Linear = nn.Linear(d_in, d_ff)
        self.relu:torch.nn.ReLU = nn.ReLU()
        self.layer2:torch.nn.Linear = nn.Linear(d_ff, d_out)
    
    def forward(self, x:torch.Tensor):
        result:torch.Tensor = self.layer1(x)
        result = self.relu(result)
        result = self.layer2(result)
        return result
