"""
Next step: add support for mask
"""

import torch
from torch import nn


class SingleHeadedAttention(nn.Module):
    """
    Single-headed attention layer

    Args:
        d_model (int): The dimensions for word embedding for this model
        d_k (int): The dimension for query and key vector
        d_v (int): The dimension for value vector
    """
    def __init__(self, d_model, d_k, d_v):
        self.d_model:int = d_model
        self.d_k:int = d_k
        self.d_v:int = d_v
        self.wq:torch.nn.Linear = nn.Linear(d_model, d_k) # Weights for query
        self.wk:torch.nn.Linear = nn.Linear(d_model, d_k) # weights for key
        self.wv:torch.nn.Linear = nn.Linear(d_model, d_v) # weights for value

    def forward(self, x):
        q:torch.Tensor = self.wq(x)
        k:torch.Tensor = self.wk(x)
        v:torch.Tensor = self.wv(x)
        scores:torch.Tensor = torch.matmul(q, torch.transpose(k, -2, -1)) / self.d_k
        result:torch.Tensor = torch.matmul(torch.softmax(scores), v)
        return result


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention layer

    Args:
        d_model (int): The dimensions for word embedding for this model
        h (int): Number of attention heades
        d_k (int): The dimension for query and key vector
        d_v (int): The dimension for value vector
    
    Note:
        `d_k * h` must equal to `d_model`
    """
    def __init__(self, d_model, h, d_k, d_v):
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.wq:torch.nn.Linear = nn.Linear(d_model, d_k) # Weights for query
        self.wk:torch.nn.Linear = nn.Linear(d_model, d_k) # weights for key
        self.wv:torch.nn.Linear = nn.Linear(d_model, d_v) # weights for value

    def forward(self, x):
        q:torch.Tensor = self.wq(x).view()
        k:torch.Tensor = self.wk(x)
        v:torch.Tensor = self.wv(x)


if __name__ == "__main__":
    pass