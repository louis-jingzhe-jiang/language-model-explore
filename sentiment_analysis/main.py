import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import transformer
import torch
from torch import nn


class Model1(nn.Module):
    """
    Transformer encoder model using multi-headed attention. Each layer is identical, with same
    input and output dimensions `d_model`.

    Args:
        num_embeds (int): Size of the dictionary of embeddings
        d_model (int): The dimension for word embedding for this model
        n_layer (int): The number of encoder layers
        h (int): Number of attention heades
        d_ff (int): The dimension for feed-forward network
    """
    def __init__(self, num_embeds:int, d_model:int, n_layer:int, h:int, d_ff:int):
        super().__init__()
        self.model = transformer.EncoderBasic(num_embeds, d_model, n_layer, h, d_ff)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x:torch.Tensor):
        x = self.model(x)
        x = self.output(x)
        return x


class Model2(nn.Module):
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
    def __init__(self, num_embeds:int, d_model:int, n_layer:int, h:int, d_ff:int):
        super().__init__()
        self.model = transformer.EncoderBasic(num_embeds, d_model, n_layer, h, d_ff)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x:torch.Tensor):
        x = self.model(x)
        x = self.output(x)
        return x
    

class Model3(nn.Module):
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
    def __init__(self, num_embeds:int, d_model:int, n_layer:int, h:int, d_ff:int):
        super().__init__()
        self.model = transformer.EncoderBasic(num_embeds, d_model, n_layer, h, d_ff)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x:torch.Tensor):
        x = self.model(x)
        x = self.output(x)
        return x
    

class Model4(nn.Module):
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
    def __init__(self, num_embeds:int, d_model:int, n_layer:int, h:int, d_ff:int):
        super().__init__()
        self.model = transformer.EncoderBasic(num_embeds, d_model, n_layer, h, d_ff)
        self.output = nn.Linear(d_model, 2)

    def forward(self, x:torch.Tensor):
        x = self.model(x)
        x = self.output(x)
        return x
    

if __name__ == "__main__":
    # model constants
    VOCAB_SIZE = 30522
    D_MODEL = 512
    N_LAYER = 1
    H = 1
    D_FF = 512

    # create model
    #model1 = Model1(VOCAB_SIZE, D_MODEL, N_LAYER, H, D_FF)