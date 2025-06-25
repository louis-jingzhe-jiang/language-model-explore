import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import transformer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TokenizedDatasetFullMask(Dataset):
    def __init__(self, data_file:str, label_file:str, mask_file:str):
        self.data_tensor:torch.Tensor = torch.load(data_file)
        self.label_tensor:torch.Tensor = torch.load(label_file)
        self.mask_tensor:torch.Tensor = torch.load(mask_file)

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx:int):
        return self.data_tensor[idx], self.label_tensor[idx], self.mask_tensor[idx]


class TokenizedDatasetFull(Dataset):
    def __init__(self, data_file:str, label_file:str):
        self.data_tensor:torch.Tensor = torch.load(data_file)
        self.label_tensor:torch.Tensor = torch.load(label_file)

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx:int):
        return self.data_tensor[idx], self.label_tensor[idx]


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
        self.output = nn.Linear(d_model * 256, 2)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        x = self.model(x, mask=mask)
        batch_size, d_context, d_model = x.shape
        x = self.output(x.view(batch_size, d_context * d_model))
        return x
    

if __name__ == "__main__":
    # model constants
    VOCAB_SIZE = 30522
    D_MODEL = 512
    N_LAYER = 1
    H = 1
    D_FF = 512
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

    # prepare dataset
    train_data = TokenizedDatasetFullMask("train_input.pt", "train_label.pt", "train_mask.pt")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_data = TokenizedDatasetFullMask("test_input.pt", "test_label.pt", "test_mask.pt")
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    print(train_data.mask_tensor)

    # create model
    model1 = Model1(VOCAB_SIZE, D_MODEL, N_LAYER, H, D_FF)
    model1.to(DEVICE)

    # training constants and parameters
    N_EPOCH = 5
    LR = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=LR)

    # training loop
    for epoch in range(N_EPOCH):
        model1.train()
        running_loss = 0.0
        for data, label, mask in train_loader:
            # process mask
            print(mask)
            output = model1(data.to(DEVICE), mask=mask.to(DEVICE))
            loss = criterion(output, label.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        epoch_loss = running_loss / len(train_data)
        # evalulation
        model1.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for data, label, mask in test_loader:
                # process mask
                
                output = model1(data.to(DEVICE), mask=mask.to(DEVICE))
                loss = criterion(output, label.to(DEVICE))
                eval_loss += loss.item() * data.size(0)
        eval_loss = eval_loss / len(test_data)
        # show result
        print(f'Epoch [{epoch+1}/{N_EPOCH}], Loss: {epoch_loss:.4f}, Eval Loss: {eval_loss:.4f}')
        