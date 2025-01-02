import torch 
import torch.nn as nn
import math
class positional_encoding(nn.Module):
    def __init__(self, dim_model : int, seq_length : int, dropout : float):
        super().__init__()
        self.dim = dim_model
        self.seq = seq_length
        self.dropout = nn.Dropout(dropout)
        # empty matrix of shape (dim, seq)
        pe = torch.zeros(self.seq, self.dim)
        # vector of shape (seq) to get the postion of the word inside the sentence
        position = torch.arange(0, self.seq, dtype = torch.float).unsqueeze(1) # shape is (seq_len, 1)
        # formula
        div_term = torch.exp(torch.arange(0, self.dim,2).float() * (-math.log(10000)/self.dim) )
        # vector for even positions (sine)
        pe[:, 0::2] = torch.sin(position * div_term)
        # vector for odd positions (cosine)
        pe[:,1::2] = torch.cos(position * div_term)

        pe.unsqueeze(0) # shape is (1, seq, dim)
        self.register_buffer('pe', pe) # a tensor that can be saved in the model but not as a parameter, will also be saved when the file is saved
        
        
    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return x
