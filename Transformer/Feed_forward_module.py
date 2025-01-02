import torch 
import torch.nn as nn
import math
class feed_forward(nn.Module):
    def __init__(self, dropout: float,  dim_model: int = 512 , d_ff: int = 2048 ):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, dim_model)
    def forward(self, x):
        # input ---> hidden ---. ouput
        # (batch, seq_len, dim_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, dim_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df f = 2048. ( in paper)        
