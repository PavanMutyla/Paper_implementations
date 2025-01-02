import torch 
import torch.nn as nn
import math
from Transformer.Layer_normalization import Layer_norm
from Transformer.Feed_forward_module import feed_forward
from Transformer.Multi_head_attention import multihead_attention
class Encoder(nn.Module):
    def __init__(self,dim:int, heads : int, dropout : float, d_ff : int = None):
        super().__init__()
        '''Block that functions for the encoder operations, not inclusive of input and positional embeddings'''
        self.attn = multihead_attention(dim, h = heads, dropout = dropout)
        self.feed_forward = feed_forward(dropout = dropout, dim_model = dim)
        self.norm1 = Layer_norm(dim)
        self.norm2 = Layer_norm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask = None):
        attn = self.attn(x, x,x , mask)
        out = self.norm1(x+ self.dropout(attn))
        print(out)
        feed_forward = self.feed_forward(out)
        out_put = self.norm2(x + self.dropout(feed_forward))
        return out_put
        
        
        
