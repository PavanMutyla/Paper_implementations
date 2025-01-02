import torch 
import torch.nn as nn
import math
from Transformer.Layer_normalization import Layer_norm
from Transformer.Feed_forward_module import feed_forward
from Transformer.Multi_head_attention import multihead_attention
class Decoder(nn.Module):
    '''V,K,Q
    Block that functions for the decoder operatiosn, not inclusive of target and positional embeddings and output probs
    '''
    def __init__(self, dim:int, heads : int, dropout : float, d_ff:int = None ):
        super().__init__()
        self.masked_attn = multihead_attention(dim, h = heads, dropout = dropout )
        self.cross_attn = multihead_attention(dim, h = heads, dropout = dropout)
        self.norm1 = Layer_norm(dim)
        self.norm2 = Layer_norm(dim)
        self.norm3 = Layer_norm(dim)
        self.feed_forward = feed_forward(dropout, dim_model = dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x , target_mask, src_mask, encoder_output):
        masked_attn = self.masked_attn(x,x,x,target_mask)
        op1 = self.norm1(x + self.dropout(masked_attn))
        # cross attention
        cross_attn = self.cross_attn(op1, encoder_output, encoder_output, src_mask)
        op2 = self.norm2(op1 + self.dropout(cross_attn))
        # feed forward and layer norm
        feed_forward  = self.feed_forward(op2)
        op3 = self.norm3(op2 + self.dropout(feed_forward))
        return op3
        
