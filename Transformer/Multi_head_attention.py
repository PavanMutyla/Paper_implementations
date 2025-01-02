import torch 
import torch.nn as nn
import math
class multihead_attention(nn.Module):
    def __init__(self, dim : int, h : int, dropout : float):
        super().__init__()
        self.dim = dim
        self.h = h
        self.dropout  = dropout
        # check of the dim is divisible by the heads
        assert dim % h ==0, "dimensions cannot be divisibly by the number of parallel heads"
        self.dim_head = self.dim // self.h
        self.w_k = nn.Linear(self.dim, self.dim, bias = False)
        self.w_q = nn.Linear(self.dim, self.dim, bias = False)
        self.w_v = nn.Linear(self.dim, self.dim, bias = False)
        self.w_o = nn.Linear(self.dim, self.dim, bias = False)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product(self, q, k, v, mask = None):
        '''
        performs scaled dot product for every head
        '''
        attention_scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask == True:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e20)
        attention_scores = torch.softmax(attention_scores, dim = -1)
        output = torch.matmul(attention_scores, v)
        return output
    def split_heads(self, x):
        # x ---> (batch_size, seq_len, dim)
        '''
        example:
        x = (32, 100, 512) and h = 8
        it returns
        x = (32, 100, 8, 64)
        after transpose
        x = (32, 8, 100, 64)
        '''
        batch_size, seq_len, dim = x.size()
        return x.view(batch_size, seq_len, self.h, self.d_k).transpose(1,2) # transpose element at 1,2 position

    def merge_heads(self, x):
        '''
        takes individual heads and combines them and return as follows:
        x = (32, 8,100,64)
        1) reshape
        x = (32, 100, 8, 64)
        2) contiguous
        ensures that the tensor's memory layout is organized in a contiguous block
        3) view
        x = (32, 100, 512)
        '''
        batch_size, _, seq_len, dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

    def forward(self, q, k, v, mask = None):
        '''takes the inputs q,k,v
        shape of each is (batch_size, seq_len, dim)
        1) splits into heads 
        2) learned weights w_q, w_k, w_v are applied , these are linear projection of the input space that seperates QKV
        3)perform attention on each head using scaled dot product
        4)combine all the heads
        5)the w_o transforms the shape (batch_size,num_heads,seq_len,dk) ---> (batch_size, seq-len, dim) 

        '''
        Q = split_heads(self.w_q(q))
        K = split_heads(self.w_k(k))
        V = split_heads(self.w_v(v))

        output_attn = self.scaled_dot_product(Q, K, V, mask)
        output = self.w_o(self.merge_heads(output_attn))
        return output
        
