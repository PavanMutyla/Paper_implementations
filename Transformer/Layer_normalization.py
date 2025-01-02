import torch 
import torch.nn as nn
import math
class Layer_norm(nn.Module):
    def __init__(self, features : int, eps : float = 10**-6): # we take epsilon value if the Standard deviations comes out as 0
        super().__init__()
        self.eps = eps 
        self.features  = features
        self.gamma = nn.Parameter(torch.ones(self.features))
        self.beta = nn.Parameter(torch.zeros(self.features))
    def forward(self, x):
        
        mean = x.mean(dim=-1, keep_dim = True )
        std = x.std(dim = -1, keep_dim = True)

        # dim = -1, refers to the last dimension of the tensor.
        # keep_dim = True, This parameter determines whether the output tensor has the same number of dimensions as the input tensor.
        # if true : the dimensions with size 1 are retained in the result tensor.
        # if false : the ouput will be squeezed to remove any dimensions of 1
        return self.gamma (x -mean) / (std + self.eps) + self.beta
        
