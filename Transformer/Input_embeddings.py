import torch 
import torch.nn as nn
import math
class input_embeddings(nn.Module):
    def __init__(self, dimensions : int, vocab_size : int ):
        super().__init__()
        self.dimensions = dimensions
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dimensions) # embedding that takes num_embeddings and embedding_dim
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dimensions)
