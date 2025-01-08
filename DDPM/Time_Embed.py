import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbed(nn.Module):

    """Takes time stam't' and the required embeddings dimnestion.
    Then 't' is passed to Embedding followed by Linear layer, acitvation function and a final Linear layer:
    This is done to project the timestamp values as vectors.
    return: BxD embedding representation of B time steps.
    """    
    
    def __init__(self, t_embed_dim):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.fc = nn.Linear(t_embed_dim, t_embed_dim)
        self.swish = Swish()
    
    def forward(self, t):

        # Factor: 10000^(2i/d_model)
        factor = 10000 ** (torch.arange(
            start=0, end=self.t_embed_dim // 2, dtype=torch.float32, device=t.device
        ) / (self.t_embed_dim // 2))

        # Compute embeddings
        t_emb = t[:, None] / factor  # Shape: (B, t_embed_dim // 2)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)  # Shape: (B, t_embed_dim)

        # Pass through fully connected layer and Swish activation
        t_emb = self.swish(self.fc(t_emb))  # Final projection with non-linearity
        return t_emb
