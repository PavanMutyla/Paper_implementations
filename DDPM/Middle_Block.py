import torch 
import torch.nn as nn
from Linear_scheduler import LinearScehduler
from Time_Embed import Swish,TimeEmbed
from Down_Block import DownBlock
class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, num_heads):
        super().__init__()
        # First Residual Block
        self.Block1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.Block2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Attention Block
        self.attention_norm = nn.GroupNorm(n_groups, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        # Input Projection and Time Embedding Projection
        self.input_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.time_projection = nn.Linear(out_channels, out_channels)

    def forward(self, x, t_emb):
        # First Residual Block
        res1_block1 = self.Block1(x)
        t_proj = self.time_projection(t_emb).view(t_emb.size(0), -1, 1, 1)  # Project time embedding
        res1_block2 = self.Block2(res1_block1 + t_proj)
        skip_connection1 = self.input_projection(x)
        out_residual = res1_block2 + skip_connection1
    
        # Attention Block
        batch, channel, h, w = out_residual.shape
        attn_input = out_residual.view(batch, channel, h * w).permute(0, 2, 1)  # Shape: (B, H*W, C)
        normalized_attn_input = attn_input.permute(0, 2, 1).view(batch, channel, h, w)  # Shape: (B, C, H, W)
        normalized_attn_input = self.attention_norm(normalized_attn_input)  # Apply GroupNorm
        normalized_attn_input = normalized_attn_input.view(batch, channel, h * w).permute(0, 2, 1)  # Shape: (B, H*W, C)
        out_attn, _ = self.attention(normalized_attn_input, normalized_attn_input, normalized_attn_input)
        out_attn = out_attn.permute(0, 2, 1).view(batch, channel, h, w)  # Reshape back to [B, C, H, W]
    
        # Second Residual Block
        res2_block1 = self.Block1(out_attn)
        t_proj2 = self.time_projection(t_emb).view(t_emb.size(0), -1, 1, 1)
        res2_block2 = self.Block2(res2_block1 + t_proj2)
        skip_connection2 = self.input_projection(x)
        out_final = res2_block2 + skip_connection2
        return out_final

