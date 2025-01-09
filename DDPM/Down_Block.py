import torch
import torch.nn as nn
from Time_Embed import Swish, TimeEmbed

class DownBlock(nn.Module):
    """
    A downsampling block with residual connections, attention, and time embeddings.
    """
    def __init__(self, n_groups, in_channels, out_channels, num_heads):
        super().__init__()
        self.Block1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.Block2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.attention_norm = nn.GroupNorm(n_groups, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        # Linear projection to match input size for skip connection
        self.linear_layer_input = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        # Linear projection for time embedding to match out_channels
        self.time_proj = nn.Linear(out_channels, out_channels)

        # Downsampling layer
        self.down_sampling = nn.Conv2d(out_channels, out_channels, kernel_size=4, padding=1, stride=2)

    def forward(self, x, t_emb):
        # Residual block
        block1 = self.Block1(x)
        
        # Project time embedding and add
        t_proj = self.time_proj(t_emb)  # Shape: (B, out_channels)
        t_proj = t_proj[:, :, None, None]  # Add spatial dimensions to match the expected shape by Block1
        block_time_sum = block1 + t_proj

        block2 = self.Block2(block_time_sum)
        skip_connection = self.linear_layer_input(x)
        out_residual = block2 + skip_connection

        # Attention block
        batch, channel, h, w = out_residual.shape
        attn_input = out_residual.reshape(batch, channel, h * w)
        attn_input = self.attention_norm(attn_input)
        attn_input = attn_input.transpose(1, 2)
        out_attn, _ = self.attention(attn_input, attn_input, attn_input)
        out_attn = out_attn.transpose(1, 2).reshape(batch, channel, h, w)

        # Final output
        out_final = out_attn + out_residual
        out_final = self.down_sampling(out_final)

        return out_final
