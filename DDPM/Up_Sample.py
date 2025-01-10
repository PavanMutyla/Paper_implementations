import torch 
import torch.nn as nn
from Linear_scheduler import LinearScehduler
from Time_Embed import TimeEmbed, Swish

class UpBlock(nn.Module):
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
        self.attention_norm = nn.LayerNorm(out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        # Linear projection to match input size for skip connection
        self.linear_layer_input = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=(1, 1))

        # Linear projection for time embedding to match out_channels
        self.time_proj = nn.Linear(out_channels, out_channels)

        # Upsampling layer
        self.up_sample = nn.ConvTranspose2d(in_channels//2, out_channels//2, kernel_size = 4, padding = 1, stride = 2)

    def forward(self, x, down_out, t):
        x = self.up_sample(x)
        print(x.shape)
        # Concatenate skip connection
        x = torch.concat([x, down_out], dim=1)

        # Residual block
        block1 = self.Block1(x)

        # Project time embedding and add
        t_proj = self.time_proj(t)
        t_proj = t_proj[:, :, None, None]  # Broadcast over spatial dimensions
        block_time_sum = block1 + t_proj

        block2 = self.Block2(block_time_sum)
        skip_connection = self.linear_layer_input(x)
        out_residual = block2 + skip_connection

        # Attention block
        batch, channel, h, w = out_residual.shape
        attn_input = out_residual.reshape(batch, h * w, channel)  # (B, H*W, C)
        out_attn, _ = self.attention(attn_input, attn_input, attn_input)
        out_attn = out_attn.reshape(batch, channel, h, w)

        # Final output
        out_final = out_attn + out_residual
        return out_final
