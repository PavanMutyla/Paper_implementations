import torch 
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, channels, features_disc):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels, features_disc, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(features_disc, features_disc*2, 4, 2, 1),
            self._block(features_disc*2, features_disc*4, 4, 2, 1),
            self._block(features_disc*4, features_disc*8, 4, 2, 1),
            self._block(features_disc*8, 1, 4, 2, 0),
            nn.Sigmoid()
            
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.disc(x)
