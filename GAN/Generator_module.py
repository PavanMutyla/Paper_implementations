import torch 
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, noise_dim, channels , features_gen):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(noise_dim, features_gen*16, 4, 1, 0), # 1024
            self._block(features_gen*16, features_gen*8, 4, 2, 1), # 512
            self._block(features_gen*8, features_gen*4, 4, 2, 1), #256
            self._block(features_gen*4, features_gen*2, 4, 2, 1), # 128
            self._block(features_gen*2, features_gen, 4, 2, 1), #64
            nn.ConvTranspose2d(
                features_gen, channels, 4, 2, 1 # 64x64x3
            ),
            nn.Tanh()
            
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.gen(x)
        
