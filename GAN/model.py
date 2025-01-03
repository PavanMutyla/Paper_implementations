import torch 
import torch.nn as nn
from GAN.Generator_module import Generator
from GAN.Discriminator_module import Discriminator
def init_weights(model):
    for i in model.modules():
        if isinstance(i,(nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
            nn.init.normal_(i.weight.data, mean = 0.0, std = 0.02)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

init_weights(gen)
init_weights(disc)

        
