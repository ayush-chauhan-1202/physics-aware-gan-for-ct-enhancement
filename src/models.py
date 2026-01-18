import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# -----------------
# Generator (UNet-style)
# -----------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            block(1, 32),
            nn.MaxPool2d(2),
            block(32, 64),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            block(32, 32),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# -----------------
# PatchGAN Discriminator
# -----------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
            )

        self.net = nn.Sequential(
            block(2, 32),
            block(32, 64),
            block(64, 128),
            nn.Conv2d(128, 1, 4, padding=1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))
