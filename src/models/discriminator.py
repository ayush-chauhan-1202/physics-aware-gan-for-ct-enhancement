import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(256, 1, 4, 1, 1)),
        )

    def forward(self, lq, hq):
        x = torch.cat([lq, hq], dim=1)
        return self.model(x)
