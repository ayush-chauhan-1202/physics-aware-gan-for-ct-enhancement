import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_norm=True):
        super().__init__()
        if down:
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        else:
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_c))

        layers.append(nn.LeakyReLU(0.2) if down else nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = UNetBlock(1, 64, down=True, use_norm=False)
        self.d2 = UNetBlock(64, 128)
        self.d3 = UNetBlock(128, 256)

        self.u1 = UNetBlock(256, 128, down=False)
        self.u2 = UNetBlock(256, 64, down=False)
        self.final = nn.ConvTranspose2d(128, 1, 4, 2, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)

        u1 = self.u1(d3)
        u2 = self.u2(torch.cat([u1, d2], dim=1))
        out = self.final(torch.cat([u2, d1], dim=1))

        return torch.sigmoid(out)
