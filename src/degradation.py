import torch
import torch.nn.functional as F
import numpy as np


def add_gaussian_noise(x, sigma=0.05):
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def add_poisson_noise(x, scale=30.0):
    # Simulate photon noise
    noisy = torch.poisson(x * scale) / scale
    return torch.clamp(noisy, 0.0, 1.0)


def apply_blur(x, kernel_size=5):
    # Simple average blur (can be replaced with PSF later)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device)
    kernel = kernel / kernel.numel()
    return F.conv2d(x, kernel)


def downsample_upsample(x, scale_factor=2):
    h, w = x.shape[-2:]
    x_low = F.interpolate(x, size=(h // scale_factor, w // scale_factor), mode="bilinear")
    x_up = F.interpolate(x_low, size=(h, w), mode="bilinear")
    return x_up


def degrade(x):
    """Full degradation pipeline."""
    x = add_poisson_noise(x, scale=40.0)
    x = add_gaussian_noise(x, sigma=0.03)
    x = apply_blur(x, kernel_size=5)
    x = downsample_upsample(x, scale_factor=2)
    return x
