import torch
import torch.nn.functional as F

def physics_degrade(img):
    # Poisson noise (photon statistics)
    noisy = torch.poisson(img * 40) / 40

    # Blur (simulating motion / optics)
    kernel = torch.ones(1, 1, 5, 5, device=img.device) / 25
    blurred = F.conv2d(noisy, kernel, padding=2)

    return blurred.clamp(0, 1)
