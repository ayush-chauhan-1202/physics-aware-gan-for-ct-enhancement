# src/losses.py
import torch
import torch.nn.functional as F

# -----------------------
# Basic losses
# -----------------------

def gan_loss(pred, target_is_real=True):
    targets = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, targets)

def l1_loss(fake, real):
    return F.l1_loss(fake, real)

# -----------------------
# SSIM loss
# -----------------------

def ssim_loss(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = x.mean(dim=(-2, -1), keepdim=True)
    mu_y = y.mean(dim=(-2, -1), keepdim=True)

    sigma_x = ((x - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(-2, -1), keepdim=True)

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

    return 1 - ssim.mean()

# -----------------------
# Frequency loss
# -----------------------

def frequency_loss(fake, real):
    fake_fft = torch.fft.fft2(fake)
    real_fft = torch.fft.fft2(real)
    return torch.mean(torch.abs(fake_fft - real_fft))

# -----------------------
# Composite loss
# -----------------------

def generator_loss(d_fake, fake, real, weights):
    loss_gan = gan_loss(d_fake, True)
    loss_l1 = l1_loss(fake, real)
    loss_ssim = ssim_loss(fake, real)
    loss_freq = frequency_loss(fake, real)

    total = (
        weights["gan"] * loss_gan +
        weights["l1"] * loss_l1 +
        weights["ssim"] * loss_ssim +
        weights["freq"] * loss_freq
    )

    return total, {
        "gan": loss_gan.item(),
        "l1": loss_l1.item(),
        "ssim": loss_ssim.item(),
        "freq": loss_freq.item()
    }

def discriminator_loss(d_real, d_fake):
    loss_real = gan_loss(d_real, True)
    loss_fake = gan_loss(d_fake, False)
    return 0.5 * (loss_real + loss_fake)
