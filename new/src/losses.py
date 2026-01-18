import torch
import torch.nn.functional as F

def gan_loss(pred, real=True):
    targets = torch.ones_like(pred) if real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, targets)

def ssim_loss(x, y):
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov = ((x - mu_x)*(y - mu_y)).mean()
    return 1 - (2*mu_x*mu_y + 1e-4)*(2*cov + 1e-4)/((mu_x**2 + mu_y**2 + 1e-4)*(var_x + var_y + 1e-4))

def frequency_loss(x, y):
    return torch.mean(torch.abs(torch.fft.fft2(x) - torch.fft.fft2(y)))

def generator_loss(d_fake, fake, real, w):
    l1 = F.l1_loss(fake, real)
    ssim = ssim_loss(fake, real)
    freq = frequency_loss(fake, real)
    gan = gan_loss(d_fake, True)

    total = w["gan"]*gan + w["l1"]*l1 + w["ssim"]*ssim + w["freq"]*freq
    return total

def discriminator_loss(d_real, d_fake):
    return 0.5*(gan_loss(d_real,True)+gan_loss(d_fake,False))
