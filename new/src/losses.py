import torch
import torch.nn.functional as F
import lpips

# Global LPIPS model (loaded once)
_lpips_model = None

def get_lpips(device):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
    return _lpips_model


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


def lpips_loss(fake, real, device):
    model = get_lpips(device)

    # LPIPS expects 3-channel [-1,1]
    fake = fake.repeat(1,3,1,1) * 2 - 1
    real = real.repeat(1,3,1,1) * 2 - 1

    return model(fake, real).mean()


def generator_loss(d_fake, fake, real, w, device):
    l1 = F.l1_loss(fake, real)
    ssim = ssim_loss(fake, real)
    freq = frequency_loss(fake, real)
    gan = gan_loss(d_fake, True)
    lp = lpips_loss(fake, real, device)

    total = (
        w["gan"] * gan +
        w["l1"] * l1 +
        w["ssim"] * ssim +
        w["freq"] * freq +
        w["lpips"] * lp
    )

    return total


def discriminator_loss(d_real, d_fake):
    return 0.5 * (gan_loss(d_real, True) + gan_loss(d_fake, False))
