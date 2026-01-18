import torch
import numpy as np
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
from dataset import GoodDataset
from models import Generator
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

# -------------------------
# Feature extractor for FID
# -------------------------
class Inception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(pretrained=True, aux_logits=False)
        self.features = torch.nn.Sequential(*list(model.children())[:-1])
        self.eval()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(299,299), mode="bilinear")
        x = x.repeat(1,3,1,1)
        return self.features(x).squeeze(-1).squeeze(-1)


def compute_fid(real_feats, fake_feats):
    mu1, mu2 = real_feats.mean(0), fake_feats.mean(0)
    sigma1, sigma2 = np.cov(real_feats.T), np.cov(fake_feats.T)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def evaluate():
    device = get_device()

    G = Generator().to(device)
    G.load_state_dict(torch.load("outputs/checkpoints/G_49.pt", map_location=device))
    G.eval()

    ds = GoodDataset("data/train/good")
    loader = DataLoader(ds, batch_size=8)

    inception = Inception().to(device)

    real_feats, fake_feats = [], []

    with torch.no_grad():
        for img in tqdm(loader):
            img = img.to(device)
            fake = G(img)

            real_f = inception(img).cpu().numpy()
            fake_f = inception(fake).cpu().numpy()

            real_feats.append(real_f)
            fake_feats.append(fake_f)

    real_feats = np.concatenate(real_feats)
    fake_feats = np.concatenate(fake_feats)

    fid = compute_fid(real_feats, fake_feats)
    print("FID:", fid)


if __name__ == "__main__":
    evaluate()
