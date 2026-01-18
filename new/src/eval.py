import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset import GoodDataset
from models import Generator

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def evaluate():
    device = get_device()
    model = Generator().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/G_49.pt", map_location=device))
    model.eval()

    ds = GoodDataset("data/train/good")
    loader = DataLoader(ds, batch_size=1)

    psnr_scores, ssim_scores = [], []

    with torch.no_grad():
        for img in loader:
            img = img.to(device)
            pred = model(img)

            gt = img.squeeze().cpu().numpy()
            out = pred.squeeze().cpu().numpy()

            psnr_scores.append(peak_signal_noise_ratio(gt, out))
            ssim_scores.append(structural_similarity(gt, out, data_range=1.0))

    print("Avg PSNR:", sum(psnr_scores)/len(psnr_scores))
    print("Avg SSIM:", sum(ssim_scores)/len(ssim_scores))

if __name__ == "__main__":
    evaluate()
