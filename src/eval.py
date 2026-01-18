# src/eval.py
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate(model, loader, device):
    model.eval()
    psnr_scores, ssim_scores = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            pred_np = pred.squeeze().cpu().numpy()
            y_np = y.squeeze().cpu().numpy()

            psnr_scores.append(peak_signal_noise_ratio(y_np, pred_np))
            ssim_scores.append(structural_similarity(y_np, pred_np, data_range=1.0))

    return sum(psnr_scores)/len(psnr_scores), sum(ssim_scores)/len(ssim_scores)
