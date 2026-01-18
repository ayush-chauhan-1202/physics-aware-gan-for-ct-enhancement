# src/train.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from dataset import PairedDataset
from losses import generator_loss, discriminator_loss
from torchvision.utils import save_image

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train(config_path="configs/baseline.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print("Using device:", device)

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["training"]["lr"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["training"]["lr"], betas=(0.5, 0.999))

    train_ds = PairedDataset(cfg["paths"]["train_input"], cfg["paths"]["train_target"])
    train_loader = DataLoader(train_ds,
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=True)

    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)
    os.makedirs(cfg["paths"]["samples"], exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # -----------------
            # Train D
            # -----------------
            fake = G(x).detach()
            d_real = D(x, y)
            d_fake = D(x, fake)

            loss_d = discriminator_loss(d_real, d_fake)
            opt_D.zero_grad()
            loss_d.backward()
            opt_D.step()

            # -----------------
            # Train G
            # -----------------
            fake = G(x)
            d_fake = D(x, fake)

            loss_g, components = generator_loss(
                d_fake, fake, y, cfg["loss_weights"]
            )

            opt_G.zero_grad()
            loss_g.backward()
            opt_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}] [Batch {i}] "
                      f"G: {loss_g.item():.4f} | D: {loss_d.item():.4f}")

        # Save samples + checkpoints
        save_image(fake[:4], f"{cfg['paths']['samples']}/epoch_{epoch}.png", normalize=True)
        torch.save(G.state_dict(), f"{cfg['paths']['checkpoints']}/G_{epoch}.pt")

if __name__ == "__main__":
    train()
