import yaml, torch, os
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from dataset import GoodDataset
from losses import *
from physics import physics_degrade
from utils import seed_all
from torchvision.utils import save_image

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def train(cfg_path="configs/baseline.yaml"):
    seed_all()
    cfg = yaml.safe_load(open(cfg_path))
    device = get_device()
    use_amp = device=="cuda"

    G, D = Generator().to(device), Discriminator().to(device)
    optG = torch.optim.Adam(G.parameters(), lr=cfg["lr"])
    optD = torch.optim.Adam(D.parameters(), lr=cfg["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ds = GoodDataset(cfg["data"])
    loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=True)

    os.makedirs("outputs/samples", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    for epoch in range(cfg["epochs"]):
        for img in loader:
            img = img.to(device)
            degraded = physics_degrade(img)

            # ---- D ----
            fake = G(degraded).detach()
            loss_d = discriminator_loss(D(degraded,img), D(degraded,fake))
            optD.zero_grad(); loss_d.backward(); optD.step()

            # ---- G ----
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = G(degraded)
                loss_g = generator_loss(
                                        D(degraded, fake),
                                        fake,
                                        img,
                                        cfg["loss_weights"],
                                        device
                                    )

            optG.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.step(optG)
            scaler.update()

        save_image(fake[:4], f"outputs/samples/{epoch}.png")
        torch.save(G.state_dict(), f"outputs/checkpoints/G_{epoch}.pt")
        print(f"Epoch {epoch}: G={loss_g.item():.4f} D={loss_d.item():.4f}")

if __name__=="__main__":
    train()
