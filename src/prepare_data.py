import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from src.degradation import degrade

# Paths
RAW_DIR = Path("data/raw")
OUT_LQ = Path("data/processed/lq")
OUT_HQ = Path("data/processed/hq")

OUT_LQ.mkdir(parents=True, exist_ok=True)
OUT_HQ.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

for img_path in tqdm(list(RAW_DIR.glob("*.png"))):
    img = Image.open(img_path).convert("L")
    x = transform(img).unsqueeze(0)  # [1,1,H,W]

    with torch.no_grad():
        lq = degrade(x)

    # Save
    name = img_path.name
    transforms.ToPILImage()(x.squeeze()).save(OUT_HQ / name)
    transforms.ToPILImage()(lq.squeeze()).save(OUT_LQ / name)

print("Data preparation complete.")
