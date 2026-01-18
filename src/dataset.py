import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, root_lq, root_hq, image_size=256, augment=True):
        self.root_lq = Path(root_lq)
        self.root_hq = Path(root_hq)
        self.files = sorted([p.name for p in self.root_lq.glob("*.png")])
        self.augment = augment

        self.base_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        lq = Image.open(self.root_lq / name).convert("L")
        hq = Image.open(self.root_hq / name).convert("L")

        lq = self.base_tf(lq)
        hq = self.base_tf(hq)

        if self.augment:
            seed = torch.randint(0, 10_000, (1,)).item()
            torch.manual_seed(seed)
            lq = self.aug_tf(lq)
            torch.manual_seed(seed)
            hq = self.aug_tf(hq)

        return lq, hq
