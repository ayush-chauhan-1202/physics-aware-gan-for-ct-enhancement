from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class GoodDataset(Dataset):
    def __init__(self, root):
        self.paths = [os.path.join(root, f) for f in os.listdir(root)]
        self.t = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = self.t(Image.open(self.paths[i]))
        return img
