import argparse, torch
from torchvision import transforms
from PIL import Image
from models import Generator
from torchvision.utils import save_image

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()

    device = get_device()

    model = Generator().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    img = transform(Image.open(args.input)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)

    save_image(out, args.output)
    print("Saved to:", args.output)

if __name__ == "__main__":
    main()
