"""Generate a grid of fashion images from a trained generator checkpoint."""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models import Generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/generated_fashion.png"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}. Run train.py first.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    generator = Generator(checkpoint["latent_dim"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    noise = torch.randn(args.count, checkpoint["latent_dim"], 1, 1, device=device)
    with torch.no_grad():
        images = generator(noise)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = min(8, args.count)
    save_image(images, args.output, nrow=columns, normalize=True, value_range=(-1, 1))
    print(f"Saved {args.count} generated images to {args.output}")


if __name__ == "__main__":
    main()
