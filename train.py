"""Train a DCGAN to generate 28 x 28 Fashion-MNIST images."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import Discriminator, Generator, initialize_weights


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # scale pixels to [-1, 1]
    ])
    dataset = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")

    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        for batch_index, (real_images, _) in enumerate(loader, start=1):
            real_images = real_images.to(device)
            real_labels = torch.ones(real_images.size(0), device=device)
            fake_labels = torch.zeros(real_images.size(0), device=device)

            # Train discriminator on real images and a detached batch of fake images.
            optimizer_d.zero_grad(set_to_none=True)
            real_loss = loss_fn(discriminator(real_images), real_labels)
            noise = torch.randn(real_images.size(0), args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_loss = loss_fn(discriminator(fake_images.detach()), fake_labels)
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            optimizer_d.step()

            # Train generator to make the discriminator predict "real".
            optimizer_g.zero_grad(set_to_none=True)
            generator_loss = loss_fn(discriminator(fake_images), real_labels)
            generator_loss.backward()
            optimizer_g.step()

            if batch_index % 100 == 0 or batch_index == len(loader):
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | batch {batch_index:03d}/{len(loader)} | "
                    f"D loss: {discriminator_loss.item():.4f} | G loss: {generator_loss.item():.4f}"
                )

        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise)
        save_image(samples, args.output_dir / f"epoch_{epoch:03d}.png", nrow=8, normalize=True, value_range=(-1, 1))
        torch.save(
            {"epoch": epoch, "latent_dim": args.latent_dim, "generator": generator.state_dict(), "discriminator": discriminator.state_dict()},
            args.checkpoint_dir / "latest.pt",
        )

    print(f"Finished. Samples are in {args.output_dir} and the model is in {args.checkpoint_dir / 'latest.pt'}.")


if __name__ == "__main__":
    main()
