"""Neural-network components for a DCGAN trained on Fashion-MNIST."""

import torch.nn as nn


class Generator(nn.Module):
    """Maps a latent noise vector to a 28 x 28 grayscale fashion image."""

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):
    """Estimates whether a 28 x 28 image is real or generated."""

    def __init__(self, feature_maps: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # The preceding layers reduce 28×28 to 4×4; this produces one logit per image.
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image):
        # BCEWithLogitsLoss expects raw logits, so sigmoid is deliberately omitted.
        return self.network(image).flatten()


def initialize_weights(module: nn.Module) -> None:
    """Apply the initialization convention used in the DCGAN paper."""
    class_name = module.__class__.__name__
    if "Conv" in class_name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in class_name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
