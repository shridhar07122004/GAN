"""Minimal web interface for exploring images made by the Fashion-MNIST GAN."""

from datetime import datetime
from pathlib import Path

import torch
from flask import Flask, flash, redirect, render_template, send_from_directory, url_for
from torchvision.utils import save_image

from models import Generator

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
app = Flask(__name__)
app.config["SECRET_KEY"] = "replace-this-local-development-key"


def gallery_images():
    """Return generated PNGs newest first, excluding intermediate training previews."""
    OUTPUTS.mkdir(exist_ok=True)
    images = [item for item in OUTPUTS.glob("design_*.png")]
    return sorted(images, key=lambda item: item.stat().st_mtime, reverse=True)


@app.route("/")
def home():
    previews = sorted(OUTPUTS.glob("epoch_*.png"), key=lambda item: item.name, reverse=True)
    latest_preview = previews[0].name if previews else None
    return render_template(
        "index.html",
        model_ready=CHECKPOINT.is_file(),
        latest_preview=latest_preview,
        design_count=len(gallery_images()),
    )


@app.post("/generate")
def generate():
    if not CHECKPOINT.is_file():
        flash("Train the model first. Run: python train.py --epochs 5", "error")
        return redirect(url_for("home"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    generator = Generator(checkpoint["latent_dim"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(16, checkpoint["latent_dim"], 1, 1, device=device)
        images = generator(noise)

    OUTPUTS.mkdir(exist_ok=True)
    filename = f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_image(images, OUTPUTS / filename, nrow=4, normalize=True, value_range=(-1, 1))
    flash("A new four-by-four design sheet is ready.", "success")
    return redirect(url_for("gallery"))


@app.route("/gallery")
def gallery():
    images = [image.name for image in gallery_images()]
    return render_template("gallery.html", images=images)


@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUTS, filename)


if __name__ == "__main__":
    app.run(debug=True)
