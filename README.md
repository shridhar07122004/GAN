# Fashion Image Generator (DCGAN)

This project trains a **Deep Convolutional GAN (DCGAN)** on the Fashion-MNIST dataset and presents its work in a clean local Flask web application. The generator learns to create new 28×28 grayscale images resembling clothing items, while the discriminator learns to distinguish generated images from real training samples.

## Project layout

| File | Purpose |
| --- | --- |
| `models.py` | Generator, discriminator, and weight initialization |
| `train.py` | Downloads Fashion-MNIST and trains the GAN |
| `generate.py` | Creates a new image grid from the saved generator |
| `app.py` | Runs the simple Flask web interface |
| `templates/` and `static/` | Plain HTML and CSS for the interface |
| `outputs/` | Training previews and generated image grids |
| `checkpoints/` | Saved model checkpoint |

## Setup

Use Python 3.10 or newer. From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands in Command Prompt (`.venv\Scripts\activate.bat`) or permit local scripts for the current session with `Set-ExecutionPolicy -Scope Process Bypass`.

## Train

```powershell
python train.py --epochs 30 --batch-size 128
```

The first run downloads Fashion-MNIST into `data/`. Each epoch writes a fixed-noise preview to `outputs/epoch_XXX.png`, so you can watch quality improve. The latest model is saved as `checkpoints/latest.pt`.

For a quick smoke test:

```powershell
python train.py --epochs 1 --batch-size 64
```

## Generate new fashion images

After training:

```powershell
python generate.py --count 64 --output outputs/generated_fashion.png
```

## Run the project website

First train the model for at least a few epochs. Then start the local site:

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Select **Generate 16 designs** to create and save a fresh design sheet. The gallery keeps the generated sheets in `outputs/`.

## Suggested report talking points

1. Explain adversarial training: the generator tries to fool the discriminator, and the discriminator tries to detect fake images.
2. Include image grids from early, middle, and final epochs.
3. Discuss training stability: GAN losses do not necessarily decrease smoothly, so image quality is the main metric here.
4. Note limitations: Fashion-MNIST is low-resolution and the unconditional model cannot select a requested clothing class.
5. Extension idea: implement a conditional GAN (cGAN) so users can generate a chosen class, such as shoes or shirts.
