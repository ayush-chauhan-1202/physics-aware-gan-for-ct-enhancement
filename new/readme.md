# Physics-Informed GAN for Industrial Image Restoration (NDE)

A research-grade, end-to-end deep learning pipeline for **industrial Non-Destructive Evaluation (NDE) image enhancement** using a physics-informed Generative Adversarial Network (GAN).

This project demonstrates how domain knowledge (physics of acquisition) can be integrated with modern deep learning techniques to build realistic, stable, and reproducible image restoration models.

---

## Motivation

Industrial inspection modalities such as:

- X-ray radiography
- X-ray CT
- Ultrasound imaging
- Thermography
- Eddy current inspection  

often suffer from acquisition artifacts including:
- Photon noise (Poisson statistics)
- Blur due to optics or motion
- Low signal-to-noise ratio
- Limited contrast for subtle defects

Rather than using purely synthetic noise, this project incorporates a **physics-inspired degradation model** and trains a GAN to learn restoration in a realistic industrial setting.

---

## Key Features

- UNet-style Generator  
- PatchGAN Discriminator  
- Spectral Normalization for GAN stability  
- Physics-informed image degradation  
- Composite loss (GAN + L1 + SSIM + Frequency loss)  
- CUDA acceleration + Apple Silicon (MPS) compatibility  
- Automatic Mixed Precision (AMP) on CUDA  
- Config-driven experiments  
- Reproducible training  
- Clean modular research codebase  

---

## Architecture Overview

**Pipeline:**
```
Clean image → Physics degradation → Generator → Restored image
↓
Discriminator (PatchGAN)
```

The discriminator operates on local patches, forcing the generator to learn high-frequency texture fidelity — critical for industrial defect morphology.


## Project Structure

```
nde-gan-pipeline/
│
├── src/
│   ├── models.py        # Generator + PatchGAN Discriminator
│   ├── dataset.py       # Dataset loader
│   ├── physics.py       # Physics-based degradation model
│   ├── losses.py        # Composite losses
│   ├── train.py         # Training pipeline
│   ├── eval.py          # Evaluation (PSNR, SSIM)
│   ├── inference.py     # Run model on new images
│   └── utils.py         # Reproducibility utils
│
├── configs/
│   ├── baseline.yaml
│   └── advanced.yaml
│
├── data/
│   └── train/good/
│
├── outputs/
│   ├── samples/
│   └── checkpoints/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/nde-gan-pipeline.git
cd nde-gan-pipeline
pip install -r requirements.txt
```

Supports:
	•	CUDA (NVIDIA GPU)
	•	Apple Silicon (MPS)
	•	CPU fallback

## Training
Put grayscale images inside:
```
data/train/good/
```

Run:
```
pythn src/train.py
```

Samples saved to:
```
outputs/samples/
```

Checkpoints saved to:
```
outputs/checkpoints/
```

## Evaluation
```
python src/eval.py
```

Metrics:
- PSNR (pixel fidelity)
- SSIM (structural fidelity)


## Inference on New images
```
python src/inference.py --input path/to/image.png --checkpoint outputs/checkpoints/G_49.pt
```

Results (examples)
Metric        Value
--------------------
PSNR          ~28-32 dB
SSIM          ~0.85-0.92

(varies with dataset and degradation severity.)



