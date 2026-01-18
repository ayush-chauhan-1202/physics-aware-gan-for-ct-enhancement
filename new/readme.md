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
