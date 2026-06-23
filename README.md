# Diffusion Models from Scratch: Building Your Own Image Generator
## Seasons of Code

**Duration:** 8 weeks 
**Prerequisites:** Python proficiency, basic ML (sklearn-level), comfort with NumPy and linear algebra

---

## Project Overview

By the end of this program, we will have built — entirely from scratch — a working diffusion model capable of generating original images, along with a polished interactive demo and a technical blog post explaining their work. No black-box API calls. Every component (UNet, noise scheduler, sampling loop) will be implemented and understood.

### What You Will Learn

- PyTorch fluency: tensors, autograd, training loops, GPU usage
- Convolutional architectures and the UNet design pattern
- The mathematics and intuition behind diffusion models (DDPM)
- Modern sampling techniques (DDIM, classifier-free guidance)
- Conditional generation and embedding-based control
- Deployment, demo-building, and technical writing

---

### Compute Resources
- **Primary:** Google Colab free tier (T4 GPU) — sufficient for all weeks
- **Backup:** Kaggle Notebooks (30 hrs/week of P100 free) — recommend for Weeks 6–7
- **Optional:** Lightning.ai free tier, Paperspace Gradient, or local GPU if available
- Datasets must be downsampled to fit Colab constraints (32×32 or 64×64 images)

---

## Week 0: Pre-Program Setup (Optional, Sent 1 Week Before Start)

- GitHub account creation + fork this repo
- Colab + Kaggle account setup
- Hugging Face account creation
- Pre-program reading: 3Blue1Brown's "But what is a neural network?" series(optional)
- Pre-program coding: Implement linear regression with gradient descent in NumPy

---

## Week 1: PyTorch Foundations

**Theme:** From sklearn to PyTorch. Same ideas, more flexibility.

### Learning Objectives
- Understand tensors, autograd, and the PyTorch computational graph
- Write a training loop from scratch (no `model.fit()`)
- Train a simple feedforward network on MNIST
- Move computations to GPU
  
---

## Week 2: Convolutional Networks and the UNet

**Theme:** Build the architecture that powers every diffusion model.

### Learning Objectives
- Understand convolutions, pooling, and feature maps
- Implement skip connections and understand why they matter
- Build a UNet from scratch — encoder, bottleneck, decoder
- Train a UNet on an image-to-image task (denoising)

---

## Week 3: The Forward Diffusion Process

**Theme:** Learn how to systematically destroy an image with noise — and why that's useful.

### Learning Objectives
- Understand the forward diffusion process mathematically
- Implement linear and cosine noise schedules
- Visualize the noising process at every timestep
- Build the dataset pipeline that produces (noisy_image, timestep, noise) tuples

---

## Week 4: Training the Reverse Process

**Theme:** Generate your first images. Likely blurry. Likely magical anyway.

### Learning Objectives
- Understand the reverse process and the noise prediction objective
- Modify the UNet to accept timestep embeddings
- Train a full DDPM on MNIST or Fashion-MNIST
- Implement the iterative sampling loop

---

## Week 5: Faster Sampling with DDIM

**Theme:** 1000 sampling steps is too many. Let's cut it to 50 with no quality loss.

### Learning Objectives
- Understand the difference between stochastic (DDPM) and deterministic (DDIM) sampling
- Implement DDIM sampling on top of a pre-trained DDPM
- Benchmark sampling speed vs. quality trade-offs
  
---

## Week 6: Conditional Generation

**Theme:** Don't just generate random digits. Generate the digit you want.

### Learning Objectives
- Implement class-conditional generation (label embeddings)
- Understand and implement classifier-free guidance (CFG)
- Build the foundation for text-to-image (using CLIP)

---

## Week 7: Custom Dataset and Final Training Run

**Theme:** Now make it yours.

### Learning Objectives
- Curate and preprocess a custom dataset
- Make practical training decisions under compute constraints
- Apply all techniques from previous weeks to a new domain

### Suggested Datasets (Pick One)
- **Pokemon sprites** (~800 images, 64×64) — small enough to overfit creatively
- **Anime faces** (Danbooru subset, downsampled to 64×64)
- **CelebA faces** (downsampled, 64×64)
- **Quick, Draw! sketches** (specific category, 28×28)
- **Logo dataset** (LLD-icon, 32×32)
- **Indian art / textile patterns** (scraped from open sources — culturally interesting)

---

## Week 8: Demo, Documentation, and Showcase

**Theme:** Ship it. Show it. Explain it.

### Learning Objectives
- Build an interactive Gradio demo
- Deploy to Hugging Face Spaces
- Communicate technical work through writing
- Present technical work to an audience

### Coding Deliverable(Any 2 out of which Gradio Demo is compulsory)

A complete, polished portfolio piece consisting of:

1. **Gradio demo** with at least: a "Generate" button, class/condition selector (if applicable), guidance scale slider, number of sampling steps slider, and seed input for reproducibility
2. **Public Hugging Face Space** with the demo deployed and accessible
3. **Technical blog post** (1500–2500 words) covering: motivation, math intuition (with one or two equations), architecture choices, training process, results gallery, lessons learned and what they'd do differently
4. **Final presentation** (12-18 slides) walking through the project

### Stretch Goals
- Submit the blog post to a publication (Medium's Towards Data Science, Hacker Noon)
- Tweet a thread with samples and tag the diffusion research community

## Reference Reading List (Curated)

**Beginner-friendly:**
- "What are Diffusion Models?" — Lilian Weng (must-read, blog)
- "The Annotated Diffusion Model" — Hugging Face (code-along)
- "Generative Modeling by Estimating Gradients of the Data Distribution" — Yang Song's blog

**Foundational papers (in suggested reading order):**
1. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (Sohl-Dickstein et al., 2015) — the OG
2. "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) — DDPM
3. "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021) — practical tricks
4. "Denoising Diffusion Implicit Models" (Song et al., 2021) — DDIM
5. "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022) — CFG
6. "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022) — Stable Diffusion

**Advanced / stretch goals:**
- "Elucidating the Design Space of Diffusion-Based Generative Models" (Karras et al., 2022) — EDM
- "Score-Based Generative Modeling through Stochastic Differential Equations" (Song et al., 2021) — unifying view

**Code references (read, don't copy):**
- lucidrains/denoising-diffusion-pytorch
- huggingface/diffusers (production-grade reference)
- openai/improved-diffusion

---
