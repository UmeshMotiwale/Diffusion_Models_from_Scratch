# Seasons of Code 2026
## Diffusion Models from Scratch: Building Your Own Image Generator

**Duration:** 8 weeks | **Time Commitment:** 8–12 hours/week | **Cohort Size:** 6 mentees
**Prerequisites:** Python proficiency, basic ML (sklearn-level), comfort with NumPy and linear algebra

---

## Project Overview

By the end of this program, each mentee will have built — entirely from scratch — a working diffusion model capable of generating original images, along with a polished interactive demo and a technical blog post explaining their work. No black-box API calls. Every component (UNet, noise scheduler, sampling loop) will be implemented and understood.

### Final Deliverables (Per Mentee)

1. A GitHub repository with a clean, documented codebase
2. A trained diffusion model on a custom dataset of their choice
3. An interactive Gradio demo deployed on Hugging Face Spaces
4. A technical blog post (Medium / personal site) explaining their implementation
5. A 5-minute final presentation video

### What Mentees Will Learn

- PyTorch fluency: tensors, autograd, training loops, GPU usage
- Convolutional architectures and the UNet design pattern
- The mathematics and intuition behind diffusion models (DDPM)
- Modern sampling techniques (DDIM, classifier-free guidance)
- Conditional generation and embedding-based control
- Deployment, demo-building, and technical writing

---

## Cohort Logistics (6 Mentees)

### Communication
- **Discord/Slack server** with channels: `#general`, `#help`, `#showcase`, `#resources`
- **Weekly group call** (1 hour, Saturdays recommended) — concept review + Q&A
- **Office hours** (1 hour midweek) — debugging and 1:1 help
- **Pair-up system:** Pair mentees for weekly code reviews (rotate pairings every 2 weeks)

### Tracking & Accountability
- Each mentee maintains a **public GitHub repo** from Week 1 with weekly commits
- Weekly **progress check-in form** (5 questions, ~5 minutes to fill)
- Mid-program (Week 4) and final (Week 8) **demo days** where each mentee presents

### Compute Resources
- **Primary:** Google Colab free tier (T4 GPU) — sufficient for all weeks
- **Backup:** Kaggle Notebooks (30 hrs/week of P100 free) — recommend for Weeks 6–7
- **Optional:** Lightning.ai free tier, Paperspace Gradient, or local GPU if available
- Datasets must be downsampled to fit Colab constraints (32×32 or 64×64 images)

---

## Week 0: Pre-Program Setup (Optional, Sent 1 Week Before Start)

Send mentees a setup email containing:

- GitHub account creation + repo template link
- Colab + Kaggle account setup
- Hugging Face account creation
- Pre-program reading: 3Blue1Brown's "But what is a neural network?" series
- Pre-program coding: Implement linear regression with gradient descent in NumPy
- Discord/Slack invite

This filters serious mentees and ensures everyone starts at the same baseline.

---

## Week 1: PyTorch Foundations

**Theme:** From sklearn to PyTorch. Same ideas, more flexibility.

### Learning Objectives
- Understand tensors, autograd, and the PyTorch computational graph
- Write a training loop from scratch (no `model.fit()`)
- Train a simple feedforward network on MNIST
- Move computations to GPU

### Resources
- **Reading:** PyTorch official "60 Minute Blitz" tutorial
- **Video:** Andrej Karpathy's "The spelled-out intro to neural networks and backpropagation" (first 1 hour)
- **Reference:** PyTorch documentation for `nn.Module`, `optim`, `DataLoader`

### Coding Deliverable
Build a feedforward neural network that classifies MNIST digits with at least 97% test accuracy. Code must include:
- Custom `Dataset` class
- Manual training loop with logging
- Train/val/test split
- Saving and loading model checkpoints
- A short README.md explaining design choices

### Checkpoint Question
"Walk me through what `loss.backward()` does internally. What gets modified, and where do gradients live?"

### Stretch Goals
- Implement dropout and batch normalization manually
- Compare SGD vs. Adam convergence on the same architecture
- Visualize learned filters in the first layer

### Common Pitfalls
- Forgetting `optimizer.zero_grad()` — gradients accumulate across batches
- Confusion between `.eval()` mode and training mode for dropout/batchnorm
- Not moving both model AND data to the same device (`.to(device)`)

---

## Week 2: Convolutional Networks and the UNet

**Theme:** Build the architecture that powers every diffusion model.

### Learning Objectives
- Understand convolutions, pooling, and feature maps
- Implement skip connections and understand why they matter
- Build a UNet from scratch — encoder, bottleneck, decoder
- Train a UNet on an image-to-image task (denoising)

### Resources
- **Reading:** "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015) — the original paper
- **Video:** Stanford CS231n Lecture 5 (Convolutional Neural Networks)
- **Blog:** "An Introduction to Different Types of Convolutions in Deep Learning" by Paul-Louis Pröve

### Coding Deliverable
Implement a UNet from scratch in PyTorch. Train it as a denoising autoencoder: take MNIST or CIFAR-10 images, add Gaussian noise, train the UNet to reconstruct clean images. Code must include:
- Modular `DoubleConv`, `Down`, and `Up` blocks
- Configurable depth and channel count
- Visualization of denoised outputs every few epochs

### Checkpoint Question
"Why are skip connections in a UNet important? What happens to the gradients and information flow without them?"

### Stretch Goals
- Add attention blocks at the bottleneck
- Compare UNet performance to a vanilla CNN encoder-decoder (no skips)
- Try replacing transposed convolutions with bilinear upsampling

### Common Pitfalls
- Mismatched spatial dimensions when concatenating skip connections (off-by-one with padding)
- Forgetting that the output should have the same number of channels as the input
- Insufficient capacity in the bottleneck for complex datasets

---

## Week 3: The Forward Diffusion Process

**Theme:** Learn how to systematically destroy an image with noise — and why that's useful.

### Learning Objectives
- Understand the forward diffusion process mathematically
- Implement linear and cosine noise schedules
- Visualize the noising process at every timestep
- Build the dataset pipeline that produces (noisy_image, timestep, noise) tuples

### Resources
- **Blog (essential):** "What are Diffusion Models?" by Lilian Weng — read sections 1 and 2
- **Paper:** "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) — focus on Sections 1–3
- **Video:** "Diffusion Models | Paper Explanation | Math Explained" by Outlier (YouTube)

### Coding Deliverable
Implement the forward diffusion process. Code must include:
- A `NoiseScheduler` class supporting linear and cosine schedules
- The closed-form `q(x_t | x_0)` sampling (using the reparameterization trick)
- A visualization script that takes one image and shows it noised at t = 0, 100, 250, 500, 750, 999
- Unit tests verifying that `x_T` is approximately pure Gaussian noise

### Checkpoint Question
"Explain why we can sample `x_t` directly from `x_0` in one step instead of iterating t times. What's the math behind that, and why does it matter for training?"

### Stretch Goals
- Implement and compare quadratic and sigmoid schedules
- Plot the signal-to-noise ratio (SNR) curve for each schedule
- Read the "Improved DDPM" paper and understand why cosine schedules help

### Common Pitfalls
- Confusing `beta`, `alpha`, and `alpha_bar` (cumulative product) — get the notation locked in early
- Numerical instability when computing square roots of `alpha_bar` near `t=T`
- Not normalizing image data to `[-1, 1]` before adding noise

---

## Week 4: Training the Reverse Process

**Theme:** Generate your first images. Likely blurry. Likely magical anyway.

### Learning Objectives
- Understand the reverse process and the noise prediction objective
- Modify the UNet to accept timestep embeddings
- Train a full DDPM on MNIST or Fashion-MNIST
- Implement the iterative sampling loop

### Resources
- **Blog:** "The Annotated Diffusion Model" by Hugging Face (this is gold — read carefully)
- **Paper:** DDPM paper, Sections 3.2 and 4
- **Code reference:** lucidrains' `denoising-diffusion-pytorch` repo (read, don't copy)

### Coding Deliverable
Train a working DDPM on MNIST. Code must include:
- Sinusoidal timestep embeddings injected into UNet blocks
- A training loop that samples random timesteps and predicts noise
- A sampling function that iteratively denoises from pure noise
- Generated samples saved every N epochs as a grid image
- A README with training curves and sample outputs

### Checkpoint Question
"During training we predict noise, but during sampling we use that noise prediction to estimate the slightly-denoised image. Walk me through the equation for one reverse step."

### Stretch Goals
- Train on Fashion-MNIST and compare sample quality
- Implement EMA (exponential moving average) of model weights — this dramatically improves sample quality
- Track FID score across epochs

### Common Pitfalls
- Timestep embedding dimension mismatch with UNet channels
- Sampling loop bugs: wrong sign on noise terms, off-by-one on timestep indexing
- Forgetting to set `model.eval()` and `torch.no_grad()` during sampling
- Insufficient training time — DDPM needs more epochs than typical classification tasks

### Mid-Program Demo Day
At the end of Week 4, host a 90-minute demo session where each mentee shares:
- Their best generated samples
- Their biggest debugging challenge of the past 4 weeks
- One thing they want to improve in the second half

---

## Week 5: Faster Sampling with DDIM

**Theme:** 1000 sampling steps is too many. Let's cut it to 50 with no quality loss.

### Learning Objectives
- Understand the difference between stochastic (DDPM) and deterministic (DDIM) sampling
- Implement DDIM sampling on top of a pre-trained DDPM
- Benchmark sampling speed vs. quality trade-offs

### Resources
- **Paper:** "Denoising Diffusion Implicit Models" (Song et al., 2021) — Sections 1–4
- **Blog:** Lilian Weng's diffusion post, the DDIM section
- **Code reference:** Hugging Face Diffusers library's `DDIMScheduler` source

### Coding Deliverable
Add DDIM sampling to the existing model from Week 4. Code must include:
- A `DDIMScheduler` class with configurable `eta` parameter
- Side-by-side comparison: DDPM at 1000 steps vs. DDIM at 10, 25, 50, 100 steps
- Wall-clock timing for each
- A short markdown report with findings

### Checkpoint Question
"DDIM uses the same trained model as DDPM but generates samples faster. What's the key insight that makes this possible? What does `eta` control?"

### Stretch Goals
- Implement DPM-Solver or DPM-Solver++
- Try interpolating between two noise vectors and visualize the smooth transition (only possible with deterministic sampling)
- Implement image-to-image generation by partially noising an existing image then denoising

### Common Pitfalls
- Mixing up DDIM's deterministic update equation with DDPM's stochastic one
- Wrong timestep subset — DDIM picks evenly spaced timesteps from the original schedule
- Not realizing DDIM with `eta=1` reduces back to DDPM (good sanity check)

---

## Week 6: Conditional Generation

**Theme:** Don't just generate random digits. Generate the digit you want.

### Learning Objectives
- Implement class-conditional generation (label embeddings)
- Understand and implement classifier-free guidance (CFG)
- Build the foundation for text-to-image (using CLIP)

### Resources
- **Paper:** "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
- **Blog:** "Classifier-Free Guidance: From Bayes' Theorem to Conditional Image Generation" by Sander Dieleman
- **Reference:** Stable Diffusion paper sections on conditioning (for context)

### Coding Deliverable
Extend the model to support conditional generation. Code must include:
- Class label embeddings concatenated/added to timestep embeddings
- Random label dropout during training (10–20%) to enable CFG
- CFG sampling with adjustable guidance scale `w`
- Sample grids showing the same noise → different classes
- Sample grids showing the same class with varying `w` (1, 3, 5, 10)

### Checkpoint Question
"Classifier-free guidance interpolates between conditional and unconditional predictions. Why does increasing the guidance scale improve sample quality but reduce diversity?"

### Stretch Goals
- Replace class labels with CLIP text embeddings for true text-to-image (small scale)
- Implement classifier guidance (the older approach) and compare
- Visualize how guidance scale affects the noise prediction direction

### Common Pitfalls
- Forgetting to randomly drop the conditioning during training — without this, CFG won't work
- Using the wrong sign on the CFG formula
- Overly high guidance scales producing oversaturated, artifact-heavy images

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
- **Mentee's own choice** (must be approved by mentor — minimum 500 images)

### Coding Deliverable
A full training run on the custom dataset producing usable samples. Code must include:
- A clean data preprocessing pipeline (resize, normalize, augment)
- A training script with logging (Weights & Biases recommended, free tier)
- At least 100 epochs of training (or until convergence)
- A samples folder with generations at different training milestones
- A `model.pt` checkpoint pushed to Hugging Face Hub

### Checkpoint Question
"What was the hardest decision you made about your dataset and architecture this week? What would you change if you had 10x the compute?"

### Stretch Goals
- Implement data augmentation (random horizontal flips, slight rotations)
- Try a larger image resolution (128×128) and report on training time and quality
- Compare your model to a fine-tuned pretrained Stable Diffusion model on the same dataset

### Common Pitfalls
- Underestimating dataset preprocessing time (often takes longer than expected)
- Training too few epochs and concluding the model "doesn't work"
- Mode collapse from poorly cleaned datasets — quality > quantity
- Not saving intermediate checkpoints — losing progress to Colab disconnects

---

## Week 8: Demo, Documentation, and Showcase

**Theme:** Ship it. Show it. Explain it.

### Learning Objectives
- Build an interactive Gradio demo
- Deploy to Hugging Face Spaces
- Communicate technical work through writing
- Present technical work to an audience

### Resources
- **Tutorial:** Gradio Quickstart
- **Tutorial:** Hugging Face Spaces deployment guide
- **Reading:** "How to write a great technical blog post" — pick any 2–3 examples from distill.pub

### Coding Deliverable

A complete, polished portfolio piece consisting of:

1. **Gradio demo** with at least: a "Generate" button, class/condition selector (if applicable), guidance scale slider, number of sampling steps slider, and seed input for reproducibility
2. **Public Hugging Face Space** with the demo deployed and accessible
3. **Technical blog post** (1500–2500 words) covering: motivation, math intuition (with one or two equations), architecture choices, training process, results gallery, lessons learned and what they'd do differently
4. **Polished GitHub README** with project description, demo GIF, installation instructions, training instructions, and credits/references
5. **Final presentation video** (5 minutes, recorded) walking through the project

### Final Demo Day
Host a 2-hour final showcase where each mentee:
- Presents for 8 minutes
- Takes 5 minutes of Q&A
- Demos their Hugging Face Space live

Invite Seasons of Code organizers, alumni, and friends. Record the session.

### Stretch Goals
- Submit the blog post to a publication (Medium's Towards Data Science, Hacker Noon)
- Tweet a thread with samples and tag the diffusion research community
- Apply the same techniques to a new task in the following months

---

## Mentor's Toolkit

### Recommended Mentor Prep (Before Program Starts)

If you haven't built a diffusion model yourself recently, spend a weekend reproducing the Hugging Face "Annotated Diffusion Model" notebook. Your debugging instincts will be invaluable for mentees in Weeks 3–4.

### Suggested Repo Template

Provide each mentee a starter template containing:

```
diffusion-soc-2026/
├── README.md
├── requirements.txt
├── data/
│   └── (gitignored)
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── scheduler.py
│   ├── train.py
│   └── sample.py
├── notebooks/
│   └── exploration.ipynb
├── scripts/
│   └── download_data.py
├── samples/
│   └── (gitignored)
└── checkpoints/
    └── (gitignored)
```

### Weekly Check-in Form (5 Questions)

1. Did you complete this week's deliverable? (Yes / Partial / No)
2. How many hours did you spend? (number)
3. What was the most confusing concept this week?
4. What's blocking you from progressing? (free text)
5. Rate your confidence with this week's material (1–5)

### Red Flags to Watch For

- A mentee skipping commits for 3+ days — reach out immediately
- Asking only "how do I..." questions and never "why does..." — push them to read papers
- Copying code from public diffusion repos verbatim — redirect to fundamentals
- Spending too long on visual polish before the model works — set hard ordering: working model → demo → polish

### Grading Rubric (If Required)

| Component | Weight |
|---|---|
| Weekly deliverables (Weeks 1–7) | 50% |
| Final demo and Hugging Face Space | 20% |
| Technical blog post | 15% |
| Code quality and documentation | 10% |
| Final presentation | 5% |

---

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

## Appendix A: Sample Schedule for Mentees

A suggested weekly time breakdown for the 8–12 hour commitment:

- **2 hours:** Reading and watching resources
- **4–6 hours:** Coding the deliverable
- **1 hour:** Group call attendance
- **1 hour:** Pair code review
- **1–2 hours:** Debugging, polishing, writing README

---

## Appendix B: Pitfall Recovery Guide

If a mentee is stuck at the end of Week 4 with non-converging loss:
1. Verify data is normalized to `[-1, 1]`
2. Verify timestep embedding is being used in the forward pass
3. Check learning rate (recommended: 1e-4 to 2e-4 for AdamW)
4. Check that loss is computed against true noise, not the image
5. Visualize a single noised sample at various timesteps to verify the scheduler

If a mentee can't deploy to Hugging Face Spaces:
1. Verify model weights are under 5GB (Spaces limit)
2. Use `gradio` (not `streamlit`) for compatibility
3. Test locally first with `gradio app.py`
4. Check the Spaces logs for missing dependencies

---

*Designed for the Seasons of Code 2026 program. Adapt freely to your cohort.*
