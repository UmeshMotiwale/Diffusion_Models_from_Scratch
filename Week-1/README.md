# Week 1 — PyTorch Foundations

> **Theme:** From sklearn to PyTorch. Same ideas, more flexibility.
> **Time commitment:** 8–12 hours
> **Deliverable due:** Friday End of day

---

## What You'll Build

A feedforward neural network that classifies MNIST digits with **≥97% test accuracy**, written entirely from scratch in PyTorch — no `model.fit()` shortcuts.

## Why This Week Matters

You can't debug a diffusion model in Week 4 if you can't debug a basic classifier in Week 1. Every concept introduced here — tensors, autograd, custom training loops, GPU usage — is a foundation you'll use every week after.

## Deliverable Checklist

- [ ] Custom `Dataset` class (don't use `torchvision.datasets.MNIST` directly)
- [ ] Manual training loop with epoch-level logging
- [ ] Train/validation/test split (not just train/test)
- [ ] Model checkpoint saving and loading
- [ ] README explaining your design choices
- [ ] ≥97% test accuracy
- [ ] Code pushed to your project GitHub repo

## Folder Structure

```
week1/
├── README.md          (this file)
├── train.py           (training script)
├── model.py           (your network architecture)
├── dataset.py         (custom Dataset class)
├── utils.py           (helpers: logging, checkpoints)
└── results/
    ├── loss_curve.png
    └── test_accuracy.txt
```

## Self-Check Questions

By Sunday, you should be able to answer these without Googling:

1. What does `loss.backward()` actually do internally?
2. Why does `optimizer.zero_grad()` exist?
3. What's the difference between `.eval()` and `.train()` mode?
4. Where do gradients live after `loss.backward()` runs?

## Common Pitfalls

- Forgetting `optimizer.zero_grad()` → gradients accumulate across batches
- Not moving both model AND data to the same device
- Confusion between `.eval()` and training mode for dropout/batchnorm

## Resources

See the resources pdf in the same directory as this

**Quick links:**
- [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Karpathy: Neural Networks Zero to Hero (Lecture 1)](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Daniel Bourke: Learn PyTorch](https://www.learnpytorch.io/)

# Week 1 — MNIST Classification

**Mentee:** Umesh Motiwale
**SoC Track:** Diffusion Models from Scratch — SoC 2026

## Final results
- **Test accuracy:** 0.9824
- **Best validation accuracy:** 0.9812 at epoch 10
- **Final train loss:** ~0.0238

## Design choices
- **Architecture:** Simple MLP — 784 → 256 → 256 → 10, with ReLU and dropout (p=0.2) after each hidden layer
- **Optimizer:** Adam, lr=1e-3
- **Batch size:** 128
- **Epochs trained:** 10
- **Validation split:** 10% of training data, seed=42

## What I learned
Writing the training loop by hand made it click that "training" is really just five repeated steps which clear gradients, forward pass, compute loss, backward pass, update weights instead of a magic `model.fit()` call. I also saw firsthand why `zero_grad()` matters forgetting it lets gradients pile up across batches and breaks training. The train/val gap in my curves also made overfitting feel concrete instead of theoretical — even a small MLP starts memorizing train-specific noise pretty early.

## What I'd do differently
I would add weight decay or bump up dropout to close the gap between train and val accuracy a bit and try swapping the MLP for a small CNN to see how much accuracy that buys for free.

## How to reproduce
1. Open `week1_mnist.ipynb` in Colab with a T4 GPU runtime.
2. Run all cells top to bottom.
3. Checkpoint will be saved to `best_model.pt`.
