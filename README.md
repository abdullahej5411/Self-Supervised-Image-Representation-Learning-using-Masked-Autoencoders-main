# Masked Autoencoder (MAE) — Self-Supervised Image Representation Learning

## GenAI Assignment 02 | Spring 2026

---

## Overview

This project implements a **Masked Autoencoder (MAE)** from scratch using base **PyTorch**. The system learns visual representations by masking 75% of image patches and training a ViT-based encoder-decoder to reconstruct them.

## Architecture

| Component | Config | Details |
|-----------|--------|---------|
| **Encoder** | ViT-Base (B/16) | 768-dim, 12 layers, 12 heads, ~86M params |
| **Decoder** | ViT-Small (S/16) | 384-dim, 12 layers, 6 heads, ~22M params |
| **Patch Size** | 16×16 | 196 patches per 224×224 image |
| **Mask Ratio** | 75% | 49 visible, 147 masked patches |

### Why this Asymmetric Architecture?
1. **Compute Efficiency:** The key insight of MAE is masking out **75%** of the input image. The heavy encoder *only* processes the remaining 25% visible patches, drastically reducing memory consumption and computation time (by 3x or more).
2. **Semantic Understanding:** By dropping 75% of the patches, the model cannot rely on simply interpolating nearby pixels. Instead, it is forced to develop a deep semantic understanding to hallucinate the missing pieces.
3. **Lightweight Decoder:** The decoder's only job is to reconstruct the original pixels from the deep latent representation for the pre-training task (we discard it during downstream fine-tuning), so we can keep it small and fast.

## 📁 Project Structure

```
Assignment2/
├── mae_assignment.py     # Complete MAE implementation (notebook-friendly)
├── gradio_app.py         # Interactive Gradio deployment app
├── README.md             # This file
└── GenAI_Assignment02.docx  # Assignment specification
```

## How to Run

### On Kaggle (Recommended)

1. Create a new **Kaggle Notebook**
2. Add the dataset: **akash2sharma/tiny-imagenet**
3. Set accelerator to **GPU T4 x2**
4. Copy the contents of `mae_assignment.py` into notebook cells (split at `# %%` markers)
5. Run all cells sequentially

### Gradio App

After training, add the `gradio_app.py` cell to your notebook or run standalone:

```bash
pip install gradio
python gradio_app.py
```

## Results & Evaluation

After 50 epochs of training on TinyImageNet:
- **PSNR (Peak Signal-to-Noise Ratio):** Achieved **~23.37 dB** (± 3.23), indicating highly competent visual fidelity.
- **SSIM (Structural Similarity Index):** Achieved **~0.9202** (± 0.0558), implying that the generated structural geometry closely matches the original ground truth.

When presented with an input image that looked like a scrambled chessboard of 75% missing pixels, the model impressively reconstructed distinct objects—such as animals, vehicles, and landscapes—with coherent shapes and textures.

## Deliverables

- [x] Complete PyTorch implementation (no HuggingFace/timm)
- [x] Training loop with mixed precision, AdamW, Cosine LR
- [x] Loss vs. epochs plot
- [x] 5+ reconstruction visualizations
- [x] PSNR and SSIM evaluation
- [x] Gradio app with image upload + masking ratio slider

## Submission Checklist

- [ ] Rename notebook: `AI_ASS01_XXF_YYYY.ipynb`
- [ ] Push to GitHub
- [ ] Write Medium blog post
- [ ] Write LinkedIn post
- [ ] Submit Word doc with all links
# Self-Supervised-Image-Representation-Learning-using-Masked-Autoencoders
