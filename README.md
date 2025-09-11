# Generalization in Diffusion Models — Reproduction and Critical Analysis

This repository archives the research internship project I conducted at **Inria (Mokaplan team), France’s National Institute for Research in Computer Science (2025)**.  

It reproduces and extends the experiments of:

> **Zahra Kadkhodaie, Florentin Guth, Eero P. Simoncelli, Stéphane Mallat (2024)**  
> *Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations*  

## 📖 Overview

Diffusion models have achieved state-of-the-art sample quality, yet their mechanisms of **generalization** remain debated.  
Kadkhodaie et al. (2024) reported a surprising finding: two denoisers trained on disjoint subsets of CelebA converged to almost the same function, producing nearly identical generations from the same noise.

This project reproduces the main results and contributes three **extensions**:

1. **Reproducible pipeline**  
   - Deterministic dataset ordering  
   - Memory-efficient duplicate removal (164GB → 400MB)  
   - Public release of ordering metadata for full reproducibility  

2. **Metric critique**  
   - Demonstrated that **pixel correlation** misrepresents originality  
   - Introduced **LPIPS (VGG)** re-ranking to better capture perceptual similarity  

3. **Dataset structure analysis**  
   - Attribute-controlled splits (Eyeglasses, Male/Female)  
   - Identity-disjoint splits  
   - Findings: convergence weakens when redundancy is reduced  

## 🗂 Repository structure

- `report/` — full PDF report with methodology, results, discussion  
- `code/` — preprocessing, training, evaluation scripts (PyTorch)  
- `datasets/` — deterministic ordering metadata (`train_filenames.txt`, etc.)  
- `notebooks/` — Jupyter notebooks to reproduce experiments and figures  
- `results/` — figures used in the report  
- `requirements.txt` — Python dependencies  

> Preprocessed tensors (.pt) and trained denoisers are hosted externally on Hugging Face.

## 📦 Data & Models Access

- **Datasets (CelebA 40×40 tensors):** [Hugging Face](https://huggingface.co/datasets/Sylvain-Topeza/inria-datasets/tree/main)  
- **Trained denoisers (N=10 … 100k):** [Hugging Face](https://huggingface.co/Sylvain-Topeza/inria-models/tree/main)
- **Ordering metadata (.txt):** included in `datasets/` here on GitHub  

## 🚀 Quick start

1. Clone the repository and install dependencies from `requirements.txt`.  
2. Download datasets and models from Hugging Face (links above).  
3. Run the notebooks in `notebooks/` to reproduce the experiments and figures.  

## 🔬 Main Findings

- **Replication:** confirmed the original phenomenon — with large N, disjoint denoisers converge to nearly identical score functions.  
- **Critique:** correlation-based evaluation overstates originality; LPIPS reveals closer matches.  
- **Extensions:** convergence is not universal; it depends on dataset redundancy (identities, attributes).  

➡️ Generalization in diffusion models arises partly from **inductive biases**, but also heavily from **dataset structure** and **metric choice**.  

## 📚 Citation

If you use this repository, please cite:

- Topeza, S. (2025). *Memorization vs. Generalization in Diffusion Models with the U-Net Architecture: A Reproduction with Perceptual Metrics and Attribute-Controlled Splits*. Inria Mokaplan.  
- Kadkhodaie, Z., Guth, F., Simoncelli, E. P., & Mallat, S. (2024). *Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations*.  

## 📜 License

Released under the **MIT License**.  

## 🙋 Acknowledgements

Conducted during a research internship at **Inria (Mokaplan team)** in 2025.  
Based on the public code and protocol of Kadkhodaie et al. (2024).
