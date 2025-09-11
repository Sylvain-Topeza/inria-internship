# Generalization in Diffusion Models — Reproduction and Critical Analysis

This repository contains the full code, notebooks, results, and report for a student-led research project conducted at **Inria Mokaplan (2025)**.

It reproduces and extends the experiments of:

> **Zahra Kadkhodaie, Florentin Guth, Eero P. Simoncelli, Stéphane Mallat (2024)**  
> *Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations*  

---

## 📖 Overview

Diffusion models have achieved state-of-the-art sample quality, yet their mechanisms of **generalization** remain debated.  
Kadkhodaie et al. (2024) reported a surprising finding: two denoisers trained on disjoint subsets of CelebA converged to almost the same function, producing nearly identical generations from the same noise.

This project reproduces the main results and contributes three **critical extensions**:

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
   - Findings: convergence weakens significantly when redundancy is reduced  

---

## 🗂 Repository structure

- `report/` — full PDF report with methodology, results, discussion  
- `code/` — preprocessing, training, evaluation scripts (PyTorch)  
- `datasets/` — deterministic ordering metadata (`train_filenames.txt`, etc.)  
- `notebooks/` — Jupyter notebooks to reproduce experiments and figures  
- `results/` — figures used in the report  
- `requirements.txt` — Python dependencies  

> Preprocessed tensors (.pt) and trained denoisers are hosted externally (see below).

---

## 📦 Data & Models Access

- **Datasets (CelebA 40×40 tensors):** hosted on Hugging Face → [link here]  
- **Trained denoisers (N=10 … 100k):** hosted on Hugging Face → [link here]  
- **Ordering metadata (.txt):** included in `datasets/` here on GitHub  

---

## 🔬 Main Findings

- **Replication**: confirmed the original phenomenon — with large N, disjoint denoisers converge to nearly identical score functions.  
- **Critique**: correlation-based evaluation overstates originality; LPIPS reveals closer matches.  
- **Extensions**: convergence is not universal; it depends on dataset redundancy (identities, attributes).  

➡️ Generalization in diffusion models arises partly from **inductive biases**, but also heavily from **dataset structure** and **metric choice**.  

---

## ⚙️ Installation

```bash
git clone https://github.com/USERNAME/diffusion-generalization-reproduction.git
cd diffusion-generalization-reproduction
pip install -r requirements.txt
```

---

## 📑 Report

Full report available in `report/report.pdf`.

---

## 📜 License

Released under the **MIT License**.  

---

## 🙋 Acknowledgements

Conducted at **Inria Mokaplan** (2025).  
Based on the public code and protocol of Kadkhodaie et al. (2024).
