# Gemini-DDI ♊: A Dual-view Framework for Drug-Drug Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-red.svg)]()

> This repository contains the official TensorFlow implementation of the paper: **"Gemini-DDI: A Dual-view Framework for Drug-Drug Interaction Prediction"**.

---

## 💡 Overview

Graph Neural Networks (GNNs) have significantly advanced drug-drug interaction (DDI) prediction. However, their reliance on extracting training-specific structural motifs often leads to **Structural Overfitting**, which severely limits generalization to novel chemical scaffolds (i.e., the cold-start problem).

**Gemini-DDI** overcomes this limitation by acting as a logical bridge between micro-structural topologies and macro-physicochemical descriptors. By utilizing a **Consistency Distillation** strategy integrated with **Stochastic Modality Occlusion ($p=0.8$)**, Gemini-DDI compels the physicochemical branch to reconstruct the predictive manifold relying strictly on invariant, continuous physicochemical laws. 

This mechanism successfully decouples predictive reasoning from specific chemical backbones, achieving state-of-the-art (SOTA) performance in strict Inductive (Unseen-Unseen) scenarios without relying on external Knowledge Graphs.

<p align="center">
  <img src="figs/Figure1_base.png" alt="Gemini-DDI Architecture" width="90%">
  <br>
  <em>Figure 1: Schematic architecture of the Gemini-DDI framework.</em>
</p>

---

## 🚀 Key Features
- **Dual-View Representation**: Integrates a Bond-Aware Graph Isomorphism Network (Micro-View) with an SE-calibrated Physicochemical Encoder (Macro-View).
- **Consistency Distillation**: Employs an offline distillation pipeline with temperature-scaled KL-divergence.
- **Robust Out-of-Distribution (OOD) Generalization**: Sets a new benchmark on the highly imbalanced, fine-grained **DrugBank-65** dataset for zero-shot scaffold hopping.
- **Hardware Optimized**: Fully supports TensorFlow Mixed Precision (`mixed_float16`) and dynamic memory growth, optimized for NVIDIA A800 GPUs.

---

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/Gemini-DDI.git
cd Gemini-DDI
```

**2. Create a Conda environment**
```bash
conda create -n gemini_ddi python=3.8
conda activate gemini_ddi
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
# Core dependencies: tensorflow>=2.8, rdkit-pypi, pandas, scikit-learn, tqdm
```

---

## 📂 Project Structure

```text
Gemini-DDI/
├── configs/                  # Configuration files for Transductive/Inductive tasks
├── data/                     # Raw datasets and processed TFRecords
│   ├── raw/                  # Original CSV files (e.g., ZhongDDI, DrugBank)
│   └── db65_exp/             # Physically isolated directory for DB-65
├── prism/                    # Core Model Architecture
│   ├── dataloader.py         # Type-safe, dynamic TFRecord parser
│   ├── layers.py             # Bond-Aware GIN, Transformer, SE-Block
│   └── model.py              # Dual-view Gemini-DDI Network
├── scripts/                  # Execution pipelines
│   ├── run_deepddi_5fold_pipeline.py    # Warm-start (S0) 5-fold CV
│   └── run_db65_s2_distillation.py      # Cold-start (S2) Consistency Distillation
└── README.md
```

---

## 🏃‍♂️ Reproducing the Experiments

### Phase 1: Transductive Evaluation (Warm-start S0)
To train the base model on the DrugBank-65 dataset and evaluate its fundamental representational capacity:

```bash
# 1. Filter the dataset to 65 core mechanisms
python scripts/filter_deepddi_65.py

# 2. Extract dual-view features (RDKit descriptors & Motif graphs)
python scripts/preprocess_deepddi_features.py

# 3. Execute the 5-fold cross-validation pipeline
python scripts/run_deepddi_5fold_pipeline.py
```
*Note: The best-performing weights from this phase will be automatically archived as `DB65_Warm_Fold_X.h5` and used as the "Teacher" in Phase 2.*

### Phase 2: Inductive Evaluation (Cold-start S2) via Distillation
To evaluate the true zero-shot generalization on novel scaffolds using Modality Occlusion:

```bash
# 1. Generate strictly isolated drug-wise split datasets (S1 & S2)
python scripts/prepare_db65_inductive.py

# 2. Execute consistency distillation and S2 Radar evaluation
python scripts/run_db65_s2_distillation.py
```

---

## 📊 Main Results

Gemini-DDI achieves SOTA performance across multiple benchmarks. Below are the finalized results on the **DrugBank-65** dataset (evaluated across 5-fold CV):

| Scenario | Method | Top-1 Accuracy | Top-3 Accuracy | Micro-AUC | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Warm-start (S0)** | DSN-DDI (2023) | 96.94% | - | 0.9947 | 0.9693 |
| | **Gemini-DDI (Ours)** | **98.13%** | **99.85%** | **0.9980** | **0.9839** |
| **Cold-start (S2)** | SSI-DDI (2021) | 54.12% | - | 0.8423 | - |
| | **Gemini-DDI (Ours)** | **55.01%** | **78.30%** | **0.8920** | **0.4634** |

*(Detailed ablation studies and latent space clustering metrics (NMI & Purity) can be found in our manuscript.)*

---

## 📝 Citation

If you find our work or this code useful for your research, please consider citing our paper:

```bibtex
@article{wang2026geminiddi,
  title={Gemini-DDI: A Dual-view Framework for Drug-Drug Interaction Prediction},
  author={Wang, Shuoxiang and Zhou, Changjian},
  journal={Briefings in Bioinformatics (Under Review)},
  year={2026}
}
```

## 📧 Contact
For any questions or issues regarding the code, please open an issue in this repository or contact the corresponding author at: `zhouchangjian@neau.edu.cn`.
```

---

