
# Gemini-DDI ♊: A Dual-view Framework for Drug-Drug Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-red.svg)]()

> **Official implementation for the paper:**  
> **"Gemini-DDI: A Dual-view Framework for Drug-Drug Interaction Prediction"**  

---

## Project Structure

We maintain a strictly modularized directory structure for easy reproduction and scalability:

```text
Gemini-DDI/
├── configs/                  # Task-specific configurations (ZhongDDI/DB65)
├── prism/                    # Core neural operators and data loaders
│   ├── layers.py             # Mixed-precision optimized layers
│   └── model.py              # Dual-view residual interaction architecture
├── scripts/
│   ├── 01_data_preprocessing/ # Data filtering, 5x Augmentation, and Serialization
│   ├── 02_zhongddi_task/      # Reproduction for 4-class ADME task
│   ├── 03_drugbank65_task/    # Reproduction for 65-class fine-grained task
│   └── 04_evaluation_and_plots/ # Plotting and quantitative clustering (NMI/Purity)
└── data/                     # Data placeholders (Raw/Processed)
```

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/harukafreeze/Gemini-DDI.git
cd Gemini-DDI
```

**2. Setup Environment**
```bash
conda create -n gemini_ddi python=3.8
conda activate gemini_ddi
pip install -r requirements.txt
# Requirements: tensorflow==2.13.0, rdkit-pypi, scikit-learn, pandas, matplotlib
```

## Reproduction Guide

Follow these stages to reproduce the results for both **ZhongDDI** (Macro-level) and **DrugBank-65** (Mechanism-specific) benchmarks.

### Stage 1: Data Preparation & Preprocessing
Regardless of the task, you must first initialize the dual-view features:

```bash
# 1. Extract 210-D descriptors and Motif graphs (Common for all tasks)
python scripts/01_data_preprocessing/extract_features.py

# 2. For DrugBank-65 task: Filter rare mechanisms (< 500 samples)
python scripts/01_data_preprocessing/filter_db_65.py
```

### Stage 2: Macro-Level ADME Task (ZhongDDI)

```bash
# 1. Execute the 5-fold cross-validation for Warm-start (S0)
python scripts/02_zhongddi_task/train_s0_zhong.py

# 2. Run the logic-alignment distillation for Cold-start (S2)
python scripts/02_zhongddi_task/distill_s2_zhong.py

# 3. (Optional) Run all ablation studies
python scripts/02_zhongddi_task/run_ablations_zhong.py --mode no_distill
python scripts/02_zhongddi_task/run_ablations_zhong.py --mode no_se
```

### Stage 3: Fine-Grained Mechanism Task (DrugBank-65)

#### A. Data Pipeline (Inductive Split + 5x Augmentation)
```bash
# 1. Generate strictly isolated drug-wise splits (Folds 0-4)
python scripts/01_data_preprocessing/split_inductive.py

# 2. Generate 5x SMILES variants for training sets
python scripts/01_data_preprocessing/augment_data.py

# 3. Serialize all folds into TFRecords
python scripts/01_data_preprocessing/create_tfrecords.py
```

#### B. Training & Distillation
```bash
# 1. Reproduce S0 Warm-start results (Table 4, 98.13% Acc)
python scripts/03_drugbank65_task/train_s0_db65.py

# 2. Reproduce S2 Cold-start results:
# First, train Inductive Teachers (Clean models for zero-shot guide)
python scripts/03_drugbank65_task/train_inductive_teacher.py
# Then, run Pro-level Consistency Distillation with Radar monitoring
python scripts/03_drugbank65_task/distill_s2_db65.py
```

---

## 📊 Quantitative Visualization
Our framework provides built-in scripts to generate high-quality figures and latent space analysis reported in the paper:

- **Cluster Analysis**: Run `scripts/04_evaluation_and_plots/calc_purity_nmi.py` to calculate Normalized Mutual Information (NMI) and Cluster Purity.
- **Latent Manifolds**: Use `scripts/04_evaluation_and_plots/plot_figures.py` to generate t-SNE visualizations and task-adaptive interpretability bar charts.

---

## 📝 Citation

```bibtex
@article{wang2026geminiddi,
  title={Gemini-DDI: A Dual-view Framework for Drug-Drug Interaction Prediction},
  author={Wang, Shuoxiang and Zhang, Jiaqing and Hu, Xiya and Song, Jia and Cheng, Heng-Da and Xiang, Wensheng and Zhou, Changjian},
  journal={Pattern Recognition (Under Review)},
  year={2026}
}
```


---

