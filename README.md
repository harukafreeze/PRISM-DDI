
# Gemini-DDI ♊: A Dual-view Framework for Drug-Drug Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-red.svg)]()

> **Official implementation for the paper:**  
> **"Gemini-DDI: A Dual-view Framework for Drug-Drug Interaction Prediction"**  
> (Currently under review at *Pattern Recognition*)

---

## 💡 Highlights
- **Theoretical Grounding**: Formulated via the **Information Bottleneck (IB)** principle to minimize spurious topological correlations while maximizing causal physicochemical information.
- **Dual-View Learning**: Synergizes discrete **Bond-Aware GINs** (Micro-View) with **SE-calibrated Continuous Descriptors** (Macro-View).
- **Invariance Distillation**: Compels the framework to internalize predictive logic within an invariant physicochemical manifold through **Stochastic Modality Occlusion ($p=0.8$)**.
- **OOD Robustness**: Achieves superior generalization on the fine-grained **DrugBank-65** benchmark, effectively mitigating structural overfitting in zero-shot scaffold hopping.

---

## 🏗️ Project Structure

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

---

## 🏃‍♂️ Reproduction Guide

### 1. Data Preparation (Common)
Generate the 210-D descriptors and motif graphs for the DrugBank chemical space:
```bash
python scripts/01_data_preprocessing/extract_features.py
```

### 2. Fine-Grained Mechanism Prediction (DrugBank-65)
To reproduce the **98.13% Accuracy** and **0.998 Micro-AUC**:
```bash
# Step A: Filter rare mechanisms and generate 5-fold inductive splits
python scripts/01_data_preprocessing/filter_db_65.py
python scripts/01_data_preprocessing/split_inductive.py

# Step B: Train Inductive Teachers (No-leakage guarantee)
python scripts/03_drugbank65_task/train_inductive_teacher.py

# Step C: Run S2 Pro Distillation with Radar Monitoring
python scripts/03_drugbank65_task/distill_s2_db65.py
```

### 3. Macro-Level Prediction (ZhongDDI)
To reproduce the cold-start benchmark of **55.45% Accuracy**:
```bash
python scripts/02_zhongddi_task/run_ablations_zhong.py --mode full
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

## 📧 Contact
For technical inquiries, please open an issue or contact the corresponding authors:  
**Wensheng Xiang**: `xiangwensheng@neau.edu.cn`  
**Changjian Zhou**: `zhouchangjian@neau.edu.cn`
```

---

