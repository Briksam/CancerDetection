# 🧬 Cancer Detection & Analysis Toolkit (Deep Learning)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-BiLSTM-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive deep learning project combining **Natural Language Processing (NLP)** and **Computer Vision (CV)** to analyze cancer data from both pathology reports and medical images.

---

## 🚀 Project Overview

This repository explores a **multi-modal approach** to cancer analysis, implemented in **Python** with **PyTorch**. It includes:

1. **Pathology Report Classification (`CancerReports.py`)** → NLP model to classify reports into four cancer grades (G1–G4).
2. **Medical Image Classification (`part1.py`)** → CNN for binary classification (benign vs. malignant).
3. **Metastasis Ratio Prediction (`part2.py`)** → CNN for regression, predicting metastasis ratios from medical images.

---

## 📑 1. Pathology Report Analysis (`CancerReports.py`)

A **BiLSTM with attention** network for classifying pathology reports.

### ✨ Key Features

* Preprocesses text, merges datasets, and encodes labels.
* Uses **pre-trained GloVe embeddings** for word representations.
* BiLSTM with attention pooling (mean + max).
* Outputs `SubmissionFile.csv` with cancer grade probabilities.

### ▶️ How to Run

```bash
python CancerReports.py
```

**Requirements**: `data.tsv`, `train.csv`, `test.csv`, and `glove.840B.300d.txt` (update its path in `load_glove`).

---

## 🖼️ 2. Medical Image Classification (`part1.py`)

A **custom CNN** for binary classification of medical images (cancerous vs. non-cancerous).

### ✨ Key Features

* Custom PyTorch `Dataset` class for efficient image loading.
* CNN with 4 convolutional layers + fully connected layers.
* Training with **Adam**, `NLLLoss`, and `ReduceLROnPlateau`.
* Best model saved as `weights.pt`.
* Outputs `submission.csv` with `cancer_score` for each test image.

### ▶️ How to Run

```bash
python part1.py
```

> ⚠️ Update hardcoded file paths (`train.csv`, `sample_submission.csv`, `train/`, `test/`) before running.

---

## 📊 3. Metastasis Ratio Prediction (`part2.py`)

A CNN-based **regression model** predicting the **metastasis ratio** from medical images.

### ✨ Key Features

* Predicts continuous values (`metastasis_ratio`).
* Same CNN as `part1.py` but with a single output node.
* Loss function: **L1 Loss** (`nn.L1Loss`).
* Outputs `submission.csv` with predicted ratios.

### ▶️ How to Run

```bash
python part2.py
```

> ⚠️ Update hardcoded file paths before running.

---

## 📦 Dependencies

Install the required libraries with:

```bash
pip install torch torchvision pandas numpy scikit-learn keras_preprocessing matplotlib pillow seaborn tqdm
```

**Main stack**:

* Python 3.x
* PyTorch
* pandas, NumPy, scikit-learn
* keras\_preprocessing
* matplotlib, seaborn
* Pillow (PIL)
* tqdm

---

## 🏆 Related Kaggle Competitions

The project components correspond to the following Kaggle competitions:

1. **Breast Cancer Metastases Detection (Medical Image Classification)**  
   [BAU AIN2001 Fall22 A3P1](https://www.kaggle.com/competitions/bau-ain2001-fall22-a3p1/overview)

2. **Breast Cancer Percent Metastases Prediction (Metastasis Ratio Regression)**  
   [BAU AIN2001 Fall22 A3P2](https://www.kaggle.com/competitions/bau-ain2001-fall22-a3p2)

3. **Cancer Grade Extraction from Pathology Reports (NLP Classification)**  
   [BAU AIN2001 Fall22 A4](https://www.kaggle.com/competitions/bau-ain2001-fall22-a4)


## 🔮 Future Improvements

* ✅ Replace hardcoded paths with **config files or CLI arguments**.
* ✅ Combine image & text models into a **multimodal system**.
* ✅ Refactor scripts into **modular, reusable components**.

---
