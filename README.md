# TMMSRec
This repository contains the official implementation of the paper:

**"TMMSRec: Time-interval-aware Multi-Modal Sequential Recommender"**  

> *If you find this work useful for your research, please consider citing our paper (citation info to be added).*

---

## 📌 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Running the Code](#running-the-code)

---

## 📖 Overview

TMMSRec is a **time-interval-aware multi-modal sequential recommendation model** that leverages both temporal patterns and multi-modal content (e.g., text, image) to predict user preferences. This repository provides the full implementation, including data preprocessing, model training, evaluation, and pre-trained checkpoints.


---

## ⚙️ Environment Setup

### Requirements
- Python 3.8
- PyTorch 1.13.1
- CUDA 12.1 (recommended for GPU training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TMMSRec.git
cd TMMSRec
```
2. Create a virtual environment (optional but recommended):
```bash
conda create -n tmmsrec python=3.8
conda activate tmmsrec
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Preparation:
1. dataset.txt --including user interaction data
user_id item_id timestamp
2. text_feature.json --pre-trained text features
3. image_feature.json --pre-trained image features

### Running the Code
```bash
python main.py --dataset='All_Beauty'
```
