# 📡 Disaster Tweet Classification with BERT

This repository contains a high-performance NLP pipeline designed to classify tweets as either **real disasters** or **non-disasters**. Built for the [Kaggle: Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) competition, this project leverages state-of-the-art transformer architecture.

## 🚀 Overview

The model distinguishes between tweets that are reporting actual emergencies and those that use disaster-related language metaphorically (e.g., "The concert was a total disaster!"). 

### Key Features:
- **Transformer-based Logic**: Utilizes `DistilBERT` (`distilbert-base-uncased`) for deep semantic understanding.
- **PyTorch Backend**: High-performance training using custom Dataset and DataLoader classes.
- **GPU Accelerated**: Optimized for CUDA execution.
- **Evaluation**: Focused on F1-Score to balance precision and recall.

## 🛠️ Performance
- **Target F1-Score**: 0.85 - 1.00 (State-of-the-art range)
- **Primary Metric**: F1-Score

## 📦 Requirements

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## 🏗️ Pipeline Architecture

1. **Preprocessing**: Minimal cleaning is performed as BERT's tokenizer handles complex patterns (punctuation, casing, etc.) natively.
2. **Dataset Class**: Custom `TweetDataset` handles tokenization and tensor conversion.
3. **Training**: 
   - Optimizer: `AdamW`
   - Batch Size: 16
   - Max Length: 128
4. **Validation**: Real-time F1-score tracking after each epoch with automatic checkpointing of the best model (`best_model.pth`).

## 📋 Usage

1. Ensure the training and test CSV files are present in the Kaggle input directory.
2. Run the `NLP-Disaster.ipynb` notebook.
3. The script will generate a `submission_bert.csv` file ready for Kaggle leaderboard submission.

---
*Created as part of the Kaggle NLP series.*
