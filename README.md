# IITJ MLOps Assignment 2 - Goodreads Genre Classification

**Name:** Prudhvi Gowroju  
**Roll No:** G25AIT2152  

## Project Description

This project implements an MLOps workflow for fine-tuning a Hugging Face transformer model on Goodreads book review genre classification. The model used is `distilbert-base-cased`, and the workflow includes dataset preparation, tokenization, model fine-tuning using Hugging Face Trainer, experiment tracking with Weights & Biases, final evaluation, artifact logging, model publishing to Hugging Face Hub, and reproducible scripts for training, evaluation, publishing, and inference.

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/PrudhviGowroju/IITJ_MLOPs_assignment2.git
cd IITJ_MLOPs_assignment2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Login to Weights & Biases:

```bash
wandb login
```

Login to Hugging Face:

```bash
huggingface-cli login
```

Run training:

```bash
python train.py
```

Run evaluation:

```bash
python eval.py
```

Push the trained model to Hugging Face Hub:

```bash
python push_to_hub.py
```

Run inference:

```bash
python predict.py --text "A magical adventure about kingdoms, dragons, friendship, and destiny."
```

## Results

| Metric | Score |
| --- | ---: |
| Accuracy | 0.5763 |
| F1 Score | 0.5838 |
| Eval Loss | 1.2141 |

## Hugging Face Model

https://huggingface.co/PrudhviG31/distilbert-goodreads-genres

## W&B Project Dashboard

https://wandb.ai/prudhvigowroju_iitj/mlops-assignment2