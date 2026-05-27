# IITJ MLOps Assignment 2 - Goodreads Genre Classification

**Name:** Prudhvi Gowroju  
**Roll No:** G25AIT2152  
**Program:** PGD AI Program, IIT Jodhpur  

## Project Description

This project implements an end-to-end MLOps workflow for fine-tuning a Hugging Face transformer model on Goodreads book review genre classification. The workflow uses `distilbert-base-cased` for sequence classification, tracks training and evaluation metrics using Weights & Biases, saves evaluation outputs as W&B artifacts, publishes the trained model to Hugging Face Hub, and provides reproducible Python scripts for training, evaluation, publishing, and inference.

## Model

The selected model is `distilbert-base-cased`.

DistilBERT was selected because it is smaller and faster than full BERT while still retaining strong language understanding ability. This makes it suitable for fine-tuning on Goodreads genre classification within limited GPU time.

## Repository Structure

```text
.
├── data.py
├── eval.py
├── predict.py
├── push_to_hub.py
├── requirements.txt
├── train.py
├── utils.py
└── README.md
```

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

Login to W&B:

```bash
wandb login
```

Login to Hugging Face:

```bash
huggingface-cli login
```

## Training

Run training:

```bash
python train.py
```

The training script downloads and samples Goodreads reviews, creates a train/test split, loads `distilbert-base-cased`, fine-tunes the model using Hugging Face Trainer, logs training to W&B using `report_to="wandb"`, and saves the trained model locally.

## Evaluation

Run evaluation:

```bash
python eval.py
```

The evaluation script evaluates the model on the test set, records accuracy, weighted F1 score, and eval loss, saves `eval_results.json`, saves `eval_report.json`, and uploads both files to W&B as an artifact.

## Results

| Metric | Score |
| --- | ---: |
| Accuracy | 0.5763 |
| F1 Score | 0.5838 |
| Eval Loss | 1.2141 |

## Inference

Run prediction with the published Hugging Face model:

```bash
python predict.py --text "A magical adventure about kingdoms, dragons, friendship, and destiny."
```

## Public Links

- GitHub Repository: https://github.com/PrudhviGowroju/IITJ_MLOPs_assignment2
- Hugging Face Model: https://huggingface.co/PrudhviG31/distilbert-goodreads-genres
- W&B Project Dashboard: https://wandb.ai/prudhvigowroju_iitj/mlops-assignment2
- W&B Training Run: https://wandb.ai/prudhvigowroju_iitj/mlops-assignment2/runs/d1fkpl6f

## Notes

The main goal of this assignment is not perfect accuracy but understanding the MLOps workflow: model fine-tuning, experiment tracking, evaluation artifact logging, model publishing, and reproducible code organization.