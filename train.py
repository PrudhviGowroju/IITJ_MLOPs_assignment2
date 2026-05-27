
import argparse
import wandb

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from data import prepare_datasets
from utils import compute_metrics, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Goodreads genre reviews.")
    parser.add_argument("--model-name", default="distilbert-base-cased")
    parser.add_argument("--output-dir", default="distilbert-goodreads-genres")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sample-size-per-genre", type=int, default=500)
    parser.add_argument("--project", default="mlops-assignment2")
    parser.add_argument("--run-name", default="distilbert-run-1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data = prepare_datasets(
        model_name=args.model_name,
        max_length=args.max_length,
        sample_size_per_genre=args.sample_size_per_genre,
        seed=args.seed,
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(data["id2label"]),
        id2label=data["id2label"],
        label2id=data["label2id"],
    )

    wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "model": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "dataset": "UCSD Goodreads",
            "sample_size_per_genre": args.sample_size_per_genre,
        },
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        run_name=args.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train_dataset"],
        eval_dataset=data["test_dataset"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    data["tokenizer"].save_pretrained(args.output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
