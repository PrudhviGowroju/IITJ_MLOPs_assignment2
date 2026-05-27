
import argparse
import json
import wandb

from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, Trainer

from data import prepare_datasets
from utils import compute_metrics, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Goodreads genre classifier.")
    parser.add_argument("--model-dir", default="distilbert-goodreads-genres")
    parser.add_argument("--base-model", default="distilbert-base-cased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sample-size-per-genre", type=int, default=500)
    parser.add_argument("--project", default="mlops-assignment2")
    parser.add_argument("--run-name", default="distilbert-evaluation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data = prepare_datasets(
        model_name=args.base_model,
        max_length=args.max_length,
        sample_size_per_genre=args.sample_size_per_genre,
        seed=args.seed,
    )

    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)

    wandb.init(
        project=args.project,
        name=args.run_name,
        job_type="evaluation",
    )

    trainer = Trainer(
        model=model,
        eval_dataset=data["test_dataset"],
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    pred_output = trainer.predict(data["test_dataset"])
    preds = pred_output.predictions.argmax(-1)
    labels = [item["labels"].item() for item in data["test_dataset"]]

    target_names = [
        data["id2label"][i]
        for i in sorted(data["id2label"].keys())
    ]

    report = classification_report(
        labels,
        preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    with open("eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    wandb.log({
        "final/loss": eval_results["eval_loss"],
        "final/accuracy": eval_results["eval_accuracy"],
        "final/f1": eval_results["eval_f1"],
    })

    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file("eval_results.json")
    artifact.add_file("eval_report.json")
    wandb.log_artifact(artifact)

    print(eval_results)
    wandb.finish()


if __name__ == "__main__":
    main()
