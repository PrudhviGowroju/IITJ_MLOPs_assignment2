
import argparse
import wandb

from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Push trained model to Hugging Face Hub.")
    parser.add_argument("--model-dir", default="distilbert-goodreads-genres")
    parser.add_argument("--repo-id", default="PrudhviG31/distilbert-goodreads-genres")
    parser.add_argument("--project", default="mlops-assignment2")
    return parser.parse_args()


def main():
    args = parse_args()

    login()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    model.push_to_hub(args.repo_id)
    tokenizer.push_to_hub(args.repo_id)

    hf_url = f"https://huggingface.co/{args.repo_id}"

    wandb.init(
        project=args.project,
        name="huggingface-model-publish",
        job_type="model-publish",
    )

    wandb.run.summary["huggingface_model"] = hf_url
    wandb.finish()

    print("Hugging Face model:", hf_url)


if __name__ == "__main__":
    main()
