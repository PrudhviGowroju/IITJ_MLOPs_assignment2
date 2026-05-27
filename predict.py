
import argparse
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Goodreads review genre.")
    parser.add_argument("--model-dir", default="PrudhviG31/distilbert-goodreads-genres")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    inputs = tokenizer(
        args.text,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_id = int(torch.argmax(probs, dim=-1).item())

    label = model.config.id2label[predicted_id]
    confidence = float(probs[0][predicted_id])

    print({
        "predicted_genre": label,
        "confidence": round(confidence, 4),
    })


if __name__ == "__main__":
    main()
