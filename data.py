
import gzip
import json
import random
import requests

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from utils import GENRE_URLS, GoodreadsGenreDataset


def load_reviews(url, head=10000, sample_size=500, seed=42):
    reviews = []

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for count, line in enumerate(file):
            if head is not None and count >= head:
                break

            data = json.loads(line)
            review_text = data.get("review_text", "").strip()

            if review_text:
                reviews.append(review_text)

    rng = random.Random(seed)
    return rng.sample(reviews, min(sample_size, len(reviews)))


def prepare_datasets(
    model_name="distilbert-base-cased",
    max_length=512,
    sample_size_per_genre=500,
    seed=42,
):
    all_texts = []
    all_labels = []

    for genre, url in GENRE_URLS.items():
        print(f"Loading reviews for genre: {genre}")
        reviews = load_reviews(
            url=url,
            head=10000,
            sample_size=sample_size_per_genre,
            seed=seed,
        )

        all_texts.extend(reviews)
        all_labels.extend([genre] * len(reviews))

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts,
        all_labels,
        test_size=0.2,
        random_state=seed,
        stratify=all_labels,
    )

    unique_labels = sorted(set(train_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    train_labels_encoded = [label2id[label] for label in train_labels]
    test_labels_encoded = [label2id[label] for label in test_labels]

    train_dataset = GoodreadsGenreDataset(train_encodings, train_labels_encoded)
    test_dataset = GoodreadsGenreDataset(test_encodings, test_labels_encoded)

    return {
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "label2id": label2id,
        "id2label": id2label,
    }
