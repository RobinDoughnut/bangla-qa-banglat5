import json
import re
import string
import collections
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from tqdm import tqdm

MODEL_PATH = Path("outputs/model/best")
DATA_DIR = Path("data/squad_bn")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 16
NUM_BEAMS = 4


def load_squad_json(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    ids, questions, contexts, answers = [], [], [], []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if not qa.get("answers"):
                    continue
                ids.append(qa["id"])
                questions.append(qa["question"])
                contexts.append(context)
                answers.append({
                    "text": [a["text"] for a in qa["answers"]],
                    "answer_start": [a["answer_start"] for a in qa["answers"]],
                })
    return Dataset.from_dict({"id": ids, "question": questions, "context": contexts, "answers": answers})


def generate_predictions(model, tokenizer, dataset, device):
    model.eval()
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(dataset["question"], dataset["context"])
    ]
    predictions = []
    for i in tqdm(range(0, len(inputs), BATCH_SIZE), desc="generating"):
        batch = inputs[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=MAX_TARGET_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(decoded)
    return predictions


def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def exact_match(gold, pred):
    return int(normalize(gold) == normalize(pred))


def token_f1(gold, pred):
    g = normalize(gold).split()
    p = normalize(pred).split()
    common = collections.Counter(g) & collections.Counter(p)
    n = sum(common.values())
    if not g or not p:
        return int(g == p)
    if n == 0:
        return 0.0
    return 2 * (n / len(p)) * (n / len(g)) / ((n / len(p)) + (n / len(g)))


def compute_metrics(predictions, dataset):
    em_total = f1_total = 0.0
    examples = list(dataset)
    for ex, pred in zip(examples, predictions):
        golds = [a for a in ex["answers"]["text"] if normalize(a)]
        if not golds:
            golds = [""]
        em_total += max(exact_match(g, pred) for g in golds)
        f1_total += max(token_f1(g, pred) for g in golds)
    n = len(examples)
    return {"EM": 100 * em_total / n, "F1": 100 * f1_total / n, "N": n}


def evaluate_split(split, tokenizer, model, device):
    print(f"\n=== {split} ===")
    ds = load_squad_json(DATA_DIR / f"{split}.json")
    preds = generate_predictions(model, tokenizer, ds, device)
    metrics = compute_metrics(preds, ds)
    print(f"  EM: {metrics['EM']:.2f}  F1: {metrics['F1']:.2f}  (N={metrics['N']:,})")
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH)).to(device)

    evaluate_split("validation", tokenizer, model, device)
    evaluate_split("test", tokenizer, model, device)


if __name__ == "__main__":
    main()
