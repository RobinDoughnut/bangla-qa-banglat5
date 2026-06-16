# Bangla QA with BanglaT5

Fine-tuning [`csebuetnlp/banglat5`](https://huggingface.co/csebuetnlp/banglat5) for extractive question answering on the Bangla SQuAD dataset ([csebuetnlp/squad_bn](https://huggingface.co/datasets/csebuetnlp/squad_bn)). Evaluated using Exact Match (EM) and F1 score.

This is a companion experiment to [bangla-qa-mbert](../bangla-qa-mbert), using a Bangla-specific T5 model instead of multilingual BERT.

## Approach

Unlike mBERT which predicts answer span positions (start/end tokens), BanglaT5 is a **seq2seq model** that *generates* the answer as text given the question and context.

| | mBERT | BanglaT5 |
|---|---|---|
| Architecture | Encoder-only | Encoder-Decoder |
| Output | Start/end token positions | Generated answer text |
| Input format | `[CLS] question [SEP] context [SEP]` | `"question: Q context: C"` |
| Long context | Sliding window (stride=128) | Truncation at 512 tokens |

## Dataset

| Split | Examples |
|-------|----------|
| Train | ~68,000 |
| Validation | ~1,250 |
| Test | ~1,250 |

(Unanswerable questions are excluded.)

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended
- `sentencepiece` is required for the BanglaT5 tokenizer

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd bangla-qa-banglat5

python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Data setup

```bash
python src/prepare_data.py
```

This downloads `csebuetnlp/squad_bn` and saves the splits under `data/squad_bn/`.

## Training

```bash
python src/train.py
```

Saves the best checkpoint to `outputs/model/best/`.

**Key hyperparameters** (edit at the top of `src/train.py`):

| Parameter | Value |
|-----------|-------|
| Base model | `csebuetnlp/banglat5` |
| Max input length | 512 tokens |
| Max target length | 64 tokens |
| Batch size | 16 |
| Epochs | 3 |
| Learning rate | 3e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Mixed precision (fp16) | Auto (enabled if CUDA available) |

## Evaluation

```bash
python src/evaluate_model.py
```

Generates predictions with beam search (4 beams) and reports EM and F1 on validation and test splits.

## Project Structure

```
bangla-qa-banglat5/
├── data/
│   └── squad_bn/          # gitignored — run prepare_data.py to populate
├── outputs/
│   └── model/             # gitignored — created during training
│       ├── best/
│       └── checkpoints/
├── src/
│   ├── prepare_data.py    # downloads dataset from HuggingFace
│   ├── train.py           # seq2seq fine-tuning script
│   └── evaluate_model.py  # generation + EM / F1 evaluation
├── .vscode/launch.json
├── requirements.txt
└── README.md
```

## Reproducing Results

```bash
git clone <repo-url> && cd bangla-qa-banglat5
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
python src/evaluate_model.py
```
