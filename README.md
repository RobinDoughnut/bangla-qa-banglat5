# Bangla QA with BanglaT5

Fine-tuning [`csebuetnlp/banglat5`](https://huggingface.co/csebuetnlp/banglat5) for extractive question answering on the Bangla SQuAD dataset ([csebuetnlp/squad_bn](https://huggingface.co/datasets/csebuetnlp/squad_bn)). Evaluated using Exact Match (EM) and F1 score.

## Related Experiments

Part of a 3-way comparison of QA architectures on Bangla SQuAD:

| Repo | Model | Architecture |
|------|-------|--------------|
| **bangla-qa-banglat5** (this repo) | BanglaT5 | Encoder-Decoder, Bangla-pretrained |
| [bangla-t5-finetune-qa](https://github.com/RobinDoughnut/bangla-t5-finetune-qa) | T5-base | Encoder-Decoder, English-pretrained |
| [mbert-finetune-banglaSQUAD](https://github.com/RobinDoughnut/mbert-finetune-banglaSQUAD) | mBERT | Encoder-only, span extraction |

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
| Train | 68,674 |
| Validation | 1,251 |
| Test | 1,252 |

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
| Batch size (per device) | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Epochs | 3 |
| Learning rate | 3e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Optimizer | Adafactor |
| Mixed precision | bf16 (if supported) |
| Gradient checkpointing | Yes |

## Evaluation

```bash
python src/evaluate_model.py
```

Generates predictions with beam search (4 beams) and reports EM, F1, and BERTScore-F1 on validation and test splits.

## Results

**Model stats:**

| Metric | Value |
|--------|-------|
| Trainable parameters | 247,577,856 |
| Total parameters | 247,577,856 |
| Model size (fp32) | 944.4 MB |

**Training** (RTX 4070, effective batch size 16):

| Epoch | Eval loss | Time |
|-------|-----------|------|
| 1 | 0.9986 | 30.9 min |
| 2 | 0.8223 | 30.9 min |
| 3 | 0.8003 | 30.8 min |

**Evaluation** (BERTScore-F1 uses `bert-base-multilingual-cased`, unrescaled, max over gold references):

| Split | EM | F1 | BERTScore-F1 | N |
|-------|-----|-----|--------------|-------|
| Validation | 54.92 | 68.34 | 91.16 | 1,251 |
| Test | 53.19 | 67.90 | 91.10 | 1,252 |

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
