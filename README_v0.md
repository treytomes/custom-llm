# Custom LLM

A lightweight research project for building and training a small (~50M parameter) language model from scratch.

The goal of this repository is to provide a simple, understandable pipeline for:

- preparing a text corpus
- training a transformer language model
- saving and resuming checkpoints
- testing the model with prompts
- running training locally or on AWS SageMaker

This repository is designed for experimentation and learning rather than production-scale training.

---

# Project Structure

| Directory / File | Purpose |
|---|---|
| `training/model.py` | Transformer model architecture |
| `training/data.py` | Corpus tokenization and dataset loader |
| `training/train.py` | Training loop with checkpointing and resume support |
| `training/config.py` | Shared configuration for model and training parameters |
| `launch_training_local.py` | Launch training locally |
| `launch_training_remote.py` | Launch training on AWS SageMaker |
| `test_checkpoint.py` | Run inference from a trained checkpoint |
| `corpus/` | Raw text training data |
| `data/` | Tokenized dataset (`corpus.pt`) |
| `checkpoints/` | Saved model checkpoints |
| `sync_corpus_to_s3.sh` | Upload corpus to S3 |
| `sync_checkpoints.sh` | Download latest checkpoint from S3 |

The project contains a full training pipeline including tokenization, training, checkpointing, and generation scripts [2].

---

# Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training Setup

## Create the Python virtual environment

```bash
python3 -m venv custom-llm-env
source custom-llm-env/bin/activate
pip install -r requirements.txt
```

---

# Running Training Locally

## 1. Prepare a corpus

Place `.txt` files inside:

```
./corpus
```

Each file will be concatenated into the training dataset.

You can also generate a small sample dataset for testing:

```bash
python data.py --make_sample
```

---

## 2. Tokenize the corpus

```bash
python data.py --input_dir ./corpus --output_dir ./data
```

This creates:

```
./data/corpus.pt
```

---

## 3. Train the model

```bash
python train.py
```

Or run a short test session:

```bash
python train.py --steps 200
```

The standard workflow is:

1. Install dependencies
2. Prepare corpus
3. Tokenize
4. Train the model
5. Generate text from the trained checkpoint [1].

---

# Checkpoints

Checkpoints are stored in:

```
./checkpoints
```

Files include:

```
model_1000.pt
model_2000.pt
latest.pt
```

`latest.pt` always points to the newest checkpoint and is used when resuming training.

Training automatically resumes if a checkpoint is present.

---

# Testing the Model

Run inference with a trained checkpoint:

```bash
python test_checkpoint.py
```

Example prompt:

```
Prompt > It was a beautiful day.
```

The script loads `latest.pt` and generates text autoregressively.

---

# AWS SageMaker Training

Training can also run on SageMaker.

## Upload the corpus to S3

```bash
chmod +x sync_corpus_to_s3.sh
./sync_corpus_to_s3.sh
```

### What it does

1. Deletes everything under:

```
s3://bitnet-training-456088019014-us-east-1-an/corpus/
```

2. Uploads all files from:

```
./corpus
```

3. Lists uploaded objects for confirmation.

### Notes

- `aws s3 sync` preserves folder structure.
- Large files use multipart uploads automatically.
- SageMaker downloads the dataset to:

```
/opt/ml/input/data/train
```

when the training job starts.

---

# Download Latest Checkpoint

To retrieve the most recent model from S3:

```bash
chmod +x sync_checkpoints.sh
./sync_checkpoints.sh
```

This downloads:

```
s3://bitnet-training-456088019014-us-east-1-an/checkpoints/latest.pt
```

to:

```
./checkpoints/latest.pt
```

---

# Launch SageMaker Training

Run:

```bash
python launch_training_remote.py
```

This script:

1. Uploads training configuration
2. Starts a SageMaker training job
3. Streams logs until completion

The training container runs `launch_training_local.py`, ensuring local and remote training use the same pipeline.

---

# Directed Pair Optimization Training

1. Run `interact_and_review.py` to interact with the model and receive candidate feedback.  Select the best response and optionally provide your own correction.

2. Run `dpo.py` to convert the reviewed sessions into DPO training pairs that can be fed into the fine-tuning process.

---

# Future Work

Planned improvements include:

- larger training corpora
- improved sampling (top‑p / repetition penalties)
- evaluation scripts
- dataset streaming
- FlashAttention support
- experiment tracking

---

# License

This repository is intended for research and educational use.