"""
train.py — Train a small GPT-style language model (Scout)

Changes from previous version
──────────────────────────────
• Linear warmup → cosine decay LR schedule (properly wired)
• Scheduler state saved and restored in checkpoints
• [Trey] turn masking — loss computed on [Scout] tokens only
• config.py values used as authoritative defaults throughout
• batch_size, block_size, max_steps consistent with config
• Gradient accumulation support for effective large batch sizes
• Validation loss logged against held-out token file if present
"""

import boto3
import time
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from transformers import AutoTokenizer

from .config import DEFAULT_CONFIG, update_vocab_size
from .model import GPT
from .data import tokenize_corpus, load_token_tensor, build_dataloader


# ─────────────────────────────────────────────────────────────
# SageMaker helpers
# ─────────────────────────────────────────────────────────────

def running_in_sagemaker() -> bool:
    return Path("/opt/ml").exists()


def detect_sagemaker_paths(args):
    if running_in_sagemaker():
        print("SageMaker environment detected")
        if args.corpus_dir is None:
            args.corpus_dir = "/opt/ml/input/data/train"
        if args.data_path is None:
            args.data_path = "/opt/ml/input/data/train/corpus.pt"
        if args.out_dir is None:
            args.out_dir = "/opt/ml/checkpoints"
    return args


def upload_checkpoint(s3, bucket, prefix, file_path):
    key = f"{prefix}/{Path(file_path).name}"
    print(f"  Uploading {file_path} → s3://{bucket}/{key}")
    s3.upload_file(str(file_path), bucket, key)


# ─────────────────────────────────────────────────────────────
# Trey-turn masking
# ─────────────────────────────────────────────────────────────

def build_scout_mask(input_ids: torch.Tensor, trey_id: int, scout_id: int) -> torch.Tensor:
    """
    Build a boolean mask that is True only for tokens that are part
    of a [Scout] turn. [Trey] turns are masked out so they do not
    contribute to the training loss.

    Assumes alternating turn structure:
        [Trey]  ... tokens ...
        [Scout] ... tokens ...

    Args:
        input_ids : (batch, seq_len) token tensor
        trey_id   : token ID for the [Trey] speaker tag
        scout_id  : token ID for the [Scout] speaker tag

    Returns:
        mask : (batch, seq_len) bool tensor
                True  → include this token in loss
                False → mask this token out
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(batch_size):
        in_scout_turn = False
        for t in range(seq_len):
            token = input_ids[b, t].item()
            if token == scout_id:
                in_scout_turn = True
            elif token == trey_id:
                in_scout_turn = False
            if in_scout_turn:
                mask[b, t] = True

    return mask


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask:   torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss over Scout tokens only.

    Args:
        logits : (batch, seq_len, vocab_size)
        labels : (batch, seq_len)
        mask   : (batch, seq_len) bool — True where loss should be computed

    Returns:
        Scalar loss tensor. Returns full cross-entropy if no Scout tokens
        are present in the batch (guards against empty-mask edge case).
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask   = mask[:, 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask   = shift_mask.view(-1)

    if flat_mask.sum() == 0:
        # No Scout tokens in this batch — fall back to full loss
        return F.cross_entropy(flat_logits, flat_labels)

    loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    return (loss * flat_mask.float()).sum() / flat_mask.float().sum()


# ─────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr: float, base_lr: float):
    """
    Linear warmup followed by cosine decay to min_lr.
    """
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=min_lr,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ─────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────

def save_checkpoint(out_dir, step, model, optimizer, scheduler, cfg, s3=None, s3_bucket=None, s3_prefix="checkpoints"):
    out_dir = Path(out_dir)
    ckpt_path  = out_dir / f"model_{step}.pt"
    latest_path = out_dir / "latest.pt"

    checkpoint = {
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config":    cfg,
    }

    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    if s3 and s3_bucket:
        upload_checkpoint(s3, s3_bucket, s3_prefix, ckpt_path)
        upload_checkpoint(s3, s3_bucket, s3_prefix, latest_path)


def try_resume_checkpoint(model, optimizer, scheduler, checkpoint_dir, device):
    latest = Path(checkpoint_dir) / "latest.pt"

    if not latest.exists():
        print("No checkpoint found — starting fresh.")
        return 0

    print(f"Resuming from checkpoint: {latest}")
    ckpt = torch.load(latest, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        print("  Warning: no scheduler state in checkpoint — scheduler reset.")

    step = ckpt.get("step", 0)
    print(f"  Resumed from step {step}")
    return step


# ─────────────────────────────────────────────────────────────
# Corpus helpers
# ─────────────────────────────────────────────────────────────

def newest_file_mtime(directory):
    newest = 0
    for p in Path(directory).rglob("*"):
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
    return newest


def corpus_needs_tokenization(corpus_dir, token_file):
    token_file = Path(token_file)
    if not token_file.exists():
        return True
    return newest_file_mtime(corpus_dir) > token_file.stat().st_mtime


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validation_loss(model, val_tokens, cfg, device, trey_id, scout_id, speaker_masking, num_batches=20):
    """
    Estimate loss on held-out validation tokens.
    Uses the same Scout-only masking as training.
    """
    model.eval()

    loader = build_dataloader(
        val_tokens,
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    total_loss = 0.0
    count      = 0

    for batch in loader:
        if count >= num_batches:
            break

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        logits = model(x)
        if speaker_masking:
            mask = build_scout_mask(x, trey_id, scout_id).to(device)
            loss = masked_cross_entropy(logits, y, mask)
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        total_loss += loss.item()
        count      += 1

    model.train()
    return total_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def main(args):
    args    = detect_sagemaker_paths(args)
    use_s3  = running_in_sagemaker() and args.s3_bucket is not None
    s3      = boto3.client("s3") if use_s3 else None

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Paths ────────────────────────────────────────────────
    corpus_dir = Path(args.corpus_dir)
    token_file = Path(args.data_path)
    token_file.parent.mkdir(parents=True, exist_ok=True)

    val_token_file = token_file.parent / "corpus_val.pt"

    # ── Config ───────────────────────────────────────────────
    cfg = DEFAULT_CONFIG.copy()

    # Allow launcher overrides
    cfg["batch_size"]    = getattr(args, "batch_size",    cfg["batch_size"])
    cfg["block_size"]    = getattr(args, "seq_len",       cfg["block_size"])
    cfg["max_steps"]     = getattr(args, "steps",         cfg["max_steps"])
    cfg["learning_rate"] = getattr(args, "lr",            cfg["learning_rate"])
    cfg["warmup_steps"]  = getattr(args, "warmup_steps",  cfg["warmup_steps"])
    cfg["min_lr"]        = getattr(args, "min_lr",        cfg.get("min_lr", 3e-5))

    # ── Tokenizer ────────────────────────────────────────────
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = update_vocab_size(cfg, tokenizer)

    # Resolve speaker tag token IDs for masking
    trey_id = None
    scout_id = None

    if args.speaker_masking:
        trey_id  = tokenizer.convert_tokens_to_ids("[Trey]")
        scout_id = tokenizer.convert_tokens_to_ids("[Scout]")

        if trey_id == tokenizer.unk_token_id or scout_id == tokenizer.unk_token_id:
            print("Warning: [Trey] or [Scout] not found in tokenizer vocabulary.")
            print("Speaker masking disabled.")
            args.speaker_masking = False
        else:
            print(f"Speaker tokens — [Trey]: {trey_id}  [Scout]: {scout_id}")
    else:
        print("Speaker masking disabled.")

    if trey_id == tokenizer.unk_token_id or scout_id == tokenizer.unk_token_id:
        print("Warning: [Trey] or [Scout] not found in tokenizer vocabulary.")
        print("  Trey-turn masking will fall back to full loss.")
        print("  Consider adding these as special tokens before training.")
    else:
        print(f"Speaker tokens — [Trey]: {trey_id}  [Scout]: {scout_id}")

    # ── Tokenize corpus ──────────────────────────────────────
    if corpus_needs_tokenization(corpus_dir, token_file):
        print("Tokenizing corpus...")
        tokenize_corpus(
            input_dir=corpus_dir,
            tokenizer=tokenizer,
            output_path=token_file,
        )
    else:
        print("Tokenized corpus is up to date.")

    # ── Load tokens ──────────────────────────────────────────
    print(f"Loading corpus: {token_file}")
    tokens = load_token_tensor(token_file)
    print(f"Corpus: {len(tokens):,} tokens ({len(tokens)/1e6:.2f}M)")

    val_tokens = None
    if val_token_file.exists():
        val_tokens = load_token_tensor(val_token_file)
        print(f"Validation corpus: {len(val_tokens):,} tokens")
    else:
        print("No validation corpus found — skipping validation loss.")
        print(f"  To enable: place held-out conversations in {val_token_file}")

    # ── Dataloader ───────────────────────────────────────────
    loader    = build_dataloader(
        tokens,
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    data_iter = iter(loader)

    # ── Model ────────────────────────────────────────────────
    model = GPT(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        max_seq=cfg["block_size"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Optimizer + scheduler ────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    scheduler = build_scheduler(
        optimizer,
        total_steps=cfg["max_steps"],
        warmup_steps=cfg["warmup_steps"],
        min_lr=cfg["min_lr"],
        base_lr=cfg["learning_rate"],
    )

    # ── Resume ───────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = try_resume_checkpoint(model, optimizer, scheduler, out_dir, device)
    step       = start_step

    # ── Gradient accumulation ────────────────────────────────
    # Effective batch size = batch_size × accum_steps
    accum_steps = getattr(args, "accum_steps", 1)
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps "
              f"(effective batch size: {cfg['batch_size'] * accum_steps})")

    # ── Training loop ─────────────────────────────────────────
    print(f"\nTraining from step {start_step} → {cfg['max_steps']}\n")

    model.train()
    start_time     = time.time()
    accum_loss     = 0.0
    optimizer.zero_grad()

    while step < cfg["max_steps"]:

        # ── Fetch batch ──────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        # ── Forward pass ─────────────────────────────────────
        logits = model(x)

        # Build Scout-only mask and compute masked loss
        if args.speaker_masking:
            mask = build_scout_mask(x, trey_id, scout_id).to(device)
            loss = masked_cross_entropy(logits, y, mask)
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Scale loss for gradient accumulation
        scaled_loss = loss / accum_steps
        scaled_loss.backward()
        accum_loss += loss.item()

        # ── Optimizer step (every accum_steps) ───────────────
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ──────────────────────────────────────────
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            avg_loss   = accum_loss / max(step - start_step + 1, 1)

            val_str = ""
            if val_tokens is not None and step % (args.log_interval * 10) == 0 and step > 0:
                v_loss = validation_loss(
                    model, val_tokens, cfg, device,
                    trey_id, scout_id,
                    args.speaker_masking
                )
                val_str = f" | val {v_loss:.4f}"

            print(
                f"step {step:6d} | "
                f"loss {loss.item():.4f} | "
                f"avg {avg_loss:.4f} | "
                f"lr {current_lr:.2e}"
                f"{val_str} | "
                f"{elapsed:.0f}s"
            )

        # ── Checkpoint ───────────────────────────────────────
        if step % args.save_interval == 0 and step > start_step:
            save_checkpoint(
                out_dir, step,
                model, optimizer, scheduler, cfg,
                s3=s3, s3_bucket=args.s3_bucket,
            )

        step += 1

    # ── Final checkpoint ─────────────────────────────────────
    print("\nTraining complete.")
    save_checkpoint(
        out_dir, step,
        model, optimizer, scheduler, cfg,
        s3=s3, s3_bucket=args.s3_bucket,
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_dir",   default="../corpus")
    parser.add_argument("--data_path",    default="../data/corpus.pt")
    parser.add_argument("--out_dir",      default="../checkpoints")
    parser.add_argument("--tokenizer",    default="mistralai/Mistral-7B-v0.1")

    parser.add_argument("--steps",        type=int,   default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--seq_len",      type=int,   default=DEFAULT_CONFIG["block_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--warmup_steps", type=int,   default=DEFAULT_CONFIG["warmup_steps"])
    parser.add_argument("--min_lr",       type=float, default=3e-5)
    parser.add_argument("--accum_steps",  type=int,   default=1)

    parser.add_argument("--log_interval",  type=int,  default=DEFAULT_CONFIG["log_interval"])
    parser.add_argument("--save_interval", type=int,  default=DEFAULT_CONFIG["save_interval"])

    parser.add_argument("--s3_bucket",    default=None)

    parser.add_argument(
        "--speaker_masking",
        action="store_true",
        help="Enable Scout-only loss masking using [Trey]/[Scout] tokens"
    )

    args = parser.parse_args()
    main(args)
