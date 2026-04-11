"""
model/loader.py

Shared model loading.
"""

import torch

import config
from ai_client.tokenizer import load_tokenizer
from .model import GPT


def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Fix compiled-model checkpoints
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Remove tensors dependent on BLOCK_SIZE
    state = {
        k: v
        for k, v in state.items()
        if "attn.mask" not in k
        and "rope_cos" not in k
        and "rope_sin" not in k
    }

    # The `strict=False` bit should help remove the warnings about the missing mask tensors.
    model.load_state_dict(state, strict=False)

    return checkpoint, state


def init_model(vocab_size, device):
    model = GPT(
        vocab_size=vocab_size,
        dim=config.MODEL_DIM,
        layers=config.MODEL_LAYERS,
        heads=config.MODEL_HEADS,
        max_seq=config.BLOCK_SIZE,
    ).to(device)
    return model


def load_model(checkpoint_path, device):
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size

    model = init_model(vocab_size, device)

    checkpoint, state = load_checkpoint(checkpoint_path, model, device)

    model.eval()

    return model, tokenizer
