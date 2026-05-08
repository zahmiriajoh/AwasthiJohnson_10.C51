"""Generate ESM-2 mean-pooled embeddings for sequences specified by a config.

Cached under cfg.embedding.cache_dir, keyed by SHA256(model_name + sequences).
Embedding is chunked and resumable: if interrupted, re-running picks up at
the last saved chunk.

Usage:
    python embed.py --config config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_utils import load_and_split, load_config


@torch.inference_mode()
def _embed_chunk(sequences, tokenizer, model, device, batch_size, max_length):
    """Mean-pool encoder output over residues for a chunk of sequences."""
    out = []
    for i in range(0, len(sequences), batch_size):
        batch = list(sequences[i:i + batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        h = model(**enc).last_hidden_state  # (B, L, D)

        # Mask out <cls>, <eos>, and padding before mean-pooling.
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0
        last = enc["attention_mask"].sum(dim=1) - 1
        mask[torch.arange(mask.size(0)), last] = 0
        mask = mask.unsqueeze(-1).float()

        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_sequences(sequences, cfg) -> np.ndarray:
    """Compute mean-pooled embeddings for an ordered list of sequences.

    Returns an (N, D) float32 array. Resumable: each chunk_NNNNN.npy is
    saved as it's computed, so a crash can pick up where it left off.
    """
    e = cfg["embedding"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sig = hashlib.sha256(e["model_name"].encode())
    for s in sequences:
        sig.update(s.encode())
    key = sig.hexdigest()[:16]
    run_dir = Path(e["cache_dir"]) / f"esm_{key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "all.npy"

    if final.exists():
        print(f"Loading cached embeddings: {final}")
        return np.load(final)

    print(f"Loading {e['model_name']} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(e["model_name"])
    model = (AutoModel.from_pretrained(e["model_name"], add_pooling_layer=False)
             .to(device).eval())

    n = len(sequences)
    chunk = e["chunk_size"]
    n_chunks = (n + chunk - 1) // chunk
    for ci in tqdm(range(n_chunks), desc="Embedding chunks"):
        path = run_dir / f"chunk_{ci:05d}.npy"
        if path.exists():
            continue
        s, t = ci * chunk, min((ci + 1) * chunk, n)
        embs = _embed_chunk(sequences[s:t], tokenizer, model, device,
                            e["batch_size"], e["max_length"])
        np.save(path, embs)

    out = np.concatenate(
        [np.load(run_dir / f"chunk_{ci:05d}.npy") for ci in range(n_chunks)],
        axis=0,
    )
    np.save(final, out)
    for ci in range(n_chunks):
        (run_dir / f"chunk_{ci:05d}.npy").unlink(missing_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_col = cfg["data"]["sequence_col"]
    df_train, df_test = load_and_split(cfg)
    sequences = (df_train[seq_col].astype(str).tolist()
                 + df_test[seq_col].astype(str).tolist())
    print(f"{len(df_train)} train + {len(df_test)} test = {len(sequences)} sequences")

    X = embed_sequences(sequences, cfg)
    print(f"Embedding shape: {X.shape}")


if __name__ == "__main__":
    main()