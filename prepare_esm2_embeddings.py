import argparse
import os
from typing import List, Tuple

import numpy as np
import torch


def _parse_label(record_id: str) -> int:
    # Expected format in provided FASTA: ">0nonAMP_1" or ">1AMP_3" etc.
    if not record_id:
        raise ValueError("Empty FASTA record id")
    if record_id[0] not in {"0", "1"}:
        raise ValueError(
            f"Cannot infer label from record id '{record_id}'. "
            "Expected it to start with '0' or '1' (like '0nonAMP_1' or '1AMP_3')."
        )
    return int(record_id[0])


def _load_fasta(fasta_path: str) -> Tuple[List[str], List[str], np.ndarray, int]:
    try:
        from Bio import SeqIO
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "biopython is required to read FASTA. Install it with: pip install biopython"
        ) from e

    ids: List[str] = []
    seqs: List[str] = []
    labels: List[int] = []
    max_len = 0

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).strip().upper()
        # ESM2 expects standard AA letters; map unknowns to X.
        seq = "".join(ch if ch.isalpha() else "X" for ch in seq)

        ids.append(record.id)
        seqs.append(seq)
        labels.append(_parse_label(record.id))
        max_len = max(max_len, len(seq))

    if not ids:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    return ids, seqs, np.asarray(labels, dtype=np.int64), max_len

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ESM2 per-residue embeddings for this repo and save as a .npy "
            "compatible with get_data.data()."
        )
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA, e.g. dataset/AMPlify/AMPlify.fasta")
    parser.add_argument("--out", required=True, help="Output .npy, e.g. dataset/AMPlify/amplify_esm2.npy")
    parser.add_argument(
        "--model",
        default="facebook/esm2_t33_650M_UR50D",
        help=(
            "HuggingFace model id. IMPORTANT: this repo's model expects embedding dim 1280, "
            "so the default ESM2-650M is recommended."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: 'cpu' or 'cuda'. Default: auto-detect.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Embedding dtype to store. float16 is smaller but may slightly change downstream results.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Directory to cache/download HuggingFace models (use a drive with several GB free). "
            "If omitted, uses the default HuggingFace cache under your user profile."
        ),
    )

    args = parser.parse_args()

    fasta_path = args.fasta
    out_path = args.out

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    ids, seqs, labels, max_len = _load_fasta(fasta_path)

    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "transformers is required to compute ESM2 embeddings. Install it with: pip install transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model, cache_dir=args.cache_dir)
    model.to(device)
    model.eval()

    want_float16 = args.dtype == "float16"

    rows: List[list] = []

    try:
        from tqdm import tqdm

        iterator = tqdm(list(zip(ids, seqs, labels)), desc="Embedding", unit="seq")
    except Exception:
        iterator = list(zip(ids, seqs, labels))

    with torch.no_grad():
        for record_id, seq, label in iterator:
            # HF ESM tokenizers accept raw AA strings.
            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (1, L+2, 1280)
            hidden = hidden[0, 1:-1, :]  # remove BOS/EOS -> (L, 1280)

            emb = hidden.detach().cpu().numpy()  # (L, 1280)
            emb = emb.astype(np.float16 if want_float16 else np.float32)

            # Store variable-length embeddings. The dataloader will pad to match
            # the fixed max_len used by the handcrafted features.
            rows.append([record_id, emb, int(label)])

    arr = np.asarray(rows, dtype=object)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, arr, allow_pickle=True)

    print(f"Wrote {len(rows)} embeddings to: {out_path}")
    print(f"Max sequence length (from FASTA): {max_len}")


if __name__ == "__main__":
    main()
