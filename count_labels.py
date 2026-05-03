import argparse
from collections import Counter

def iter_fasta_labels(fasta_path: str):
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                record_id = line[1:]
                if not record_id:
                    raise ValueError("Empty FASTA record id")
                if record_id[0] not in {"0", "1"}:
                    raise ValueError(
                        f"Cannot infer label from record id '{record_id}'. "
                        "Expected it to start with '0' or '1' (like '0nonAMP_1' or '1AMP_3')."
                    )
                yield int(record_id[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Count AMP vs non-AMP labels from a FASTA file.")
    parser.add_argument("--fasta", required=True, help="Input FASTA, e.g. dataset/AMPlify/AMPlify.fasta")
    args = parser.parse_args()

    counts = Counter(iter_fasta_labels(args.fasta))
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    total = pos + neg

    print(f"Total: {total}")
    print(f"AMP (label=1): {pos}")
    print(f"non-AMP (label=0): {neg}")
    if total > 0:
        print(f"Positive ratio: {pos / total:.4f}")
        print(f"Negative ratio: {neg / total:.4f}")


if __name__ == "__main__":
    main()
