import argparse
import json
from bisect import bisect_left
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dacs-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    shortcut = []
    faithful = []

    for split in ("train", "test"):
        path = args.dacs_dir.expanduser() / f"{split}.jsonl"
        with path.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if record["AUC"] is None:
                    continue
                group = record["shortcut"]
                if not isinstance(group, bool):
                    raise ValueError(f"Invalid shortcut label: {group!r}")
                (shortcut if group else faithful).append(float(record["AUC"]))

    if not shortcut or not faithful:
        raise ValueError("Both shortcut and faithful groups must contain valid samples.")

    faithful.sort()
    wins = sum(bisect_left(faithful, auc) for auc in shortcut)
    auroc = wins / (len(shortcut) * len(faithful))
    print(f"AUROC: {auroc:.6f}")


if __name__ == "__main__":
    main()
