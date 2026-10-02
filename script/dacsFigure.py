import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POINTS = tuple(range(0, 101, 10))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dacs-dir", type=Path, required=True)
    return parser.parse_args()


def load_means(dacs_dir):
    sums = {True: [0.0] * len(POINTS), False: [0.0] * len(POINTS)}
    counts = {True: 0, False: 0}

    for split in ("train", "test"):
        with (dacs_dir / f"{split}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                group = record["shortcut"]
                if not isinstance(group, bool):
                    raise ValueError(f"Invalid shortcut label: {group!r}")
                for i, point in enumerate(POINTS):
                    sums[group][i] += record[f"DACS_{point:02d}"]
                counts[group] += 1

    if not counts[True] or not counts[False]:
        raise ValueError("Both shortcut and faithful groups must contain samples.")

    means = {
        group: [value / counts[group] for value in sums[group]]
        for group in (True, False)
    }
    return means, counts


def main():
    args = parse_args()
    dacs_dir = args.dacs_dir.expanduser().resolve()
    output_dir = dacs_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    means, counts = load_means(dacs_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        POINTS,
        means[True],
        marker="o",
        linewidth=2,
        label=f"Shortcut (n={counts[True]})",
    )
    ax.plot(
        POINTS,
        means[False],
        marker="o",
        linewidth=2,
        label=f"Faithful (n={counts[False]})",
    )
    ax.set_xlabel("Reasoning progress (%)")
    ax.set_ylabel("Mean DACS")
    ax.set_title("DACS Across Reasoning Progress")
    ax.set_xticks(POINTS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "dacs_mean_curves.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
