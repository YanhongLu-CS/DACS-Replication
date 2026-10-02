import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    return parser.parse_args()


def is_correct(gold, prediction):
    return prediction == gold


def load_by_id(path):
    records = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id in records:
                raise ValueError(f"Duplicate sample_id in {path}: {sample_id}")
            records[sample_id] = record
    return records


def classify_split(generation_dir, output_dir, split):
    hint_path = generation_dir / "hint" / f"{split}.jsonl"
    no_hint_path = generation_dir / "no-hint" / f"{split}.jsonl"
    no_hint_records = load_by_id(no_hint_path)
    output_path = output_dir / f"{split}.jsonl"
    temporary_path = output_path.with_suffix(".jsonl.tmp")

    stats = {
        "both_format_ok": 0,
        "hint_correct": 0,
        "shortcut": 0,
    }
    seen_ids = set()

    with hint_path.open(encoding="utf-8") as source, temporary_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            hint = json.loads(line)
            sample_id = hint["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id in {hint_path}: {sample_id}")
            seen_ids.add(sample_id)

            if sample_id not in no_hint_records:
                raise ValueError(f"Missing no-hint record: {sample_id}")
            no_hint = no_hint_records[sample_id]
            if hint["ground_truth"] != no_hint["ground_truth"]:
                raise ValueError(f"Ground-truth mismatch: {sample_id}")

            if not (hint["format_ok"] and no_hint["format_ok"]):
                continue
            stats["both_format_ok"] += 1

            gold = hint["ground_truth"]
            if not is_correct(gold, hint["predicted_answer"]):
                continue
            stats["hint_correct"] += 1

            shortcut = not is_correct(gold, no_hint["predicted_answer"])
            result = {**hint, "shortcut": shortcut}
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

            stats["shortcut"] += int(shortcut)

    if seen_ids != set(no_hint_records):
        missing = sorted(set(no_hint_records) - seen_ids)
        raise ValueError(f"Missing hint records, first missing sample: {missing[0]}")

    temporary_path.replace(output_path)
    return stats


def main():
    args = parse_args()
    generation_dir = args.generation_dir.expanduser().resolve()
    output_dir = generation_dir.parent / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        split: classify_split(generation_dir, output_dir, split)
        for split in ("train", "test")
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
