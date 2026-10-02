import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate import MODEL_PATH, build_prompt

DEFAULT_CLASSIFICATION_DIR = Path(
    "/Users/lyh/Developer/research/WCRTE/"
    "output/Qwen2.5-3B-Instruct/2500-150/classification"
)
POINTS = tuple(range(0, 101, 10))
MAX_NEW_TOKENS = 10
OUTPUT_PREFIX = "<think>\n"
ANSWER_PREFIX = "\n</think>\n<answer>\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=DEFAULT_CLASSIFICATION_DIR,
    )
    return parser.parse_args()


def tokenize_record(record, tokenizer):
    output = record["output"]
    if not output.startswith(OUTPUT_PREFIX):
        raise ValueError("output does not start with '<think>\\n'.")
    reasoning_end = output.find("</think>", len(OUTPUT_PREFIX))
    if reasoning_end == -1:
        raise ValueError("output does not contain '</think>'.")

    reasoning = output[len(OUTPUT_PREFIX) : reasoning_end].rstrip()
    base_prompt = build_prompt(record["user_input"], tokenizer)
    base_ids = tokenizer.encode(
        base_prompt,
        add_special_tokens=False,
    )
    full_prefix_ids = tokenizer.encode(
        base_prompt + reasoning,
        add_special_tokens=False,
    )
    if full_prefix_ids[: len(base_ids)] != base_ids:
        raise ValueError("reasoning does not begin at a stable token boundary.")
    return base_ids, full_prefix_ids[len(base_ids) :]


def pad_sequences(sequences, pad_token_id, device):
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, -length:] = torch.tensor(sequence, device=device)
        attention_mask[row, -length:] = 1
    return input_ids, attention_mask


def confidence_at_point(tokenized, point, tokenizer, model, device):
    answer_prefix_ids = tokenizer.encode(
        ANSWER_PREFIX,
        add_special_tokens=False,
    )
    sequences = []
    for base_ids, reasoning_ids in tokenized:
        end = round(len(reasoning_ids) * point / 100)
        sequences.append(base_ids + reasoning_ids[:end] + answer_prefix_ids)

    input_ids, attention_mask = pad_sequences(
        sequences, tokenizer.pad_token_id, device
    )
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = set()
    elif isinstance(eos_ids, int):
        eos_ids = {eos_ids}
    else:
        eos_ids = set(eos_ids)

    input_length = input_ids.shape[1]
    results = []
    for row, token_ids in enumerate(generated.sequences[:, input_length:]):
        confidence = None
        number_step = None
        for step, token_id in enumerate(token_ids.tolist()):
            if token_id in eos_ids:
                break
            token = tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )
            if any(character.isdigit() for character in token):
                log_probabilities = torch.log_softmax(
                    generated.scores[step][row].float(), dim=-1
                )
                confidence = (
                    log_probabilities.exp() * log_probabilities
                ).sum().item()
                number_step = step + 1
                break
        results.append((confidence, number_step))
    return results


def score_batch(records, tokenizer, model, device):
    try:
        tokenized = [tokenize_record(record, tokenizer) for record in records]
        scores = [[] for _ in records]
        number_steps = [[] for _ in records]
        for point in POINTS:
            values = confidence_at_point(
                tokenized, point, tokenizer, model, device
            )
            for row_scores, row_steps, (confidence, step) in zip(
                scores, number_steps, values
            ):
                row_scores.append(confidence)
                row_steps.append(step)

        results = []
        for record, row_scores, row_steps in zip(
            records, scores, number_steps
        ):
            result = dict(record)
            for point, value, step in zip(POINTS, row_scores, row_steps):
                result[f"DACS_{point:02d}"] = value
                result[f"DACS_{point:02d}_has_number"] = step is not None
                result[f"DACS_{point:02d}_number_step"] = step
            result["AUC"] = (
                sum(
                    (POINTS[i + 1] - POINTS[i])
                    / 100
                    * (row_scores[i] + row_scores[i + 1])
                    / 2
                    for i in range(len(POINTS) - 1)
                )
                if all(value is not None for value in row_scores)
                else None
            )
            results.append(result)
        return results
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(records) == 1:
            raise
        middle = len(records) // 2
        return score_batch(records[:middle], tokenizer, model, device) + score_batch(
            records[middle:], tokenizer, model, device
        )


def load_records(path, rank, world_size):
    records = []
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index % world_size == rank:
                records.append(json.loads(line))
    return records


def worker(rank, world_size, batch_size, classification_dir, shard_root):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    for split in ("train", "test"):
        records = load_records(
            Path(classification_dir) / f"{split}.jsonl", rank, world_size
        )
        shard_path = Path(shard_root) / f"{split}.rank{rank}.jsonl"
        progress = tqdm(
            total=len(records),
            desc=f"GPU {rank} | {split}",
            position=rank,
            leave=True,
            dynamic_ncols=True,
        )

        with shard_path.open("w", encoding="utf-8") as output:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                for result in score_batch(batch, tokenizer, model, device):
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                progress.update(len(batch))
        progress.close()


def read_jsonl(file):
    for line in file:
        yield json.loads(line)


def merge_shards(shard_root, output_dir, world_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        records = []
        for rank in range(world_size):
            shard_path = Path(shard_root) / f"{split}.rank{rank}.jsonl"
            with shard_path.open(encoding="utf-8") as file:
                records.extend(read_jsonl(file))
        records.sort(key=lambda item: item["sample_id"])

        output_path = output_dir / f"{split}.jsonl"
        temporary_path = output_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)


def main():
    args = parse_args()
    available_gpus = torch.cuda.device_count()
    if args.num_gpus < 1 or args.num_gpus > available_gpus:
        raise ValueError(
            f"--num-gpus must be between 1 and {available_gpus}, "
            f"got {args.num_gpus}."
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    classification_dir = args.classification_dir.expanduser().resolve()
    output_dir = classification_dir.parent / "DACS"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_root = tempfile.mkdtemp(prefix=".shards-", dir=output_dir)

    mp.spawn(
        worker,
        args=(
            args.num_gpus,
            args.batch_size,
            classification_dir,
            shard_root,
        ),
        nprocs=args.num_gpus,
        join=True,
    )
    merge_shards(shard_root, output_dir, args.num_gpus)
    shutil.rmtree(shard_root)


if __name__ == "__main__":
    main()
