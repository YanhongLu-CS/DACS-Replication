import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/data4/lyh/models/Qwen2.5-3B-Instruct"
DEFAULT_CLASSIFICATION_DIR = Path(
    "/Users/lyh/Developer/research/WCRTE/"
    "output/Qwen2.5-3B-Instruct/2500-150/classification"
)
POINTS = tuple(range(0, 101, 10))

SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud.
You are a helpful assistant.

You must follow this exact output structure:
<think>Your reasoning process</think><answer>Your final answer</answer>

Rules:
- Put all reasoning inside <think> tags.
- Put only the final answer inside <answer> tags.
- Do not output any text outside these tags.
- Do not repeat the opening <think> tag.
- Always close both tags.
"""


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


def build_base_prompt(user_input, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input.rstrip()},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<think>"


def tokenize_record(record, tokenizer):
    base_prompt = build_base_prompt(record["user_input"], tokenizer)
    reasoning_ids = tokenizer.encode(
        record["reasoning"],
        add_special_tokens=False,
    )
    return base_prompt, reasoning_ids


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
    sequences = []
    for base_prompt, reasoning_ids in tokenized:
        end = round(len(reasoning_ids) * point / 100)
        partial_reasoning = tokenizer.decode(
            reasoning_ids[:end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        complete_prompt = base_prompt + partial_reasoning + "</think><answer>"
        sequences.append(
            tokenizer.encode(complete_prompt, add_special_tokens=False)
        )

    input_ids, attention_mask = pad_sequences(
        sequences, tokenizer.pad_token_id, device
    )
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits[:, -1, :].float()
        log_probabilities = torch.log_softmax(logits, dim=-1)
        confidence = (
            log_probabilities.exp() * log_probabilities
        ).sum(dim=-1)
    return confidence.cpu().tolist()


def score_batch(records, tokenizer, model, device):
    try:
        tokenized = [tokenize_record(record, tokenizer) for record in records]
        scores = [[] for _ in records]
        for point in POINTS:
            values = confidence_at_point(
                tokenized, point, tokenizer, model, device
            )
            for row_scores, value in zip(scores, values):
                row_scores.append(value)

        results = []
        for record, row_scores in zip(records, scores):
            result = dict(record)
            for point, value in zip(POINTS, row_scores):
                result[f"DACS_{point:02d}"] = value
            result["AUC"] = sum(
                (POINTS[i + 1] - POINTS[i])
                / 100
                * (row_scores[i] + row_scores[i + 1])
                / 2
                for i in range(len(POINTS) - 1)
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
