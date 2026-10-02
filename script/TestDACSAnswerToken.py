import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate import MODEL_PATH, build_prompt


DATA_PATH = (
    ROOT
    / "output/Qwen2.5-3B-Instruct/2500-150/classification/train.jsonl"
)
POINTS = tuple(range(0, 101, 10))


def find_number_token(row, token_ids, scores, tokenizer, eos_ids):
    for step, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            break
        token = tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
        if any(character.isdigit() for character in token):
            probability = torch.softmax(scores[step][row].float(), dim=-1)[
                token_id
            ].item()
            return step + 1, token, probability
    return None, None, None


def main():
    with DATA_PATH.open(encoding="utf-8") as file:
        records = [json.loads(next(file)) for _ in range(10)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)

    for record in records:
        reasoning_ids = tokenizer.encode(
            record["reasoning"], add_special_tokens=False
        )
        base_prompt = build_prompt(record["user_input"], tokenizer)
        prompts = []

        for point in POINTS:
            end = round(len(reasoning_ids) * point / 100)
            partial_reasoning = tokenizer.decode(
                reasoning_ids[:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts.append(
                base_prompt
                + partial_reasoning
                + "\n</think>\n<answer>\n"
            )

        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        with torch.inference_mode():
            result = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        input_length = inputs["input_ids"].shape[1]
        print(f"\n{'=' * 20} {record['sample_id']} {'=' * 20}")
        print(f"Ground truth: {record['ground_truth']}")

        for row, point in enumerate(POINTS):
            token_ids = result.sequences[row, input_length:].tolist()
            content_ids = []
            for token_id in token_ids:
                if token_id in eos_ids:
                    break
                content_ids.append(token_id)
            generated = tokenizer.decode(
                content_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            step, number_token, probability = find_number_token(
                row, token_ids, result.scores, tokenizer, eos_ids
            )

            print(f"DACS_{point:02d}: output={generated!r}")
            if number_token is None:
                print("  number token: not found within 10 tokens")
            else:
                print(
                    f"  number token={number_token!r}, "
                    f"step={step}, confidence={probability:.8f}"
                )


if __name__ == "__main__":
    main()
