import re

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/data4/lyh/models/Qwen2.5-3B-Instruct"
DATA_PATH = "data/Math-Reasoning/train-00000-of-00001.parquet"

SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud.
You are a helpful assistant.

You must follow this exact output structure:

<think>
Your reasoning process
</think>
<answer>
Your final answer
</answer>

Rules:
- Put all reasoning inside <think> tags.
- Put only the final answer inside <answer> tags.
- Do not output any text outside these tags.
- Do not repeat the opening <think> tag.
- Always close both tags.
"""


def build_prompt(problem, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.rstrip()},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<think>\n"


def format_ok(response):
    match = re.fullmatch(
        r"\s*<think>\s*(.+?)\s*</think>\s*"
        r"<answer>\s*(.+?)\s*</answer>\s*",
        response,
        re.DOTALL,
    )
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    return bool(match) and all(response.count(tag) == 1 for tag in tags)


problems = pq.read_table(DATA_PATH, columns=["problem"]).slice(0, 10)["problem"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

passed = 0
for i, problem in enumerate(problems.to_pylist(), 1):
    prompt = build_prompt(problem, tokenizer)
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    response = "<think>\n" + generated
    valid = format_ok(response)
    passed += valid

    print(f"\n--- Sample {i} ---")
    print(f"Problem:\n{problem}")
    print(f"\nOutput:\n{response}")
    print(f"\nFormat: {'PASS' if valid else 'FAIL'}")

print(f"\nFormat-compliant outputs: {passed}/10")
