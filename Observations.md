1. Prompt format
I identified an effective prompt format that consistently produces the required `<think>...</think>` and `<answer>...</answer>` structure (10/10 samples passed). Minor answer formatting, such as `\boxed{...}`, can be normalized during post-processing.

input format:
```
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

def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.rstrip()},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<think>\n"
```