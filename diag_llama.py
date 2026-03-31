"""Diagnose Llama chat template role parsing."""
from transformers import AutoTokenizer
from semantic_analyzer import SemanticAnalyzer, RoleTag
import torch

from modelscope import snapshot_download
model_dir = snapshot_download("LLM-Research/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained(model_dir)
sa = SemanticAnalyzer(tok)
print(f"chat_format: {sa.chat_format}")
print(f"im_start_id: {sa.im_start_id}, im_end_id: {sa.im_end_id}")

msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "The magic number is 42. What is the magic number?"},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(f"\nTemplate text:\n{repr(text[:300])}")

input_ids = tok.encode(text, return_tensors="pt")[0]
tags = sa.compute_role_tags(input_ids)
for t in [RoleTag.SYSTEM, RoleTag.USER_LATEST, RoleTag.FILLER, RoleTag.CONTEXT]:
    print(f"{t.name}: {(tags == t).sum().item()}")
print(f"Total: {len(input_ids)}, tagged: {(tags != RoleTag.FILLER).sum().item()}")
