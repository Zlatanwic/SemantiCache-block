"""Deep diagnosis: why SemantiCache fails at shallow depths on Llama."""
import json
import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import ExperimentConfig
from semantic_analyzer import SemanticAnalyzer, RoleTag
from eval_ruler_niah import build_pg_haystack, make_ruler_needle, insert_needle_at_depth
import random

# Load model
model_dir = snapshot_download("LLM-Research/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained(model_dir)
sa = SemanticAnalyzer(tok)

# Build the same RULER prompt at depth=0.0 (where it fails)
rng = random.Random(42)
needle = make_ruler_needle("numbers", rng)
haystack = build_pg_haystack(tok, 4096)
text_with_needle = insert_needle_at_depth(haystack, needle.fact, depth=0.0)

msgs = [
    {"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided text."},
    {"role": "user", "content": f"{text_with_needle}\n\n{needle.question}"},
]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
input_ids = tok.encode(prompt, return_tensors="pt")[0]

print(f"=== Prompt Stats ===")
print(f"Total tokens: {len(input_ids)}")
print(f"Needle: key={needle.key}, value={needle.value}")
print(f"Needle fact: {needle.fact}")

# Find needle position in token ids
needle_token_ids = tok.encode(needle.value, add_special_tokens=False)
needle_positions = []
ids_list = input_ids.tolist()
for i in range(len(ids_list) - len(needle_token_ids) + 1):
    if ids_list[i:i+len(needle_token_ids)] == needle_token_ids:
        needle_positions.append(i)
print(f"Needle value token ids: {needle_token_ids}")
print(f"Needle positions in prompt: {needle_positions}")

# Role tags
tags = sa.compute_role_tags(input_ids)
for t in [RoleTag.SYSTEM, RoleTag.USER_LATEST, RoleTag.USER_HISTORY, RoleTag.ASSISTANT, RoleTag.FILLER, RoleTag.CONTEXT]:
    count = (tags == t).sum().item()
    if count > 0:
        print(f"  {t.name}: {count} tokens")

# Check pinned mask
pinned = sa.get_pinned_mask(tags, pin_system=True, pin_latest_user=True, latest_user_tail_tokens=16)
print(f"\nPinned tokens: {pinned.sum().item()}")

# Budget calculation
budget_tokens = int(len(input_ids) * 0.2)
print(f"Budget at 20%: {budget_tokens} tokens")
print(f"Pinned / budget: {pinned.sum().item()} / {budget_tokens} = {pinned.sum().item()/budget_tokens:.1%}")

# Check if needle is in pinned region
for pos in needle_positions:
    for offset in range(len(needle_token_ids)):
        p = pos + offset
        print(f"  Needle token at pos {p}: pinned={pinned[p].item()}, tag={RoleTag(tags[p].item()).name}")

# Semantic signals
info_density = sa.compute_info_density(input_ids)
query_relevance = sa.compute_query_relevance(input_ids, needle.question)
factual_bonus = sa.compute_factual_bonus(input_ids)

# Show signal values at needle positions
print(f"\n=== Signals at needle positions ===")
for pos in needle_positions:
    for offset in range(len(needle_token_ids)):
        p = pos + offset
        tok_text = tok.decode([ids_list[p]])
        print(f"  pos={p} '{tok_text}': density={info_density[p]:.3f}, query_rel={query_relevance[p]:.3f}, factual={factual_bonus[p]:.3f}")

# Compare with signals at other positions (sample)
print(f"\n=== Signal distribution (percentiles) ===")
for name, signal in [("info_density", info_density), ("query_relevance", query_relevance), ("factual_bonus", factual_bonus)]:
    vals = signal.float()
    p25, p50, p75, p95 = torch.quantile(vals, torch.tensor([0.25, 0.5, 0.75, 0.95]))
    needle_vals = [signal[p].item() for p in needle_positions for _ in range(1)]
    print(f"  {name}: p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} p95={p95:.3f} | needle={needle_vals}")

# Recent window and generated retention
from config import CacheConfig
cc = CacheConfig()
print(f"\n=== Protection windows ===")
print(f"  recent_window: {cc.semantic_recent_window}")
print(f"  latest_user_tail: {cc.semantic_latest_user_tail_tokens}")
print(f"  generated_retention: {cc.semantic_generated_retention_window}")

# How many USER_LATEST tokens
user_latest_count = (tags == RoleTag.USER_LATEST).sum().item()
print(f"  USER_LATEST tokens: {user_latest_count}")
print(f"  SYSTEM tokens: {(tags == RoleTag.SYSTEM).sum().item()}")

# Simulate what gets kept: pinned + recent_window + top scored
remaining_budget = budget_tokens - pinned.sum().item()
print(f"\n=== Budget breakdown ===")
print(f"  Total budget: {budget_tokens}")
print(f"  Pinned (system+user): {pinned.sum().item()}")
print(f"  Remaining for scoring: {remaining_budget}")
if remaining_budget < 0:
    print(f"  WARNING: Pinned tokens EXCEED budget by {-remaining_budget}!")
