"""
Entry point for text generation with manual KV-cache eviction.

This script avoids `model.generate()` so we can:
1. run a manual decode loop
2. inspect attentions
3. prune KV cache between decode steps
"""

import argparse
import time

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

from attention_tracker import AttentionTracker
from config import CacheConfig, ExperimentConfig, ModelConfig
from eviction_policies import (
    FullCachePolicy,
    H2OPolicy,
    LocalWindowPolicy,
    SemantiCachePolicy,
    StreamingLLMPolicy,
)
from kv_cache_manager import (
    KVCacheManager,
    get_cache_layer_count,
    get_cache_seq_len,
    get_first_layer_cache_shape,
    prune_dynamic_cache,
)
from semantic_analyzer import SemanticAnalyzer


def load_model(cfg: ModelConfig):
    """Load model and tokenizer."""
    print(f"Loading model: {cfg.model_name}")
    model_dir = snapshot_download(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    kwargs = {
        "device_map": cfg.device_map,
        "attn_implementation": cfg.attn_implementation,
    }

    if cfg.use_bnb_4bit:
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )
    else:
        kwargs["torch_dtype"] = cfg.torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    print(f"Model loaded. Device: {model.device}")
    return model, tokenizer


def build_policy(cache_cfg: CacheConfig, tracker: AttentionTracker, analyzer: SemanticAnalyzer):
    """Build the requested eviction policy."""
    budget_tokens = 512

    if cache_cfg.policy == "full":
        return FullCachePolicy()
    if cache_cfg.policy == "window":
        return LocalWindowPolicy(window_size=budget_tokens)
    if cache_cfg.policy == "streaming":
        return StreamingLLMPolicy(
            sink_tokens=cache_cfg.sink_tokens,
            window_size=budget_tokens,
        )
    if cache_cfg.policy == "h2o":
        return H2OPolicy(tracker=tracker)
    if cache_cfg.policy == "semantic":
        return SemantiCachePolicy(
            tracker=tracker,
            analyzer=analyzer,
            alpha=cache_cfg.alpha,
            beta=cache_cfg.beta,
            gamma=cache_cfg.gamma,
            pin_system=cache_cfg.pin_system,
            pin_latest_user=cache_cfg.pin_latest_user,
            recent_window_size=cache_cfg.semantic_recent_window,
            block_size=cache_cfg.semantic_block_size,
            latest_user_tail_tokens=cache_cfg.semantic_latest_user_tail_tokens,
        )

    raise ValueError(f"Unknown policy: {cache_cfg.policy}")


def _update_tracker_or_raise(attentions, tracker: AttentionTracker, policy_name: str, phase: str):
    if attentions and attentions[0] is not None:
        tracker.update(attentions)
        return

    if policy_name in {"h2o", "semantic"}:
        raise RuntimeError(
            f"Model did not return attentions during {phase}. "
            "Set ModelConfig.attn_implementation='eager'."
        )


@torch.no_grad()
def generate_with_eviction(model, tokenizer, messages: list[dict], config: ExperimentConfig) -> dict:
    """Run a manual decode loop and prune KV cache between steps."""
    model_cfg = config.model
    cache_cfg = config.cache

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    print(f"Prompt length: {prompt_len} tokens")
    print(f"Policy: {cache_cfg.policy}, Budget: {cache_cfg.cache_budget:.0%}")

    tracker = AttentionTracker(
        num_layers=model_cfg.num_layers,
        num_kv_heads=model_cfg.num_kv_heads,
    )
    analyzer = SemanticAnalyzer(tokenizer)
    policy = build_policy(cache_cfg, tracker, analyzer)

    cache_manager = KVCacheManager(
        config=cache_cfg,
        policy=policy,
        tracker=tracker,
        num_layers=model_cfg.num_layers,
    )

    t_start = time.time()
    past_key_values = DynamicCache(config=model.config)
    prompt_cache_position = torch.arange(prompt_len, device=model.device, dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=prompt_cache_position,
        position_ids=prompt_cache_position.unsqueeze(0),
        output_attentions=True,
        return_dict=True,
    )

    _update_tracker_or_raise(outputs.attentions, tracker, cache_cfg.policy, "prefill")
    cache_manager.set_initial_seq_len(prompt_len)

    if isinstance(policy, SemantiCachePolicy):
        policy.setup_semantic_signals(input_ids[0])

    next_token_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    generated_ids = []
    eos_token_id = tokenizer.eos_token_id
    next_cache_position = torch.tensor([prompt_len], device=model.device, dtype=torch.long)

    for step in range(model_cfg.max_new_tokens):
        if model_cfg.do_sample:
            probs = torch.softmax(next_token_logits / model_cfg.temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

        token_id = next_token_id.item()
        generated_ids.append(token_id)

        if token_id == eos_token_id:
            break

        past_key_values = cache_manager.evict(past_key_values)

        if isinstance(policy, SemantiCachePolicy):
            policy.extend_signals(num_new_tokens=1)

        outputs = model(
            input_ids=next_token_id,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=next_cache_position,
            position_ids=next_cache_position.unsqueeze(0),
            output_attentions=True,
            return_dict=True,
        )

        _update_tracker_or_raise(outputs.attentions, tracker, cache_cfg.policy, "decode")
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_cache_position = next_cache_position + 1

        if (step + 1) % 50 == 0:
            stats = cache_manager.get_stats()
            print(
                f"  Step {step + 1}: cache_len={stats['current_cache_len']}, "
                f"evicted={stats['total_evicted']}"
            )

    elapsed = time.time() - t_start
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stats = cache_manager.get_stats()

    print(
        f"\nGeneration complete: {len(generated_ids)} tokens in {elapsed:.2f}s "
        f"({len(generated_ids) / elapsed:.1f} tok/s)"
    )
    print(f"Eviction stats: {stats}")

    return {
        "output_text": output_text,
        "output_ids": generated_ids,
        "stats": stats,
        "elapsed_time": elapsed,
    }


def test_environment():
    """Quick environment test for model load, attentions and KV pruning."""
    print("=" * 60)
    print("Environment Test: validate model load and KV cache pruning")
    print("=" * 60)

    cfg = ModelConfig()
    model, tokenizer = load_model(cfg)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 1+1?"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"\n[1/4] Prompt tokens: {inputs['input_ids'].shape[1]}")

    past = DynamicCache(config=model.config)
    prompt_cache_position = torch.arange(
        inputs["input_ids"].shape[1],
        device=model.device,
        dtype=torch.long,
    )
    out = model(
        input_ids=inputs["input_ids"],
        past_key_values=past,
        use_cache=True,
        cache_position=prompt_cache_position,
        position_ids=prompt_cache_position.unsqueeze(0),
        output_attentions=True,
        return_dict=True,
    )

    kv = out.past_key_values
    kv_shape = get_first_layer_cache_shape(kv)
    print(f"[2/4] KV cache layers: {get_cache_layer_count(kv)}")
    print(f"       Per-layer shape: {kv_shape}")
    print("       (batch, num_kv_heads, seq_len, head_dim)")

    attns = out.attentions
    if not attns or attns[0] is None:
        raise RuntimeError(
            "Model did not return attentions during environment test. "
            "Set ModelConfig.attn_implementation='eager'."
        )
    print(f"[3/4] Attention layers: {len(attns)}")
    print(f"       Per-layer shape: {attns[0].shape}")
    print("       (batch, num_q_heads, q_len, kv_len)")

    seq_len = get_cache_seq_len(kv)
    keep = torch.arange(0, seq_len, 2, device=inputs["input_ids"].device)
    kv = prune_dynamic_cache(kv, keep)

    new_len = get_cache_seq_len(kv)
    print(f"[4/4] KV cache pruned: {seq_len} -> {new_len} (kept {new_len}/{seq_len})")

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    next_cache_position = torch.tensor([seq_len], device=model.device, dtype=torch.long)
    out2 = model(
        input_ids=next_token,
        past_key_values=kv,
        use_cache=True,
        cache_position=next_cache_position,
        position_ids=next_cache_position.unsqueeze(0),
        return_dict=True,
    )
    print(f"\nOK: forward after pruning succeeded, output shape: {out2.logits.shape}")
    print(f"   Next token: {tokenizer.decode(next_token[0])!r}")

    analyzer = SemanticAnalyzer(tokenizer)
    role_tags = analyzer.compute_role_tags(inputs["input_ids"][0])
    from semantic_analyzer import RoleTag

    for role in RoleTag:
        count = (role_tags == role.value).sum().item()
        if count > 0:
            print(f"   Role {role.name}: {count} tokens")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SemantiCache: KV Cache Eviction")
    parser.add_argument("--test", action="store_true", help="Run environment validation")
    parser.add_argument(
        "--policy",
        type=str,
        default="full",
        choices=["full", "window", "streaming", "h2o", "semantic"],
    )
    parser.add_argument("--budget", type=float, default=0.5, help="Cache budget ratio (0.1 ~ 1.0)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", type=str, default=None, help="Custom user prompt")
    args = parser.parse_args()

    if args.test:
        test_environment()
        return

    config = ExperimentConfig()
    config.cache.policy = args.policy
    config.cache.cache_budget = args.budget
    config.model.max_new_tokens = args.max_tokens

    model, tokenizer = load_model(config.model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    else:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": "My name is Kuo Li and I'm working on an OS kernel called NovaOS written in Rust.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Nice to meet you, Kuo Li! Working on NovaOS sounds like a fascinating project. "
                        "Building an OS kernel in Rust is a great choice given Rust's memory safety guarantees. "
                        "What specific aspects of NovaOS are you currently working on?"
                    ),
                },
                {
                    "role": "user",
                    "content": "I'm implementing a batch syscall mechanism. Can you tell me what's the name of my project again?",
                },
            ]
        )

    result = generate_with_eviction(model, tokenizer, messages, config)

    print(f"\n{'=' * 60}")
    print("Generated text:")
    print(f"{'=' * 60}")
    print(result["output_text"])


if __name__ == "__main__":
    main()
