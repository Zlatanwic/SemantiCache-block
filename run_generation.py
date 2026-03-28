"""
Entry point for text generation with manual KV-cache eviction.

This script avoids `model.generate()` so we can:
1. run a manual decode loop
2. inspect attentions
3. prune KV cache between decode steps
"""

import argparse
import sys
import time

import torch
from modelscope import snapshot_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

from attention_tracker import AttentionTracker
from config import CacheConfig, ExperimentConfig, ModelConfig
from eviction_policies import (
    DefensiveKVPolicy,
    FullCachePolicy,
    H2OPolicy,
    KVzipPolicy,
    LocalWindowPolicy,
    SemantiCachePolicy,
    SnapKVPolicy,
    TieredSemantiCachePolicy,
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


def _print_console_safe(text: str) -> None:
    """Print text using the active console encoding without crashing on Windows."""
    encoding = sys.stdout.encoding or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_text)


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


def build_policy(cache_cfg: CacheConfig, tracker: AttentionTracker, analyzer: SemanticAnalyzer, model_cfg=None):
    """Build the requested eviction policy."""
    budget_tokens = 512

    if cache_cfg.policy == "full":
        return FullCachePolicy()
    if cache_cfg.policy == "window":
        return LocalWindowPolicy(window_size=budget_tokens)
    if cache_cfg.policy == "streaming":
        return StreamingLLMPolicy(sink_tokens=cache_cfg.sink_tokens)
    if cache_cfg.policy == "h2o":
        return H2OPolicy(tracker=tracker)
    if cache_cfg.policy == "snapkv":
        return SnapKVPolicy(tracker=tracker)
    if cache_cfg.policy == "kvzip":
        return KVzipPolicy(tracker=tracker)
    if cache_cfg.policy == "defensivekv":
        return DefensiveKVPolicy(
            tracker=tracker,
            num_kv_heads=model_cfg.num_kv_heads if model_cfg else 2,
            num_attention_heads=model_cfg.num_attention_heads if model_cfg else 16,
        )
    if cache_cfg.policy == "semantic":
        return SemantiCachePolicy(
            tracker=tracker,
            analyzer=analyzer,
            alpha=cache_cfg.alpha,
            beta=cache_cfg.beta,
            gamma=cache_cfg.gamma,
            query_weight=cache_cfg.query_weight,
            factual_weight=cache_cfg.factual_weight,
            pin_system=cache_cfg.pin_system,
            pin_latest_user=cache_cfg.pin_latest_user,
            recent_window_size=cache_cfg.semantic_recent_window,
            hot_recent_window=cache_cfg.semantic_hot_recent_window,
            hot_block_size=cache_cfg.semantic_hot_block_size,
            block_size=cache_cfg.semantic_block_size,
            warm_promotable_reserve=cache_cfg.semantic_warm_promotable_reserve,
            latest_user_tail_tokens=cache_cfg.semantic_latest_user_tail_tokens,
            generated_retention_window=cache_cfg.semantic_generated_retention_window,
        )
    if cache_cfg.policy == "tiered_semantic":
        return TieredSemantiCachePolicy(
            tracker=tracker,
            analyzer=analyzer,
            alpha=cache_cfg.alpha,
            beta=cache_cfg.beta,
            gamma=cache_cfg.gamma,
            query_weight=cache_cfg.query_weight,
            factual_weight=cache_cfg.factual_weight,
            pin_system=cache_cfg.pin_system,
            pin_latest_user=cache_cfg.pin_latest_user,
            recent_window_size=cache_cfg.semantic_recent_window,
            hot_recent_window=cache_cfg.semantic_hot_recent_window,
            hot_block_size=cache_cfg.semantic_hot_block_size,
            block_size=cache_cfg.semantic_block_size,
            warm_promotable_reserve=cache_cfg.semantic_warm_promotable_reserve,
            latest_user_tail_tokens=cache_cfg.semantic_latest_user_tail_tokens,
            generated_retention_window=cache_cfg.semantic_generated_retention_window,
            hot_ratio=cache_cfg.semantic_hot_ratio,
        )

    raise ValueError(f"Unknown policy: {cache_cfg.policy}")


def _update_tracker_or_raise(
    attentions,
    tracker: AttentionTracker,
    policy_name: str,
    phase: str,
    kv_positions: torch.Tensor | None = None,
    new_token_position: int | None = None,
):
    if attentions and attentions[0] is not None:
        tracker.update(
            attentions,
            kv_positions=kv_positions,
            new_token_position=new_token_position,
        )
        return

    if policy_name in {"h2o", "semantic", "tiered_semantic"}:
        raise RuntimeError(
            f"Model did not return attentions during {phase}. "
            "Set ModelConfig.attn_implementation='eager'."
        )


def _extract_latest_user_query(messages: list[dict]) -> str:
    """Extract the actual question text from the latest user message when possible."""
    latest_user_content = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            latest_user_content = message.get("content", "")
            break

    if not latest_user_content:
        return ""

    markers = [
        "Now answer this question:",
        "Question:",
        "Q:",
    ]
    for marker in markers:
        if marker in latest_user_content:
            return latest_user_content.rsplit(marker, maxsplit=1)[-1].strip()

    return latest_user_content


def _decode_token_window(
    tokenizer,
    token_history: list[int],
    position: int,
    radius: int = 6,
) -> str:
    """Decode a compact token window around one logical cache position."""
    if not token_history:
        return ""

    start = max(0, position - radius)
    end = min(len(token_history), position + radius + 1)
    snippet = tokenizer.decode(token_history[start:end], skip_special_tokens=False)
    return " ".join(snippet.split())


def _print_promotion_debug(
    tokenizer,
    token_history: list[int],
    promoted_positions: torch.Tensor,
    step: int,
    top_n: int,
) -> None:
    """Print which warm positions were promoted for the upcoming decode step."""
    if promoted_positions.numel() == 0:
        print(f"  Promotion debug step {step}: no warm positions promoted")
        return

    print(f"  Promotion debug step {step}:")
    for position in promoted_positions[:top_n].tolist():
        snippet = _decode_token_window(tokenizer, token_history, int(position))
        _print_console_safe(f"    pos={position}: {snippet}")


def _print_tier_debug(
    tokenizer,
    token_history: list[int],
    hot_positions: torch.Tensor,
    warm_positions: torch.Tensor,
    step: int,
    top_n: int,
) -> None:
    """Print compact snapshots of hot and warm retained spans."""
    print(f"  Tier debug step {step}:")
    hot_preview = hot_positions[:top_n].tolist()
    warm_preview = warm_positions[:top_n].tolist()

    print("    hot:")
    for position in hot_preview:
        snippet = _decode_token_window(tokenizer, token_history, int(position))
        _print_console_safe(f"      pos={position}: {snippet}")

    print("    warm:")
    if not warm_preview:
        print("      <empty>")
        return
    for position in warm_preview:
        snippet = _decode_token_window(tokenizer, token_history, int(position))
        _print_console_safe(f"      pos={position}: {snippet}")


def _print_top_logits(tokenizer, logits: torch.Tensor, label: str, top_n: int) -> None:
    """Print the top next-token candidates for a logits tensor."""
    if top_n <= 0:
        return

    top_values, top_indices = logits[0].topk(min(top_n, logits.shape[-1]))
    print(f"  {label}:")
    for rank, (token_id, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
        token_text = tokenizer.decode([token_id]).replace("\n", "\\n")
        _print_console_safe(f"    {rank}. id={token_id} logit={value:.4f} text={token_text!r}")


def _build_polite_opener_token_ids(tokenizer) -> list[int]:
    """Return token ids for common polite opener starters."""
    opener_phrases = [
        "Of",
        "Sure",
        "Certainly",
        "Absolutely",
        "Yes",
        "当然",
        "Your",
        "The",
        "You",
    ]
    token_ids: set[int] = set()
    for phrase in opener_phrases:
        encoded = tokenizer.encode(phrase, add_special_tokens=False)
        if encoded:
            token_ids.add(int(encoded[0]))
    return sorted(token_ids)


def _mask_immediate_repeat_candidates(
    tokenizer,
    logits: torch.Tensor,
    previous_token_id: int,
    scan_top_n: int = 256,
) -> torch.Tensor:
    """Mask top repeat-like candidates whose stripped text matches the previous token."""
    previous_text = tokenizer.decode([previous_token_id]).strip()
    if not previous_text:
        return logits

    masked_logits = logits.clone()
    top_n = min(scan_top_n, masked_logits.shape[-1])
    _, top_indices = masked_logits[0].topk(top_n)
    repeat_ids: list[int] = []
    for token_id in top_indices.tolist():
        candidate_text = tokenizer.decode([token_id]).strip()
        if candidate_text == previous_text:
            repeat_ids.append(token_id)

    if repeat_ids:
        masked_logits[:, repeat_ids] = -torch.inf
    return masked_logits


def _collect_follow_token_counts(
    tokenizer,
    token_history: list[int],
    source_positions: torch.Tensor | None,
    previous_token_id: int,
) -> dict[int, int]:
    """Collect retained continuation tokens that followed the previous token."""
    if source_positions is None or source_positions.numel() == 0 or not token_history:
        return {}

    retained_positions = {
        int(position)
        for position in source_positions.detach().cpu().tolist()
        if 0 <= int(position) < len(token_history)
    }
    if not retained_positions:
        return {}

    token_text_cache: dict[int, str] = {}

    def normalized_token_text(token_id: int) -> str:
        if token_id not in token_text_cache:
            token_text_cache[token_id] = tokenizer.decode([token_id]).strip()
        return token_text_cache[token_id]

    previous_token_text = normalized_token_text(previous_token_id)
    previous_raw_text = tokenizer.decode([previous_token_id])
    follower_counts: dict[int, int] = {}
    for position in retained_positions:
        next_position = position + 1
        current_token_id = int(token_history[position])
        current_token_text = normalized_token_text(current_token_id)
        exact_or_normalized_match = current_token_id == previous_token_id or (
            previous_token_text and current_token_text == previous_token_text
        )
        if exact_or_normalized_match and next_position in retained_positions:
            follower_token_id = int(token_history[next_position])
            follower_counts[follower_token_id] = follower_counts.get(follower_token_id, 0) + 1

        # If the retained prompt token is a longer fused token, allow the model to
        # continue into its suffix, e.g. prompt " Rust" vs generated "R" -> "ust".
        if (
            previous_token_text
            and current_token_text
            and current_token_text.startswith(previous_token_text)
            and len(current_token_text) > len(previous_token_text)
        ):
            suffix_text = current_token_text[len(previous_token_text) :]
            suffix_token_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
            if suffix_token_ids:
                follower_token_id = int(suffix_token_ids[0])
                follower_counts[follower_token_id] = follower_counts.get(follower_token_id, 0) + 1
            continue

    # Narrow fallback for date/number-like chains when the retained span loses
    # one link, e.g. "March 15" where " " or "1" may drop out between steps.
    if not follower_counts and (previous_raw_text == " " or previous_raw_text.isdigit()):
        # Scan only a window around retained positions instead of entire history
        scan_positions = sorted(retained_positions)
        for position in scan_positions:
            if position + 1 >= len(token_history):
                continue
            current_token_id = int(token_history[position])
            current_token_text = normalized_token_text(current_token_id)
            exact_or_normalized_match = current_token_id == previous_token_id or (
                previous_token_text and current_token_text == previous_token_text
            )
            if not exact_or_normalized_match:
                continue
            follower_token_id = int(token_history[position + 1])
            follower_counts[follower_token_id] = follower_counts.get(follower_token_id, 0) + 1

    return follower_counts


def _apply_follow_token_bias(
    follower_counts: dict[int, int],
    logits: torch.Tensor,
    bias: float,
) -> torch.Tensor:
    """Bias logits toward retained continuation tokens that followed the previous token."""
    if bias <= 0 or not follower_counts:
        return logits

    if not follower_counts:
        return logits

    biased_logits = logits.clone()
    for follower_token_id, count in follower_counts.items():
        biased_logits[0, follower_token_id] += bias * count
    return biased_logits


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
    if cache_cfg.policy == "tiered_semantic":
        print(
            f"Hot ratio: {cache_cfg.semantic_hot_ratio:.2f}, "
            f"Warm top-k: {cache_cfg.semantic_warm_top_k}"
        )
        if cache_cfg.semantic_follow_token_bias > 0:
            print(f"Follow-token bias: {cache_cfg.semantic_follow_token_bias:.2f}")
    decode_mode = "greedy" if not model_cfg.do_sample else f"sample(temp={model_cfg.temperature:.2f})"
    print(f"Decoding: {decode_mode}")
    if model_cfg.mask_polite_openers:
        print("First-token mask: polite openers enabled")
    if model_cfg.demo_stop_on_repeat:
        print("Demo repeat-stop: enabled")
    stop_substrings = [item.lower() for item in model_cfg.stop_when_output_contains if item]
    if stop_substrings:
        print(f"Early stop on output match: {stop_substrings}")
    verbose_decode_progress = model_cfg.max_new_tokens <= 16
    decode_progress_bar = None
    if model_cfg.show_progress_bar:
        decode_progress_bar = tqdm(
            total=model_cfg.max_new_tokens,
            desc="Decode",
            leave=False,
            dynamic_ncols=True,
        )

    tracker = AttentionTracker(
        num_layers=model_cfg.num_layers,
        num_kv_heads=model_cfg.num_kv_heads,
    )
    analyzer = SemanticAnalyzer(tokenizer)
    policy = build_policy(cache_cfg, tracker, analyzer, model_cfg)

    cache_manager = KVCacheManager(
        config=cache_cfg,
        policy=policy,
        tracker=tracker,
        num_layers=model_cfg.num_layers,
        model_config=model.config,
    )

    t_start = time.time()
    past_key_values = DynamicCache(config=model.config)
    prompt_cache_position = torch.arange(prompt_len, device=model.device, dtype=torch.long)
    print("Prefill: running prompt forward pass...")
    prefill_start = time.time()

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        cache_position=prompt_cache_position,
        position_ids=prompt_cache_position.unsqueeze(0),
        output_attentions=True,
        return_dict=True,
    )
    print(f"Prefill: done in {time.time() - prefill_start:.2f}s")

    _update_tracker_or_raise(outputs.attentions, tracker, cache_cfg.policy, "prefill")
    cache_manager.set_initial_seq_len(prompt_len)

    if isinstance(policy, SemantiCachePolicy):
        policy.setup_semantic_signals(
            input_ids[0],
            latest_query_text=_extract_latest_user_query(messages),
        )
    if isinstance(policy, SnapKVPolicy):
        policy.snapshot_prefill_attention()
    if isinstance(policy, KVzipPolicy):
        policy.snapshot_prefill_attention(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids[0],
            past_key_values=outputs.past_key_values,
        )
    if isinstance(policy, DefensiveKVPolicy):
        policy.snapshot_prefill_attention(attentions=outputs.attentions)

    next_token_logits = outputs.logits[:, -1, :]
    if cache_cfg.semantic_debug_logits_top_n > 0:
        _print_top_logits(
            tokenizer,
            next_token_logits,
            label="Prefill next-token logits (policy-independent)",
            top_n=cache_cfg.semantic_debug_logits_top_n,
        )
    past_key_values = outputs.past_key_values

    generated_ids = []
    logical_token_history = input_ids[0].detach().cpu().tolist()
    # Track the original absolute sequence position for correct RoPE after eviction.
    # cache_position (physical write index) and position_ids (RoPE) must diverge
    # once tokens are evicted: RoPE must continue from the original sequence length.
    next_absolute_position = prompt_len
    eos_token_id = tokenizer.eos_token_id
    polite_opener_token_ids = _build_polite_opener_token_ids(tokenizer) if model_cfg.mask_polite_openers else []
    try:
        for step in range(model_cfg.max_new_tokens):
            step_start = time.time()
            if verbose_decode_progress:
                print(f"Decode step {step + 1}/{model_cfg.max_new_tokens}: selecting next token...")
            if decode_progress_bar is not None:
                decode_progress_bar.set_postfix_str(f"step={step + 1}/{model_cfg.max_new_tokens}")
            current_token_logits = next_token_logits
            if step == 0 and polite_opener_token_ids:
                current_token_logits = current_token_logits.clone()
                current_token_logits[:, polite_opener_token_ids] = -torch.inf
                if cache_cfg.semantic_debug_logits_top_n > 0:
                    _print_top_logits(
                        tokenizer,
                        current_token_logits,
                        label="Prefill next-token logits after polite-opener mask",
                        top_n=cache_cfg.semantic_debug_logits_top_n,
                    )
            if (
                cache_cfg.policy == "tiered_semantic"
                and step > 0
                and generated_ids
                and cache_cfg.semantic_follow_token_bias > 0
            ):
                follow_token_counts = _collect_follow_token_counts(
                    tokenizer=tokenizer,
                    token_history=logical_token_history,
                    source_positions=cache_manager.last_prepared_origin_positions,
                    previous_token_id=generated_ids[-1],
                )
                current_token_logits = _apply_follow_token_bias(
                    follower_counts=follow_token_counts,
                    logits=current_token_logits,
                    bias=cache_cfg.semantic_follow_token_bias,
                )
                if cache_cfg.semantic_debug_logits_top_n > 0:
                    if follow_token_counts:
                        print("  Follow-token candidates:")
                        sorted_candidates = sorted(
                            follow_token_counts.items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                        for follower_token_id, count in sorted_candidates[: cache_cfg.semantic_debug_logits_top_n]:
                            token_text = tokenizer.decode([follower_token_id]).replace("\n", "\\n")
                            _print_console_safe(
                                f"    id={follower_token_id} count={count} text={token_text!r}"
                            )
                    else:
                        print("  Follow-token candidates: <none>")
                    _print_top_logits(
                        tokenizer,
                        current_token_logits,
                        label=f"Decode step {step + 1} logits after follow-token bias",
                        top_n=cache_cfg.semantic_debug_logits_top_n,
                    )
            if model_cfg.demo_stop_on_repeat and generated_ids:
                current_token_logits = _mask_immediate_repeat_candidates(
                    tokenizer,
                    current_token_logits,
                    previous_token_id=generated_ids[-1],
                )

            if model_cfg.do_sample:
                probs = torch.softmax(current_token_logits / model_cfg.temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = current_token_logits.argmax(dim=-1, keepdim=True)

            token_id = next_token_id.item()
            if model_cfg.demo_stop_on_repeat and generated_ids and token_id == generated_ids[-1]:
                break
            generated_ids.append(token_id)
            if decode_progress_bar is not None:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
                decode_progress_bar.update(1)
                decode_progress_bar.set_postfix_str(
                    f"step={step + 1}/{model_cfg.max_new_tokens}, token={token_text!r}"
                )

            if token_id == eos_token_id:
                if verbose_decode_progress:
                    print(f"Decode step {step + 1}/{model_cfg.max_new_tokens}: hit EOS in {time.time() - step_start:.2f}s")
                break

            if stop_substrings:
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
                if any(stop_text in generated_text for stop_text in stop_substrings):
                    if verbose_decode_progress:
                        print(
                            f"Decode step {step + 1}/{model_cfg.max_new_tokens}: "
                            f"matched early-stop text in {time.time() - step_start:.2f}s"
                        )
                    break

            if step == 0 or not cache_manager.uses_tiered_cache:
                past_key_values = cache_manager.evict(past_key_values)
            if cache_cfg.semantic_debug_tiers and step == 0:
                _print_tier_debug(
                    tokenizer,
                    logical_token_history,
                    cache_manager.origin_positions[cache_manager.hot_positions],
                    cache_manager.origin_positions[cache_manager.warm_tier.positions]
                    if cache_manager.warm_tier.size > 0
                    else torch.empty(0, dtype=torch.long),
                    step + 1,
                    cache_cfg.semantic_debug_tiers_top_n,
                )
            model_past_key_values = cache_manager.prepare_past_key_values(past_key_values)
            if (
                cache_cfg.semantic_debug_promotion
                and cache_manager.last_promoted_positions.numel() > 0
                and ((step + 1) <= 3 or (step + 1) % max(1, cache_cfg.semantic_debug_promotion_every) == 0)
            ):
                _print_promotion_debug(
                    tokenizer,
                    logical_token_history,
                    cache_manager.last_promoted_origin_positions,
                    step + 1,
                    cache_cfg.semantic_debug_promotion_top_n,
                )
            # Physical write position in the (possibly pruned) cache
            next_cache_position = torch.tensor(
                [cache_manager.current_cache_len],
                device=model.device,
                dtype=torch.long,
            )
            # Absolute position for RoPE — continues from original sequence length
            next_position_id = torch.tensor(
                [next_absolute_position],
                device=model.device,
                dtype=torch.long,
            )

            if isinstance(policy, SemantiCachePolicy):
                policy.extend_signals(num_new_tokens=1)

            outputs = model(
                input_ids=next_token_id,
                past_key_values=model_past_key_values,
                use_cache=True,
                cache_position=next_cache_position,
                position_ids=next_position_id.unsqueeze(0),
                output_attentions=True,
                return_dict=True,
            )
            next_absolute_position += 1

            _update_tracker_or_raise(
                outputs.attentions,
                tracker,
                cache_cfg.policy,
                "decode",
                kv_positions=cache_manager.last_prepared_positions,
                new_token_position=cache_manager.current_cache_len,
            )
            next_token_logits = outputs.logits[:, -1, :]
            if cache_cfg.semantic_debug_logits_top_n > 0 and step == 0:
                _print_top_logits(
                    tokenizer,
                    next_token_logits,
                    label="Decode step 1 next-token logits (policy-dependent)",
                    top_n=cache_cfg.semantic_debug_logits_top_n,
                )
            past_key_values = cache_manager.finalize_decode_step(
                past_key_values,
                outputs.past_key_values,
            )
            logical_token_history.append(token_id)

            if verbose_decode_progress:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
                print(
                    f"Decode step {step + 1}/{model_cfg.max_new_tokens}: "
                    f"token={token_text!r} id={token_id} in {time.time() - step_start:.2f}s"
                )

            if (step + 1) % 50 == 0:
                stats = cache_manager.get_stats()
                if cache_cfg.policy == "tiered_semantic":
                    print(
                        f"  Step {step + 1}: hot={stats['hot_cache_len']}, warm={stats['warm_cache_len']}, "
                        f"cold={stats['cold_cache_len']}, evicted={stats['total_evicted']}, "
                        f"promoted={stats['last_promoted_warm_count']}/{stats['warm_top_k']}, "
                        f"quant_s={stats['warm_quantize_time_s']:.3f}, "
                        f"dequant_s={stats['warm_dequantize_time_s']:.3f}"
                    )
                else:
                    print(
                        f"  Step {step + 1}: cache_len={stats['current_cache_len']}, "
                        f"evicted={stats['total_evicted']}"
                    )
    finally:
        if decode_progress_bar is not None:
            decode_progress_bar.close()

    elapsed = time.time() - t_start
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stats = cache_manager.get_stats()

    print(
        f"\nGeneration complete: {len(generated_ids)} tokens in {elapsed:.2f}s "
        f"({len(generated_ids) / elapsed:.1f} tok/s)"
    )
    print(f"Eviction stats: {stats}")
    if cache_cfg.policy == "tiered_semantic":
        print(
            "Tier summary: "
            f"hot={stats['hot_cache_len']}/{stats['hot_budget_tokens']}, "
            f"warm={stats['warm_cache_len']}, cold={stats['cold_cache_len']}, "
            f"peak_hot={stats['peak_hot_cache_len']}, peak_warm={stats['peak_warm_cache_len']}, "
            f"peak_cold={stats['peak_cold_cache_len']}"
        )
        print(
            "Quantization overhead: "
            f"quant={stats['warm_quantize_time_s']:.4f}s ({stats['warm_quantize_ops']} ops), "
            f"dequant={stats['warm_dequantize_time_s']:.4f}s ({stats['warm_dequantize_ops']} ops)"
        )
        print(
            "Warm promotion: "
            f"top_k={stats['warm_top_k']}, last={stats['last_promoted_warm_count']}, "
            f"peak={stats['peak_promoted_warm_count']}, steps={stats['promotion_steps']}"
        )

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
        choices=["full", "window", "streaming", "h2o", "semantic", "tiered_semantic"],
    )
    parser.add_argument("--budget", type=float, default=0.5, help="Cache budget ratio (0.1 ~ 1.0)")
    parser.add_argument(
        "--hot-ratio",
        type=float,
        default=0.5,
        help="Hot-tier ratio within the total retained budget for tiered_semantic",
    )
    parser.add_argument(
        "--warm-top-k",
        type=int,
        default=16,
        help="How many warm-tier tokens to promote per decode step for tiered_semantic",
    )
    parser.add_argument("--debug-promotion", action="store_true", help="Print promoted warm snippets during decoding")
    parser.add_argument("--debug-promotion-every", type=int, default=25, help="Promotion debug print interval")
    parser.add_argument("--debug-promotion-top-n", type=int, default=4, help="How many promoted snippets to print each time")
    parser.add_argument("--debug-tiers", action="store_true", help="Print hot/warm retained snippets after the first eviction")
    parser.add_argument("--debug-tiers-top-n", type=int, default=8, help="How many hot/warm snippets to print")
    parser.add_argument("--debug-logits-top-n", type=int, default=0, help="Print top logits for prefill and first decode step")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--mask-polite-openers", action="store_true", help="Mask common polite opener tokens for the first generated token")
    parser.add_argument(
        "--follow-token-bias",
        type=float,
        default=None,
        help="Bias next-token logits toward retained continuation tokens in tiered_semantic",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", type=str, default=None, help="Custom user prompt")
    args = parser.parse_args()

    if args.test:
        test_environment()
        return

    config = ExperimentConfig()
    config.cache.policy = args.policy
    config.cache.cache_budget = args.budget
    config.cache.semantic_hot_ratio = args.hot_ratio
    config.cache.semantic_warm_top_k = args.warm_top_k
    config.cache.semantic_debug_promotion = args.debug_promotion
    config.cache.semantic_debug_promotion_every = args.debug_promotion_every
    config.cache.semantic_debug_promotion_top_n = args.debug_promotion_top_n
    config.cache.semantic_debug_tiers = args.debug_tiers
    config.cache.semantic_debug_tiers_top_n = args.debug_tiers_top_n
    config.cache.semantic_debug_logits_top_n = args.debug_logits_top_n
    if args.follow_token_bias is not None:
        config.cache.semantic_follow_token_bias = args.follow_token_bias
    if args.temperature is not None:
        config.model.temperature = args.temperature
    if args.greedy:
        config.model.do_sample = False
    if args.mask_polite_openers:
        config.model.mask_polite_openers = True
    if not args.prompt:
        config.model.demo_stop_on_repeat = True
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
                    "content": (
                        "I'm implementing a batch syscall mechanism. "
                        "What is the name of my project? Answer with only the project name."
                    ),
                },
            ]
        )

    result = generate_with_eviction(model, tokenizer, messages, config)

    print(f"\n{'=' * 60}")
    print("Generated text:")
    print(f"{'=' * 60}")
    _print_console_safe(result["output_text"])


if __name__ == "__main__":
    main()
