"""Eviction policy implementations for KV-cache pruning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from attention_tracker import AttentionTracker
from op_policy_model import FEATURE_DIM, load_policy_checkpoint
from semantic_analyzer import RoleTag, SemanticAnalyzer


@dataclass
class KVTipStats:
    """Per-decision diagnostics for OP-SieveKV-style retention training."""

    entropy: torch.Tensor
    divergence: torch.Tensor
    soft_or: torch.Tensor
    quadrant: torch.Tensor


class EvictionPolicy(ABC):
    """Base class for all cache-eviction policies."""

    @abstractmethod
    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        """
        Return per-token eviction scores.

        Higher scores mean a token is more likely to be evicted.
        `torch.inf` means evict first. `-torch.inf` means never evict.
        """

    def select_eviction_indices(self, scores: torch.Tensor, num_to_evict: int) -> torch.Tensor:
        """Select the token indices with the largest eviction scores."""
        _, indices = scores.topk(num_to_evict, largest=True)
        return indices.sort().values

    def select_keep_indices(self, scores: torch.Tensor, budget: int) -> torch.Tensor:
        """Select the token indices to keep under a fixed budget."""
        _, keep_indices = scores.topk(budget, largest=False)
        return keep_indices.sort().values

    def compute_promotion_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        """Return per-token scores for warm-to-hot promotion. Higher is better."""
        return -self.compute_eviction_scores(seq_len, **kwargs)

    def compute_hot_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        """Return per-token scores for hot-tier selection. Higher is better."""
        return self.compute_promotion_scores(seq_len, **kwargs)


class FullCachePolicy(EvictionPolicy):
    """Upper-bound baseline: never evict any token."""

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        return torch.full((seq_len,), -torch.inf)


class LocalWindowPolicy(EvictionPolicy):
    """Keep only the most recent `window_size` tokens."""

    def __init__(self, window_size: int):
        self.window_size = window_size

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        scores = torch.zeros(seq_len)
        if seq_len > self.window_size:
            scores[: seq_len - self.window_size] = torch.inf
        return scores


class StreamingLLMPolicy(EvictionPolicy):
    """Keep sink tokens plus a recent sliding window.

    Eviction scores use a graded scheme so that `select_keep_indices(budget)`
    naturally keeps sinks first, then the most-recent tokens, regardless of
    the exact budget value.

    Reference: Xiao et al., "Efficient Streaming Language Models with
    Attention Sinks", ICLR 2024.
    """

    def __init__(self, sink_tokens: int = 4):
        self.sink_tokens = sink_tokens

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        # Lower score = keep. Sinks get -inf, recent tokens get linearly
        # decreasing scores (most recent = lowest), middle tokens get +inf.
        scores = torch.full((seq_len,), torch.inf)
        # Sinks: always keep
        sink_end = min(self.sink_tokens, seq_len)
        scores[:sink_end] = -torch.inf
        # Remaining tokens: recency gradient (newest = lowest eviction score)
        if seq_len > sink_end:
            remaining = seq_len - sink_end
            # Score from (remaining-1) down to 0 — newest token gets 0
            scores[sink_end:] = torch.arange(remaining - 1, -1, -1, dtype=torch.float)
        return scores


class H2OPolicy(EvictionPolicy):
    """Heavy-Hitter Oracle baseline based on cumulative attention scores."""

    def __init__(self, tracker: AttentionTracker):
        self.tracker = tracker

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        cumulative = self.tracker.get_cumulative_scores()
        if len(cumulative) == 0 or len(cumulative) < seq_len:
            return torch.zeros(seq_len)

        cumulative = cumulative[:seq_len]
        max_score = cumulative.max().clamp(min=1e-10)
        return 1.0 - (cumulative / max_score)


class KVzipPolicy(EvictionPolicy):
    """KVzip baseline: reconstruction-based one-shot scoring.

    After prefill, runs a context-reconstruction forward pass ("Repeat the
    previous context:") and scores each KV pair by the maximum attention it
    receives across all query positions, layers, and heads.

    Reference: He et al., "KVzip: Query-Agnostic KV Cache Compression with
    Context Reconstruction", NeurIPS 2025.
    Official code: https://github.com/snu-mllab/KVzip
    """

    def __init__(
        self,
        tracker: AttentionTracker,
        chunk_size: int = 2048,
    ):
        self.tracker = tracker
        self.chunk_size = chunk_size
        self._prefill_scores: Optional[torch.Tensor] = None

    def snapshot_prefill_attention(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        past_key_values,
    ) -> None:
        """Run the KVzip reconstruction self-task and compute importance scores.

        Follows Algorithm 1 from the paper:
        1. Partition context into chunks of size `chunk_size`
        2. For each chunk, prepend a repeat prompt and forward through the model
           using the prefilled KV cache (without updating it)
        3. Score each KV pair as the max attention weight it receives across all
           query positions and GQA groups, then aggregate across layers/heads
           to produce a 1D shared-index importance vector.

        Args:
            model: The language model (for running the reconstruction forward pass).
            tokenizer: Tokenizer (for encoding the repeat prompt).
            input_ids: The original prefill input_ids, shape [seq_len].
            past_key_values: The prefilled KV cache (will NOT be modified).
        """
        device = input_ids.device
        context_ids = input_ids  # shape [seq_len]
        n_c = context_ids.shape[0]
        num_layers = self.tracker.num_layers
        num_kv_heads = self.tracker.num_kv_heads

        # Score tensor: [L, H, n_c]
        S = torch.zeros(num_layers, num_kv_heads, n_c, device=device, dtype=torch.float32)

        # Partition context into chunks
        chunk_size = min(self.chunk_size, n_c)
        chunks = []
        for start in range(0, n_c, chunk_size):
            end = min(start + chunk_size, n_c)
            chunks.append((start, end, context_ids[start:end]))

        for t, (c_start, c_end, chunk_ids) in enumerate(chunks):
            # Build repeat prompt
            if t == 0:
                prompt_text = "Repeat the previous context:"
            else:
                # Use last 8 tokens of preceding chunk as anchor
                prev_start = max(0, c_start - 8)
                anchor_ids = context_ids[prev_start:c_start]
                anchor_text = tokenizer.decode(anchor_ids, skip_special_tokens=True)
                prompt_text = f"Repeat the previous context starting with {anchor_text}:"

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, device=device)
            else:
                prompt_ids = prompt_ids.to(device)
            if prompt_ids.dim() == 2:
                prompt_ids = prompt_ids[0]

            # Reconstruction input = repeat prompt + chunk context (teacher-forced)
            recon_ids = torch.cat([prompt_ids, chunk_ids], dim=0).unsqueeze(0)  # [1, n_in]
            n_in = recon_ids.shape[1]
            m = chunk_ids.shape[0]  # chunk length

            # Position ids continue from where the prefill ended
            recon_position = torch.arange(
                n_c, n_c + n_in, device=device, dtype=torch.long
            ).unsqueeze(0)
            recon_cache_position = torch.arange(
                n_c, n_c + n_in, device=device, dtype=torch.long
            )

            # Forward pass with the prefilled cache.
            # Save original tensor references so we can restore after the
            # reconstruction pass (torch.cat in DynamicLayer.update creates
            # NEW tensors, so originals are preserved).
            saved_keys = []
            saved_values = []
            if hasattr(past_key_values, 'layers'):
                for layer_obj in past_key_values.layers:
                    saved_keys.append(layer_obj.keys)
                    saved_values.append(layer_obj.values)
            elif hasattr(past_key_values, 'key_cache'):
                for li in range(len(past_key_values.key_cache)):
                    saved_keys.append(past_key_values.key_cache[li])
                    saved_values.append(past_key_values.value_cache[li])

            recon_out = model(
                input_ids=recon_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=recon_cache_position,
                position_ids=recon_position,
                output_attentions=True,
                return_dict=True,
            )

            # Extract scores from attention: for each layer, get max attention
            # each context KV pair receives from the reconstruction queries.
            for layer_idx, attn in enumerate(recon_out.attentions):
                # attn shape: [batch, num_attention_heads, n_in, n_c + n_in]
                # We want attention to the original context positions [0, n_c)
                # from our reconstruction query positions.
                attn_to_context = attn[0, :, :, :n_c]  # [num_attn_heads, n_in, n_c]

                # Only keep attention to the current chunk's positions
                attn_to_chunk = attn_to_context[:, :, c_start:c_end]  # [num_attn_heads, n_in, m]

                # Group attention heads into KV head groups and take max across GQA groups
                num_attn_heads = attn_to_chunk.shape[0]
                group_size = num_attn_heads // num_kv_heads
                attn_grouped = attn_to_chunk.reshape(
                    num_kv_heads, group_size, n_in, m
                )
                # Max across GQA groups and query positions → [H, m]
                chunk_scores = attn_grouped.amax(dim=(1, 2))
                S[layer_idx, :, c_start:c_end] = torch.max(
                    S[layer_idx, :, c_start:c_end], chunk_scores.float()
                )

            # Restore original cache tensors (undo reconstruction tokens).
            if hasattr(past_key_values, 'layers'):
                for i, layer_obj in enumerate(past_key_values.layers):
                    layer_obj.keys = saved_keys[i]
                    layer_obj.values = saved_values[i]
            elif hasattr(past_key_values, 'key_cache'):
                for li in range(len(past_key_values.key_cache)):
                    past_key_values.key_cache[li] = saved_keys[li]
                    past_key_values.value_cache[li] = saved_values[li]

            del recon_out, saved_keys, saved_values

        # Aggregate across layers and heads to get 1D shared-index scores.
        # Use mean: max is too flat (every token has at least one head giving
        # high attention), mean preserves discrimination across the full
        # layer×head landscape.
        importance_1d = S.mean(dim=(0, 1))  # [n_c]

        self._prefill_scores = importance_1d
        print(f"KVzip: reconstruction scoring done, {n_c} positions, "
              f"{len(chunks)} chunk(s), "
              f"score range [{importance_1d.min():.4f}, {importance_1d.max():.4f}]")

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        """Return eviction scores: high = evict first."""
        if self._prefill_scores is None or len(self._prefill_scores) == 0:
            return torch.zeros(seq_len)

        scores = self._prefill_scores
        if len(scores) < seq_len:
            # Pad with max importance so generated tokens are protected
            extra = torch.full((seq_len - len(scores),), scores.max(), device=scores.device)
            scores = torch.cat([scores, extra])
        elif len(scores) > seq_len:
            scores = scores[:seq_len]

        max_score = scores.max().clamp(min=1e-10)
        eviction = 1.0 - (scores / max_score)
        # Protect generated tokens from decode-time re-eviction (one-shot method)
        if seq_len > len(self._prefill_scores):
            eviction[len(self._prefill_scores):] = -torch.inf
        return eviction


class SnapKVPolicy(EvictionPolicy):
    """SnapKV baseline: one-shot prefill-time pruning via observation window.

    After prefill, uses attention from the last `observation_window` query tokens
    to score each KV position. Positions with low attention are evicted.

    Reference: Li et al., "SnapKV: LLM Knows What You Are Looking For Before
    Generation", ICML 2024.
    """

    def __init__(
        self,
        tracker: AttentionTracker,
        observation_window: int = 64,
        kernel_size: int = 5,
    ):
        self.tracker = tracker
        self.observation_window = observation_window
        self.kernel_size = kernel_size
        self._prefill_scores: Optional[torch.Tensor] = None

    def snapshot_prefill_attention(self) -> None:
        """Capture attention scores from the observation window at end of prefill.

        Called once after the prefill forward pass. Computes per-position
        importance as the sum of attention received from the last
        `observation_window` query positions, averaged across all heads/layers.
        """
        cumulative = self.tracker.get_cumulative_scores()
        if cumulative is None or len(cumulative) == 0:
            return
        # cumulative already sums over decode steps; for SnapKV we want the
        # prefill-time attention which is the first (and only) update so far.
        # Apply optional pooling to smooth the scores.
        scores = cumulative.clone().float()
        if self.kernel_size > 1 and scores.numel() >= self.kernel_size:
            pad = self.kernel_size // 2
            padded = torch.nn.functional.pad(
                scores.unsqueeze(0).unsqueeze(0), (pad, pad), mode="reflect"
            )
            kernel = torch.ones(1, 1, self.kernel_size, device=scores.device) / self.kernel_size
            scores = torch.nn.functional.conv1d(padded, kernel).squeeze()
        self._prefill_scores = scores

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        """Return eviction scores: high = evict first."""
        if self._prefill_scores is None or len(self._prefill_scores) == 0:
            return torch.zeros(seq_len)

        scores = self._prefill_scores
        if len(scores) < seq_len:
            extra = torch.full((seq_len - len(scores),), scores.max(), device=scores.device)
            scores = torch.cat([scores, extra])
        elif len(scores) > seq_len:
            scores = scores[:seq_len]

        max_score = scores.max().clamp(min=1e-10)
        eviction = 1.0 - (scores / max_score)
        # Protect generated tokens from decode-time re-eviction (one-shot method)
        if seq_len > len(self._prefill_scores):
            eviction[len(self._prefill_scores):] = -torch.inf
        return eviction


class DefensiveKVPolicy(EvictionPolicy):
    """DefensiveKV baseline: defensive aggregation with observation window.

    Faithfully follows the official implementation:
    1. Extract attention from the last `window_size` queries to all keys
    2. Average across GQA groups, apply avg_pool1d smoothing per head
    3. Defensive mechanism: max across heads, clamp to mean as floor
    4. Observation window tokens are always protected

    Reference: Fan et al., "DefensiveKV: Optimizing KV Cache Eviction for
    LLMs via Defensive Aggregation", ICLR 2026.
    Official code: https://github.com/FFY0/DefensiveKV
    """

    def __init__(
        self,
        tracker: AttentionTracker,
        kernel_size: int = 5,
        window_size: int = 32,
        num_kv_heads: int = 2,
        num_attention_heads: int = 16,
    ):
        self.tracker = tracker
        self.kernel_size = kernel_size
        self.window_size = window_size
        self.num_kv_heads = num_kv_heads
        self.num_attention_heads = num_attention_heads
        self.num_kv_groups = num_attention_heads // num_kv_heads
        self._prefill_scores: Optional[torch.Tensor] = None
        self._is_one_shot = True  # only evict once after prefill

    def snapshot_prefill_attention(
        self, attentions: tuple[torch.Tensor, ...] | None = None
    ) -> None:
        """Compute defensive scores from raw prefill attention matrices.

        Follows the official DefensiveKV implementation:
        1. Extract attention from last window_size queries to all preceding keys
        2. Operate at full attention-head granularity (not KV head) for max diversity
        3. Pool with avg_pool1d, then max across heads + clamp to mean
        4. Protect observation window tokens

        Args:
            attentions: tuple of per-layer attention tensors, each with shape
                (batch, num_attention_heads, seq_len, seq_len).
        """
        if attentions is None or len(attentions) == 0:
            return

        device = attentions[0].device
        seq_len = attentions[0].shape[-1]
        window = min(self.window_size, seq_len)
        kv_len = seq_len - window
        if kv_len <= 0:
            return

        # Collect per-attention-head window scores across all layers.
        # Use full attention heads (16) not KV heads (2) for better diversity.
        all_layer_head_scores = []  # list of (num_attn_heads, kv_len)

        for attn in attentions:
            # attn: (1, num_attn_heads, seq_len, seq_len)
            # Last `window` queries attending to first `kv_len` keys
            window_attn = attn[0, :, -window:, :kv_len]  # (num_attn_heads, window, kv_len)

            # Average across window queries -> (num_attn_heads, kv_len)
            head_scores = window_attn.mean(dim=1)
            all_layer_head_scores.append(head_scores)

        # Average across layers -> (num_attn_heads, kv_len)
        head_scores = torch.stack(all_layer_head_scores, dim=0).mean(dim=0).float()

        # Apply avg_pool1d smoothing per head (following official impl)
        if self.kernel_size > 1 and kv_len >= self.kernel_size:
            pad = self.kernel_size // 2
            padded = torch.nn.functional.pad(
                head_scores.unsqueeze(1), (pad, pad), mode="reflect"
            )
            kernel = torch.ones(1, 1, self.kernel_size, device=device) / self.kernel_size
            smoothed = torch.nn.functional.conv1d(padded, kernel)
            head_scores = smoothed.squeeze(1)

        # --- Defensive mechanism (core of DefensiveKV) ---
        # Max across ALL attention heads (16 heads, not 2 KV heads)
        max_scores = head_scores.max(dim=0).values  # (kv_len,)
        # Clamp to mean as floor
        defensive_scores = max_scores.clamp(min=max_scores.mean().item())

        # Append window tokens with max score (always protected, per official impl)
        window_protection = torch.full(
            (window,), defensive_scores.max().item(), device=device
        )
        self._prefill_scores = torch.cat([defensive_scores, window_protection])

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        if self._prefill_scores is None or len(self._prefill_scores) == 0:
            return torch.zeros(seq_len)

        scores = self._prefill_scores
        if len(scores) < seq_len:
            # Newly generated tokens get -inf (never evict) — one-shot method
            extra = torch.full((seq_len - len(scores),), -1.0, device=scores.device)
            scores = torch.cat([scores, extra])
        elif len(scores) > seq_len:
            scores = scores[:seq_len]

        max_score = scores.max().clamp(min=1e-10)
        # Invert: high attention -> low eviction score (keep)
        eviction = 1.0 - (scores / max_score)
        # Protect generated tokens from decode-time re-eviction
        if seq_len > len(self._prefill_scores):
            eviction[len(self._prefill_scores):] = -torch.inf
        return eviction


class SemantiCachePolicy(EvictionPolicy):
    """
    SemantiCache: combine attention, information density, and head-entropy signals.

    On top of semantic role protection, this policy keeps a recent decode window
    to avoid collapsing generation quality when the cache budget is capped.
    """

    def __init__(
        self,
        tracker: AttentionTracker,
        analyzer: SemanticAnalyzer,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        query_weight: float = 0.25,
        factual_weight: float = 0.2,
        pin_system: bool = True,
        pin_latest_user: bool = True,
        recent_window_size: int = 64,
        hot_recent_window: int = 8,
        hot_block_size: int = 6,
        block_size: int = 16,
        warm_promotable_reserve: int = 8,
        latest_user_tail_tokens: int = 64,
        generated_retention_window: int = 12,
    ):
        self.tracker = tracker
        self.analyzer = analyzer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.query_weight = query_weight if query_weight == 0.0 else max(query_weight, 0.30)
        self.factual_weight = factual_weight if factual_weight == 0.0 else max(factual_weight, 0.40)
        self.pin_system = pin_system
        self.pin_latest_user = pin_latest_user
        self.recent_window_size = recent_window_size
        self.hot_recent_window = max(0, hot_recent_window)
        self.hot_block_size = max(1, hot_block_size)
        self.block_size = min(block_size, 3)
        self.warm_promotable_reserve = max(0, warm_promotable_reserve)
        self.latest_user_tail_tokens = min(latest_user_tail_tokens, 56)
        self.generated_retention_window = max(0, generated_retention_window)

        self.role_tags: Optional[torch.Tensor] = None
        self.info_density: Optional[torch.Tensor] = None
        self.query_relevance: Optional[torch.Tensor] = None
        self.factual_bonus: Optional[torch.Tensor] = None
        self.authority_bonus: Optional[torch.Tensor] = None
        self.pinned_mask: Optional[torch.Tensor] = None
        self.chat_boundary_mask: Optional[torch.Tensor] = None
        self.chat_template_mask: Optional[torch.Tensor] = None
        self.question_tail_mask: Optional[torch.Tensor] = None
        self.question_like_mask: Optional[torch.Tensor] = None
        self.generated_assistant_mask: Optional[torch.Tensor] = None
        self.prompt_token_ids: Optional[torch.Tensor] = None

    def _build_recent_generated_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a mask for the most recent generated assistant tokens."""
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if (
            self.generated_retention_window <= 0
            or self.generated_assistant_mask is None
            or len(self.generated_assistant_mask) < seq_len
        ):
            return mask

        generated_indices = torch.nonzero(
            self.generated_assistant_mask[:seq_len],
            as_tuple=False,
        ).flatten()
        if generated_indices.numel() == 0:
            return mask

        mask[generated_indices[-self.generated_retention_window :].to(device=device)] = True
        return mask

    def _build_protected_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return the tokens that must remain in the hot tier."""
        protected_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            protected_mask |= self.pinned_mask[:seq_len].to(device=device)
        protected_mask |= self._build_recent_generated_mask(seq_len, device)
        if self.recent_window_size > 0:
            protected_mask[max(0, seq_len - self.recent_window_size) :] = True
        return protected_mask

    @staticmethod
    def _inverse_normalize(values: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
        """Map a signal to eviction pressure where larger source values are safer to keep."""
        if values is None or len(values) < seq_len:
            return None

        window = values[:seq_len]
        scale = window.max().clamp(min=1e-10)
        return 1.0 - (window / scale)

    @staticmethod
    def _normalize(values: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
        """Map a signal to [0, 1] where larger source values stay larger."""
        if values is None or len(values) < seq_len:
            return None

        window = values[:seq_len].float()
        min_value = window.min()
        max_value = window.max()
        if torch.isclose(max_value, min_value):
            return torch.zeros(seq_len, dtype=torch.float32)
        return (window - min_value) / (max_value - min_value)

    def setup_semantic_signals(self, input_ids: torch.Tensor, latest_query_text: str = "") -> None:
        """Compute semantic signals once after prefill."""
        self.prompt_token_ids = input_ids.detach().cpu().to(dtype=torch.long)
        self.role_tags = self.analyzer.compute_role_tags(input_ids)
        self.info_density = self.analyzer.compute_info_density(input_ids)
        self.query_relevance = self.analyzer.compute_query_relevance(input_ids, latest_query_text)
        self.factual_bonus = self.analyzer.compute_factual_bonus(input_ids)
        self.authority_bonus = self.analyzer.compute_authority_bonus(input_ids, latest_query_text)
        self.chat_boundary_mask = self.analyzer.compute_chat_boundary_mask(input_ids)
        self.chat_template_mask = self.analyzer.compute_chat_template_mask(input_ids)
        self.question_tail_mask = self.analyzer.compute_latest_question_mask(input_ids)
        self.question_like_mask = self.analyzer.compute_question_like_mask(input_ids)
        self.generated_assistant_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        self.pinned_mask = self.analyzer.get_pinned_mask(
            self.role_tags,
            self.pin_system,
            self.pin_latest_user,
            latest_user_tail_tokens=self.latest_user_tail_tokens,
        )

    def extend_signals(self, num_new_tokens: int = 1) -> None:
        """Append semantic defaults for newly generated assistant tokens."""
        if self.role_tags is not None:
            new_roles = torch.full((num_new_tokens,), RoleTag.ASSISTANT, dtype=torch.long)
            self.role_tags = torch.cat([self.role_tags, new_roles])

        if self.info_density is not None:
            new_density = torch.full((num_new_tokens,), 0.5, dtype=torch.float32)
            self.info_density = torch.cat([self.info_density, new_density])

        if self.query_relevance is not None:
            new_query = torch.zeros(num_new_tokens, dtype=torch.float32)
            self.query_relevance = torch.cat([self.query_relevance, new_query])

        if self.factual_bonus is not None:
            new_factual = torch.zeros(num_new_tokens, dtype=torch.float32)
            self.factual_bonus = torch.cat([self.factual_bonus, new_factual])
        if self.authority_bonus is not None:
            new_authority = torch.zeros(num_new_tokens, dtype=torch.float32)
            self.authority_bonus = torch.cat([self.authority_bonus, new_authority])

        if self.pinned_mask is not None:
            new_pinned = torch.zeros(num_new_tokens, dtype=torch.bool)
            self.pinned_mask = torch.cat([self.pinned_mask, new_pinned])
        if self.chat_boundary_mask is not None:
            new_boundary = torch.zeros(num_new_tokens, dtype=torch.bool)
            self.chat_boundary_mask = torch.cat([self.chat_boundary_mask, new_boundary])
        if self.chat_template_mask is not None:
            new_template = torch.zeros(num_new_tokens, dtype=torch.bool)
            self.chat_template_mask = torch.cat([self.chat_template_mask, new_template])
        if self.question_tail_mask is not None:
            new_question_tail = torch.zeros(num_new_tokens, dtype=torch.bool)
            self.question_tail_mask = torch.cat([self.question_tail_mask, new_question_tail])
        if self.question_like_mask is not None:
            new_question_like = torch.zeros(num_new_tokens, dtype=torch.bool)
            self.question_like_mask = torch.cat([self.question_like_mask, new_question_like])
        if self.generated_assistant_mask is not None:
            new_generated = torch.ones(num_new_tokens, dtype=torch.bool)
            self.generated_assistant_mask = torch.cat([self.generated_assistant_mask, new_generated])
        if self.prompt_token_ids is not None:
            new_token_ids = torch.full((num_new_tokens,), -1, dtype=torch.long)
            self.prompt_token_ids = torch.cat([self.prompt_token_ids, new_token_ids])

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        scores = torch.zeros(seq_len, dtype=torch.float32)

        attn_pressure = self._inverse_normalize(self.tracker.get_cumulative_scores(), seq_len)
        if attn_pressure is not None:
            scores += self.alpha * attn_pressure.cpu()

        density_pressure = self._inverse_normalize(self.info_density, seq_len)
        if density_pressure is not None:
            scores += self.beta * density_pressure

        entropy_pressure = self._inverse_normalize(self.tracker.get_head_entropy(), seq_len)
        if entropy_pressure is not None:
            scores += self.gamma * entropy_pressure.cpu()

        query_pressure = self._inverse_normalize(self.query_relevance, seq_len)
        if query_pressure is not None:
            scores += self.query_weight * query_pressure

        factual_pressure = self._inverse_normalize(self.factual_bonus, seq_len)
        if factual_pressure is not None:
            scores += self.factual_weight * factual_pressure

        authority_pressure = self._inverse_normalize(self.authority_bonus, seq_len)
        if authority_pressure is not None:
            scores += 0.65 * authority_pressure

        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            scores[self.pinned_mask[:seq_len]] = -torch.inf

        recent_generated_mask = self._build_recent_generated_mask(seq_len, scores.device)
        if recent_generated_mask.any():
            scores[recent_generated_mask] = -torch.inf

        if self.recent_window_size > 0:
            recent_start = max(0, seq_len - self.recent_window_size)
            scores[recent_start:] = -torch.inf

        return scores

    def compute_promotion_scores(
        self,
        seq_len: int,
        hot_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Score which logical tokens are most worth promoting from warm to hot.

        This is intentionally different from eviction scoring:
        promotion should prefer tokens likely to matter for the *next* decode
        step, not tokens that are merely globally protected.
        """
        scores = torch.zeros(seq_len, dtype=torch.float32)

        attn_signal = self._normalize(self.tracker.get_cumulative_scores(), seq_len)
        if attn_signal is not None:
            scores += 0.50 * attn_signal.cpu()

        recent_attn_signal = self._normalize(self.tracker.get_last_step_scores(), seq_len)
        if recent_attn_signal is not None:
            scores += 1.10 * recent_attn_signal.cpu()

        query_signal = self._normalize(self.query_relevance, seq_len)
        if query_signal is not None:
            scores += 1.05 * query_signal

        factual_signal = self._normalize(self.factual_bonus, seq_len)
        if factual_signal is not None:
            scores += 0.95 * factual_signal

        authority_signal = self._normalize(self.authority_bonus, seq_len)
        if authority_signal is not None:
            scores += 1.15 * authority_signal

        density_signal = self._normalize(self.info_density, seq_len)
        if density_signal is not None:
            scores += 0.15 * density_signal

        entropy_signal = self._normalize(self.tracker.get_head_entropy(), seq_len)
        if entropy_signal is not None:
            scores += 0.10 * entropy_signal.cpu()

        if seq_len > 1:
            recency_signal = torch.linspace(0.0, 1.0, seq_len, dtype=torch.float32)
            scores += 0.10 * recency_signal

        if self.role_tags is not None and len(self.role_tags) >= seq_len:
            role_tags = self.role_tags[:seq_len]
            role_bonus = torch.zeros(seq_len, dtype=torch.float32)
            role_bonus[role_tags == RoleTag.FILLER] = -0.35
            role_bonus[role_tags == RoleTag.ASSISTANT] = -0.70
            role_bonus[role_tags == RoleTag.CONTEXT] = 0.08
            role_bonus[role_tags == RoleTag.USER_HISTORY] = 0.42
            role_bonus[role_tags == RoleTag.USER_LATEST] = 0.18
            role_bonus[role_tags == RoleTag.SYSTEM] = -1.25
            scores += role_bonus

            informative_signal = torch.zeros(seq_len, dtype=torch.float32)
            if query_signal is not None:
                informative_signal += query_signal
            if factual_signal is not None:
                informative_signal += factual_signal
            if authority_signal is not None:
                informative_signal += authority_signal
            informative_signal = informative_signal.clamp(max=1.5)

            scores[role_tags == RoleTag.USER_HISTORY] += 0.75 * informative_signal[role_tags == RoleTag.USER_HISTORY]
            scores[role_tags == RoleTag.USER_LATEST] += 0.28 * informative_signal[role_tags == RoleTag.USER_LATEST]
            scores[role_tags == RoleTag.ASSISTANT] -= 0.55 * informative_signal[role_tags == RoleTag.ASSISTANT]

        if self.chat_template_mask is not None and len(self.chat_template_mask) >= seq_len:
            scores[self.chat_template_mask[:seq_len]] = -torch.inf

        if self.question_tail_mask is not None and len(self.question_tail_mask) >= seq_len:
            scores[self.question_tail_mask[:seq_len]] = -torch.inf
        if self.question_like_mask is not None and len(self.question_like_mask) >= seq_len:
            scores[self.question_like_mask[:seq_len]] -= 1.20

        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            scores[self.pinned_mask[:seq_len]] = -torch.inf

        recent_generated_mask = self._build_recent_generated_mask(seq_len, scores.device)
        if recent_generated_mask.any():
            scores[recent_generated_mask] += 1.75

        if self.role_tags is not None and len(self.role_tags) >= seq_len:
            assistant_history_mask = self.role_tags[:seq_len] == RoleTag.ASSISTANT
            if recent_generated_mask.any():
                assistant_history_mask &= ~recent_generated_mask.cpu()
            scores[assistant_history_mask] = -torch.inf

        if hot_positions is not None and hot_positions.numel() > 0:
            scores[hot_positions.to(dtype=torch.long)] = -torch.inf

        return scores

    def compute_hot_scores(
        self,
        seq_len: int,
        hot_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Score which logical tokens deserve scarce full-precision hot capacity.

        Hot selection should skew even more aggressively toward factual,
        user-provided anchor spans than warm promotion does.
        """
        scores = self.compute_promotion_scores(
            seq_len,
            hot_positions=hot_positions,
            **kwargs,
        ).clone()

        factual_signal = self._normalize(self.factual_bonus, seq_len)
        authority_signal = self._normalize(self.authority_bonus, seq_len)
        query_signal = self._normalize(self.query_relevance, seq_len)

        if factual_signal is not None:
            scores += 0.85 * factual_signal
        if authority_signal is not None:
            scores += 0.75 * authority_signal
        if query_signal is not None:
            scores += 0.30 * query_signal

        if self.role_tags is not None and len(self.role_tags) >= seq_len:
            role_tags = self.role_tags[:seq_len]
            scores[role_tags == RoleTag.USER_HISTORY] += 0.90
            scores[role_tags == RoleTag.USER_LATEST] -= 0.20
            scores[role_tags == RoleTag.ASSISTANT] -= 0.45
            scores[role_tags == RoleTag.SYSTEM] = -torch.inf

            informative_signal = torch.zeros(seq_len, dtype=torch.float32)
            if factual_signal is not None:
                informative_signal += factual_signal
            if authority_signal is not None:
                informative_signal += authority_signal
            if query_signal is not None:
                informative_signal += 0.5 * query_signal
            informative_signal = informative_signal.clamp(max=1.5)

            user_history_mask = role_tags == RoleTag.USER_HISTORY
            assistant_mask = role_tags == RoleTag.ASSISTANT
            scores[user_history_mask] += 0.90 * informative_signal[user_history_mask]
            scores[assistant_mask] -= 0.35 * informative_signal[assistant_mask]

        if self.question_tail_mask is not None and len(self.question_tail_mask) >= seq_len:
            scores[self.question_tail_mask[:seq_len]] = -torch.inf
        if self.question_like_mask is not None and len(self.question_like_mask) >= seq_len:
            scores[self.question_like_mask[:seq_len]] -= 1.00
        if self.chat_boundary_mask is not None and len(self.chat_boundary_mask) >= seq_len:
            scores[self.chat_boundary_mask[:seq_len]] = -torch.inf
        if self.chat_template_mask is not None and len(self.chat_template_mask) >= seq_len:
            scores[self.chat_template_mask[:seq_len]] -= 1.25

        return scores

    def select_keep_indices(self, scores: torch.Tensor, budget: int) -> torch.Tensor:
        """Keep protected tokens plus the best-scoring contiguous blocks."""
        seq_len = scores.shape[0]
        if budget >= seq_len:
            return torch.arange(seq_len, device=scores.device)

        protected_mask = self._build_protected_mask(seq_len, scores.device)
        return self._select_keep_indices_with_mask(scores, budget, protected_mask)

    def _select_keep_indices_with_mask(
        self,
        scores: torch.Tensor,
        budget: int,
        protected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Keep protected tokens plus the best-scoring contiguous blocks under a custom mask."""
        seq_len = scores.shape[0]
        if budget >= seq_len:
            return torch.arange(seq_len, device=scores.device)

        protected_indices = torch.nonzero(protected_mask, as_tuple=False).flatten()
        if protected_indices.numel() >= budget:
            return protected_indices[-budget:].sort().values

        keep_indices = protected_indices.tolist()
        remaining_budget = budget - len(keep_indices)

        candidate_blocks: list[tuple[float, list[int]]] = []
        block_size = max(1, self.block_size)
        for block_start in range(0, seq_len, block_size):
            block_end = min(seq_len, block_start + block_size)
            block_positions = torch.arange(block_start, block_end, device=scores.device)
            candidate_mask = ~protected_mask[block_start:block_end]
            candidate_positions = block_positions[candidate_mask]
            if candidate_positions.numel() == 0:
                continue

            block_score = scores[candidate_positions].min().item()
            candidate_blocks.append((block_score, candidate_positions.tolist()))

        candidate_blocks.sort(key=lambda item: item[0])

        for _, block_positions in candidate_blocks:
            if remaining_budget <= 0:
                break

            if len(block_positions) <= remaining_budget:
                keep_indices.extend(block_positions)
                remaining_budget -= len(block_positions)
            else:
                block_tensor = torch.tensor(block_positions, device=scores.device, dtype=torch.long)
                block_scores = scores[block_tensor]
                _, best_local = block_scores.topk(remaining_budget, largest=False)
                keep_indices.extend(block_tensor[best_local].sort().values.tolist())
                remaining_budget = 0

        if remaining_budget > 0:
            all_indices = torch.arange(seq_len, device=scores.device)
            already_kept = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
            already_kept[torch.tensor(keep_indices, device=scores.device, dtype=torch.long)] = True
            fallback_candidates = all_indices[~already_kept]
            fallback_scores = scores[fallback_candidates]
            _, extra_order = fallback_scores.topk(min(remaining_budget, fallback_candidates.numel()), largest=False)
            keep_indices.extend(fallback_candidates[extra_order].tolist())

        keep_tensor = torch.tensor(keep_indices, device=scores.device, dtype=torch.long).unique(sorted=True)
        if keep_tensor.numel() > budget:
            keep_tensor = keep_tensor[-budget:]
        return keep_tensor.sort().values

    def evict_positions(self, keep_mask: torch.Tensor) -> None:
        """Keep semantic signal tensors aligned with the pruned cache."""
        if self.role_tags is not None:
            self.role_tags = self.role_tags[keep_mask]
        if self.info_density is not None:
            self.info_density = self.info_density[keep_mask]
        if self.query_relevance is not None:
            self.query_relevance = self.query_relevance[keep_mask]
        if self.factual_bonus is not None:
            self.factual_bonus = self.factual_bonus[keep_mask]
        if self.authority_bonus is not None:
            self.authority_bonus = self.authority_bonus[keep_mask]
        if self.pinned_mask is not None:
            self.pinned_mask = self.pinned_mask[keep_mask]
        if self.chat_boundary_mask is not None:
            self.chat_boundary_mask = self.chat_boundary_mask[keep_mask]
        if self.chat_template_mask is not None:
            self.chat_template_mask = self.chat_template_mask[keep_mask]
        if self.question_tail_mask is not None:
            self.question_tail_mask = self.question_tail_mask[keep_mask]
        if self.question_like_mask is not None:
            self.question_like_mask = self.question_like_mask[keep_mask]
        if self.generated_assistant_mask is not None:
            self.generated_assistant_mask = self.generated_assistant_mask[keep_mask]
        if self.prompt_token_ids is not None:
            self.prompt_token_ids = self.prompt_token_ids[keep_mask]


class OPSieveKVLitePolicy(SemantiCachePolicy):
    """
    OP-SieveKV-Lite: semantic segment retention with adaptive gated scoring.

    This is the online-serving half of the research plan's MVP. It keeps the
    runtime path oracle-free, but exposes KV-TIP diagnostics so offline scripts
    can attach counterfactual oracle labels and select informative decisions.
    """

    def __init__(
        self,
        *args,
        max_segment_tokens: int = 32,
        min_segment_tokens: int = 4,
        uncertainty_weight: float = 0.15,
        budget_ratio: float = 0.5,
        policy_ckpt: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_segment_tokens = max(4, int(max_segment_tokens))
        self.min_segment_tokens = max(1, int(min_segment_tokens))
        self.uncertainty_weight = max(0.0, float(uncertainty_weight))
        self.budget_ratio = min(max(float(budget_ratio), 0.0), 1.0)

        self.segment_ids: Optional[torch.Tensor] = None
        self.last_keep_prob: Optional[torch.Tensor] = None
        self.last_segment_scores: Optional[dict[int, float]] = None
        self.last_kvtip: Optional[KVTipStats] = None
        self.learned_policy = None
        self.learned_feature_mean: Optional[torch.Tensor] = None
        self.learned_feature_std: Optional[torch.Tensor] = None
        self.learned_policy_metadata: dict = {}
        if policy_ckpt:
            (
                self.learned_policy,
                self.learned_feature_mean,
                self.learned_feature_std,
                self.learned_policy_metadata,
            ) = load_policy_checkpoint(policy_ckpt, map_location="cpu")

    def setup_semantic_signals(self, input_ids: torch.Tensor, latest_query_text: str = "") -> None:
        super().setup_semantic_signals(input_ids, latest_query_text)
        self.segment_ids = self._build_semantic_segment_ids(input_ids)

    def extend_signals(self, num_new_tokens: int = 1) -> None:
        next_segment_id = 0
        if self.segment_ids is not None and self.segment_ids.numel() > 0:
            next_segment_id = int(self.segment_ids.max().item()) + 1
        super().extend_signals(num_new_tokens)
        if self.segment_ids is not None:
            new_segments = torch.arange(
                next_segment_id,
                next_segment_id + num_new_tokens,
                dtype=torch.long,
            )
            self.segment_ids = torch.cat([self.segment_ids, new_segments])

    def evict_positions(self, keep_mask: torch.Tensor) -> None:
        if self.segment_ids is not None:
            self.segment_ids = self.segment_ids[keep_mask.cpu()]
        super().evict_positions(keep_mask)

    def _build_semantic_segment_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build sentence/role-aware segment ids with a fixed-block fallback."""
        ids = input_ids.detach().cpu()
        seq_len = int(ids.numel())
        segment_ids = torch.empty(seq_len, dtype=torch.long)
        if seq_len == 0:
            return segment_ids

        role_tags = self.role_tags if self.role_tags is not None and len(self.role_tags) >= seq_len else None
        boundary = self.chat_boundary_mask if self.chat_boundary_mask is not None and len(self.chat_boundary_mask) >= seq_len else None
        template = self.chat_template_mask if self.chat_template_mask is not None and len(self.chat_template_mask) >= seq_len else None

        segment_id = 0
        start = 0
        for pos in range(seq_len):
            segment_ids[pos] = segment_id
            token_text = self.analyzer.tokenizer.decode([int(ids[pos].item())], skip_special_tokens=False)
            role_changed = bool(role_tags is not None and pos > start and role_tags[pos] != role_tags[pos - 1])
            hard_boundary = bool(boundary is not None and boundary[pos] and pos > start)
            template_boundary = bool(template is not None and template[pos] and pos > start)
            sentence_boundary = any(mark in token_text for mark in (".", "?", "!", "\n", "\r", ";"))
            too_long = (pos - start + 1) >= self.max_segment_tokens
            long_enough = (pos - start + 1) >= self.min_segment_tokens

            if role_changed or hard_boundary or template_boundary or too_long or (sentence_boundary and long_enough):
                if pos + 1 < seq_len:
                    segment_id += 1
                    start = pos + 1

        return segment_ids

    @staticmethod
    def _safe_signal(signal: Optional[torch.Tensor], seq_len: int) -> torch.Tensor:
        if signal is None or len(signal) < seq_len:
            return torch.zeros(seq_len, dtype=torch.float32)
        return signal[:seq_len].detach().cpu().float()

    def _segment_positions(self, seq_len: int) -> list[torch.Tensor]:
        if self.segment_ids is None or len(self.segment_ids) < seq_len:
            return [torch.arange(start, min(seq_len, start + self.max_segment_tokens), dtype=torch.long) for start in range(0, seq_len, self.max_segment_tokens)]

        segment_ids = self.segment_ids[:seq_len].cpu()
        positions: list[torch.Tensor] = []
        for segment_id in segment_ids.unique(sorted=True).tolist():
            segment_positions = torch.nonzero(segment_ids == segment_id, as_tuple=False).flatten()
            if segment_positions.numel() > 0:
                positions.append(segment_positions)
        return positions

    def compute_segment_features(
        self,
        seq_len: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Return segment features, positions, and heuristic keep probabilities."""
        attn = self._normalize(self.tracker.get_cumulative_scores(), seq_len)
        entropy = self._normalize(self.tracker.get_head_entropy(), seq_len)
        density = self._normalize(self.info_density, seq_len)
        query = self._normalize(self.query_relevance, seq_len)
        factual = self._normalize(self.factual_bonus, seq_len)
        authority = self._normalize(self.authority_bonus, seq_len)

        signals = {
            "attn": self._safe_signal(attn, seq_len),
            "entropy": self._safe_signal(entropy, seq_len),
            "density": self._safe_signal(density, seq_len),
            "query": self._safe_signal(query, seq_len),
            "factual": self._safe_signal(factual, seq_len),
            "authority": self._safe_signal(authority, seq_len),
        }
        recency = torch.linspace(0.0, 1.0, seq_len, dtype=torch.float32) if seq_len > 1 else torch.ones(seq_len)
        signals["recency"] = recency

        query_peak = float(signals["query"].max().item()) if seq_len else 0.0
        factual_peak = float(signals["factual"].max().item()) if seq_len else 0.0
        sparse_query = query_peak > 0.15
        factual_task = factual_peak > 0.20 or float(signals["authority"].max().item()) > 0.0

        weights = {
            "attn": max(0.05, self.alpha),
            "density": max(0.05, self.beta * 0.75),
            "entropy": max(0.05, self.gamma * 0.50),
            "query": max(0.05, self.query_weight),
            "factual": max(0.05, self.factual_weight),
            "authority": 0.35,
            "recency": 0.08,
        }
        if sparse_query:
            weights["query"] += 0.45
            weights["factual"] += 0.15
        if factual_task:
            weights["factual"] += 0.45
            weights["authority"] += 0.35
        if seq_len > self.max_segment_tokens * 4:
            weights["recency"] += 0.05

        segment_features: list[torch.Tensor] = []
        segment_positions = self._segment_positions(seq_len)
        heuristic_probs: list[float] = []
        for positions in self._segment_positions(seq_len):
            positions = positions.to(dtype=torch.long)
            pooled: dict[str, float] = {}
            for name, values in signals.items():
                segment_values = values[positions]
                pooled[name] = 0.55 * float(segment_values.max().item()) + 0.35 * float(segment_values.mean().item())
                if segment_values.numel() > 1:
                    pooled[name] += 0.10 * float(segment_values.std(unbiased=False).item())

            score = sum(weights[name] * pooled[name] for name in weights)

            if self.role_tags is not None and len(self.role_tags) >= seq_len:
                role_values = self.role_tags[:seq_len][positions]
                role = int(torch.mode(role_values).values.item())
                if role == RoleTag.SYSTEM:
                    score += 0.25
                elif role == RoleTag.USER_LATEST:
                    score += 0.55
                elif role == RoleTag.USER_HISTORY:
                    score += 0.30
                elif role == RoleTag.ASSISTANT:
                    score -= 0.20
                elif role == RoleTag.FILLER:
                    score -= 0.18

            length_penalty = max(0, int(positions.numel()) - self.max_segment_tokens) / max(1, self.max_segment_tokens)
            score -= 0.10 * length_penalty

            prob = torch.sigmoid(torch.tensor(2.8 * (score - 0.55))).item()
            heuristic_probs.append(float(prob))

            start = float(positions[0].item())
            end = float(positions[-1].item())
            denom = max(1.0, float(seq_len - 1))
            feature_values: list[float] = []
            for signal_name in signals:
                segment_values = signals[signal_name][positions]
                feature_values.extend(
                    [
                        float(segment_values.max().item()),
                        float(segment_values.mean().item()),
                        float(segment_values.std(unbiased=False).item()) if segment_values.numel() > 1 else 0.0,
                    ]
                )

            feature_values.extend(
                [
                    start / denom,
                    end / denom,
                    ((start + end) * 0.5) / denom,
                    float(positions.numel()) / max(1.0, float(seq_len)),
                    self.budget_ratio,
                ]
            )

            role_fractions = [0.0] * 6
            if self.role_tags is not None and len(self.role_tags) >= seq_len:
                role_values = self.role_tags[:seq_len][positions]
                for role_idx in range(6):
                    role_fractions[role_idx] = float((role_values == role_idx).float().mean().item())
            feature_values.extend(role_fractions)

            def mask_fraction(mask: Optional[torch.Tensor]) -> float:
                if mask is None or len(mask) < seq_len:
                    return 0.0
                return float(mask[:seq_len][positions].float().mean().item())

            feature_values.extend(
                [
                    mask_fraction(self.pinned_mask),
                    mask_fraction(self.question_tail_mask),
                    mask_fraction(self.question_like_mask),
                    mask_fraction(self.chat_template_mask),
                    mask_fraction(self.chat_boundary_mask),
                    float(prob),
                ]
            )
            segment_features.append(torch.tensor(feature_values, dtype=torch.float32))

        if segment_features:
            features = torch.stack(segment_features, dim=0)
            if features.shape[1] != FEATURE_DIM:
                raise RuntimeError(f"OP segment feature dim {features.shape[1]} != expected {FEATURE_DIM}")
            heuristic_tensor = torch.tensor(heuristic_probs, dtype=torch.float32)
            return features, segment_positions, heuristic_tensor

        return torch.empty(0, FEATURE_DIM, dtype=torch.float32), [], torch.empty(0, dtype=torch.float32)

    def _compute_token_keep_prob(self, seq_len: int) -> torch.Tensor:
        """Predict per-token keep probability via heuristic or learned segment scoring."""
        device = torch.device("cpu")
        features, segment_positions, heuristic_probs = self.compute_segment_features(seq_len)
        keep_prob = torch.zeros(seq_len, dtype=torch.float32, device=device)
        self.last_segment_scores = {}

        if self.learned_policy is not None and features.numel() > 0:
            assert self.learned_feature_mean is not None
            assert self.learned_feature_std is not None
            normalized = (features - self.learned_feature_mean) / self.learned_feature_std
            with torch.no_grad():
                segment_probs = torch.sigmoid(self.learned_policy(normalized)).cpu()
        else:
            segment_probs = heuristic_probs

        for positions, prob in zip(segment_positions, segment_probs.tolist()):
            positions = positions.to(dtype=torch.long)
            keep_prob[positions] = float(prob)
            self.last_segment_scores[int(positions[0].item())] = float(prob)

        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            keep_prob[self.pinned_mask[:seq_len]] = 1.0
        recent_generated_mask = self._build_recent_generated_mask(seq_len, keep_prob.device)
        if recent_generated_mask.any():
            keep_prob[recent_generated_mask] = 1.0
        if self.recent_window_size > 0:
            keep_prob[max(0, seq_len - self.recent_window_size) :] = 1.0

        self.last_keep_prob = keep_prob
        return keep_prob

    def compute_eviction_scores(self, seq_len: int, **kwargs) -> torch.Tensor:
        keep_prob = self._compute_token_keep_prob(seq_len)
        entropy = -(keep_prob * torch.log(keep_prob.clamp(min=1e-6)) + (1.0 - keep_prob) * torch.log((1.0 - keep_prob).clamp(min=1e-6)))
        scores = 1.0 - keep_prob
        scores -= self.uncertainty_weight * (entropy / torch.log(torch.tensor(2.0)))

        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            scores[self.pinned_mask[:seq_len]] = -torch.inf
        recent_generated_mask = self._build_recent_generated_mask(seq_len, scores.device)
        if recent_generated_mask.any():
            scores[recent_generated_mask] = -torch.inf
        if self.recent_window_size > 0:
            scores[max(0, seq_len - self.recent_window_size) :] = -torch.inf
        return scores

    def select_keep_indices(self, scores: torch.Tensor, budget: int) -> torch.Tensor:
        seq_len = int(scores.shape[0])
        if budget >= seq_len:
            return torch.arange(seq_len, device=scores.device)

        protected_mask = self._build_protected_mask(seq_len, scores.device)
        protected_indices = torch.nonzero(protected_mask, as_tuple=False).flatten()
        if protected_indices.numel() >= budget:
            return self._trim_op_protected_indices(seq_len, budget, scores.device)

        keep_indices = protected_indices.tolist()
        remaining = budget - len(keep_indices)
        selected = protected_mask.detach().cpu().clone()

        segment_candidates: list[tuple[float, torch.Tensor]] = []
        for positions in self._segment_positions(seq_len):
            optional = positions[~selected[positions]]
            if optional.numel() == 0:
                continue
            segment_score = float(scores[optional].min().item())
            segment_candidates.append((segment_score, optional))
        segment_candidates.sort(key=lambda item: item[0])

        for _, positions in segment_candidates:
            if remaining <= 0:
                break
            positions = positions.to(device=scores.device, dtype=torch.long)
            if positions.numel() <= remaining:
                keep_indices.extend(positions.tolist())
                remaining -= int(positions.numel())
                continue
            local_scores = scores[positions]
            _, best = local_scores.topk(remaining, largest=False)
            keep_indices.extend(positions[best].sort().values.tolist())
            remaining = 0

        keep_tensor = torch.tensor(keep_indices, device=scores.device, dtype=torch.long).unique(sorted=True)
        if keep_tensor.numel() > budget:
            keep_tensor = keep_tensor[-budget:]
        return keep_tensor.sort().values

    def _trim_op_protected_indices(self, seq_len: int, budget: int, device: torch.device) -> torch.Tensor:
        """Trim oversized protected sets without collapsing into a pure recent window."""
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            keep_mask |= self.pinned_mask[:seq_len].to(device=device)

        if self.generated_retention_window > 0:
            recent_generated_mask = self._build_recent_generated_mask(seq_len, device)
            keep_mask |= recent_generated_mask

        hard_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
        if hard_indices.numel() >= budget:
            scores = self.last_keep_prob if self.last_keep_prob is not None else torch.zeros(seq_len)
            hard_scores = scores[hard_indices.cpu()].to(device=device)
            _, order = hard_scores.topk(budget, largest=True)
            return hard_indices[order].sort().values

        keep_indices = hard_indices.tolist()
        remaining = budget - len(keep_indices)
        if remaining <= 0:
            return hard_indices.sort().values

        recent_quota = min(remaining, max(1, min(self.recent_window_size, budget // 3)))
        recent_start = max(0, seq_len - recent_quota)
        for idx in range(recent_start, seq_len):
            if idx not in keep_indices:
                keep_indices.append(idx)

        remaining = budget - len(keep_indices)
        if remaining > 0:
            score_source = self.last_keep_prob if self.last_keep_prob is not None else torch.zeros(seq_len)
            candidates = torch.arange(seq_len, dtype=torch.long)
            already = torch.zeros(seq_len, dtype=torch.bool)
            if keep_indices:
                already[torch.tensor(keep_indices, dtype=torch.long)] = True
            candidates = candidates[~already]
            if candidates.numel() > 0:
                candidate_scores = score_source[candidates]
                _, order = candidate_scores.topk(min(remaining, candidates.numel()), largest=True)
                keep_indices.extend(candidates[order].tolist())

        return torch.tensor(keep_indices[:budget], device=device, dtype=torch.long).unique(sorted=True)

    def compute_kvtip_stats(self, oracle_keep_prob: torch.Tensor) -> KVTipStats:
        """Compute KV-TIP entropy/divergence quadrants for offline distillation."""
        if self.last_keep_prob is None:
            raise RuntimeError("compute_eviction_scores must run before KV-TIP diagnostics.")

        policy_prob = self.last_keep_prob.detach().cpu().float()
        oracle = oracle_keep_prob.detach().cpu().float()
        if oracle.numel() != policy_prob.numel():
            raise ValueError(f"oracle_keep_prob length {oracle.numel()} != policy length {policy_prob.numel()}")

        p = policy_prob.clamp(1e-6, 1.0 - 1e-6)
        y = oracle.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / torch.log(torch.tensor(2.0))
        divergence = torch.abs(y - p)
        soft_or = entropy + divergence - entropy * divergence

        high_entropy = entropy > entropy.median()
        high_divergence = divergence > divergence.median()
        quadrant = torch.ones_like(policy_prob, dtype=torch.long)
        quadrant[high_entropy & ~high_divergence] = 2
        quadrant[~high_entropy & high_divergence] = 3
        quadrant[high_entropy & high_divergence] = 4

        stats = KVTipStats(entropy=entropy, divergence=divergence, soft_or=soft_or, quadrant=quadrant)
        self.last_kvtip = stats
        return stats


class TieredSemantiCachePolicy(SemantiCachePolicy):
    """Three-tier SemantiCache policy with hot, warm, and cold token assignments."""

    def __init__(
        self,
        *args,
        hot_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hot_ratio = min(max(hot_ratio, 0.1), 1.0)

    def select_tier_indices(
        self,
        scores: torch.Tensor,
        total_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split the current sequence into:
        - hot indices: full-precision active cache
        - warm indices: quantized retained cache
        Tokens outside both sets fall into the cold tier and are evicted.
        """
        seq_len = scores.shape[0]
        if total_budget >= seq_len:
            return (
                torch.arange(seq_len, device=scores.device, dtype=torch.long),
                torch.empty(0, device=scores.device, dtype=torch.long),
            )

        total_budget = max(1, min(total_budget, seq_len))
        pinned_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        if self.pinned_mask is not None and len(self.pinned_mask) >= seq_len:
            pinned_mask |= self.pinned_mask[:seq_len].to(device=scores.device)

        retain_only_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        if self.role_tags is not None and len(self.role_tags) >= seq_len:
            role_tags = self.role_tags[:seq_len].to(device=scores.device)
            retain_only_mask |= role_tags == RoleTag.SYSTEM
        if self.chat_boundary_mask is not None and len(self.chat_boundary_mask) >= seq_len:
            retain_only_mask |= self.chat_boundary_mask[:seq_len].to(device=scores.device)
        reserved_warm_mask = retain_only_mask.clone()
        reserved_warm_mask |= pinned_mask

        retain_only_indices = torch.nonzero(retain_only_mask, as_tuple=False).flatten()
        all_indices = torch.arange(seq_len, device=scores.device)
        recent_generated_mask = self._build_recent_generated_mask(seq_len, scores.device)
        recent_generated_indices = torch.nonzero(recent_generated_mask, as_tuple=False).flatten()
        available_non_retain = max(1, total_budget - retain_only_indices.numel())
        promotable_warm_reserve = min(
            self.warm_promotable_reserve,
            max(0, available_non_retain - 1),
        )
        max_hot_budget = max(1, available_non_retain - promotable_warm_reserve)
        hot_budget = max(1, int(total_budget * self.hot_ratio), recent_generated_indices.numel())
        if recent_generated_indices.numel() > max_hot_budget:
            hot_budget = min(available_non_retain, recent_generated_indices.numel())
        else:
            hot_budget = min(hot_budget, max_hot_budget)
        hot_budget = min(hot_budget, total_budget)

        # Cache promotion scores to avoid redundant recomputation
        _cached_promotion_scores = self.compute_promotion_scores(seq_len)

        promotable_reserve_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        if promotable_warm_reserve > 0:
            promotion_scores = _cached_promotion_scores
            promotable_candidates = all_indices[
                (~reserved_warm_mask)
                & (~recent_generated_mask)
                & torch.isfinite(promotion_scores)
                & ~torch.isneginf(promotion_scores)
            ]
            if promotable_candidates.numel() > 0:
                reserve_count = min(promotable_warm_reserve, promotable_candidates.numel())
                reserve_indices = self._select_promotable_reserve_indices(
                    promotable_candidates,
                    promotion_scores[promotable_candidates],
                    reserve_count,
                )
                promotable_reserve_mask[reserve_indices] = True
                reserved_warm_mask |= promotable_reserve_mask

        hot_protected_mask = recent_generated_mask.clone()
        hot_priority_scores = self.compute_hot_scores(seq_len)
        hot_candidate_scores = -hot_priority_scores
        hot_candidate_scores[retain_only_mask] = torch.inf
        hot_candidate_scores[pinned_mask] = torch.inf
        hot_candidate_scores[promotable_reserve_mask] = torch.inf
        if self.role_tags is not None and len(self.role_tags) >= seq_len:
            assistant_history_mask = (
                (self.role_tags[:seq_len].to(device=scores.device) == RoleTag.ASSISTANT)
                & (~recent_generated_mask)
            )
            hot_candidate_scores[assistant_history_mask] = torch.inf
        if self.question_tail_mask is not None and len(self.question_tail_mask) >= seq_len:
            hot_candidate_scores[self.question_tail_mask[:seq_len].to(device=scores.device)] = torch.inf

        generated_hot_quota = min(recent_generated_indices.numel(), hot_budget)
        hot_recent_cap = min(self.hot_recent_window, self.recent_window_size)
        hot_recent_quota = min(max(0, hot_budget - generated_hot_quota), hot_recent_cap)
        if hot_recent_quota > 0:
            recent_positions = torch.arange(
                max(0, seq_len - hot_recent_quota),
                seq_len,
                device=scores.device,
                dtype=torch.long,
            )
            recent_positions = recent_positions[~retain_only_mask[recent_positions]]
            recent_positions = recent_positions[~pinned_mask[recent_positions]]
            if self.chat_template_mask is not None and len(self.chat_template_mask) >= seq_len:
                recent_positions = recent_positions[
                    ~self.chat_template_mask[:seq_len].to(device=scores.device)[recent_positions]
                ]
            if self.question_tail_mask is not None and len(self.question_tail_mask) >= seq_len:
                recent_positions = recent_positions[
                    ~self.question_tail_mask[:seq_len].to(device=scores.device)[recent_positions]
                ]
            if self.question_like_mask is not None and len(self.question_like_mask) >= seq_len:
                recent_positions = recent_positions[
                    ~self.question_like_mask[:seq_len].to(device=scores.device)[recent_positions]
                ]
            if self.role_tags is not None and len(self.role_tags) >= seq_len:
                recent_positions = recent_positions[
                    self.role_tags[:seq_len].to(device=scores.device)[recent_positions] != RoleTag.USER_LATEST
                ]
            hot_protected_mask[recent_positions] = True

        hot_indices = self._select_hot_indices(
            hot_candidate_scores,
            hot_budget,
            hot_protected_mask,
        )
        hot_mask = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        hot_mask[hot_indices] = True

        warm_budget = total_budget - hot_indices.numel()
        if warm_budget <= 0:
            return hot_indices, torch.empty(0, device=scores.device, dtype=torch.long)

        warm_candidates = all_indices[~hot_mask]
        if warm_candidates.numel() == 0:
            return hot_indices, torch.empty(0, device=scores.device, dtype=torch.long)

        reserved_warm = warm_candidates[reserved_warm_mask[warm_candidates]]
        warm_keep: list[torch.Tensor] = []
        if reserved_warm.numel() > 0:
            reserved_warm = reserved_warm.sort().values
            if reserved_warm.numel() >= warm_budget:
                return hot_indices.sort().values, reserved_warm[:warm_budget]
            warm_keep.append(reserved_warm)
            warm_budget -= reserved_warm.numel()

        warm_scores = _cached_promotion_scores[warm_candidates].clone()
        # Mask hot positions in the warm score view
        hot_in_warm = torch.isin(warm_candidates, hot_indices)
        warm_scores[hot_in_warm] = -torch.inf
        warm_scores[reserved_warm_mask[warm_candidates]] = -torch.inf
        eligible_mask = torch.isfinite(warm_scores) & ~torch.isneginf(warm_scores)
        eligible_candidates = warm_candidates[eligible_mask]
        eligible_scores = warm_scores[eligible_mask]

        if eligible_candidates.numel() == 0:
            if warm_keep:
                return hot_indices.sort().values, torch.cat(warm_keep).sort().values
            return hot_indices, torch.empty(0, device=scores.device, dtype=torch.long)

        warm_budget = min(warm_budget, eligible_candidates.numel())
        _, warm_order = eligible_scores.topk(warm_budget, largest=True)
        warm_indices = eligible_candidates[warm_order].sort().values
        if warm_keep:
            warm_indices = torch.cat([*warm_keep, warm_indices]).sort().values
        warm_indices = self._rescue_adjacent_warm_indices(
            seq_len=seq_len,
            hot_indices=hot_indices,
            warm_indices=warm_indices,
            reserved_warm_mask=reserved_warm_mask,
            promotion_scores=_cached_promotion_scores,
        )
        return hot_indices.sort().values, warm_indices

    def _rescue_adjacent_warm_indices(
        self,
        seq_len: int,
        hot_indices: torch.Tensor,
        warm_indices: torch.Tensor,
        reserved_warm_mask: torch.Tensor,
        promotion_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Swap in adjacent continuation tokens when they outscore weak optional warm picks."""
        if warm_indices.numel() <= 1:
            return warm_indices.sort().values

        selected_mask = torch.zeros(seq_len, dtype=torch.bool, device=warm_indices.device)
        selected_mask[hot_indices] = True
        selected_mask[warm_indices] = True

        reserved_warm_indices = warm_indices[reserved_warm_mask[warm_indices]]
        optional_warm_indices = warm_indices[~reserved_warm_mask[warm_indices]]
        if optional_warm_indices.numel() == 0:
            return warm_indices.sort().values

        max_rescue_iters = min(8, optional_warm_indices.numel())
        for _rescue_iter in range(max_rescue_iters):
            all_indices = torch.arange(seq_len, device=warm_indices.device)
            candidate_positions = all_indices[
                (~selected_mask)
                & torch.isfinite(promotion_scores)
                & ~torch.isneginf(promotion_scores)
            ]
            if candidate_positions.numel() == 0:
                break

            adjacent_mask = torch.zeros(candidate_positions.numel(), dtype=torch.bool, device=warm_indices.device)
            for idx, position in enumerate(candidate_positions.tolist()):
                if (position > 0 and selected_mask[position - 1]) or (
                    position + 1 < seq_len and selected_mask[position + 1]
                ):
                    adjacent_mask[idx] = True
            candidate_positions = candidate_positions[adjacent_mask]
            if candidate_positions.numel() == 0:
                break

            optional_scores = promotion_scores[optional_warm_indices]
            lowest_optional_order = optional_scores.argmin()
            lowest_optional_index = optional_warm_indices[lowest_optional_order]
            lowest_optional_score = promotion_scores[lowest_optional_index]

            candidate_scores = promotion_scores[candidate_positions]
            best_candidate_order = candidate_scores.argmax()
            best_candidate_index = candidate_positions[best_candidate_order]
            best_candidate_score = candidate_scores[best_candidate_order]
            adjacency_bonus = 0.0
            best_candidate_pos = int(best_candidate_index.item())
            if best_candidate_pos > 0 and selected_mask[best_candidate_pos - 1]:
                adjacency_bonus += 0.12
            if best_candidate_pos + 1 < seq_len and selected_mask[best_candidate_pos + 1]:
                adjacency_bonus += 0.12
            adjacency_bonus += self._continuation_rescue_bonus(seq_len, selected_mask, best_candidate_pos)

            if best_candidate_score + adjacency_bonus < lowest_optional_score:
                break

            selected_mask[lowest_optional_index] = False
            selected_mask[best_candidate_index] = True
            optional_warm_indices = optional_warm_indices.clone()
            optional_warm_indices[lowest_optional_order] = best_candidate_index

        if reserved_warm_indices.numel() > 0:
            warm_indices = torch.cat([reserved_warm_indices, optional_warm_indices])
        else:
            warm_indices = optional_warm_indices
        return warm_indices.sort().values

    def _select_promotable_reserve_indices(
        self,
        candidate_positions: torch.Tensor,
        candidate_scores: torch.Tensor,
        reserve_count: int,
    ) -> torch.Tensor:
        """Reserve a small set of high-promotion candidates for warm-tier recall."""
        if candidate_positions.numel() == 0 or reserve_count <= 0:
            return torch.empty(0, device=candidate_positions.device, dtype=torch.long)

        reserve_count = min(reserve_count, candidate_positions.numel())
        if candidate_positions.numel() <= 1 or reserve_count <= 1:
            top_order = candidate_scores.topk(reserve_count, largest=True).indices
            return candidate_positions[top_order].sort().values

        block_size = max(1, self.hot_block_size)
        blocks: list[dict] = []
        current_positions: list[int] = []
        current_scores: list[float] = []
        previous_position: int | None = None
        for logical_position, score in zip(candidate_positions.tolist(), candidate_scores.tolist()):
            should_flush = (
                current_positions
                and (
                    logical_position != (previous_position + 1 if previous_position is not None else logical_position)
                    or len(current_positions) >= block_size
                )
            )
            if should_flush:
                blocks.append(
                    {
                        "positions": current_positions.copy(),
                        "score": max(current_scores) + 0.25 * (sum(current_scores) / len(current_scores)),
                    }
                )
                current_positions.clear()
                current_scores.clear()

            current_positions.append(logical_position)
            current_scores.append(float(score))
            previous_position = logical_position

        if current_positions:
            blocks.append(
                {
                    "positions": current_positions.copy(),
                    "score": max(current_scores) + 0.25 * (sum(current_scores) / len(current_scores)),
                }
            )

        blocks.sort(key=lambda block: block["score"], reverse=True)
        selected_positions: list[int] = []
        for block in blocks:
            remaining_budget = reserve_count - len(selected_positions)
            if remaining_budget <= 0:
                break
            block_positions = block["positions"]
            take_count = min(len(block_positions), remaining_budget)
            if take_count >= len(block_positions):
                selected_positions.extend(block_positions)
                continue

            score_lookup = {
                int(position): float(score)
                for position, score in zip(candidate_positions.tolist(), candidate_scores.tolist())
            }
            best_start = 0
            best_sum: float | None = None
            max_start = len(block_positions) - take_count
            for candidate_start in range(max_start + 1):
                candidate_end = candidate_start + take_count
                candidate_sum = sum(
                    score_lookup[int(position)]
                    for position in block_positions[candidate_start:candidate_end]
                )
                if (
                    best_sum is None
                    or candidate_sum > best_sum
                    or (abs(candidate_sum - best_sum) < 1e-6 and candidate_start > best_start)
                ):
                    best_sum = candidate_sum
                    best_start = candidate_start
            selected_positions.extend(block_positions[best_start : best_start + take_count])

        if len(selected_positions) < reserve_count:
            top_order = candidate_scores.topk(reserve_count, largest=True).indices
            for idx in top_order.tolist():
                logical_position = int(candidate_positions[idx].item())
                if logical_position not in selected_positions:
                    selected_positions.append(logical_position)
                    if len(selected_positions) >= reserve_count:
                        break

        return torch.tensor(
            selected_positions[:reserve_count],
            device=candidate_positions.device,
            dtype=torch.long,
        ).sort().values

    def _select_hot_indices(
        self,
        hot_candidate_scores: torch.Tensor,
        hot_budget: int,
        hot_protected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select hot positions with block-aware packing instead of raw token ranking."""
        seq_len = hot_candidate_scores.shape[0]
        protected_indices = torch.nonzero(hot_protected_mask, as_tuple=False).flatten()
        if protected_indices.numel() >= hot_budget:
            return protected_indices[-hot_budget:].sort().values

        keep_indices = protected_indices.tolist()
        remaining_budget = hot_budget - len(keep_indices)
        if remaining_budget <= 0:
            return torch.tensor(keep_indices, device=hot_candidate_scores.device, dtype=torch.long).sort().values

        all_indices = torch.arange(seq_len, device=hot_candidate_scores.device)
        candidate_positions = all_indices[
            (~hot_protected_mask)
            & torch.isfinite(hot_candidate_scores)
            & ~torch.isinf(hot_candidate_scores)
        ]
        if candidate_positions.numel() == 0:
            return torch.tensor(keep_indices, device=hot_candidate_scores.device, dtype=torch.long).sort().values

        candidate_scores = (-hot_candidate_scores[candidate_positions]).float()
        block_size = max(1, self.hot_block_size)

        blocks: list[dict] = []
        current_positions: list[int] = []
        current_scores: list[float] = []
        previous_position: int | None = None
        for logical_position, score in zip(candidate_positions.tolist(), candidate_scores.tolist()):
            should_flush = (
                current_positions
                and (
                    logical_position != (previous_position + 1 if previous_position is not None else logical_position)
                    or len(current_positions) >= block_size
                )
            )
            if should_flush:
                blocks.append(
                    {
                        "positions": current_positions.copy(),
                        "score": max(current_scores) + 0.25 * (sum(current_scores) / len(current_scores)),
                    }
                )
                current_positions.clear()
                current_scores.clear()

            current_positions.append(logical_position)
            current_scores.append(float(score))
            previous_position = logical_position

        if current_positions:
            blocks.append(
                {
                    "positions": current_positions.copy(),
                    "score": max(current_scores) + 0.25 * (sum(current_scores) / len(current_scores)),
                }
            )

        blocks.sort(key=lambda block: block["score"], reverse=True)
        selected_blocks: list[list[int]] = []
        score_lookup = {
            int(position): float(score)
            for position, score in zip(candidate_positions.tolist(), candidate_scores.tolist())
        }
        block_gap = max(1, block_size // 2)
        for block in blocks:
            if remaining_budget <= 0:
                break

            block_positions = block["positions"]
            block_start = block_positions[0]
            block_end = block_positions[-1]
            overlaps_existing = any(
                not (block_end + block_gap < existing[0] or block_start - block_gap > existing[-1])
                for existing in selected_blocks
            )
            if overlaps_existing:
                continue

            take_count = min(len(block_positions), remaining_budget)
            if take_count >= len(block_positions):
                selected_slice = block_positions
            else:
                best_start = 0
                best_sum: float | None = None
                max_start = len(block_positions) - take_count
                for candidate_start in range(max_start + 1):
                    candidate_end = candidate_start + take_count
                    candidate_sum = sum(
                        score_lookup[int(position)]
                        for position in block_positions[candidate_start:candidate_end]
                    )
                    if (
                        best_sum is None
                        or candidate_sum > best_sum
                        or (abs(candidate_sum - best_sum) < 1e-6 and candidate_start > best_start)
                    ):
                        best_sum = candidate_sum
                        best_start = candidate_start
                selected_slice = block_positions[best_start : best_start + take_count]
            keep_indices.extend(selected_slice)
            selected_blocks.append(selected_slice)
            remaining_budget -= take_count

        if remaining_budget > 0:
            ranked_order = candidate_scores.topk(min(remaining_budget, candidate_positions.numel()), largest=True).indices
            for idx in ranked_order.tolist():
                logical_position = int(candidate_positions[idx].item())
                if logical_position not in keep_indices:
                    keep_indices.append(logical_position)
                    remaining_budget -= 1
                    if remaining_budget <= 0:
                        break

        return torch.tensor(
            sorted(set(keep_indices)),
            device=hot_candidate_scores.device,
            dtype=torch.long,
        )

    def _continuation_rescue_bonus(
        self,
        seq_len: int,
        selected_mask: torch.Tensor,
        candidate_index: int,
    ) -> float:
        """Small token-aware bonus for adjacent answer-span continuation candidates."""
        if self.prompt_token_ids is None or len(self.prompt_token_ids) < seq_len:
            return 0.0

        token_ids = self.prompt_token_ids[:seq_len]
        token_id = int(token_ids[candidate_index].item())
        if token_id < 0:
            return 0.0

        bonus = 0.0
        candidate_text = self.analyzer.tokenizer.decode([token_id])
        candidate_is_space = candidate_text == " "
        candidate_is_digit = 0 <= token_id < len(self.analyzer.is_digit_token) and bool(
            self.analyzer.is_digit_token[token_id].item()
        )
        candidate_is_entity = 0 <= token_id < len(self.analyzer.is_entityish_token) and bool(
            self.analyzer.is_entityish_token[token_id].item()
        )
        candidate_is_punct = 0 <= token_id < len(self.analyzer.is_punct_token) and bool(
            self.analyzer.is_punct_token[token_id].item()
        )

        for neighbor in (candidate_index - 1, candidate_index + 1):
            if neighbor < 0 or neighbor >= seq_len or not selected_mask[neighbor]:
                continue

            neighbor_token_id = int(token_ids[neighbor].item())
            if neighbor_token_id < 0:
                continue

            neighbor_is_digit = 0 <= neighbor_token_id < len(self.analyzer.is_digit_token) and bool(
                self.analyzer.is_digit_token[neighbor_token_id].item()
            )
            neighbor_is_month = 0 <= neighbor_token_id < len(self.analyzer.is_month_token) and bool(
                self.analyzer.is_month_token[neighbor_token_id].item()
            )
            neighbor_is_entity = 0 <= neighbor_token_id < len(self.analyzer.is_entityish_token) and bool(
                self.analyzer.is_entityish_token[neighbor_token_id].item()
            )

            if candidate_is_space and (neighbor_is_month or neighbor_is_entity or neighbor_is_digit):
                bonus += 0.20
            if candidate_is_digit and (neighbor_is_digit or neighbor_is_month or neighbor_is_entity):
                bonus += 0.24
            if candidate_is_punct and neighbor_is_digit:
                bonus += 0.08
            if candidate_is_entity and neighbor_is_entity:
                bonus += 0.12

        return bonus
