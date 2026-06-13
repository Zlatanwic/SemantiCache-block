"""Map segment-level retention scores to paged-attention block scores.

This is the bridge between the RetentionPlan IR (segment decisions) and
vLLM's block-granular KV cache: a block's score is the token-weighted mean
of the segment scores covering it, with pinned segments forcing 1.0.
"""
from __future__ import annotations

import torch


def segment_scores_to_block_scores(
    segment_bounds: list[tuple[int, int]],
    segment_scores: list[float],
    *,
    seq_len: int,
    block_size: int,
    pinned_bounds: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Return per-block scores of shape (ceil(seq_len / block_size),)."""
    token_scores = torch.zeros(seq_len, dtype=torch.float32)
    token_weight = torch.zeros(seq_len, dtype=torch.float32)
    for (start, end), score in zip(segment_bounds, segment_scores):
        start, end = max(0, start), min(seq_len, end)
        token_scores[start:end] += score
        token_weight[start:end] += 1.0
    token_scores = token_scores / token_weight.clamp(min=1.0)

    num_blocks = (seq_len + block_size - 1) // block_size
    pad = num_blocks * block_size - seq_len
    padded = torch.nn.functional.pad(token_scores, (0, pad))
    mask = torch.nn.functional.pad(torch.ones(seq_len), (0, pad))
    block_scores = (padded.view(num_blocks, block_size).sum(-1)
                    / mask.view(num_blocks, block_size).sum(-1).clamp(min=1.0))

    if pinned_bounds:
        for start, end in pinned_bounds:
            b0, b1 = start // block_size, (max(start, end - 1)) // block_size
            block_scores[b0 : b1 + 1] = 1.0
    return block_scores


def block_aligned_keep_indices(
    eviction_scores: torch.Tensor,
    *,
    budget: int,
    block_size: int,
) -> torch.Tensor:
    """Select whole blocks to keep under a token budget (paged-KV realistic).

    Eviction granularity is the whole block: a block is either fully kept or
    fully evicted, matching what a paged KV cache (vLLM) can release. Any block
    containing a ``-inf`` (never-evict / pinned / protected) token is
    force-kept regardless of budget; remaining budget is filled with the
    lowest-eviction-pressure blocks first.

    Args:
        eviction_scores: per-token scores; higher means more evictable,
            ``-inf`` means never evict.
        budget: maximum number of tokens to keep (soft; rounded to whole blocks).
        block_size: paged block size (e.g. 16/32/64).

    Returns:
        Sorted 1-D LongTensor of kept token indices.
    """
    seq_len = eviction_scores.shape[0]
    device = eviction_scores.device
    if budget >= seq_len:
        return torch.arange(seq_len, device=device)

    block_size = max(1, block_size)
    num_blocks = (seq_len + block_size - 1) // block_size

    forced: list[tuple[int, int]] = []
    candidates: list[tuple[float, int, int]] = []
    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, seq_len)
        block = eviction_scores[start:end]
        finite = block[~torch.isneginf(block)]
        if finite.numel() < (end - start):
            forced.append((start, end))
        else:
            candidates.append((float(finite.mean()), start, end))

    kept: list[int] = []
    used = 0
    for start, end in forced:
        kept.extend(range(start, end))
        used += end - start

    candidates.sort(key=lambda item: item[0])
    for _, start, end in candidates:
        n = end - start
        if used + n > budget:
            break
        kept.extend(range(start, end))
        used += n

    return torch.tensor(sorted(kept), dtype=torch.long, device=device)
