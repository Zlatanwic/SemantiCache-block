"""Tests for segment-score -> block-score mapping."""
import math

import torch

from semserve.block_mapping import (
    block_aligned_keep_indices,
    segment_scores_to_block_scores,
)


def test_uniform_segment_covers_blocks():
    # 一个 segment 覆盖 token [0, 32)，分数 0.8，block_size=16 -> 两个 block 都是 0.8
    seg_bounds = [(0, 32)]
    seg_scores = [0.8]
    out = segment_scores_to_block_scores(seg_bounds, seg_scores, seq_len=32, block_size=16)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([0.8, 0.8]))


def test_partial_overlap_uses_token_weighted_mean():
    # segment A [0,8) 分数 1.0，segment B [8,16) 分数 0.0 -> block0 = 0.5
    out = segment_scores_to_block_scores([(0, 8), (8, 16)], [1.0, 0.0], seq_len=16, block_size=16)
    assert torch.allclose(out, torch.tensor([0.5]))


def test_pinned_segment_forces_max_score():
    # pinned segment（如 system prompt / 最新 user query）所在 block 强制 1.0
    out = segment_scores_to_block_scores(
        [(0, 16), (16, 32)], [0.2, 0.3], seq_len=32, block_size=16,
        pinned_bounds=[(0, 16)],
    )
    assert torch.allclose(out, torch.tensor([1.0, 0.3]))


def test_tail_block_padding():
    # seq_len=20, block_size=16 -> 2 个 block，尾块按实际 4 个 token 加权
    out = segment_scores_to_block_scores([(0, 20)], [0.6], seq_len=20, block_size=16)
    assert out.shape == (2,)


# --- block_aligned_keep_indices: paged-KV realistic whole-block eviction ---

def test_whole_block_selection_keeps_lowest_eviction_blocks():
    # 4 个 size-4 block，eviction 压力 block0<block2<block3<block1；budget=8 -> 保留 block0+block2
    scores = torch.tensor([0.1] * 4 + [0.9] * 4 + [0.2] * 4 + [0.8] * 4)
    keep = block_aligned_keep_indices(scores, budget=8, block_size=4)
    assert keep.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]


def test_forced_block_with_neg_inf_kept_under_tight_budget():
    # block3 含一个 -inf（pinned/protected）token -> 整块强制保留，即使分数差
    scores = torch.tensor([0.1] * 4 + [0.1] * 4 + [0.1] * 4 + [-math.inf, 0.5, 0.5, 0.5])
    keep = block_aligned_keep_indices(scores, budget=4, block_size=4)
    assert keep.tolist() == [12, 13, 14, 15]


def test_kept_indices_are_block_aligned():
    # 保留的 index 必须成完整块（不留半块），符合 paged 显存约束
    torch.manual_seed(0)
    scores = torch.rand(64)
    keep = set(block_aligned_keep_indices(scores, budget=32, block_size=16).tolist())
    for block_start in range(0, 64, 16):
        block = set(range(block_start, block_start + 16))
        # 该块要么全保留要么全不保留
        assert block <= keep or not (block & keep)


def test_budget_covers_all_returns_everything():
    scores = torch.rand(16)
    keep = block_aligned_keep_indices(scores, budget=16, block_size=4)
    assert keep.tolist() == list(range(16))
