"""Tests for the cross-request water-filling allocator (Task 3.1)."""
import torch

from semserve.allocator import SemanticAllocator, RequestCacheState


def _req(rid, scores, pinned=1, floor=0):
    return RequestCacheState(
        request_id=rid,
        block_scores=torch.tensor(scores),
        num_pinned_blocks=pinned,
        floor_blocks=floor,
    )


def test_pressure_reclaims_from_low_value_request():
    # A 高密度（全 0.9），B 低密度（全 0.1），各持有 4 block，需回收 3 个
    alloc = SemanticAllocator()
    plan = alloc.reclaim(
        [_req("A", [1.0, 0.9, 0.9, 0.9]), _req("B", [1.0, 0.1, 0.1, 0.1])],
        num_blocks_needed=3,
    )
    # B 的 3 个低分 block 全部被收走，A 不动
    assert plan.reclaim_per_request == {"B": 3}


def test_pinned_and_floor_are_never_reclaimed():
    alloc = SemanticAllocator()
    plan = alloc.reclaim(
        [_req("A", [1.0, 0.2, 0.2, 0.2], pinned=1, floor=2)],
        num_blocks_needed=4,
    )
    # floor=2 -> 最多收走 2 个；剩余缺口标记为需要抢占
    assert plan.reclaim_per_request == {"A": 2}
    assert plan.unmet_blocks == 2
    assert plan.preempt_candidates == ["A"]


def test_marginal_value_interleaving():
    # A=[1.0,0.8,0.3], B=[1.0,0.5,0.4]，收 2 个 -> 先收 A 的 0.3，再收 B 的 0.4
    alloc = SemanticAllocator()
    plan = alloc.reclaim(
        [_req("A", [1.0, 0.8, 0.3]), _req("B", [1.0, 0.5, 0.4])],
        num_blocks_needed=2,
    )
    assert plan.reclaim_per_request == {"A": 1, "B": 1}
