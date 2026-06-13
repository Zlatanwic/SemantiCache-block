"""Cross-request water-filling KV block allocator (Task 3.1).

OS analogy: page reclaim. Under global memory pressure, instead of preempting
whole requests (vLLM's OOM-killer), reclaim the globally lowest-marginal-value
blocks across all running requests.

Each request exposes per-block semantic scores. The top
``max(num_pinned_blocks, floor_blocks)`` blocks are protected (pinned head/tail +
the tenant's cgroup floor); the rest are reclaimable, cheapest-score-first.
Because each request's value-vs-blocks curve is concave (descending prefix sum),
greedy reclaim of the globally lowest marginal score is optimal.

Pure algorithm, vLLM-independent, CPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RequestCacheState:
    """A running request's KV state as seen by the allocator."""

    request_id: str
    block_scores: torch.Tensor          # per-block semantic value, shape (num_blocks,)
    num_pinned_blocks: int = 1          # protected head/tail blocks (never reclaimed)
    floor_blocks: int = 0               # tenant cgroup floor share (never reclaimed)


@dataclass
class ReclaimPlan:
    """Output of a reclaim pass: who gives up how many blocks, and the shortfall."""

    reclaim_per_request: dict[str, int] = field(default_factory=dict)
    unmet_blocks: int = 0
    preempt_candidates: list[str] = field(default_factory=list)


class SemanticAllocator:
    """Reclaim blocks by ascending global marginal semantic value."""

    def reclaim(
        self, states: list[RequestCacheState], num_blocks_needed: int
    ) -> ReclaimPlan:
        """Return a ReclaimPlan satisfying up to ``num_blocks_needed`` blocks."""
        # (score, request_id) for every reclaimable block across all requests.
        reclaimable: list[tuple[float, str]] = []
        protected_value: dict[str, float] = {}
        for st in states:
            scores = st.block_scores.tolist()
            protect = max(st.num_pinned_blocks, st.floor_blocks)
            # Protect the highest-value blocks; the rest (lowest scores) are reclaimable.
            ordered = sorted(scores, reverse=True)
            protected_value[st.request_id] = sum(ordered[:protect])
            for score in ordered[protect:]:
                reclaimable.append((score, st.request_id))

        reclaimable.sort(key=lambda x: x[0])  # cheapest marginal value first

        reclaim_per_request: dict[str, int] = {}
        for score, rid in reclaimable[:num_blocks_needed]:
            reclaim_per_request[rid] = reclaim_per_request.get(rid, 0) + 1

        reclaimed = sum(reclaim_per_request.values())
        unmet = max(0, num_blocks_needed - reclaimed)

        preempt_candidates: list[str] = []
        if unmet > 0:
            # Least-valuable retained requests are the best preemption victims.
            preempt_candidates = sorted(
                (st.request_id for st in states),
                key=lambda rid: protected_value[rid],
            )

        return ReclaimPlan(
            reclaim_per_request=reclaim_per_request,
            unmet_blocks=unmet,
            preempt_candidates=preempt_candidates,
        )
