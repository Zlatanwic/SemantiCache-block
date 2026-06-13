"""Multi-tenant KV-cache contention simulator.

A calibrated, CPU-only discrete-event model of multiple tenants sharing one GPU's
KV budget. It instantiates the OS-scheduling analogy as runnable code and lets us
compare three policies under load:

  * ``vanilla``  — whole-request preemption + recompute (vLLM's behavior).
  * ``uniform``  — compress every request to 0.5 budget (SnapKV/StreamingLLM-style,
                   no semantics).
  * ``semserve`` — ``TenantPriorityTracker`` (cgroup budgets + aging) +
                   ``SemanticAllocator`` (cross-request water-filling): reclaim the
                   globally lowest-value blocks; preempt only as last resort.

Honesty note for the talk: *quality* is data-grounded — the accuracy of a
compressed request is read from a **measured** accuracy-vs-budget curve
(``block_quality_curve``). *Latency* (TTFT/TPOT) is an explicit analytical model
(prefill ∝ blocks; a preemption pays a re-prefill penalty). This is a simulation
calibrated by real measurements, not a vLLM run.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from semserve.allocator import RequestCacheState, SemanticAllocator
from semserve.priority import TenantPriorityTracker

PREFILL_PER_BLOCK = 1.0  # sim steps of prefill latency per KV block


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested in test_sim.py)
# --------------------------------------------------------------------------- #
@dataclass
class ReqResult:
    """Per-request outcome record produced by the simulator."""

    tenant: str
    ttft: float          # first-token latency (sim steps), incl. recompute penalty
    tpot: float          # time-per-output-token (sim steps)
    retained: float      # final retained KV fraction in [0, 1]
    correct: bool        # sampled from the measured quality curve at `retained`
    slo_met: bool        # ttft <= tenant SLO


def interp_quality(curve: list[tuple[float, float]], frac: float) -> float:
    """Piecewise-linear interpolate accuracy at retained fraction ``frac``.

    ``curve`` is a list of (budget, accuracy) points; values outside the measured
    range clamp to the nearest endpoint (we never extrapolate beyond data).
    """
    pts = sorted(curve)
    if frac <= pts[0][0]:
        return pts[0][1]
    if frac >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= frac <= x1:
            t = (frac - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def jain_index(values: list[float]) -> float:
    """Jain's fairness index: 1.0 = perfectly equal, 1/n = maximally unfair."""
    if not values:
        return 1.0
    s = sum(values)
    sq = sum(v * v for v in values)
    if sq == 0:
        return 1.0
    return (s * s) / (len(values) * sq)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0, 1])."""
    if not sorted_vals:
        return 0.0
    rank = max(1, math.ceil(p * len(sorted_vals)))
    return sorted_vals[rank - 1]


def aggregate_metrics(records: list[ReqResult]) -> dict:
    """Aggregate per-request records into the headline metrics for one policy."""
    if not records:
        return {"n": 0, "ttft_p50": 0.0, "ttft_p99": 0.0, "mean_tpot": 0.0,
                "goodput": 0.0, "fairness": 1.0}
    ttfts = sorted(r.ttft for r in records)
    by_tenant: dict[str, list[ReqResult]] = {}
    for r in records:
        by_tenant.setdefault(r.tenant, []).append(r)
    tenant_goodput = [
        sum(1 for r in rs if r.slo_met and r.correct) / len(rs) for rs in by_tenant.values()
    ]
    return {
        "n": len(records),
        "ttft_p50": _percentile(ttfts, 0.50),
        "ttft_p99": _percentile(ttfts, 0.99),
        "mean_tpot": sum(r.tpot for r in records) / len(records),
        "goodput": sum(1 for r in records if r.slo_met and r.correct) / len(records),
        "fairness": jain_index(tenant_goodput),
    }


# --------------------------------------------------------------------------- #
# Simulation entities
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TenantSpec:
    """Static description of a tenant's workload and SLO."""

    tenant_id: str
    slo_class: int          # 0/1/2 (higher = stricter / more important)
    slo_ttft: float         # TTFT SLO threshold (sim steps)
    prompt_blocks: int      # KV footprint per request
    decode_tokens: int      # decode length (sim steps occupied)
    mean_score: float       # semantic density of this tenant's content in [0, 1]


@dataclass(frozen=True)
class SimConfig:
    """Simulation-wide parameters."""

    total_blocks: int
    arrival_rate: float                         # expected new requests per step (Poisson)
    horizon: int                                # number of sim steps
    block_quality_curve: list[tuple[float, float]]  # measured accuracy vs retained budget
    uniform_quality: float                      # accuracy of uniform compression
    full_quality: float                         # accuracy with no eviction


@dataclass
class SimRun:
    """Output of one policy run."""

    records: list[ReqResult] = field(default_factory=list)
    preemptions: int = 0
    admitted: int = 0


@dataclass
class _Req:
    """Mutable in-flight request state."""

    rid: int
    spec: TenantSpec
    arrival: int
    orig_blocks: int
    blocks: int
    block_scores: list[float]
    remaining: int
    penalty: float = 0.0     # accumulated recompute penalty (re-prefill)
    admit_step: int = -1
    ttft: float = 0.0
    retained: float = 1.0


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm for a Poisson sample."""
    if lam <= 0:
        return 0
    target = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            return k - 1


def _make_scores(spec: TenantSpec, rng: random.Random) -> list[float]:
    """Synthesize per-block semantic scores: pinned head + tenant-mean filler."""
    scores = [1.0]  # pinned head (system prompt / instruction)
    for _ in range(1, spec.prompt_blocks):
        v = spec.mean_score + rng.uniform(-0.1, 0.1)
        scores.append(min(1.0, max(0.0, v)))
    return scores


def _relieve_vanilla(deficit: int, running: list[_Req], now: int) -> tuple[int, list[_Req]]:
    """Preempt most-recently-admitted *prior-step* requests until ``deficit`` freed.

    Excluding requests admitted in the current step is essential: otherwise, when
    several queued requests each need more than the free budget, the scheduler
    ping-pongs (admit A -> preempt A to admit B -> preempt B to admit A -> ...).
    """
    freed = 0
    preempted: list[_Req] = []
    victims = sorted((r for r in running if r.admit_step < now),
                     key=lambda r: r.admit_step, reverse=True)
    for v in victims:
        if freed >= deficit:
            break
        freed += v.blocks
        running.remove(v)
        preempted.append(v)
    return freed, preempted


def _relieve_uniform(deficit: int, running: list[_Req], now: int) -> tuple[int, list[_Req]]:
    """Compress every request to 0.5 budget; preempt only if still short."""
    freed = 0
    for r in running:
        if freed >= deficit:
            break
        target = max(1, round(r.orig_blocks * 0.5))
        give = r.blocks - target
        if give > 0:
            r.blocks = target
            r.retained = target / r.orig_blocks
            freed += give
    preempted: list[_Req] = []
    if freed < deficit:
        extra, preempted = _relieve_vanilla(deficit - freed, running, now)
        freed += extra
    return freed, preempted


def _relieve_semserve(
    deficit: int,
    running: list[_Req],
    alloc: SemanticAllocator,
    tracker: TenantPriorityTracker,
    now: int,
) -> tuple[int, list[_Req]]:
    """Cross-request water-filling reclaim; preempt only the allocator's fallback."""
    budgets = tracker.compute_budgets()
    counts: dict[str, int] = {}
    for r in running:
        counts[r.spec.tenant_id] = counts.get(r.spec.tenant_id, 0) + 1

    states: list[RequestCacheState] = []
    by_id: dict[str, _Req] = {}
    for r in running:
        tenant_budget = budgets.get(r.spec.tenant_id, 0.0)
        floor = int(tenant_budget / counts[r.spec.tenant_id]) if counts[r.spec.tenant_id] else 0
        floor = min(floor, r.blocks)
        cur_scores = sorted(r.block_scores, reverse=True)[: r.blocks]
        states.append(RequestCacheState(str(r.rid), torch.tensor(cur_scores),
                                        num_pinned_blocks=1, floor_blocks=floor))
        by_id[str(r.rid)] = r

    plan = alloc.reclaim(states, deficit)
    freed = 0
    for rid, k in plan.reclaim_per_request.items():
        r = by_id[rid]
        r.blocks -= k
        r.retained = r.blocks / r.orig_blocks
        freed += k

    preempted: list[_Req] = []
    if plan.unmet_blocks > 0:
        for rid in plan.preempt_candidates:
            if freed >= deficit:
                break
            r = by_id[rid]
            if r.admit_step >= now:          # never preempt a same-step admit
                continue
            freed += r.blocks
            running.remove(r)
            preempted.append(r)
    return freed, preempted


def _quality(policy: str, r: _Req, cfg: SimConfig) -> float:
    """Expected correctness probability for a completed request under ``policy``."""
    if policy == "vanilla" or r.retained >= 0.999:
        return cfg.full_quality
    if policy == "uniform":
        return cfg.uniform_quality
    return interp_quality(cfg.block_quality_curve, r.retained)


def simulate(policy: str, tenants: list[TenantSpec], cfg: SimConfig, seed: int = 0) -> SimRun:
    """Run one policy over ``cfg.horizon`` steps and return its outcome records."""
    rng = random.Random(seed)
    alloc = SemanticAllocator()
    tracker = TenantPriorityTracker(cfg.total_blocks, w_sem=1.0, w_slo=0.5, w_age=0.02)
    run = SimRun()
    free = cfg.total_blocks
    running: list[_Req] = []
    queue: list[_Req] = []
    next_id = 0

    for t in range(cfg.horizon):
        # 1. Poisson arrivals.
        for _ in range(_poisson(rng, cfg.arrival_rate)):
            spec = tenants[rng.randrange(len(tenants))]
            queue.append(_Req(next_id, spec, t, spec.prompt_blocks, spec.prompt_blocks,
                              _make_scores(spec, rng), spec.decode_tokens))
            next_id += 1

        # 2. Refresh tenant priorities from current demand (semserve uses budgets).
        for r in running + queue:
            tracker.update_tenant(r.spec.tenant_id, r.spec.mean_score, r.spec.slo_class)

        # 3. Admission (with policy-driven contention relief).
        while queue:
            head = queue[0]
            need = head.blocks
            if free < need:
                deficit = need - free
                if policy == "vanilla":
                    freed, preempted = _relieve_vanilla(deficit, running, t)
                elif policy == "uniform":
                    freed, preempted = _relieve_uniform(deficit, running, t)
                else:
                    freed, preempted = _relieve_semserve(deficit, running, alloc, tracker, t)
                free += freed
                run.preemptions += len(preempted)
                for pr in preempted:                 # requeue with recompute penalty
                    pr.penalty += pr.orig_blocks * PREFILL_PER_BLOCK
                    pr.blocks = pr.orig_blocks
                    pr.retained = 1.0
                    queue.append(pr)
                if free < need:
                    break                            # cannot admit this step
            queue.pop(0)
            free -= need
            head.admit_step = t
            head.ttft = (t - head.arrival) + head.orig_blocks * PREFILL_PER_BLOCK + head.penalty
            running.append(head)
            run.admitted += 1

        # 4. Decode one token; retire finished requests.
        still: list[_Req] = []
        for r in running:
            r.remaining -= 1
            if r.remaining <= 0:
                free += r.blocks
                correct = rng.random() < _quality(policy, r, cfg)
                run.records.append(ReqResult(
                    tenant=r.spec.tenant_id, ttft=r.ttft, tpot=1.0,
                    retained=r.retained, correct=correct, slo_met=r.ttft <= r.spec.slo_ttft,
                ))
            else:
                still.append(r)
        running = still

        # 5. Age all tenants (starvation clock).
        tracker.tick()

    return run
