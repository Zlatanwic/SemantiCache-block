"""Integration test: the simulator reproduces the core SemServe contrast.

Under a load that forces KV contention, vanilla vLLM preempts whole requests
(recompute storms -> tail latency), while SemServe compresses the lowest-value
blocks instead (no recompute), and uniform compression sacrifices more quality.
"""
from semserve.sim import SimConfig, TenantSpec, aggregate_metrics, simulate


def _tenants() -> list[TenantSpec]:
    return [
        # low-density RAG tenant: big footprint, lots of reclaimable filler
        TenantSpec("rag", slo_class=1, slo_ttft=40, prompt_blocks=8, decode_tokens=10, mean_score=0.2),
        # high-density chat tenant: small footprint, dense content
        TenantSpec("chat", slo_class=1, slo_ttft=40, prompt_blocks=2, decode_tokens=5, mean_score=0.85),
    ]


def _cfg() -> SimConfig:
    return SimConfig(
        total_blocks=20,
        arrival_rate=0.7,
        horizon=400,
        # block-gate-like: semantic retention holds quality down to ~25% budget
        block_quality_curve=[(0.1, 0.32), (0.25, 0.95), (0.5, 0.95), (1.0, 0.95)],
        uniform_quality=0.45,   # uniform/attention-only compression at 0.5 budget
        full_quality=0.95,
    )


def test_semserve_compresses_instead_of_preempting():
    tenants, cfg = _tenants(), _cfg()
    van = simulate("vanilla", tenants, cfg, seed=0)
    sem = simulate("semserve", tenants, cfg, seed=0)
    uni = simulate("uniform", tenants, cfg, seed=0)

    # Contention is real for vanilla; SemServe avoids most preemptions.
    assert van.preemptions > 0
    assert sem.preemptions < van.preemptions

    a_van = aggregate_metrics(van.records)
    a_sem = aggregate_metrics(sem.records)
    a_uni = aggregate_metrics(uni.records)

    # No recompute storms -> lower p99 first-token latency.
    assert a_sem["ttft_p99"] < a_van["ttft_p99"]
    # Semantic compression preserves quality better than uniform compression.
    assert a_sem["goodput"] > a_uni["goodput"]
