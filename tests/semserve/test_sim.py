"""Tests for the multi-tenant simulator's pure helpers."""
from semserve.sim import ReqResult, aggregate_metrics, interp_quality, jain_index


def test_interp_quality_exact_points():
    curve = [(0.1, 0.80), (0.25, 0.72), (0.5, 0.64)]
    assert interp_quality(curve, 0.1) == 0.80
    assert interp_quality(curve, 0.5) == 0.64


def test_interp_quality_linear_midpoint():
    curve = [(0.25, 0.72), (0.5, 0.64)]
    # halfway between 0.25 and 0.5 -> halfway between 0.72 and 0.64
    assert abs(interp_quality(curve, 0.375) - 0.68) < 1e-9


def test_interp_quality_clamps_to_endpoints():
    curve = [(0.1, 0.80), (0.5, 0.64)]
    assert interp_quality(curve, 0.01) == 0.80   # below min budget
    assert interp_quality(curve, 0.99) == 0.64   # above max budget


def test_jain_index_equal_allocation_is_one():
    assert abs(jain_index([5, 5, 5, 5]) - 1.0) < 1e-9


def test_jain_index_single_winner_is_one_over_n():
    assert abs(jain_index([10, 0, 0, 0]) - 0.25) < 1e-9


def test_aggregate_metrics_percentiles_and_goodput():
    recs = [
        ReqResult(tenant="A", ttft=1.0, tpot=1.0, retained=1.0, correct=True, slo_met=True),
        ReqResult(tenant="A", ttft=3.0, tpot=1.0, retained=0.5, correct=False, slo_met=True),
        ReqResult(tenant="B", ttft=9.0, tpot=1.0, retained=1.0, correct=True, slo_met=False),
    ]
    m = aggregate_metrics(recs)
    assert m["n"] == 3
    assert m["ttft_p50"] == 3.0                  # median of {1,3,9}
    assert m["ttft_p99"] == 9.0                  # nearest-rank top
    assert abs(m["goodput"] - 1 / 3) < 1e-9      # only rec1 is slo_met AND correct
