"""Tests for dynamic tenant priority and budget assignment (Task 3.0)."""
from semserve.priority import TenantPriorityTracker


def test_high_density_tenant_gets_larger_budget():
    tr = TenantPriorityTracker(total_blocks=100)
    tr.update_tenant("A", mean_density=0.9, slo_class=1)
    tr.update_tenant("B", mean_density=0.1, slo_class=1)
    budgets = tr.compute_budgets()
    assert budgets["A"] > budgets["B"]
    assert budgets["A"] + budgets["B"] <= 100


def test_aging_prevents_starvation():
    tr = TenantPriorityTracker(total_blocks=100, w_age=0.05)
    tr.update_tenant("A", mean_density=0.9, slo_class=1)
    tr.update_tenant("B", mean_density=0.1, slo_class=1)
    b0 = tr.compute_budgets()["B"]
    for _ in range(50):           # B 长期拿不到提升，aging 累积
        tr.tick()
    b1 = tr.compute_budgets()["B"]
    assert b1 > b0                # 饿死保护生效


def test_floor_and_ceiling_respected():
    tr = TenantPriorityTracker(total_blocks=100, floor_blocks=10, ceil_blocks=80)
    tr.update_tenant("A", mean_density=1.0, slo_class=2)
    tr.update_tenant("B", mean_density=0.0, slo_class=0)
    budgets = tr.compute_budgets()
    assert budgets["B"] >= 10 and budgets["A"] <= 80


def test_priority_is_dynamic_across_turns():
    # 同一租户新 turn 带来高密度请求 -> 优先级上升
    tr = TenantPriorityTracker(total_blocks=100)
    tr.update_tenant("A", mean_density=0.2, slo_class=1)
    p0 = tr.priorities()["A"]
    tr.update_tenant("A", mean_density=0.9, slo_class=1)
    assert tr.priorities()["A"] > p0
