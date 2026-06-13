"""Dynamic tenant priority and cgroup-style KV budget assignment (Task 3.0).

OS analogy: tenants are processes; priority is a dynamic-priority scheduler with
aging; budget is a cgroup memory quota. Priority blends semantic value (the
profiler's per-tenant mean density), a static SLO class, and an aging term that
prevents low-density tenants from starving:

    priority(t) = w_sem * ema_density(t) + w_slo * slo_class(t) + w_age * age(t)
    budget(t)   = clamp(total_blocks * priority(t) / Σ priority, floor, ceil)

Pure algorithm, vLLM-independent, CPU-only — unit-tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _TenantState:
    """Mutable per-tenant scheduling state."""

    ema_density: float
    slo_class: int
    age: int


class TenantPriorityTracker:
    """Track per-tenant dynamic priority and derive KV block budgets."""

    def __init__(
        self,
        total_blocks: int,
        *,
        w_sem: float = 1.0,
        w_slo: float = 1.0,
        w_age: float = 0.0,
        ema_alpha: float = 0.5,
        floor_blocks: float = 0.0,
        ceil_blocks: float | None = None,
    ) -> None:
        self.total_blocks = total_blocks
        self.w_sem = w_sem
        self.w_slo = w_slo
        self.w_age = w_age
        self.ema_alpha = ema_alpha
        self.floor_blocks = floor_blocks
        self.ceil_blocks = ceil_blocks if ceil_blocks is not None else float(total_blocks)
        self._tenants: dict[str, _TenantState] = {}

    def update_tenant(self, tenant_id: str, mean_density: float, slo_class: int) -> None:
        """Register a (new or updated) tenant; EMA-smooth its density, reset aging.

        Resetting age on update encodes "this tenant just got served / refreshed",
        so aging only accrues while a tenant is passed over.
        """
        st = self._tenants.get(tenant_id)
        if st is None:
            self._tenants[tenant_id] = _TenantState(mean_density, slo_class, 0)
        else:
            st.ema_density = self.ema_alpha * mean_density + (1.0 - self.ema_alpha) * st.ema_density
            st.slo_class = slo_class
            st.age = 0

    def tick(self) -> None:
        """Advance the aging clock for every tracked tenant by one step."""
        for st in self._tenants.values():
            st.age += 1

    def priorities(self) -> dict[str, float]:
        """Return current priority score per tenant."""
        return {
            tid: self.w_sem * st.ema_density + self.w_slo * st.slo_class + self.w_age * st.age
            for tid, st in self._tenants.items()
        }

    def compute_budgets(self) -> dict[str, float]:
        """Return per-tenant KV block budget (priority-proportional, floor/ceil clamped)."""
        priorities = self.priorities()
        total_priority = sum(priorities.values())
        n = len(priorities)
        budgets: dict[str, float] = {}
        for tid, priority in priorities.items():
            if total_priority > 0:
                raw = self.total_blocks * (priority / total_priority)
            else:
                raw = self.total_blocks / n if n else 0.0
            budgets[tid] = min(max(raw, self.floor_blocks), self.ceil_blocks)
        return budgets
