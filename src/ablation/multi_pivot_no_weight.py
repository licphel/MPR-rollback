"""Ablation A: Multi-Pivot without recency weighting.

Identical to multi_pivot, but steps are ranked by raw risk contribution c_i
instead of the recency-weighted score w_i = c_i * (step_loc / t).
Purpose: isolate the contribution of the position weight to performance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.base import CaseResult, run_with_repair_fn
from evaluate import evaluate_risk
from trajectory import Context, risk_tolerance, risk_decrease_threshold


def attribute(ctx: Context, dry_run: bool = False) -> list[int]:
    if ctx.risk < risk_tolerance(ctx):
        return []

    if ctx.steps == 0:
        return [0] if ctx.step_risks.get(0, 0.0) > 0.0 else []

    target = risk_decrease_threshold(ctx)

    # Rank by raw c_i only — no position weight
    contributions: list[tuple[int, float]] = []
    for i in range(ctx.steps):
        cached = ctx.step_risks.get(i)
        if cached is None:
            cached = evaluate_risk(ctx.trajectories[i], dry_run=dry_run)
            ctx.step_risks[i] = cached
        contributions.append((i, cached))   # <-- no influence_factor

    contributions.sort(key=lambda x: x[1], reverse=True)
    selected: list[int] = []
    zeroed: set[int] = set()
    for idx, _ in contributions:
        selected.append(idx)
        zeroed.add(idx)
        recomputed = sum(
            0.0 if i in zeroed else ctx.step_risks.get(i, 0.0)
            for i in range(ctx.steps)
        )
        if recomputed <= target:
            break

    return selected


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    def _repair_fn(ctx: Context) -> list[int]:
        return attribute(ctx, dry_run=dry_run)
    return run_with_repair_fn(traj_json, _repair_fn, dry_run=dry_run, verbose=verbose)
