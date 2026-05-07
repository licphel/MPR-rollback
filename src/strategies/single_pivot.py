"""Single-Pivot baseline.

When risk exceeds the tolerance threshold, selects the single step with the
highest influence-weighted risk contribution and sanitizes only that step.
Represents a minimal-intervention approach without multi-step attribution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .base import CaseResult, run_with_repair_fn
from trajectory import Context, influence_factor


def _repair_fn(ctx: Context) -> list[int]:
    """Return the single highest influence-weighted risk step."""
    if ctx.steps == 0:
        return []

    best_idx   = -1
    best_score = -1.0

    for i in range(ctx.steps):
        risk  = ctx.step_risks.get(i, 0.0)
        step_loc = ctx.trajectories[i].step   # 1-indexed from JSON
        score = risk * influence_factor(step_loc, ctx)
        if score > best_score:
            best_score = score
            best_idx   = i

    return [best_idx] if best_idx >= 0 and best_score > 0.0 else []


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_repair_fn(traj_json, _repair_fn, dry_run=dry_run, verbose=verbose)
