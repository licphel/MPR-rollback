"""Multi-Pivot (MPR) — proposed method.

Greedy multi-step attribution: selects the minimal set of steps whose
combined desensitisation is projected to reduce accumulated risk below
the target threshold.  Allows localized repair across multiple pivots.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .base import CaseResult, run_with_repair_fn
from evaluate import evaluate_risk
from trajectory import Context, influence_factor, risk_tolerance, risk_decrease_threshold


def attribute(ctx: Context, dry_run: bool = False) -> list[int]:
    """Return the minimal greedy set of step indices to sanitize.

    Steps are ranked by influence-weighted risk contribution; the lowest-cost
    subset that projects ctx.risk below risk_decrease_threshold(ctx) is chosen.
    """
    if ctx.risk < risk_tolerance(ctx):
        return []

    if ctx.steps == 0:
        return [0] if ctx.step_risks.get(0, 0.0) > 0.0 else []

    target = risk_decrease_threshold(ctx)

    # Build (index, weighted_contribution) for all processed steps
    contributions: list[tuple[int, float]] = []
    for i in range(ctx.steps):
        cached = ctx.step_risks.get(i)
        if cached is None:
            # Evaluate any step not yet scored (should be rare — normally all
            # processed steps are cached before attribution is called)
            from trajectory import Trajectory as _T  # avoid circular at module level
            cached = evaluate_risk(ctx.trajectories[i], dry_run=dry_run)
            ctx.step_risks[i] = cached
        step_loc = ctx.trajectories[i].step   # 1-indexed field from JSON
        weight   = influence_factor(step_loc, ctx)
        contributions.append((i, cached * weight))

    # Greedy: pick highest-contribution steps until projected risk ≤ target
    contributions.sort(key=lambda x: x[1], reverse=True)
    selected: list[int] = []
    projected = ctx.risk
    for idx, contrib in contributions:
        if projected <= target:
            break
        selected.append(idx)
        projected -= contrib

    return selected


def _make_repair_fn(dry_run: bool):
    def _repair_fn(ctx: Context) -> list[int]:
        return attribute(ctx, dry_run=dry_run)
    return _repair_fn


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_repair_fn(
        traj_json,
        _make_repair_fn(dry_run),
        dry_run=dry_run,
        verbose=verbose,
    )
