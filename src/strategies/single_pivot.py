"""Single-Pivot — minimal-intervention baseline.

Attributes only the single highest-risk step (argmax over influence-weighted
risk scores), gated by the same global trigger as Multi-Pivot.  Represents
the smallest possible intervention: one sanitisation per triggered trajectory.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .base import CaseResult, run_with_attribution_fn
from trajectory import Context, influence_factor, weighted_risk, risk_tolerance


def _attribution_fn(ctx: Context) -> list[int]:
    """Return argmax_i risk[i]*phi(i), or [] if total risk < tau."""
    if weighted_risk(ctx) < risk_tolerance(ctx):
        return []

    best_idx   = -1
    best_score = -1.0
    for i in range(ctx.steps):
        score = ctx.step_risks.get(i, 0.0) * influence_factor(ctx.trajectories[i].step, ctx)
        if score >= best_score and score > 0.0:  # tie-break: prefer later step
            best_score = score
            best_idx   = i

    return [best_idx] if best_idx >= 0 and best_score > 0.0 else []


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_attribution_fn(traj_json, _attribution_fn, dry_run=dry_run, verbose=verbose)
