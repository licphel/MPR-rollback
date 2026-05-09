"""Multi-Pivot (MPR) — proposed method.

Greedy global attribution: selects the minimal set S of steps whose combined
sanitization projects the total weighted risk below the target threshold theta.
Achieves higher FCR than single-pivot while incurring lower cost than full
attribution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .base import CaseResult, run_with_attribution_fn
from trajectory import (
    Context, influence_factor,
    weighted_risk, projected_risk,
    risk_tolerance, risk_decrease_threshold,
)


def attribute(ctx: Context) -> list[int]:
    """Greedy multi-pivot attribution.

    Steps are ranked by influence-weighted risk contribution w_i = c_i * phi(i).
    We greedily add the highest-scoring step to S, recompute projected risk with
    those steps zeroed out, and stop once projected_risk(S) <= theta.

    Returns [] if total weighted risk is below the trigger threshold.
    """
    if weighted_risk(ctx) < risk_tolerance(ctx):
        return []

    theta = risk_decrease_threshold(ctx)

    # Build ranked list: primary sort by score descending, tie-break by index descending
    # (prefer later steps — same risk reduction but smaller rollback depth)
    contributions = [
        (i, ctx.step_risks.get(i, 0.0) * influence_factor(ctx.trajectories[i].step, ctx))
        for i in range(ctx.steps)
    ]
    contributions.sort(key=lambda x: (x[1], x[0]), reverse=True)

    selected: list[int] = []
    zeroed: set[int] = set()

    for idx, score in contributions:
        if score <= 0.0:
            continue  # zero-risk step, skip but don't stop
        selected.append(idx)
        zeroed.add(idx)
        if projected_risk(ctx, zeroed) <= theta:
            break

    return selected


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_attribution_fn(traj_json, attribute, dry_run=dry_run, verbose=verbose)
