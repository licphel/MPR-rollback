"""MPA-Random — Multi-Pivot Attribution with random step selection.

Same framework and stopping criterion as MPA-Greedy, but steps are selected
in random order instead of descending risk score.  Serves as a comparison
point to isolate the contribution of greedy ranking within the MPA framework.
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .base import CaseResult, run_with_attribution_fn
from trajectory import Context, weighted_risk, projected_risk, risk_tolerance, risk_decrease_threshold


def attribute(ctx: Context) -> list[int]:
    """Random-order attribution with same stopping criterion as multi_pivot."""
    if weighted_risk(ctx) < risk_tolerance(ctx):
        return []

    theta = risk_decrease_threshold(ctx)

    indices = list(range(ctx.steps))
    # Per-trajectory seed: time + hash of risk scores for reproducible variance
    seed = int(time.time() * 1000) ^ hash(
        tuple(ctx.step_risks.get(i, 0.0) for i in range(ctx.steps))
    )
    random.Random(seed).shuffle(indices)

    selected: list[int] = []
    zeroed: set[int] = set()
    for idx in indices:
        if ctx.step_risks.get(idx, 0.0) <= 0.0:
            continue
        selected.append(idx)
        zeroed.add(idx)
        if projected_risk(ctx, zeroed) <= theta:
            break

    return selected


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_attribution_fn(traj_json, attribute, dry_run=dry_run, verbose=verbose)
