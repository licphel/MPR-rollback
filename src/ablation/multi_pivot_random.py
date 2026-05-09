"""Ablation: Multi-Pivot with random step selection.

Identical stopping criterion to multi_pivot, but steps are selected in
random order instead of descending weighted score.  Purpose: verify that
greedy ranking matters — random selection should achieve lower FCR and/or
higher SC than the greedy baseline.
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.base import CaseResult, run_with_attribution_fn
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
