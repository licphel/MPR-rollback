"""Ablation B: Multi-Pivot with random step selection.

Identical to multi_pivot, but steps are selected in random order instead of
descending recency-weighted score. Purpose: verify that greedy attribution
outperforms random selection, i.e. the ranking matters.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.base import CaseResult, run_with_repair_fn
from evaluate import evaluate_risk
from trajectory import Context, risk_tolerance, risk_decrease_threshold

# Fixed seed for reproducibility across runs; set to None for true randomness.
_SEED = 42


def attribute(ctx: Context, dry_run: bool = False) -> list[int]:
    if ctx.risk < risk_tolerance(ctx):
        return []

    if ctx.steps == 0:
        return [0] if ctx.step_risks.get(0, 0.0) > 0.0 else []

    target = risk_decrease_threshold(ctx)

    indices = list(range(ctx.steps))
    for i in indices:
        cached = ctx.step_risks.get(i)
        if cached is None:
            cached = evaluate_risk(ctx.trajectories[i], dry_run=dry_run)
            ctx.step_risks[i] = cached

    # Shuffle instead of sorting by score
    rng = random.Random(_SEED)
    rng.shuffle(indices)

    selected: list[int] = []
    zeroed: set[int] = set()
    for idx in indices:
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
