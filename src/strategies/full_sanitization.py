"""Full Attribution — safety upper bound.

Sanitises every step in the trajectory unconditionally (no risk-threshold gate,
no dependence on the risk estimator).  Because ALL steps are attributed, the
GT violation steps are always a subset of S, so FCR = 1.0 by construction.
This is the maximum-cost baseline against which Multi-Pivot is compared.
"""
from .base import CaseResult, run_with_attribution_fn
from trajectory import Context


def _attribution_fn(ctx: Context) -> list[int]:
    """Return all steps unconditionally."""
    return list(range(ctx.steps))


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_attribution_fn(traj_json, _attribution_fn, dry_run=dry_run, verbose=verbose)
