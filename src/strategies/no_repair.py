"""No Attribution baseline.

Never attributes any step; all violations pass through unchanged.
Serves as the lower-bound safety baseline (FCR = 0).
"""
from .base import CaseResult, run_with_attribution_fn
from trajectory import Context


def _attribution_fn(ctx: Context) -> list[int]:
    return []


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_attribution_fn(traj_json, _attribution_fn, dry_run=dry_run, verbose=verbose)
