"""No Repair baseline.

Never intervenes; all violations pass through unchanged.
Serves as the lower-bound safety baseline.
"""
from .base import CaseResult, run_with_repair_fn
from trajectory import Context


def _repair_fn(ctx: Context) -> list[int]:
    return []


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_repair_fn(traj_json, _repair_fn, dry_run=dry_run, verbose=verbose)
