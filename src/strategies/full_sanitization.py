"""Full Sanitization baseline.

When risk exceeds the tolerance threshold, sanitizes every step processed
so far and rolls back to the beginning. Maximises safety at the cost of
discarding all trajectory context.
"""
from .base import CaseResult, run_with_repair_fn
from trajectory import Context


def _repair_fn(ctx: Context) -> list[int]:
    """Return all step indices processed so far (0 … ctx.steps-1)."""
    return list(range(ctx.steps))


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_repair_fn(traj_json, _repair_fn, dry_run=dry_run, verbose=verbose)
