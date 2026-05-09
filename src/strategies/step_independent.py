"""Step-Independent Sanitization baseline.

Each step is sanitized independently if its risk score exceeds zero.
No global attribution, no stopping criterion — every risky step is
sanitized regardless of whether the cumulative risk is already below
the threshold.

This represents the naive per-step approach: "sanitize anything that
looks risky." Compared to multi_pivot, it achieves similar or higher
FCR but at significantly higher sanitization cost (SC), because it
cannot identify the minimal sufficient set of steps to sanitize.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .base import CaseResult, run_with_attribution_fn
from trajectory import Context


def _attribution_fn(ctx: Context) -> list[int]:
    """Return all steps with risk > 0, regardless of cumulative risk."""
    return [i for i in range(ctx.steps) if ctx.step_risks.get(i, 0.0) > 0.0]


def run_trajectory(traj_json: dict, dry_run: bool = False, verbose: bool = False) -> CaseResult:
    return run_with_attribution_fn(traj_json, _attribution_fn, dry_run=dry_run, verbose=verbose)
