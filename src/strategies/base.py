"""Shared infrastructure for all repair strategies.

Provides:
  - CaseResult  — per-trajectory result container
  - count_tokens — simple word-count token approximation
  - run_with_repair_fn — unified execution loop parameterised by repair policy
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

# Allow `from strategies.base import ...` whether run as a package or directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import evaluate_risk, desensitize_act
from trajectory import Trajectory, Context, risk_tolerance, risk_decrease_threshold


# ---------------------------------------------------------------------------
# CaseResult
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    trajectory_id:        str
    task_category:        str
    ground_truth_violated: bool
    post_repair_violated:  bool
    steps_sanitized:       list[int]   # 0-based indices of sanitized steps
    rollback_count:        int
    rollback_depth:        int         # cumulative steps rolled back across all rollbacks
    tokens_total:          int         # word-count sum over all observations
    tokens_sanitized:      int         # word-count of sanitized steps (original content)
    sc:                    float       # sanitization cost = tokens_sanitized / tokens_total
    final_risk:            float
    step_risks:            dict        # str-keyed for JSON serialisation

    def to_dict(self) -> dict:
        return {
            "trajectory_id":          self.trajectory_id,
            "task_category":          self.task_category,
            "ground_truth_violated":  self.ground_truth_violated,
            "post_repair_violated":   self.post_repair_violated,
            "steps_sanitized":        self.steps_sanitized,
            "rollback_count":         self.rollback_count,
            "rollback_depth":         self.rollback_depth,
            "tokens_total":           self.tokens_total,
            "tokens_sanitized":       self.tokens_sanitized,
            "sc":                     round(self.sc, 4),
            "final_risk":             round(self.final_risk, 4),
            "step_risks":             {str(k): round(v, 4) for k, v in self.step_risks.items()},
        }


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Word-level token approximation."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Shared execution loop
# ---------------------------------------------------------------------------

def run_with_repair_fn(
    traj_json: dict,
    repair_fn,           # (ctx: Context) -> list[int]
    dry_run: bool = False,
    verbose: bool = False,
) -> CaseResult:
    """Execute a trajectory under an arbitrary repair policy.

    Args:
        traj_json:  one trajectory dict from the dataset JSON.
        repair_fn:  callable(ctx) -> list[int]: given the current context,
                    return the 0-based step indices to sanitize.
                    Return [] to skip intervention.
        dry_run:    use pre-labelled risk scores; skip LLM calls.
        verbose:    print step-by-step progress.

    Returns:
        A CaseResult with all evaluation metrics populated.
    """
    raw_steps   = traj_json["steps"]
    steps       = [Trajectory(s) for s in raw_steps]
    ctx         = Context(steps)

    rollback_count  = 0
    rollback_depth  = 0
    sanitized_set: set[int] = set()

    # Pre-compute original token counts from raw JSON (before any in-place edits)
    orig_tokens = [count_tokens(str(s.get("observation", ""))) for s in raw_steps]

    # Original per-step violated flags (ground truth for attribution quality)
    orig_violated_indices = {
        i for i, s in enumerate(raw_steps) if s.get("violated", False)
    }

    while True:
        cur_idx = ctx.steps
        traj    = ctx.current_trajectory()

        # Evaluate risk (use cache if this step was already processed + possibly sanitised)
        if cur_idx in ctx.step_risks:
            step_risk = ctx.step_risks[cur_idx]
        else:
            step_risk = evaluate_risk(traj, dry_run=dry_run)
            ctx.step_risks[cur_idx] = step_risk

        ctx.risk += step_risk
        done = ctx.next()   # advances ctx.steps; True when we've passed the last step

        if verbose:
            print(f"  [{cur_idx}] tool={traj.tool} risk={step_risk:.2f} "
                  f"cumul={ctx.risk:.2f}", flush=True)

        # --- Repair decision ---
        if ctx.risk >= risk_tolerance(ctx):
            to_sanitize = repair_fn(ctx)

            if to_sanitize:
                for idx in to_sanitize:
                    desensitize_act(idx, ctx, dry_run=dry_run)
                    sanitized_set.add(idx)

                rollback_pt    = min(to_sanitize)
                prev_steps     = ctx.steps          # steps count before rollback
                rollback_depth += prev_steps - rollback_pt

                # Recompute accumulated risk from the desensitised prefix
                ctx.risk = sum(ctx.step_risks.get(i, 0.0) for i in range(rollback_pt))
                ctx.rollback_to(rollback_pt)
                rollback_count += 1

                if verbose:
                    print(f"  → rollback to {rollback_pt}, sanitized {to_sanitize}", flush=True)
                continue   # restart loop from rollback_pt

        if done:
            break

    # --- Build metrics ---
    ground_truth_violated = bool(traj_json.get("violated", False))

    # post_repair_violated: still violated iff we never sanitized a violated step
    if not ground_truth_violated:
        post_repair_violated = False
    else:
        post_repair_violated = not bool(orig_violated_indices & sanitized_set)

    tokens_total     = sum(orig_tokens)
    tokens_sanitized = sum(orig_tokens[i] for i in sanitized_set if i < len(orig_tokens))
    sc               = tokens_sanitized / tokens_total if tokens_total > 0 else 0.0

    return CaseResult(
        trajectory_id         = traj_json.get("trajectory_id", ""),
        task_category         = traj_json.get("task_category", ""),
        ground_truth_violated = ground_truth_violated,
        post_repair_violated  = post_repair_violated,
        steps_sanitized       = sorted(sanitized_set),
        rollback_count        = rollback_count,
        rollback_depth        = rollback_depth,
        tokens_total          = tokens_total,
        tokens_sanitized      = tokens_sanitized,
        sc                    = sc,
        final_risk            = ctx.risk,
        step_risks            = dict(ctx.step_risks),
    )
