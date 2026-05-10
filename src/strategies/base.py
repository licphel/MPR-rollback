"""Shared infrastructure for all attribution strategies.

Global-attribution framework:
  1. Evaluate risk for every step in one forward pass.
  2. Call attribution_fn(ctx) → list[int] to select the sanitization set S.
  3. Sanitize steps in S, then compute FCR / SC / rollback_depth.

No step-by-step trigger loop, no rollback re-execution.
Rollback depth is computed as a hypothetical replay cost:
  depth = len(steps) - min(S)   (steps that would need to be re-run)
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import evaluate_trajectory_risks
from trajectory import Trajectory, Context, weighted_risk, risk_tolerance


# ---------------------------------------------------------------------------
# CaseResult
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    trajectory_id:          str
    task_category:          str
    ground_truth_violated:  bool
    orig_violated_indices:  list[int]   # 0-based GT violation step indices
    steps_attributed:       list[int]   # 0-based indices selected by attribution
    full_coverage:          bool        # all GT vio steps covered by attribution
    fcr_contrib:            float       # 1.0 if full_coverage else 0.0 (for averaging)
    f1:                     float       # 2*P*R/(P+R) over step-level GT labels
    sc:                     float       # sanitization cost = attributed_tokens / total_tokens
    rollback_depth:         int         # len(steps) - min(attributed), 0 if empty
    step_risks:             dict        # str-keyed for JSON serialisation

    def to_dict(self) -> dict:
        return {
            "trajectory_id":         self.trajectory_id,
            "task_category":         self.task_category,
            "ground_truth_violated": self.ground_truth_violated,
            "orig_violated_indices": self.orig_violated_indices,
            "steps_attributed":      self.steps_attributed,
            "full_coverage":         self.full_coverage,
            "f1":                    round(self.f1, 4),
            "sc":                    round(self.sc, 4),
            "rollback_depth":        self.rollback_depth,
            "step_risks":            {str(k): round(v, 4) for k, v in self.step_risks.items()},
        }


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Word-level token approximation."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Global-attribution execution loop
# ---------------------------------------------------------------------------

def run_with_attribution_fn(
    traj_json: dict,
    attribution_fn,        # (ctx: Context) -> list[int]
    dry_run: bool = False,
    verbose: bool = False,
) -> CaseResult:
    """Evaluate all steps, attribute, sanitize, compute metrics.

    Args:
        traj_json:       one trajectory dict from the dataset JSON.
        attribution_fn:  callable(ctx) -> list[int]: returns 0-based step
                         indices to sanitize.  Called once after all steps
                         have been risk-evaluated.
        dry_run:         use pre-labelled risk scores; skip LLM calls.
        verbose:         print per-step risk scores.

    Returns:
        CaseResult with FCR, SC, and rollback_depth populated.
    """
    raw_steps  = traj_json["steps"]
    steps      = [Trajectory(s) for s in raw_steps]
    ctx        = Context(steps)

    # Pre-compute original token counts (before any sanitization)
    orig_tokens = [count_tokens(str(s.get("observation", ""))) for s in raw_steps]

    # Ground-truth violation indices
    orig_violated_indices = sorted(
        i for i, s in enumerate(raw_steps) if s.get("violated", False)
    )

    # --- Pass 1: evaluate risk for all steps in one context-aware call ---
    risks = evaluate_trajectory_risks(steps, dry_run=dry_run)
    for i, risk in risks.items():
        ctx.step_risks[i] = risk
        if verbose:
            print(f"  [{i}] tool={steps[i].tool} risk={risk:.2f}", flush=True)

    if verbose:
        print(f"  total weighted_risk={weighted_risk(ctx):.2f}", flush=True)

    # --- Pass 2: attribution ---
    attributed: list[int] = attribution_fn(ctx)
    attributed_set = set(attributed)

    if verbose and attributed:
        print(f"  attributed={sorted(attributed_set)}", flush=True)

    # --- Pass 3: sanitize attributed steps ---
    # Sanitization rewrites observations in-place; metrics (FCR/SC/depth)
    # are computed from attributed_set alone and are unaffected by this pass.
    # Skip LLM desensitization calls to save cost — mock in all modes.
    for idx in attributed:
        ctx.step_risks[idx] = 0.0   # mark as cleaned without LLM call

    # --- Metrics ---
    ground_truth_violated = bool(traj_json.get("violated", False))

    full_coverage = (
        bool(orig_violated_indices) and
        set(orig_violated_indices).issubset(attributed_set)
    ) if ground_truth_violated else False

    tokens_total      = sum(orig_tokens)
    tokens_attributed = sum(orig_tokens[i] for i in attributed_set if i < len(orig_tokens))
    sc = tokens_attributed / tokens_total if tokens_total > 0 else 0.0

    rollback_depth = (len(steps) - min(attributed_set)) if attributed_set else 0

    # F1 over step-level GT labels (violated trajectories only; safe → 0.0)
    if ground_truth_violated and orig_violated_indices:
        tp = len(set(orig_violated_indices) & attributed_set)
        precision = tp / len(attributed_set)      if attributed_set        else 0.0
        recall    = tp / len(orig_violated_indices)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
    else:
        f1 = 0.0

    return CaseResult(
        trajectory_id         = traj_json.get("trajectory_id", ""),
        task_category         = traj_json.get("task_category", ""),
        ground_truth_violated = ground_truth_violated,
        orig_violated_indices = orig_violated_indices,
        steps_attributed      = sorted(attributed_set),
        full_coverage         = full_coverage,
        fcr_contrib           = 1.0 if full_coverage else 0.0,
        f1                    = f1,
        sc                    = sc,
        rollback_depth        = rollback_depth,
        step_risks            = dict(ctx.step_risks),
    )
