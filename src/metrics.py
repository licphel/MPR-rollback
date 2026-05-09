"""Metric computation for attribution strategy evaluation.

Four core metrics:
  FCR  — Full-Coverage Rate: fraction of violated trajectories where ALL
          ground-truth violation steps are covered by the attribution set.
  F1   — mean per-trajectory F1 over step-level GT labels (violated only).
  SC   — Sanitization Cost: attributed tokens / total tokens (across all
          trajectories, including safe ones).
  RD   — Rollback Depth: mean hypothetical replay cost = len(steps) - min(S),
          reflecting how far back execution would need to restart.
"""
from __future__ import annotations
from typing import Sequence


def compute_summary(cases: Sequence[dict], strategy_name: str) -> dict:
    n = len(cases)
    if n == 0:
        return {"strategy": strategy_name, "n_trajectories": 0}

    violated_cases = [c for c in cases if c["ground_truth_violated"]]
    n_violated = len(violated_cases)

    def _mean(values) -> float:
        vals = list(values)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    # FCR: over violated trajectories only
    fcr = (
        round(sum(1 for c in violated_cases if c["full_coverage"]) / n_violated, 4)
        if n_violated > 0 else 0.0
    )

    # F1: mean over violated trajectories only
    avg_f1 = (
        _mean(c["f1"] for c in violated_cases)
        if n_violated > 0 else 0.0
    )

    # SC and RD: over all trajectories (safe trajectories contribute 0)
    avg_sc             = _mean(c["sc"]             for c in cases)
    avg_rollback_depth = _mean(c["rollback_depth"] for c in cases)

    return {
        "strategy":            strategy_name,
        "n_trajectories":      n,
        "n_violated":          n_violated,
        "n_safe":              n - n_violated,
        "fcr":                 fcr,
        "avg_f1":              avg_f1,
        "avg_sc":              avg_sc,
        "avg_rollback_depth":  avg_rollback_depth,
    }
