"""MPR (Multi-Pivot Rollback) — original entry point.

This file is kept for backwards compatibility.
The core logic has been factored into:
  - evaluate.py     (LLM helpers)
  - trajectory.py   (data classes + risk helpers)
  - strategies/     (all repair strategies)

To run the full benchmark comparison use runner.py instead:
  python src/runner.py --dry-run --data trajectory/real_trajectories.json
"""
import os
import sys

from evaluate   import evaluate_risk, desensitize_act          # noqa: F401
from trajectory import (                                        # noqa: F401
    Trajectory, Context,
    influence_factor, risk_tolerance, risk_decrease_threshold,
)
from strategies.multi_pivot import attribute                    # noqa: F401

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Thin wrapper kept for backward-compat (returns old-style dict)
# ---------------------------------------------------------------------------

def desensitize(target_step: int, ctx: Context) -> float:
    desensitize_act(target_step, ctx)
    return sum(ctx.step_risks.get(i, 0.0) for i in range(ctx.steps))


def run_trajectory(traj_json: dict) -> dict:
    """Run a single trajectory under the MPR strategy (original interface)."""
    from strategies.multi_pivot import run_trajectory as _mpr_run

    result = _mpr_run(traj_json, dry_run=False, verbose=False)
    case   = result.to_dict()

    ground_truth = traj_json.get("violated", False)
    detected     = bool(result.rollback_count > 0)

    return {
        "trajectory_id":     case["trajectory_id"],
        "final_risk":        case["final_risk"],
        "rollbacks":         case["rollback_count"],
        "step_risks":        case["step_risks"],
        "detected_violated": detected,
        "ground_truth":      ground_truth,
        "accurate":          detected == ground_truth,
    }


# ---------------------------------------------------------------------------
# __main__ — original CLI (unchanged behaviour)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Run MPR on trajectory data.")
    parser.add_argument("--data", default="trajectory/hf_trajectories.json",
                        help="Path to trajectories JSON file")
    parser.add_argument("-n", type=int, default=None,
                        help="Number of trajectories to run (default: all)")
    parser.add_argument("--out", default="results.json",
                        help="Output file for results (default: results.json)")
    args = parser.parse_args()

    # Resolve path relative to project root
    data = args.data
    if not os.path.isabs(data):
        data = os.path.join(os.path.dirname(_SRC), data)

    with open(data, encoding="utf-8") as f:
        trajectories = _json.load(f)

    if args.n is not None:
        trajectories = trajectories[: args.n]

    results = []
    for traj in trajectories:
        print(f"Running {traj['trajectory_id']} ...", flush=True)
        r = run_trajectory(traj)
        print(f"  final_risk={r['final_risk']:.3f}  rollbacks={r['rollbacks']}"
              f"  accurate={r['accurate']}")
        results.append(r)

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(_SRC), out)

    with open(out, "w", encoding="utf-8") as f:
        _json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} trajectories → {out}")

    n   = len(results)
    tp  = sum(1 for r in results if     r["detected_violated"] and     r["ground_truth"])
    tn  = sum(1 for r in results if not r["detected_violated"] and not r["ground_truth"])
    fp  = sum(1 for r in results if     r["detected_violated"] and not r["ground_truth"])
    fn  = sum(1 for r in results if not r["detected_violated"] and     r["ground_truth"])

    accuracy  = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    print("\n=== Violation-prediction statistics ===")
    print(f"  Accuracy  : {accuracy:.3f}  ({tp+tn}/{n})")
    print(f"  Precision : {precision:.3f}  (TP={tp}, FP={fp})")
    print(f"  Recall    : {recall:.3f}  (TP={tp}, FN={fn})")
    print(f"  F1        : {f1:.3f}")
    print(f"  FP rate   : {fp/(fp+tn) if (fp+tn) else 0.0:.3f}  ({fp} safe traj. flagged)")
