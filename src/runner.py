"""Benchmark runner for all repair strategies.

Usage examples
--------------
# Dry run all strategies (no API calls):
python src/runner.py --dry-run --data trajectory/real_trajectories.json

# Real run, all strategies, first 10 trajectories:
python src/runner.py --data trajectory/real_trajectories.json -n 10

# Single strategy, verbose:
python src/runner.py --strategy multi_pivot --data trajectory/real_trajectories.json --verbose

Output
------
results/strategy_{name}.json  for each selected strategy.
Each file contains:
  {
    "summary": { ...metrics... },
    "cases":   [ ...per-case dicts... ]
  }
"""
import argparse
import json
import os
import sys
import time

# Ensure src/ is on the path regardless of working directory
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

from strategies import STRATEGIES
from metrics import compute_summary
from evaluate import set_model_source, get_active_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_strategy(
    name: str,
    run_fn,
    trajectories: list[dict],
    dry_run: bool,
    verbose: bool,
) -> dict:
    """Run one strategy over all trajectories and return the full output dict."""
    cases: list[dict] = []
    n = len(trajectories)

    for i, traj in enumerate(trajectories):
        tid = traj.get("trajectory_id", f"traj_{i}")
        print(f"[{name}] ({i+1}/{n}) {tid} ...", flush=True)
        t0 = time.time()

        result = run_fn(traj, dry_run=dry_run, verbose=verbose)
        case   = result.to_dict()
        cases.append(case)

        elapsed = time.time() - t0
        marker  = "X" if (case["ground_truth_violated"] and not case["full_coverage"]) else "O"
        print(
            f"  {marker} vio={case['ground_truth_violated']}"
            f"  cov={case['full_coverage']}"
            f"  sc={case['sc']:.2f}  attributed={case['steps_attributed']}"
            f"  depth={case['rollback_depth']}  [{elapsed:.1f}s]",
            flush=True,
        )

    summary = compute_summary(cases, name)
    return {"summary": summary, "cases": cases}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repair-strategy benchmarks on agent trajectory data.",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all"] + list(STRATEGIES.keys()),
        help="Which strategy to run (default: all).",
    )
    parser.add_argument(
        "--data",
        default="trajectory/real_trajectories.json",
        help="Path to trajectory JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory for output JSON files (default: results/).",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Limit to the first N trajectories (default: all).",
    )
    parser.add_argument(
        "--model-source",
        default=None,
        help=(
            "Model source for risk evaluation: "
            "deepseek (default), deepseek-r1, qwen-7b, qwen-72b, custom. "
            "Or set EVAL_MODEL_SOURCE in .env."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use pre-labelled step.violated flags as risk proxy; "
            "skip all LLM API calls."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step risk scores during execution.",
    )
    args = parser.parse_args()

    if args.model_source:
        set_model_source(args.model_source)

    # Resolve data path relative to the project root (parent of src/)
    data_path = args.data
    if not os.path.isabs(data_path):
        project_root = os.path.dirname(_SRC)
        data_path = os.path.join(project_root, data_path)

    trajectories = _load_data(data_path)
    if args.n is not None:
        trajectories = trajectories[: args.n]

    print(
        f"Loaded {len(trajectories)} trajectories from {data_path}"
        + (" [DRY RUN]" if args.dry_run else f" [model={get_active_model()}]"),
        flush=True,
    )

    # Resolve output directory
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        project_root = os.path.dirname(_SRC)
        out_dir = os.path.join(project_root, out_dir)
    _ensure_dir(out_dir)

    # Determine which strategies to run
    to_run = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    # ---------------------------------------------------------------------------
    # Run each strategy
    # ---------------------------------------------------------------------------
    all_summaries: list[dict] = []

    for name in to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"Strategy: {name}", flush=True)
        print(f"{'='*60}", flush=True)

        run_fn = STRATEGIES[name]
        output = _run_strategy(name, run_fn, trajectories, args.dry_run, args.verbose)
        all_summaries.append(output["summary"])

        out_path = os.path.join(out_dir, f"strategy_{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  → saved to {out_path}", flush=True)

    # ---------------------------------------------------------------------------
    # Cross-strategy comparison table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary comparison")
    print(f"{'='*60}")
    header = f"{'strategy':<22} {'FCR':>6} {'avg_SC':>7} {'avg_depth':>10}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(
            f"{s['strategy']:<22} "
            f"{s.get('fcr', 0.0):>6.3f} "
            f"{s['avg_sc']:>7.3f} "
            f"{s['avg_rollback_depth']:>10.2f}"
        )

    print(f"\nDone. Results written to {out_dir}/")


if __name__ == "__main__":
    main()
