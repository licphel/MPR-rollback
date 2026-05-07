"""run_all.py — run every strategy N times and aggregate results.

Usage
-----
# Dry-run, 3 repetitions (no API calls):
python src/run_all.py -N 3 --dry-run

# Real LLM calls, 5 repetitions, first 10 trajectories only:
python src/run_all.py -N 5 --data trajectory/real_trajectories.json -n 10

Output
------
results/
  run_001/strategy_no_repair.json
  run_001/strategy_full_sanitization.json
  ...
  run_00N/strategy_multi_pivot.json
  aggregate.json            <- mean ± std across runs for every metric
"""
import argparse
import json
import math
import os
import sys
import time

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

from strategies import STRATEGIES
from metrics    import compute_summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_one_strategy(name, run_fn, trajectories, dry_run, verbose):
    """Run a single strategy over all trajectories; return output dict."""
    cases = []
    for traj in trajectories:
        result = run_fn(traj, dry_run=dry_run, verbose=verbose)
        cases.append(result.to_dict())
    summary = compute_summary(cases, name)
    return {"summary": summary, "cases": cases}


SCALAR_METRICS = [
    "violation_reduction",
    "unsafe_output_rate",
    "avg_sc",
    "avg_sanitized_steps",
    "avg_rollback_depth",
    "avg_final_risk",
]


def _aggregate(all_runs: dict[str, list[dict]]) -> dict:
    """
    all_runs: { strategy_name -> [summary_dict_run1, summary_dict_run2, ...] }
    Returns: { strategy_name -> { metric -> {mean, std, runs} } }
    """
    result = {}
    for name, summaries in all_runs.items():
        entry = {
            "strategy":  name,
            "n_runs":    len(summaries),
        }
        for metric in SCALAR_METRICS:
            vals = [s[metric] for s in summaries]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            std  = math.sqrt(variance)
            entry[metric] = {
                "mean": round(mean, 4),
                "std":  round(std,  4),
                "runs": [round(v, 4) for v in vals],
            }
        result[name] = entry
    return result


def _print_aggregate_table(agg: dict) -> None:
    header = (
        f"{'strategy':<22} "
        f"{'VR mean±std':>14} "
        f"{'unsafe% mean±std':>18} "
        f"{'SC mean±std':>13} "
        f"{'san_steps m±s':>14} "
        f"{'depth m±s':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, entry in agg.items():
        def _ms(k):
            return f"{entry[k]['mean']:.3f}±{entry[k]['std']:.3f}"
        print(
            f"{name:<22} "
            f"{_ms('violation_reduction'):>14} "
            f"{_ms('unsafe_output_rate'):>18} "
            f"{_ms('avg_sc'):>13} "
            f"{_ms('avg_sanitized_steps'):>14} "
            f"{_ms('avg_rollback_depth'):>10}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all strategies N times and aggregate results."
    )
    parser.add_argument("-N", type=int, default=3,
                        help="Number of repetitions (default: 3).")
    parser.add_argument("--strategy", default="all",
                        choices=["all"] + list(STRATEGIES.keys()),
                        help="Which strategy to run (default: all).")
    parser.add_argument("--data", default="trajectory/real_trajectories.json",
                        help="Path to trajectory JSON file.")
    parser.add_argument("--out-dir", default="results",
                        help="Base output directory (default: results/).")
    parser.add_argument("-n", type=int, default=None,
                        help="Limit to first N trajectories (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use pre-labelled risk scores; skip LLM calls.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step risk scores.")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(_SRC)
    data_path = args.data if os.path.isabs(args.data) else os.path.join(project_root, args.data)
    out_base  = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(project_root, args.out_dir)

    trajectories = _load_data(data_path)
    if args.n is not None:
        trajectories = trajectories[: args.n]

    to_run = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    print(
        f"Loaded {len(trajectories)} trajectories | "
        f"strategies: {to_run} | "
        f"N={args.N}"
        + (" [DRY RUN]" if args.dry_run else ""),
        flush=True,
    )

    # { strategy -> [summary_run1, summary_run2, ...] }
    all_runs: dict[str, list[dict]] = {name: [] for name in to_run}

    total_start = time.time()

    for run_idx in range(1, args.N + 1):
        run_dir = os.path.join(out_base, f"run_{run_idx:03d}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*60}", flush=True)
        print(f"Run {run_idx}/{args.N}", flush=True)
        print(f"{'='*60}", flush=True)

        for name in to_run:
            print(f"  [{name}] ...", end="  ", flush=True)
            t0 = time.time()

            output = _run_one_strategy(
                name, STRATEGIES[name], trajectories,
                dry_run=args.dry_run, verbose=args.verbose,
            )
            all_runs[name].append(output["summary"])

            out_path = os.path.join(run_dir, f"strategy_{name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            s = output["summary"]
            print(
                f"VR={s['violation_reduction']:.3f}  "
                f"SC={s['avg_sc']:.3f}  "
                f"steps={s['avg_sanitized_steps']:.2f}  "
                f"[{time.time()-t0:.1f}s]",
                flush=True,
            )

    # Save aggregate
    agg = _aggregate(all_runs)
    agg_path = os.path.join(out_base, "aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Aggregate ({args.N} runs)")
    print(f"{'='*60}")
    _print_aggregate_table(agg)
    print(f"\nAggregate saved to {agg_path}")
    print(f"Total time: {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    main()
