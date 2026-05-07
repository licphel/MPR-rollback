"""Ablation runner.

Compares multi_pivot (full method) against:
  A) multi_pivot_no_weight  — no recency weighting
  B) multi_pivot_random     — random step selection

Usage
-----
# Dry run, 30 trajectories (15 violated + 15 safe sampled from real data):
python src/ablation/run_ablation.py --dry-run

# Real run:
python src/ablation/run_ablation.py --data trajectory/real_trajectories.json

Output
------
results/ablation/ablation_results.json
"""
import argparse, json, os, random, sys, time

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from strategies.multi_pivot import run_trajectory as multi_pivot
from ablation import ABLATIONS
from metrics import compute_summary


# ---------------------------------------------------------------------------
# Sampling: 15 violated + 15 safe
# ---------------------------------------------------------------------------

def _sample(trajectories: list[dict], n_violated: int, n_safe: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    violated = [t for t in trajectories if t.get("violated", False)]
    safe     = [t for t in trajectories if not t.get("violated", False)]
    return (
        rng.sample(violated, min(n_violated, len(violated))) +
        rng.sample(safe,     min(n_safe,     len(safe)))
    )


# ---------------------------------------------------------------------------
# Run one strategy
# ---------------------------------------------------------------------------

def _run(name: str, run_fn, trajectories: list[dict], dry_run: bool) -> dict:
    cases = []
    for i, traj in enumerate(trajectories):
        tid = traj.get("trajectory_id", f"traj_{i}")
        print(f"  [{name}] ({i+1}/{len(trajectories)}) {tid}", flush=True)
        result = run_fn(traj, dry_run=dry_run)
        cases.append(result.to_dict())
    return {"summary": compute_summary(cases, name), "cases": cases}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="trajectory/real_trajectories.json")
    parser.add_argument("--out-dir", default="results/ablation")
    parser.add_argument("--n-violated", type=int, default=15)
    parser.add_argument("--n-safe",     type=int, default=15)
    parser.add_argument("--n-runs",     type=int, default=2)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    data_path = args.data if os.path.isabs(args.data) else \
        os.path.join(os.path.dirname(_SRC), args.data)
    with open(data_path, encoding="utf-8") as f:
        all_trajs = json.load(f)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else \
        os.path.join(os.path.dirname(_SRC), args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    strategies = {"multi_pivot": multi_pivot, **ABLATIONS}

    all_runs: dict[str, list[dict]] = {name: [] for name in strategies}

    for run_idx in range(args.n_runs):
        print(f"\n=== Run {run_idx + 1}/{args.n_runs} ===", flush=True)
        sample = _sample(all_trajs, args.n_violated, args.n_safe, seed=run_idx)
        print(f"Sample: {len(sample)} trajectories "
              f"({sum(t.get('violated',False) for t in sample)} violated)", flush=True)

        for name, run_fn in strategies.items():
            output = _run(name, run_fn, sample, args.dry_run)
            all_runs[name].append(output["summary"])

    # Aggregate across runs
    def _mean(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0

    aggregated = {}
    for name, summaries in all_runs.items():
        keys = ["violation_reduction", "full_coverage_rate", "unsafe_output_rate",
                "avg_sc", "avg_sanitized_steps", "avg_rollback_depth", "avg_final_risk"]
        aggregated[name] = {
            k: {
                "mean": _mean([s[k] for s in summaries if k in s]),
                "runs": [s[k] for s in summaries if k in s],
            }
            for k in keys
        }

    out_path = os.path.join(out_dir, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Print comparison table
    print(f"\n{'strategy':<26} {'FCR':>6} {'VR':>6} {'SC':>7} {'san_steps':>10} {'depth':>7} {'risk':>7}")
    print("-" * 72)
    for name, agg in aggregated.items():
        print(
            f"{name:<26} "
            f"{agg['full_coverage_rate']['mean']:>6.3f} "
            f"{agg['violation_reduction']['mean']:>6.3f} "
            f"{agg['avg_sc']['mean']:>7.3f} "
            f"{agg['avg_sanitized_steps']['mean']:>10.2f} "
            f"{agg['avg_rollback_depth']['mean']:>7.2f} "
            f"{agg['avg_final_risk']['mean']:>7.3f}"
        )


if __name__ == "__main__":
    main()
