"""run_ablation.py — run ablation studies against multi_pivot baseline.

Usage
-----
python src/ablation/run_ablation.py --dry-run -n 20
python src/ablation/run_ablation.py -N 3 --data trajectory/real_trajectories.json

Output
------
results/ablation/
  run_001/multi_pivot.json
  run_001/multi_pivot_random.json
  ...
  aggregate.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from strategies import STRATEGIES
from ablation   import ABLATIONS
from metrics    import compute_summary

RUNS: dict[str, callable] = {
    "multi_pivot":        STRATEGIES["multi_pivot"],
    **ABLATIONS,
}

SCALAR_METRICS = ["fcr", "avg_f1", "avg_sc", "avg_rollback_depth"]


def _load_data(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_one(name, run_fn, trajectories, dry_run, verbose):
    cases = []
    for traj in trajectories:
        result = run_fn(traj, dry_run=dry_run, verbose=verbose)
        cases.append(result.to_dict())
    return {"summary": compute_summary(cases, name), "cases": cases}


def _aggregate(all_runs: dict[str, list[dict]]) -> dict:
    result = {}
    for name, summaries in all_runs.items():
        entry = {"strategy": name, "n_runs": len(summaries)}
        for metric in SCALAR_METRICS:
            vals = [s[metric] for s in summaries]
            mean = sum(vals) / len(vals)
            std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            entry[metric] = {
                "mean": round(mean, 4),
                "std":  round(std,  4),
                "runs": [round(v, 4) for v in vals],
            }
        result[name] = entry
    return result


def _print_table(agg: dict) -> None:
    header = f"{'strategy':<22} {'FCR mean±std':>14} {'F1 mean±std':>13} {'SC mean±std':>13} {'depth mean±std':>15}"
    print(header)
    print("-" * len(header))
    for name, entry in agg.items():
        def _ms(k):
            return f"{entry[k]['mean']:.3f}±{entry[k]['std']:.3f}"
        print(f"{name:<22} {_ms('fcr'):>14} {_ms('avg_f1'):>13} {_ms('avg_sc'):>13} {_ms('avg_rollback_depth'):>15}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies.")
    parser.add_argument("-N", type=int, default=3)
    parser.add_argument("--data", default="trajectory/real_trajectories.json")
    parser.add_argument("--out-dir", default="results/ablation")
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--balanced", action="store_true",
                        help="Each run samples n/2 violated + n/2 safe at random.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    project_root = os.path.dirname(_SRC)
    data_path = args.data if os.path.isabs(args.data) else os.path.join(project_root, args.data)
    out_base  = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(project_root, args.out_dir)

    trajectories = _load_data(data_path)
    if args.n is not None and not args.balanced:
        trajectories = trajectories[: args.n]

    _violated_pool = [t for t in trajectories if t.get("violated")]
    _safe_pool     = [t for t in trajectories if not t.get("violated")]

    print(
        f"Loaded {len(trajectories)} trajectories | "
        f"runs: {list(RUNS.keys())} | N={args.N}"
        + (" [BALANCED]" if args.balanced else "")
        + (" [DRY RUN]" if args.dry_run else ""),
        flush=True,
    )

    all_runs: dict[str, list[dict]] = {name: [] for name in RUNS}
    total_start = time.time()

    for run_idx in range(1, args.N + 1):
        run_dir = os.path.join(out_base, f"run_{run_idx:03d}")
        os.makedirs(run_dir, exist_ok=True)

        if args.balanced and args.n is not None:
            half = args.n // 2
            rng  = random.Random(run_idx)
            run_trajs = (
                rng.sample(_violated_pool, min(half, len(_violated_pool))) +
                rng.sample(_safe_pool,     min(half, len(_safe_pool)))
            )
            rng.shuffle(run_trajs)
        else:
            run_trajs = trajectories

        print(f"\n{'='*60}", flush=True)
        print(f"Run {run_idx}/{args.N}  ({len(run_trajs)} trajectories, "
              f"{sum(t.get('violated', False) for t in run_trajs)} violated)", flush=True)
        print(f"{'='*60}", flush=True)

        for name, fn in RUNS.items():
            print(f"  [{name}] ...", end="  ", flush=True)
            t0 = time.time()
            output = _run_one(name, fn, run_trajs, args.dry_run, args.verbose)
            all_runs[name].append(output["summary"])

            out_path = os.path.join(run_dir, f"{name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            s = output["summary"]
            print(
                f"FCR={s['fcr']:.3f}  F1={s['avg_f1']:.3f}  "
                f"SC={s['avg_sc']:.3f}  depth={s['avg_rollback_depth']:.2f}  [{time.time()-t0:.1f}s]",
                flush=True,
            )

    agg = _aggregate(all_runs)
    agg_path = os.path.join(out_base, "aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Ablation aggregate ({args.N} runs)")
    print(f"{'='*60}")
    _print_table(agg)
    print(f"\nSaved to {agg_path}")
    print(f"Total time: {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    main()
