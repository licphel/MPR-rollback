"""Sensitivity analysis for MPA-Greedy: sweep tau × theta.

Uses REAL LLM-scored step_risks from existing run results (no additional API calls).
Reads step_risks from final_res/deepseek-v4-flash/run_001/strategy_no_repair.json,
re-runs greedy attribution in pure Python under each (tau, theta) combination.
The trajectory sample is fixed (same 180 cases as run_001).

Usage
-----
python3 final_res/sensitivity.py
python3 final_res/sensitivity.py --run final_res/deepseek-v4-flash/run_002
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "figure.titlesize": 18,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
TAUS   = [0.65, 0.70, 0.75, 0.80, 0.85]
THETAS = [0.15, 0.20, 0.25, 0.30, 0.35]
N      = 180   # use all cases to capture tau sensitivity (safe trajs have low risk)
SEED   = 42

# ---------------------------------------------------------------------------
# Token counter (matches src/strategies/base.py)
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(str(text).split())


# ---------------------------------------------------------------------------
# Pure-Python greedy attribution (mirrors src/strategies/multi_pivot.py)
# ---------------------------------------------------------------------------

def greedy_attribute(step_risks: dict[str, float],
                     orig_tokens: list[int],
                     tau: float,
                     theta: float) -> list[int]:
    """
    step_risks: {str(idx): score}  — real LLM scores
    orig_tokens: token count per step
    Returns list of attributed step indices, or [] if not triggered.
    """
    n = len(orig_tokens)
    risks = {int(k): v for k, v in step_risks.items()}

    # weighted_risk = sum of c_i (influence_factor ignored here;
    # mirrors the actual behaviour since influence_factor ≈ 1 for most steps)
    total_risk = sum(risks.get(i, 0.0) for i in range(n))

    if total_risk < tau:
        return []

    # rank by score desc, tie-break by index desc (prefer later steps)
    ranked = sorted(
        [(i, risks.get(i, 0.0)) for i in range(n)],
        key=lambda x: (x[1], x[0]),
        reverse=True,
    )

    selected: list[int] = []
    zeroed: set[int] = set()

    for idx, score in ranked:
        if score <= 0.0:
            continue
        selected.append(idx)
        zeroed.add(idx)
        projected = sum(risks.get(i, 0.0) for i in range(n) if i not in zeroed)
        if projected <= theta:
            break

    return selected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_case_metrics(case: dict,
                         attributed: list[int],
                         orig_tokens: list[int]) -> dict:
    attributed_set = set(attributed)
    gt_violated    = case["ground_truth_violated"]
    gt_indices     = set(case.get("orig_violated_indices") or [])

    # FCR component: full coverage if all GT violation steps are attributed
    full_coverage = bool(gt_violated and gt_indices and gt_indices <= attributed_set)

    # F1
    if gt_violated and gt_indices:
        tp        = len(gt_indices & attributed_set)
        precision = tp / len(attributed_set) if attributed_set else 0.0
        recall    = tp / len(gt_indices)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
    else:
        f1 = 0.0

    # SC
    total_tok = sum(orig_tokens)
    attr_tok  = sum(orig_tokens[i] for i in attributed_set if i < len(orig_tokens))
    sc        = attr_tok / total_tok if total_tok > 0 else 0.0

    # Rollback depth
    depth = (len(orig_tokens) - min(attributed_set)
             if attributed_set else 0)

    return dict(
        full_coverage  = full_coverage,
        gt_violated    = gt_violated,
        f1             = f1,
        sc             = sc,
        rollback_depth = depth,
    )


def aggregate_metrics(case_metrics: list[dict]) -> dict:
    violated = [m for m in case_metrics if m["gt_violated"]]
    fcr      = (sum(m["full_coverage"] for m in violated) / len(violated)
                if violated else 0.0)
    avg_f1   = sum(m["f1"] for m in violated) / len(violated) if violated else 0.0
    avg_sc   = sum(m["sc"] for m in case_metrics) / len(case_metrics)
    avg_depth = sum(m["rollback_depth"] for m in case_metrics) / len(case_metrics)
    return dict(fcr=fcr, avg_f1=avg_f1, avg_sc=avg_sc, avg_rollback_depth=avg_depth)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default=os.path.join(_HERE, "deepseek-v4-flash", "run_001"),
        help="Path to a run directory containing strategy_no_repair.json",
    )
    args = parser.parse_args()

    no_repair_path = os.path.join(args.run, "strategy_no_repair.json")
    with open(no_repair_path, encoding="utf-8") as f:
        no_repair_data = json.load(f)

    all_cases = no_repair_data["cases"]   # 180 cases with real step_risks

    # Use all 180 cases — needed so tau variation affects safe trajectories
    # (safe trajs have low total_risk; different tau determines whether they trigger)
    run_cases = all_cases

    n_v = sum(c["ground_truth_violated"] for c in run_cases)
    print(f"Sample: {len(run_cases)} cases ({n_v} violated, {len(run_cases)-n_v} safe)")
    print(f"Risk scores:  real LLM scores from {no_repair_path}")
    print(f"Grid: {len(TAUS)} τ × {len(THETAS)} θ = {len(TAUS)*len(THETAS)} combinations\n")

    # Load original trajectories to get observation token counts
    traj_path = os.path.join(_ROOT, "trajectory", "real_trajectories.json")
    with open(traj_path, encoding="utf-8") as f:
        raw_trajs = json.load(f)
    traj_by_id = {t["trajectory_id"]: t for t in raw_trajs}

    results = {}

    for tau in TAUS:
        for theta in THETAS:
            case_metrics = []
            for case in run_cases:
                traj_id   = case["trajectory_id"]
                raw_steps = traj_by_id[traj_id]["steps"]
                orig_tokens = [count_tokens(s.get("observation", ""))
                               for s in raw_steps]
                attributed = greedy_attribute(
                    case["step_risks"], orig_tokens, tau, theta
                )
                m = compute_case_metrics(case, attributed, orig_tokens)
                case_metrics.append(m)

            summary = aggregate_metrics(case_metrics)
            results[(tau, theta)] = summary
            print(
                f"τ={tau:.2f}  θ={theta:.2f}  →  "
                f"FCR={summary['fcr']:.3f}  "
                f"F1={summary['avg_f1']:.3f}  "
                f"SC={summary['avg_sc']:.3f}  "
                f"Depth={summary['avg_rollback_depth']:.2f}"
            )

    # Save JSON
    out_json = {f"tau={tau},theta={theta}": v
                for (tau, theta), v in results.items()}
    json_path = os.path.join(_HERE, "sensitivity_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\nResults → {json_path}")

    plot_heatmaps(results)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_heatmaps(results: dict) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    # Desaturated single-hue colormaps — light (low) → deep (high)
    # "better" metrics use blue-slate; "worse-is-better" metrics use same but flipped
    # All colours are muted/greyish so the grid reads as calm, academic
    _blue = LinearSegmentedColormap.from_list(
        "slate_blue",
        ["#eef1f7", "#c9d4e8", "#97aed1", "#6689bb", "#3a64a3", "#1e3f72"],
    )
    _blue_r = _blue.reversed()

    metrics = [
        ("fcr",                "FCR",   "Full-Coverage Rate",  _blue,   True),
        ("avg_f1",             "F1",    "Attribution F1",      _blue,   True),
        ("avg_sc",             "SC",    "Sanitization Cost",   _blue_r, False),
        ("avg_rollback_depth", "Depth", "Rollback Depth",      _blue_r, False),
    ]

    tau_labels   = [f"{t:.2f}" for t in TAUS]
    theta_labels = [f"{t:.2f}" for t in THETAS]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        "MPA-Greedy Sensitivity Analysis  (n=30, real LLM scores)\n"
        r"Rows: $\tau$ (trigger tolerance)   Cols: $\theta$ (stopping threshold)",
        fontweight="bold",
    )

    for ax, (mk, short, long, cmap, higher_better) in zip(axes.flatten(), metrics):
        # grid[row=tau_idx, col=theta_idx], then flip rows so large tau is at top
        grid = np.array([
            [results[(tau, theta)][mk] for theta in THETAS]
            for tau in TAUS
        ])
        grid_plot = grid[::-1]

        vmin, vmax = grid.min(), grid.max()
        # If all values identical, give a tiny range so imshow doesn't crash
        if vmax == vmin:
            vmin -= 0.01; vmax += 0.01

        im = ax.imshow(grid_plot, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        for i in range(len(TAUS)):
            for j in range(len(THETAS)):
                val = grid_plot[i, j]
                norm = (val - vmin) / (vmax - vmin)
                if not higher_better:
                    norm = 1 - norm
                txt_color = "#e8edf5" if norm > 0.55 else "#2c3550"
                fmt = f"{val:.3f}" if mk != "avg_rollback_depth" else f"{val:.2f}"
                ax.text(j, i, fmt, ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt_color)

        # Bold border on default cell (τ=0.75, θ=0.25)
        di = len(TAUS) - 1 - TAUS.index(0.75)
        dj = THETAS.index(0.25)
        rect = plt.Rectangle((dj - 0.5, di - 0.5), 1, 1,
                              fill=False, edgecolor="#1e3f72",
                              linewidth=3.0, zorder=5)
        ax.add_patch(rect)

        ax.set_xticks(range(len(THETAS)))
        ax.set_xticklabels(theta_labels)
        ax.set_yticks(range(len(TAUS)))
        ax.set_yticklabels(tau_labels[::-1])
        ax.set_xlabel(r"$\theta$  (stopping threshold)")
        ax.set_ylabel(r"$\tau$  (trigger tolerance)")
        direction = "↑ higher is better" if higher_better else "↓ lower is better"
        ax.set_title(f"{short} — {long}\n{direction}", fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(_HERE, "sensitivity_heatmap.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Heatmap → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
