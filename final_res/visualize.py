"""Visualisation for Multi-Pivot Attribution benchmark results.

Usage
-----
python final_res/visualize.py                        # DeepSeek results (default)
python final_res/visualize.py --model qwen-7b        # Qwen-7B results
python final_res/visualize.py --both                 # Both models + comparison
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Global font sizes — governs all text unless overridden below
matplotlib.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   17,
    "axes.labelsize":   14,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  12,
    "figure.titlesize": 18,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STRATEGY_LABELS = {
    "no_repair":          "No Repair",
    "single_pivot":       "Single-Pivot",
    "multi_pivot":        "MPA-Greedy",
    "multi_pivot_random": "MPA-Random",
    "full_attribution":   "Full Attrib.",
}

STRATEGY_ORDER = list(STRATEGY_LABELS.keys())

COLORS = {
    "no_repair":          "#b0b0b0",
    "single_pivot":       "#6baed6",
    "multi_pivot":        "#2171b5",
    "multi_pivot_random": "#74c476",
    "full_attribution":   "#d62728",
}

METRICS = {
    "fcr":                ("FCR",   "Full-Coverage Rate"),
    "avg_f1":             ("F1",    "Step-level F1"),
    "avg_sc":             ("SC",    "Sanitization Cost"),
    "avg_rollback_depth": ("Depth", "Rollback Depth"),
}

_HERE = os.path.dirname(os.path.abspath(__file__))


def load(model: str) -> dict:
    path = os.path.join(_HERE, model, "aggregate.json")
    with open(path) as f:
        return json.load(f)


def save(fig: plt.Figure, path: str) -> None:
    pdf_path = os.path.splitext(path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 0: All-metrics grouped bar chart (2×2 subplots, one per metric)
# ---------------------------------------------------------------------------

def plot_all_metrics_bar(agg: dict, model_name: str, out_dir: str) -> None:
    """2×2 grid: one subplot per metric, strategies as grouped bars with error bars."""
    strategies = [s for s in STRATEGY_ORDER if s in agg]
    x = np.arange(len(strategies))
    metric_keys = list(METRICS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"All Metrics — {model_name}", fontweight="bold")
    axes = axes.flatten()

    for ax, mk in zip(axes, metric_keys):
        short, long = METRICS[mk]
        means = np.array([agg[s][mk]["mean"] for s in strategies])
        stds  = np.array([agg[s][mk]["std"]  for s in strategies])
        colors = [COLORS[s] for s in strategies]

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.88,
                      error_kw={"elinewidth": 1.5, "ecolor": "#444"})

        # Value labels above each bar
        for bar, val, err in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + 0.012,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.set_title(f"{short}  ({long})", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies],
                           rotation=20, ha="right")
        ax.set_ylabel(short)
        ax.grid(axis="y", alpha=0.3, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Depth has no natural [0,1] ceiling
        if mk != "avg_rollback_depth":
            ax.set_ylim(0, max(means + stds) * 1.25 + 0.05)
        else:
            ax.set_ylim(0, max(means + stds) * 1.3 + 1)

        # Shade better direction
        if mk in ("avg_sc", "avg_rollback_depth"):
            ax.set_facecolor("#fff8f8")  # red tint: lower is better
        else:
            ax.set_facecolor("#f8fff8")  # green tint: higher is better

    # Shared strategy legend at bottom
    handles = [mpatches.Patch(color=COLORS[s], label=STRATEGY_LABELS[s])
               for s in strategies]
    fig.legend(handles=handles, loc="lower center", ncol=len(strategies),
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save(fig, os.path.join(out_dir, "0_all_metrics_bar.png"))


# ---------------------------------------------------------------------------
# Plot 1: T-bar — FCR vs SC
# ---------------------------------------------------------------------------

def plot_tbar(agg: dict, model_name: str, out_dir: str) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in agg]
    x = np.arange(len(strategies))
    width = 0.35

    fcr_means = [agg[s]["fcr"]["mean"]    for s in strategies]
    fcr_stds  = [agg[s]["fcr"]["std"]     for s in strategies]
    sc_means  = [agg[s]["avg_sc"]["mean"] for s in strategies]
    sc_stds   = [agg[s]["avg_sc"]["std"]  for s in strategies]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, fcr_means, width, yerr=fcr_stds,
                   capsize=4, color=[COLORS[s] for s in strategies],
                   alpha=0.9, error_kw={"elinewidth": 1.2})
    bars2 = ax2.bar(x + width/2, sc_means, width, yerr=sc_stds,
                    capsize=4, color=[COLORS[s] for s in strategies],
                    alpha=0.45, hatch="//", error_kw={"elinewidth": 1.2})

    ax.set_ylabel("FCR (Full-Coverage Rate)")
    ax2.set_ylabel("SC (Sanitization Cost)")
    ax.set_ylim(0, 1.22)
    ax2.set_ylim(0, 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies],
                       rotation=15, ha="right")
    ax.set_title(f"FCR vs Sanitization Cost  [{model_name}]", fontweight="bold")

    for bar, val in zip(bars1, fcr_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, sc_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, color="#555")

    solid = mpatches.Patch(color="steelblue", alpha=0.9, label="FCR (left axis)")
    hatch = mpatches.Patch(facecolor="steelblue", alpha=0.45, hatch="//", label="SC (right axis)")
    ax.legend(handles=[solid, hatch], loc="upper left")

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "1_tbar_fcr_sc.png"))


# ---------------------------------------------------------------------------
# Plot 2: Radar chart
# ---------------------------------------------------------------------------

def plot_radar(agg: dict, model_name: str, out_dir: str) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in agg]
    metric_keys = list(METRICS.keys())
    metric_labels = [METRICS[m][0] for m in metric_keys]
    N = len(metric_keys)

    max_depth = max(agg[s]["avg_rollback_depth"]["mean"] for s in strategies) + 1e-9

    def get_vals(s):
        vals = []
        for m in metric_keys:
            v = agg[s][m]["mean"]
            if m == "avg_rollback_depth":
                v = 1 - v / max_depth
            elif m == "avg_sc":
                v = 1 - v
            vals.append(v)
        return vals

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], color="grey")
    ax.set_title(f"Multi-Metric Profile  [{model_name}]", fontweight="bold", pad=20)

    for s in strategies:
        vals = get_vals(s)
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[s], linewidth=2, label=STRATEGY_LABELS[s])
        ax.fill(angles, vals, color=COLORS[s], alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.18))
    ax.text(0.5, -0.10, "SC and Depth are inverted (higher = better)",
            ha="center", transform=ax.transAxes, fontsize=12, color="grey", style="italic")

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "2_radar.png"))


# ---------------------------------------------------------------------------
# Plot 3: F1 vs SC scatter — efficiency frontier
# ---------------------------------------------------------------------------

def plot_scatter(agg: dict, model_name: str, out_dir: str) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in agg]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Soft background gradient quad lines
    ax.axvspan(0, 0.5,   ymin=0.5, ymax=1, alpha=0.04, color="green")   # ideal zone
    ax.axvspan(0.5, 1.1, ymin=0,   ymax=0.5, alpha=0.04, color="red")   # bad zone

    ax.axvline(0.5, color="#cccccc", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=1.0, zorder=1)
    ax.grid(True, alpha=0.2, linewidth=0.7, zorder=0)

    # Label placement: (dx, dy) relative to point, and anchor side
    label_cfg = {
        "no_repair":          dict(dx=-0.01, dy= 0.06, ha="center"),
        "single_pivot":       dict(dx= 0.03, dy= 0.06, ha="left"),
        "multi_pivot":        dict(dx= 0.03, dy= 0.06, ha="left"),
        "multi_pivot_random": dict(dx= 0.03, dy=-0.08, ha="left"),
        "full_attribution":   dict(dx=-0.01, dy=-0.09, ha="center"),
    }

    for s in strategies:
        f1  = agg[s]["avg_f1"]["mean"]
        sc  = agg[s]["avg_sc"]["mean"]
        f1e = agg[s]["avg_f1"]["std"]
        sce = agg[s]["avg_sc"]["std"]

        # Error bars
        ax.errorbar(sc, f1, xerr=sce, yerr=f1e,
                    fmt="none", color=COLORS[s], capsize=5,
                    elinewidth=1.5, capthick=1.5, zorder=4)

        # Marker with white ring for pop
        ax.scatter(sc, f1, s=220, color=COLORS[s],
                   edgecolors="white", linewidths=2.0, zorder=6)
        ax.scatter(sc, f1, s=320, color=COLORS[s], alpha=0.18, zorder=5)

        # Annotate with a styled box
        cfg = label_cfg.get(s, dict(dx=0.03, dy=0.06, ha="left"))
        ax.annotate(
            STRATEGY_LABELS[s],
            xy=(sc, f1),
            xytext=(sc + cfg["dx"], f1 + cfg["dy"]),
            ha=cfg["ha"], va="center",
            fontsize=13, fontweight="bold", color=COLORS[s],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS[s], alpha=0.85, linewidth=1.2),
            zorder=7,
        )

    ax.set_xlabel("Sanitization Cost (SC)   ← lower is cheaper", labelpad=8)
    ax.set_ylabel("Step-level F1   → higher is more precise", labelpad=8)
    ax.set_xlim(-0.12, 1.18)
    ax.set_ylim(-0.08, 1.08)
    ax.set_title(f"F1 vs SC Efficiency Frontier  [{model_name}]", fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ideal-corner arrow + label
    ax.annotate(
        "", xy=(0.03, 0.97), xytext=(0.20, 0.80),
        arrowprops=dict(arrowstyle="-|>", color="#888888",
                        lw=1.8, mutation_scale=16),
        zorder=8,
    )
    ax.text(0.22, 0.79, "cheaper &\nmore precise",
            fontsize=12, color="#888888", style="italic", ha="left", va="top")

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "3_scatter_f1_sc.png"))


# ---------------------------------------------------------------------------
# Plot 4: Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(agg: dict, model_name: str, out_dir: str) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in agg]
    metric_keys = list(METRICS.keys())
    metric_labels = [f"{METRICS[m][0]}\n({METRICS[m][1]})" for m in metric_keys]

    data = np.array([[agg[s][m]["mean"] for m in metric_keys] for s in strategies])
    stds = np.array([[agg[s][m]["std"]  for m in metric_keys] for s in strategies])

    norm_data = data.copy()
    for j, m in enumerate(metric_keys):
        col = norm_data[:, j]
        mn, mx = col.min(), col.max()
        if mx > mn:
            norm_data[:, j] = (col - mn) / (mx - mn)
            if m in ("avg_sc", "avg_rollback_depth"):
                norm_data[:, j] = 1 - norm_data[:, j]

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in strategies])
    ax.set_title(f"Metric Heatmap (N=3 runs)  [{model_name}]", fontweight="bold")

    for i in range(len(strategies)):
        for j in range(len(metric_keys)):
            val = data[i, j]
            std = stds[i, j]
            cell_text = f"{val:.3f}" if std == 0 else f"{val:.3f}\n±{std:.3f}"
            brightness = norm_data[i, j]
            text_color = "white" if brightness < 0.3 else "black"
            ax.text(j, i, cell_text, ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label="Relative performance (green = better, SC/Depth inverted)")

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "4_heatmap.png"))


# ---------------------------------------------------------------------------
# Plot 5: Four-metric grouped bar chart (horizontal, publication-ready)
# ---------------------------------------------------------------------------

def plot_four_metric_bar(agg: dict, model_name: str, out_dir: str) -> None:
    """
    4 horizontal subplots side-by-side, one per metric.
    Strategies shown as vertical bars with error caps and value labels.
    Colour-coded by strategy (consistent palette); ↑/↓ direction annotated.
    """
    strategies = [s for s in STRATEGY_ORDER if s in agg]
    metric_keys  = ["fcr", "avg_f1", "avg_sc", "avg_rollback_depth"]
    metric_info  = [
        ("FCR",   "Full-Coverage Rate",  "↑ higher is safer",  "#f0f8f0"),
        ("F1",    "Attribution F1",       "↑ higher is better", "#f0f8f0"),
        ("SC",    "Sanitization Cost",    "↓ lower is cheaper", "#fff5f5"),
        ("Depth", "Rollback Depth",       "↓ lower is cheaper", "#fff5f5"),
    ]

    n_s = len(strategies)
    x   = np.arange(n_s)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=False)
    fig.suptitle(f"Strategy Comparison Across All Metrics  [{model_name}]",
                 fontweight="bold", y=1.01)

    for ax, mk, (short, long, direction, bg) in zip(axes, metric_keys, metric_info):
        means  = np.array([agg[s][mk]["mean"] for s in strategies])
        stds   = np.array([agg[s][mk]["std"]  for s in strategies])
        colors = [COLORS[s] for s in strategies]

        bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.62,
                      color=colors, alpha=0.88,
                      error_kw={"elinewidth": 1.8, "ecolor": "#333", "capthick": 1.8},
                      zorder=3)

        # value labels
        for bar, val, err in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + 0.008,
                    f"{val:.3f}" if mk != "avg_rollback_depth" else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_facecolor(bg)
        ax.set_title(f"{short}\n{direction}", fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s].replace(" ", "\n")
                            for s in strategies], fontsize=11)
        ax.set_ylabel(long, fontsize=12)
        ax.grid(axis="y", alpha=0.28, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # y-axis ceiling
        ceiling = max(means + stds) * 1.28 + (0.06 if mk != "avg_rollback_depth" else 0.8)
        ax.set_ylim(0, ceiling)

    # shared legend at bottom
    handles = [mpatches.Patch(color=COLORS[s], label=STRATEGY_LABELS[s])
               for s in strategies]
    fig.legend(handles=handles, loc="lower center", ncol=n_s,
               bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=12)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "6_four_metric_bar.png"))


# ---------------------------------------------------------------------------
# Plot 7: 3-D trade-off — SC × Depth × F1 (bubble size = FCR)
# ---------------------------------------------------------------------------

def plot_cost_f1_bubble(agg: dict, model_name: str, out_dir: str) -> None:
    """
    2-D bubble chart:
      X = composite cost  = 0.5 * SC + 0.5 * (Depth / max_depth)
      Y = F1
      bubble area ∝ FCR
    Ideal corner: bottom-left (low cost) + top (high F1).
    """
    strategies = [s for s in STRATEGY_ORDER if s in agg]

    sc    = np.array([agg[s]["avg_sc"]["mean"]             for s in strategies])
    depth = np.array([agg[s]["avg_rollback_depth"]["mean"]  for s in strategies])
    f1    = np.array([agg[s]["avg_f1"]["mean"]              for s in strategies])
    fcr   = np.array([agg[s]["fcr"]["mean"]                 for s in strategies])
    sc_std    = np.array([agg[s]["avg_sc"]["std"]            for s in strategies])
    f1_std    = np.array([agg[s]["avg_f1"]["std"]            for s in strategies])
    colors = [COLORS[s] for s in strategies]

    max_depth = depth.max() if depth.max() > 0 else 1.0
    cost = 0.5 * sc + 0.5 * (depth / max_depth)

    # bubble area: FCR [0,1] → [200, 2200]
    sizes = 200 + fcr * 2000

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f9f9fb")
    fig.patch.set_facecolor("#f9f9fb")

    # soft quadrant shading
    ax.axvspan(0.0, 0.5, ymin=0.5, ymax=1.0, alpha=0.06, color="green",  zorder=0)
    ax.axvspan(0.5, 1.0, ymin=0.0, ymax=0.5, alpha=0.06, color="red",    zorder=0)
    ax.axvline(0.5, color="#cccccc", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=1.0, zorder=1)

    # error bars (X = SC std as proxy; exact cost-std not critical)
    for i, s in enumerate(strategies):
        ax.errorbar(cost[i], f1[i],
                    xerr=sc_std[i] * 0.5, yerr=f1_std[i],
                    fmt="none", color=colors[i],
                    capsize=5, elinewidth=1.8, capthick=1.8, zorder=3)

    # bubbles (two layers: glow + solid)
    for i, s in enumerate(strategies):
        ax.scatter(cost[i], f1[i], s=sizes[i] * 1.5,
                   c=colors[i], alpha=0.12, zorder=4)
        ax.scatter(cost[i], f1[i], s=sizes[i],
                   c=colors[i], alpha=0.90,
                   edgecolors="white", linewidths=2.2, zorder=5)

    # labels — placed with per-strategy offsets to avoid overlap
    label_cfg = {
        "no_repair":          dict(dx= 0.00, dy= 0.06, ha="center"),
        "single_pivot":       dict(dx= 0.03, dy= 0.06, ha="left"),
        "multi_pivot":        dict(dx= 0.04, dy= 0.06, ha="left"),
        "multi_pivot_random": dict(dx= 0.04, dy=-0.07, ha="left"),
        "full_attribution":   dict(dx=-0.01, dy=-0.08, ha="center"),
    }
    for i, s in enumerate(strategies):
        cfg = label_cfg.get(s, dict(dx=0.04, dy=0.05, ha="left"))
        ax.annotate(
            STRATEGY_LABELS[s],
            xy=(cost[i], f1[i]),
            xytext=(cost[i] + cfg["dx"], f1[i] + cfg["dy"]),
            ha=cfg["ha"], va="center",
            fontsize=14, fontweight="bold", color=colors[i],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=colors[i], alpha=0.88, linewidth=1.3),
            zorder=7,
        )

    # ideal-corner arrow
    ax.annotate("", xy=(0.03, 0.78), xytext=(0.18, 0.63),
                arrowprops=dict(arrowstyle="-|>", color="#888",
                                lw=1.8, mutation_scale=16), zorder=8)
    ax.text(0.20, 0.62, "lower cost &\nhigher F1",
            fontsize=15, color="#888", style="italic", ha="left", va="top")

    # FCR bubble-size legend — removed (fixed size now)

    ax.set_xlabel("Composite Cost  =  0.5 · SC  +  0.5 · (Depth / max)   ← lower is better",
                  labelpad=10, fontsize=16)
    ax.set_ylabel("Attribution F1   → higher is better", labelpad=10, fontsize=16)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.04, 0.90)
    ax.set_title(f"Cost–F1 Trade-off  [{model_name}]",
                 fontweight="bold", pad=14, fontsize=19)
    ax.grid(True, alpha=0.20, linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=14)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "7_cost_f1_bubble.png"))


def plot_3d_tradeoff(agg: dict, model_name: str, out_dir: str) -> None:
    """
    3-D scatter: X = SC, Y = F1, Z = Depth.
    Bubble size encodes FCR.
    View angle chosen so all 5 strategies are clearly separated.
    Labels are offset away from the bubble, never behind it.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    strategies = [s for s in STRATEGY_ORDER if s in agg]

    sc    = np.array([agg[s]["avg_sc"]["mean"]             for s in strategies])
    f1    = np.array([agg[s]["avg_f1"]["mean"]              for s in strategies])
    depth = np.array([agg[s]["avg_rollback_depth"]["mean"]  for s in strategies])
    fcr   = np.array([agg[s]["fcr"]["mean"]                 for s in strategies])
    colors = [COLORS[s] for s in strategies]

    # bubble area: FCR [0,1] → [120, 1100]
    sizes = 120 + fcr * 980

    fig = plt.figure(figsize=(13, 10))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # --- draw drop-lines from each point down to z=0 floor (depth=0)
    for i in range(len(strategies)):
        ax.plot([sc[i], sc[i]], [f1[i], f1[i]], [0, depth[i]],
                color=colors[i], linewidth=1.2, alpha=0.35, linestyle=":")

    # --- scatter
    for i, s in enumerate(strategies):
        ax.scatter(sc[i], f1[i], depth[i],
                   s=sizes[i], c=colors[i], alpha=0.90,
                   edgecolors="white", linewidths=2.0, zorder=5)

    # --- labels: push each label in a fixed (dx, dy, dz) direction away from bubble
    # offsets tuned to the actual data layout so nothing overlaps
    label_offsets = {
        "no_repair":          (-0.10,  0.06, -0.8),
        "single_pivot":       (-0.12,  0.07, -1.2),
        "multi_pivot":        (-0.14, -0.07,  1.0),
        "multi_pivot_random": ( 0.10, -0.07,  1.0),
        "full_attribution":   ( 0.10,  0.06, -0.8),
    }
    for i, s in enumerate(strategies):
        dx, dy, dz = label_offsets.get(s, (0.08, 0.05, 0.5))
        lx, ly, lz = sc[i] + dx, f1[i] + dy, depth[i] + dz
        ax.text(lx, ly, lz,
                STRATEGY_LABELS[s],
                fontsize=13, fontweight="bold", color=colors[i],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=colors[i], alpha=0.85, linewidth=1.2))

    # --- axis labels & limits
    ax.set_xlabel("SC  (← lower is cheaper)", fontsize=13, labelpad=14)
    ax.set_ylabel("F1  (→ higher is better)", fontsize=13, labelpad=14)
    ax.set_zlabel("Depth  (↓ lower is cheaper)", fontsize=13, labelpad=14)

    ax.set_xlim(0.0,  1.05)
    ax.set_ylim(0.0,  0.85)
    ax.set_zlim(0.0, 11.5)

    ax.tick_params(axis="both", labelsize=11)

    ax.set_title(
        f"3-D Trade-off: SC  ×  F1  ×  Depth\n"
        f"[{model_name}]   bubble area ∝ FCR",
        fontsize=15, fontweight="bold", pad=20,
    )

    # --- ideal corner star
    ax.scatter([0.0], [0.80], [0.0], marker="*", s=380,
               c="gold", edgecolors="#999", linewidths=1.2, zorder=6)
    ax.text(0.0, 0.80, 0.6, "Ideal", fontsize=12,
            color="#888", ha="center", style="italic")

    # --- FCR bubble-size reference legend (proxy circles in figure space)
    from matplotlib.lines import Line2D
    size_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
               markersize=np.sqrt(120 + v * 980) * 0.55,
               label=f"FCR = {v:.1f}", alpha=0.75)
        for v in [0.0, 0.5, 1.0]
    ]
    strategy_legend = [
        mpatches.Patch(color=COLORS[s], label=STRATEGY_LABELS[s])
        for s in strategies
    ]
    leg1 = ax.legend(handles=strategy_legend, loc="upper left",
                     bbox_to_anchor=(0.0, 1.0), fontsize=12,
                     frameon=True, framealpha=0.9, title="Strategy")
    ax.add_artist(leg1)
    ax.legend(handles=size_legend, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), fontsize=11,
              frameon=True, framealpha=0.9, title="Bubble size")

    # view: elev tilts up/down, azim rotates left/right
    # SC on X (left→right), F1 on Y (near→far), Depth on Z (up)
    # azim=-40 shows SC spread clearly; elev=18 keeps depth axis readable
    ax.view_init(elev=18, azim=-42)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "7_3d_tradeoff.png"))


# ---------------------------------------------------------------------------
# Plot 8: Model comparison — DeepSeek vs Qwen-7B side by side
# ---------------------------------------------------------------------------

def plot_model_comparison(loaded: dict, out_dir: str) -> None:
    models = list(loaded.keys())
    strategies = [s for s in STRATEGY_ORDER if s in loaded[models[0]]]
    metric_keys = ["fcr", "avg_f1", "avg_sc", "avg_rollback_depth"]
    metric_names = ["FCR", "F1", "SC", "Depth"]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(20, 6))
    fig.suptitle("DeepSeek-V3 vs Qwen-7B — Evaluator Quality Comparison",
                 fontweight="bold")

    x = np.arange(len(strategies))
    width = 0.35
    model_colors = {"deepseek-v4-flash": "#2171b5", "qwen-7b": "#e6550d"}

    for ax, mk, mn in zip(axes, metric_keys, metric_names):
        for i, model in enumerate(models):
            means = [loaded[model][s][mk]["mean"] for s in strategies]
            stds  = [loaded[model][s][mk]["std"]  for s in strategies]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds,
                          capsize=3, color=model_colors[model],
                          alpha=0.8, label=model, error_kw={"elinewidth": 1.0})
            for bar, val, err in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + err + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(mn, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s].replace(" ", "\n") for s in strategies],
                           fontsize=10)
        if mk != "avg_rollback_depth":
            ax.set_ylim(0, 1.25)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.legend()

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "5_model_comparison.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    models = ["deepseek-v4-flash", "qwen-7b"] if args.both else [args.model]

    loaded = {}
    for model in models:
        try:
            loaded[model] = load(model)
        except FileNotFoundError:
            print(f"No aggregate.json for {model}, skipping.")
            continue
        out_dir = os.path.join(_HERE, model)
        plot_all_metrics_bar(loaded[model],   model, out_dir)
        plot_tbar(loaded[model],              model, out_dir)
        plot_radar(loaded[model],             model, out_dir)
        plot_scatter(loaded[model],           model, out_dir)
        plot_heatmap(loaded[model],           model, out_dir)
        plot_four_metric_bar(loaded[model],   model, out_dir)
        plot_cost_f1_bubble(loaded[model],    model, out_dir)
        plot_3d_tradeoff(loaded[model],       model, out_dir)

    if args.both and len(loaded) == 2:
        plot_model_comparison(loaded, _HERE)


if __name__ == "__main__":
    main()
