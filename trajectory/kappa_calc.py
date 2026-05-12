"""Compute Cohen's Kappa between Claude LLM labels and manual human labels.

Reads:
  trajectory/subset_claude.json   — original trajectories (violated labels from
                                    real_trajectories.json are the LLM labels)
  trajectory/subset_manual.json   — human labels in the same format

Expected format for subset_manual.json:
  [
    {
      "trajectory_id": "real_0012_r06",
      "violated": false,                         <- trajectory-level label
      "steps": [
        {"step": 1, "violated": false},          <- step-level label
        {"step": 2, "violated": true},
        ...
      ]
    },
    ...
  ]

Outputs:
  - Trajectory-level Cohen's Kappa  (binary: violated / not violated)
  - Step-level Cohen's Kappa        (binary: violated / not violated)
  - Confusion matrices for both levels
  - Per-category breakdown
"""
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------

def cohen_kappa(y1: list[int], y2: list[int]) -> float:
    """Binary Cohen's Kappa from two label lists (0/1)."""
    assert len(y1) == len(y2) and len(y1) > 0
    n = len(y1)

    # Confusion matrix counts
    tp = sum(a == 1 and b == 1 for a, b in zip(y1, y2))
    tn = sum(a == 0 and b == 0 for a, b in zip(y1, y2))
    fp = sum(a == 0 and b == 1 for a, b in zip(y1, y2))  # manual=1, claude=0
    fn = sum(a == 1 and b == 0 for a, b in zip(y1, y2))  # manual=0, claude=1

    p_o = (tp + tn) / n  # observed agreement

    # Expected agreement under independence
    p_pos_a = (tp + fn) / n   # claude positive rate
    p_pos_b = (tp + fp) / n   # manual positive rate
    p_neg_a = 1 - p_pos_a
    p_neg_b = 1 - p_pos_b
    p_e = p_pos_a * p_pos_b + p_neg_a * p_neg_b

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 1.0
    return kappa, p_o, p_e, tp, tn, fp, fn


def kappa_interp(k: float) -> str:
    if k < 0:      return "poor (worse than chance)"
    if k < 0.20:   return "slight"
    if k < 0.40:   return "fair"
    if k < 0.60:   return "moderate"
    if k < 0.80:   return "substantial"
    return         "almost perfect"


def print_confusion(tp, tn, fp, fn, label="") -> None:
    print(f"\n  Confusion matrix{' — ' + label if label else ''}:")
    print(f"                    Manual +   Manual −")
    print(f"    Claude  +   {tp:>8}   {fn:>8}")
    print(f"    Claude  −   {fp:>8}   {tn:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    claude_path = os.path.join(_HERE, "subset_claude.json")
    manual_path = os.path.join(_HERE, "subset_manual.json")
    source_path = os.path.join(_HERE, "real_trajectories.json")

    # Load manual labels
    if not os.path.exists(manual_path):
        print(f"ERROR: {manual_path} not found. Please complete manual labeling first.")
        sys.exit(1)

    with open(claude_path,  encoding="utf-8") as f:
        subset = json.load(f)
    with open(manual_path,  encoding="utf-8") as f:
        manual_data = json.load(f)
    with open(source_path,  encoding="utf-8") as f:
        all_trajs = json.load(f)

    # Build lookup tables
    source_by_id = {t["trajectory_id"]: t for t in all_trajs}
    manual_by_id = {t["trajectory_id"]: t for t in manual_data}
    claude_by_id = {t["trajectory_id"]: t for t in subset}

    # Verify IDs match
    subset_ids  = {t["trajectory_id"] for t in subset}
    manual_ids  = set(manual_by_id.keys())
    missing     = subset_ids - manual_ids
    extra       = manual_ids - subset_ids
    if missing:
        print(f"WARNING: {len(missing)} trajectories in subset not found in manual labels:")
        for tid in sorted(missing):
            print(f"  {tid}")
    if extra:
        print(f"WARNING: {len(extra)} extra trajectories in manual labels (not in subset):")
        for tid in sorted(extra):
            print(f"  {tid}")

    # -----------------------------------------------------------------------
    # Trajectory-level labels
    # -----------------------------------------------------------------------
    traj_claude, traj_manual, traj_ids, traj_cats = [], [], [], []

    for t in subset:
        tid = t["trajectory_id"]
        if tid not in manual_by_id:
            continue
        claude_label = int(claude_by_id[tid]["violated"])
        manual_label = int(manual_by_id[tid]["violated"])
        traj_claude.append(claude_label)
        traj_manual.append(manual_label)
        traj_ids.append(tid)
        traj_cats.append(source_by_id[tid]["task_category"])

    k_traj, po_t, pe_t, tp_t, tn_t, fp_t, fn_t = cohen_kappa(traj_claude, traj_manual)

    print("=" * 60)
    print("TRAJECTORY-LEVEL AGREEMENT")
    print("=" * 60)
    print(f"  N trajectories     : {len(traj_claude)}")
    print(f"  Observed agreement : {po_t:.3f}")
    print(f"  Expected agreement : {pe_t:.3f}")
    print(f"  Cohen's Kappa      : {k_traj:.3f}  [{kappa_interp(k_traj)}]")
    print_confusion(tp_t, tn_t, fp_t, fn_t, "trajectory-level")

    # List disagreements
    disagree_traj = [(traj_ids[i], traj_claude[i], traj_manual[i], traj_cats[i])
                     for i in range(len(traj_ids)) if traj_claude[i] != traj_manual[i]]
    if disagree_traj:
        print(f"\n  Disagreements ({len(disagree_traj)}):")
        for tid, c, m, cat in disagree_traj:
            print(f"    {tid}  claude={'violated' if c else 'safe':8s}  "
                  f"manual={'violated' if m else 'safe':8s}  [{cat}]")

    # -----------------------------------------------------------------------
    # Step-level labels
    # -----------------------------------------------------------------------
    step_claude, step_manual = [], []
    step_details = []   # (tid, step_idx, claude, manual) for disagreement report

    for t in subset:
        tid = t["trajectory_id"]
        if tid not in manual_by_id:
            continue
        claude_steps = {s["step"]: s for s in claude_by_id[tid]["steps"]}
        manual_steps = {s["step"]: s for s in manual_by_id[tid].get("steps", [])}

        for sidx in sorted(claude_steps.keys()):
            if sidx not in manual_steps:
                continue
            c = int(claude_steps[sidx]["violated"])
            m = int(manual_steps[sidx]["violated"])
            step_claude.append(c)
            step_manual.append(m)
            step_details.append((tid, sidx, c, m))

    if step_claude:
        k_step, po_s, pe_s, tp_s, tn_s, fp_s, fn_s = cohen_kappa(step_claude, step_manual)

        print("\n" + "=" * 60)
        print("STEP-LEVEL AGREEMENT")
        print("=" * 60)
        print(f"  N steps            : {len(step_claude)}")
        print(f"  Observed agreement : {po_s:.3f}")
        print(f"  Expected agreement : {pe_s:.3f}")
        print(f"  Cohen's Kappa      : {k_step:.3f}  [{kappa_interp(k_step)}]")
        print_confusion(tp_s, tn_s, fp_s, fn_s, "step-level")

        disagree_steps = [(tid, si, c, m) for tid, si, c, m in step_details if c != m]
        if disagree_steps:
            print(f"\n  Disagreements ({len(disagree_steps)} / {len(step_claude)} steps):")
            for tid, si, c, m in disagree_steps:
                print(f"    {tid}  step {si:>2}  "
                      f"claude={'V' if c else 'S'}  manual={'V' if m else 'S'}")
    else:
        print("\nNo step-level labels found in subset_manual.json.")

    # -----------------------------------------------------------------------
    # Per-category breakdown (trajectory-level)
    # -----------------------------------------------------------------------
    cats = sorted(set(traj_cats))
    if len(cats) > 1:
        print("\n" + "=" * 60)
        print("PER-CATEGORY TRAJECTORY-LEVEL KAPPA")
        print("=" * 60)
        for cat in cats:
            idx = [i for i, c in enumerate(traj_cats) if c == cat]
            yc  = [traj_claude[i] for i in idx]
            ym  = [traj_manual[i] for i in idx]
            if len(set(yc + ym)) < 2:
                # All same label — kappa undefined
                agree = sum(a == b for a, b in zip(yc, ym))
                print(f"  {cat:<18}  n={len(idx)}  agree={agree}/{len(idx)}  kappa=N/A (single class)")
                continue
            k, po, _, tp, tn, fp, fn = cohen_kappa(yc, ym)
            print(f"  {cat:<18}  n={len(idx)}  kappa={k:+.3f}  [{kappa_interp(k)}]")

    print()


if __name__ == "__main__":
    main()
