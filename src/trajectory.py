"""Trajectory data classes and shared attribution helpers."""


class Trajectory:
    """A single step in an agent trajectory."""

    def __init__(self, json: dict):
        self.step        = json["step"]          # 1-indexed (from JSON field)
        self.thought     = json.get("thought", "")
        self.output      = json.get("output", "")
        self.tool        = json["action"]["tool"]
        self.args        = json["action"]["args"]
        self.observation = json["observation"]
        self.violated    = json.get("violated", False)  # per-step ground-truth label


class Context:
    """Execution context for a single trajectory.

    In the global-attribution framework the full trajectory is evaluated
    in one pass before any attribution decision is made.  Context stores
    the complete step list and their cached risk scores.
    """

    def __init__(self, trajectories: list[Trajectory]):
        self.trajectories  = trajectories
        self.steps         = len(trajectories)   # total step count (fixed)
        self.step_risks: dict[int, float] = {}   # cached per-step risk scores


# ---------------------------------------------------------------------------
# Shared attribution helpers
# ---------------------------------------------------------------------------

def influence_factor(step_loc: int, ctx: Context) -> float:
    """Weight for a step's contribution to attributed risk.

    step_loc is the 1-indexed step number from the JSON.
    Returns 1.0 (uniform weighting).
    """
    return 1.0


def weighted_risk(ctx: Context) -> float:
    """Total influence-weighted risk across all steps."""
    return sum(
        ctx.step_risks.get(i, 0.0) * influence_factor(ctx.trajectories[i].step, ctx)
        for i in range(ctx.steps)
    )


def projected_risk(ctx: Context, zeroed: set[int]) -> float:
    """Weighted risk with the steps in *zeroed* treated as sanitized (c_i = 0)."""
    return sum(
        0.0 if i in zeroed
        else ctx.step_risks.get(i, 0.0) * influence_factor(ctx.trajectories[i].step, ctx)
        for i in range(ctx.steps)
    )


def risk_tolerance(ctx: Context) -> float:
    """Total-risk threshold above which attribution is triggered."""
    return 0.75


def risk_decrease_threshold(ctx: Context) -> float:
    """Target projected risk for multi-pivot greedy stopping criterion."""
    return 0.25
