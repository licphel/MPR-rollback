"""Trajectory data classes and shared risk-accumulation helpers."""


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
    """Mutable execution state for a single trajectory run."""

    def __init__(self, trajectories: list[Trajectory]):
        self.trajectories  = trajectories
        self.steps         = 0      # pointer = index of the *next* step to process
        self.risk          = 0.0    # cumulative accumulated risk
        self.step_risks: dict[int, float] = {}  # cached per-step risk scores

    # ------------------------------------------------------------------

    def next(self) -> bool:
        """Advance the step pointer by one.

        Returns True when the pointer has moved past the last step
        (i.e. the trajectory is finished).
        """
        self.steps += 1
        return self.steps >= len(self.trajectories)

    def rollback_to(self, step: int) -> None:
        """Reset the step pointer to *step* (0-based index)."""
        self.steps = step

    def current_trajectory(self) -> Trajectory:
        return self.trajectories[self.steps]


# ---------------------------------------------------------------------------
# Shared risk-accumulation helpers
# ---------------------------------------------------------------------------

def influence_factor(step_loc: int, ctx: Context) -> float:
    """Weight for a step's contribution: later steps have more influence.

    step_loc is the 1-indexed step number from the JSON.
    ctx.steps is the count of steps already processed at the point of calling.
    """
    return step_loc / ctx.steps


def risk_tolerance(ctx: Context) -> float:
    """Accumulated-risk threshold above which a repair is triggered."""
    return 0.5


def risk_decrease_threshold(ctx: Context) -> float:
    """Target accumulated-risk level for multi-pivot repair.

    After greedy sanitisation, risk must fall to AT MOST this value.
    We aim to restore well into safe territory (half the tolerance ceiling)
    rather than merely reducing by a small delta, so that multiple pivot points
    are selected when the accumulated risk significantly exceeds tolerance.
    """
    return risk_tolerance(ctx) * 0.5
