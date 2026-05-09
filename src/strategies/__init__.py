"""Strategy package exports."""
from .no_repair          import run_trajectory as no_repair
from .single_pivot       import run_trajectory as single_pivot
from .multi_pivot        import run_trajectory as multi_pivot
from .full_sanitization  import run_trajectory as full_attribution

STRATEGIES: dict[str, callable] = {
    "no_repair":        no_repair,
    "single_pivot":     single_pivot,
    "multi_pivot":      multi_pivot,
    "full_attribution": full_attribution,
}
