from .multi_pivot_no_weight import run_trajectory as multi_pivot_no_weight
from .multi_pivot_random    import run_trajectory as multi_pivot_random

ABLATIONS = {
    "multi_pivot_no_weight": multi_pivot_no_weight,
    "multi_pivot_random":    multi_pivot_random,
}
