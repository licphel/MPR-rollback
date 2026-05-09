"""Ablation studies for Multi-Pivot attribution."""
from .multi_pivot_random import run_trajectory as multi_pivot_random

ABLATIONS: dict[str, callable] = {
    "multi_pivot_random": multi_pivot_random,
}
