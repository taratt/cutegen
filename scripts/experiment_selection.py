"""Shared experiment labeling and selection for canvas/error payloads."""

from __future__ import annotations

SAMPLE19 = [1, 4, 9, 21, 22, 33, 40, 49, 53, 54, 55, 58, 59, 80, 83, 88, 99, 102, 103]
NEW8 = [6, 14, 50, 61, 70, 75, 105, 107]
ALL27 = sorted(set(SAMPLE19 + NEW8))
# Primary 25-kernel cohort: all 27 minus conv kernels 55 and 59.
COHORT25_EXCLUDED = (55, 59)
COHORT25 = [k for k in ALL27 if k not in COHORT25_EXCLUDED]

# When a CuTe/Sonnet minimal tree exists, hide the matching full-prompt tree.
CUTE_SONNET_FULL_TO_MINIMAL: dict[str, str] = {
    "cute/level1-no-profile-sonnet5": "cute/level1-no-profile-sonnet5-minimal",
    "cute/level1-profiled-from-start-sonnet5": "cute/level1-profiled-from-start-sonnet5-minimal",
    "cute/level1-profiled-sonnet5": "cute/level1-profiled-sonnet5-minimal",
}


def experiment_method(leaf: str) -> str:
    leaf_l = leaf.lower()
    if "minimal" in leaf_l:
        if "from-start" in leaf_l or "from_start" in leaf_l:
            return "Minimal from start"
        return "Minimal"
    if "no-profile" in leaf or "nopf" in leaf:
        return "No profiling"
    if "from-start" in leaf or "from_start" in leaf or leaf == "level1_from_start":
        return "Profiling from start"
    if leaf == "level1-profiled" or (
        leaf.startswith("level1-profiled-") and "from-start" not in leaf
    ):
        return "Delayed profiling"
    if "profiled" in leaf:
        return "Delayed profiling"
    return leaf


def experiment_model(rel: str, leaf: str, backend: str = "") -> str:
    del backend  # unused; kept for call-site compatibility
    s = f"{rel}/{leaf}".lower()
    if "gpt5" in s or "gpt-5" in s:
        return "GPT-5"
    if "sonnet" in s:
        return "Sonnet 5"
    return "Kimi K3"


def model_tag(model: str) -> str:
    return {"Sonnet 5": "S", "GPT-5": "G"}.get(model, "K")


def paths_superseded_by_minimal(available_paths: set[str]) -> set[str]:
    """Full-prompt CuTe/Sonnet paths to drop when a minimal counterpart exists."""
    exclude: set[str] = set()
    for full, minimal in CUTE_SONNET_FULL_TO_MINIMAL.items():
        if full in available_paths and minimal in available_paths:
            exclude.add(full)
    return exclude


# Active fleet runs — omit from canvas payloads until finished.
IN_PROGRESS_PATHS = frozenset(
    {
        "ptx/level1-no-profile-gpt5",  # 10707: abandoned GPT-5 nopf partial
        "ptx/level1-profiled-from-start-gpt5",  # 10707: PTX GPT-5 from-start (still running)
    }
)


def paths_in_progress() -> frozenset[str]:
    return IN_PROGRESS_PATHS
