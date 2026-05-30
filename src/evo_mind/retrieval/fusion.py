"""Reciprocal Rank Fusion with weighted strategies."""

from __future__ import annotations

from evo_mind.core.models import Memory


def reciprocal_rank_fusion(
    results_groups: list[list[tuple[str, float, dict[str, float]]]],
    strategy_weights: list[float],
    k: int = 60,
) -> list[dict]:
    """Fuse multiple ranked result lists using weighted Reciprocal Rank Fusion.

    RRF score = sum over strategies: weight_s * 1 / (k + rank_position_s)

    Args:
        results_groups: One list per strategy, each element is (memory_id, raw_score, score_breakdown).
        strategy_weights: Weight per strategy (0.0 to disable).
        k: RRF constant (default 60, from literature).

    Returns:
        Merged, scored list with memory objects and score breakdowns.
    """
    # Accumulate scores
    fused: dict[str, dict] = {}

    for group_idx, results in enumerate(results_groups):
        weight = strategy_weights[group_idx] if group_idx < len(strategy_weights) else 0.0
        if weight <= 0 or not results:
            continue

        for rank, (mem_id, raw_score, breakdown) in enumerate(results):
            rrf_contribution = weight * (1.0 / (k + rank + 1))

            if mem_id not in fused:
                fused[mem_id] = {
                    "memory_id": mem_id,
                    "memory": None,  # Hydrated later by caller
                    "score": 0.0,
                    "score_breakdown": {},
                }

            fused[mem_id]["score"] += rrf_contribution
            # Merge breakdowns
            for strat_name, strat_score in breakdown.items():
                fused[mem_id]["score_breakdown"][strat_name] = strat_score

    # Sort by fused score descending
    sorted_results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

    return sorted_results


# Re-export for convenience
__all__ = ["reciprocal_rank_fusion"]
