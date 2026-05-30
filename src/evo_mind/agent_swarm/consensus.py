"""ConsensusEngine — agreement mechanisms for multi-agent decisions."""

from __future__ import annotations

import logging
from typing import Any

from evo_mind.agent_swarm.task import TaskResult

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """Evaluates multi-agent results and builds consensus.

    Methods:
    - Majority voting
    - Weighted scoring (by agent reliability)
    - Agreement level calculation
    """

    def __init__(self) -> None:
        self._agent_reliability: dict[str, float] = {}  # agent_id -> reliability score

    def evaluate(self, results: list[TaskResult]) -> dict[str, Any]:
        """Evaluate task results and produce a consensus verdict."""
        if not results:
            return {
                "successful": 0,
                "failed": 0,
                "agreement_level": 0.0,
                "recommendation": "no_data",
                "verdict": "inconclusive",
            }

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        # Agreement level: 1.0 if all agree, lower if split
        if len(results) == 1:
            agreement = 1.0
        else:
            majority = max(successful, failed)
            agreement = majority / len(results)

        # Weighted success score
        weighted_score = self._weighted_score(results)

        # Verdict
        if agreement >= 0.8:
            verdict = "strong_consensus"
        elif agreement >= 0.6:
            verdict = "moderate_consensus"
        elif agreement >= 0.4:
            verdict = "weak_consensus"
        else:
            verdict = "no_consensus"

        # Recommendation
        if verdict in ("strong_consensus", "moderate_consensus") and weighted_score >= 0.6:
            recommendation = "proceed"
        elif verdict == "weak_consensus":
            recommendation = "review_and_retry"
        else:
            recommendation = "replan"

        # Update reliability scores
        for r in results:
            self._update_reliability(r.agent_id, r.success)

        return {
            "successful": successful,
            "failed": failed,
            "agreement_level": round(agreement, 3),
            "weighted_score": round(weighted_score, 3),
            "verdict": verdict,
            "recommendation": recommendation,
            "agent_reliability": {
                aid: round(rel, 3)
                for aid, rel in self._agent_reliability.items()
            },
        }

    def _weighted_score(self, results: list[TaskResult]) -> float:
        """Compute success score weighted by agent reliability."""
        if not results:
            return 0.0

        total_weight = 0.0
        weighted_success = 0.0

        for r in results:
            reliability = self._agent_reliability.get(r.agent_id, 0.5)
            weight = reliability
            total_weight += weight
            if r.success:
                weighted_success += weight

        if total_weight == 0:
            return sum(1 for r in results if r.success) / len(results)

        return weighted_success / total_weight

    def _update_reliability(self, agent_id: str, success: bool) -> None:
        """Update agent reliability score using exponential moving average."""
        if not agent_id:
            return

        current = self._agent_reliability.get(agent_id, 0.5)
        target = 1.0 if success else 0.1
        # EMA with alpha=0.1
        updated = current + 0.1 * (target - current)
        self._agent_reliability[agent_id] = max(0.0, min(1.0, updated))

    def elect_leader(self, agents: list[dict[str, Any]]) -> str | None:
        """Elect a swarm leader based on reliability and experience."""
        if not agents:
            return None

        scored = []
        for agent in agents:
            agent_id = agent.get("id", "")
            reliability = self._agent_reliability.get(agent_id, 0.5)
            completed = agent.get("tasks_completed", 0)
            # Score: 0.6 * reliability + 0.4 * experience_normalized
            exp_score = min(1.0, completed / 50.0)
            score = 0.6 * reliability + 0.4 * exp_score
            scored.append((agent_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None


# Re-export
__all__ = ["ConsensusEngine"]
