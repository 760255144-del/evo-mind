"""EvolutionEngine: detects patterns across memories and generates adaptive rules."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from evo_mind.utils import uuid7

from evo_mind.core.models import EvolutionRule, Memory
from evo_mind.core.store import MemoryStore
from evo_mind.persistence.database import Database
from evo_mind.persistence.memory_repo import MemoryRepo
from evo_mind.types import MemoryType, RelationType, RuleStatus, RuleType

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(val: str | None) -> datetime | None:
    if val is None:
        return None
    return datetime.fromisoformat(val)


class EvolutionEngine:
    """Detects patterns across memories and generates adaptive rules.

    Pipeline:
    1. Detect correction patterns (feedback → correction pairs)
    2. Mine procedural patterns (common solutions)
    3. Evaluate existing rules against new evidence
    4. Optimize retrieval/importance strategies
    """

    def __init__(
        self,
        store: MemoryStore,
        db: Database,
        config: dict | None = None,
    ) -> None:
        self.store = store
        self.db = db
        self.repo = MemoryRepo(db)

        cfg = config or {}
        self.min_support = cfg.get("min_support", 3)
        self.min_confidence = cfg.get("min_confidence", 0.6)
        self.evaluation_interval_hours = cfg.get("evaluation_interval_hours", 24)

    async def evolve(
        self,
        *,
        min_support: int | None = None,
        min_confidence: float | None = None,
    ) -> list[EvolutionRule]:
        """Run the full evolution pipeline. Returns new/updated rules."""
        support = min_support or self.min_support
        confidence = min_confidence or self.min_confidence

        logger.info("evolution_started")

        all_rules: list[EvolutionRule] = []

        # 1. Correction pattern detection
        correction_rules = await self.detect_correction_patterns()
        await self._persist_rules(correction_rules)
        all_rules.extend(correction_rules)

        # 2. Procedural pattern mining
        procedural_rules = await self.mine_procedural_patterns()
        await self._persist_rules(procedural_rules)
        all_rules.extend(procedural_rules)

        # 3. Evaluate existing rules
        await self.evaluate_rules(support, confidence)

        # 4. Record metrics
        await self._record_metrics(all_rules)

        logger.info("evolution_completed", new_rules=len(all_rules))
        return all_rules

    async def detect_correction_patterns(self) -> list[EvolutionRule]:
        """Find chains: episodic(error) → feedback(correction) → procedural(fix).

        Generalizes repeated corrections into rules:
        "When <failure_pattern>, apply <correction_strategy>"
        """
        # Find feedback memories that "correct" other memories
        rows = await self.db.fetch_all(
            """SELECT
                mr.source_id as source_id,
                mr.target_id as target_id,
                m_src.content_json as src_content,
                m_tgt.content_json as tgt_content,
                m_src.memory_type as src_type,
                m_tgt.memory_type as tgt_type
            FROM memory_relationships mr
            JOIN memories m_src ON m_src.id = mr.source_id
            JOIN memories m_tgt ON m_tgt.id = mr.target_id
            WHERE mr.relation_type = 'corrects'
              AND m_src.deleted_at IS NULL
              AND m_tgt.deleted_at IS NULL
            ORDER BY mr.created_at DESC
            LIMIT 200"""
        )

        if not rows:
            logger.debug("no_correction_pairs_found")
            return []

        # Group corrections by error pattern
        error_patterns: dict[str, list[dict]] = {}
        for row in rows:
            try:
                src_content = json.loads(row["src_content"])
                tgt_content = json.loads(row["tgt_content"])
            except json.JSONDecodeError:
                continue

            # Extract the error description as the pattern key
            error_text = self._extract_text(tgt_content)
            if not error_text:
                continue

            # Use first 100 chars as pattern signature
            pattern_key = error_text[:100].lower().strip()
            if pattern_key not in error_patterns:
                error_patterns[pattern_key] = []
            error_patterns[pattern_key].append({
                "correction": self._extract_text(src_content),
                "source_id": row["source_id"],
                "target_id": row["target_id"],
            })

        # Generate rules for patterns with sufficient support
        rules: list[EvolutionRule] = []
        for pattern_key, corrections in error_patterns.items():
            if len(corrections) < self.min_support:
                continue

            # Find the most common correction strategy
            correction_texts = [c["correction"] for c in corrections]
            from collections import Counter
            most_common = Counter(correction_texts).most_common(1)
            if not most_common:
                continue

            common_correction, support_count = most_common
            confidence_score = min(1.0, support_count / len(corrections))

            if confidence_score >= self.min_confidence:
                rules.append(EvolutionRule(
                    id=str(uuid7()),
                    rule_type=RuleType.CORRECTION_PATTERN,
                    label=f"Correction: {pattern_key[:60]}...",
                    condition={"error_pattern": pattern_key, "match_method": "semantic_similarity"},
                    action={"correction_strategy": common_correction},
                    confidence=confidence_score,
                    support_count=support_count,
                    contradiction_count=0,
                    status=RuleStatus.ACTIVE.value,
                ))

        return rules

    async def mine_procedural_patterns(self) -> list[EvolutionRule]:
        """Cluster procedural memories to find common solution patterns."""
        # Get procedural memories
        proc_memories = await self.repo.list_recent(limit=200, memory_type=MemoryType.PROCEDURAL)
        if len(proc_memories) < 3:
            return []

        # Simple frequency-based pattern mining on content
        from collections import Counter

        # Extract code snippets or workflow steps
        action_verbs: Counter = Counter()
        patterns: dict[str, list[str]] = {}

        for mem in proc_memories:
            content = mem.content
            # Look for action patterns in content
            action = content.get("action") or content.get("pattern") or content.get("name")
            if action:
                action_verbs[action] += 1

            # Look for category/task grouping
            task = content.get("task") or content.get("category") or "general"
            if task not in patterns:
                patterns[task] = []
            if mem.content_plain:
                patterns[task].append(mem.content_plain)

        rules: list[EvolutionRule] = []
        for task, examples in patterns.items():
            if len(examples) < self.min_support:
                continue

            # Create a template rule
            common_action = action_verbs.most_common(1)
            action_name = common_action[0][0] if common_action else task

            rules.append(EvolutionRule(
                id=str(uuid7()),
                rule_type=RuleType.PROCEDURAL_TEMPLATE,
                label=f"Template: {action_name}",
                condition={
                    "task_type": task,
                    "trigger_keywords": self._extract_keywords(examples),
                },
                action={
                    "template_type": action_name,
                    "example_count": len(examples),
                    "reference_ids": [m.id for m in proc_memories if m.content.get("task") == task][:10],
                },
                confidence=min(0.8, len(examples) / 20.0),
                support_count=len(examples),
                contradiction_count=0,
                status=RuleStatus.ACTIVE.value,
            ))

        return rules

    async def evaluate_rules(
        self, min_support: int | None = None, min_confidence: float | None = None
    ) -> None:
        """Re-evaluate all active rules against new evidence since last evaluation."""
        support = min_support or self.min_support
        confidence = min_confidence or self.min_confidence
        now = _now()

        rows = await self.db.fetch_all(
            """SELECT * FROM evolution_rules WHERE status = 'active'"""
        )

        for row in rows:
            rule_id = row["id"]
            last_eval = _parse_dt(row["last_evaluated_at"])

            # Query new memories since last evaluation for supporting/contradicting evidence
            condition_json = row["condition_json"]
            try:
                condition = json.loads(condition_json)
            except json.JSONDecodeError:
                condition = {}

            error_pattern = condition.get("error_pattern", "")
            last_eval_time = last_eval.isoformat() if last_eval else "1970-01-01T00:00:00"

            if error_pattern:
                # Escape SQL LIKE wildcards in the pattern
                safe_pattern = error_pattern[:50].replace("%", r"\%").replace("_", r"\_")
                supporting_rows = await self.db.fetch_all(
                    """SELECT COUNT(*) as cnt FROM memories
                       WHERE created_at > ? AND deleted_at IS NULL
                         AND (content_plain LIKE ? ESCAPE '\\' OR content_json LIKE ? ESCAPE '\\')""",
                    (last_eval_time, f"%{safe_pattern}%", f"%{safe_pattern}%"),
                )
                new_support = supporting_rows[0]["cnt"] if supporting_rows else 0

                # Count contradicting memories (corrections that were marked as failed)
                contradicting_rows = await self.db.fetch_all(
                    """SELECT COUNT(*) as cnt FROM memory_relationships mr
                       JOIN memories m ON m.id = mr.source_id
                       WHERE mr.relation_type = 'contradicts'
                         AND m.created_at > ? AND m.deleted_at IS NULL""",
                    (last_eval_time,),
                )
                new_contradictions = contradicting_rows[0]["cnt"] if contradicting_rows else 0
            else:
                new_support = 0
                new_contradictions = 0

            # Update counts
            total_support = row["support_count"] + new_support
            total_contradictions = row["contradiction_count"] + new_contradictions
            new_confidence = (
                total_support / (total_support + total_contradictions + 0.001)
            )

            await self.db.execute(
                """UPDATE evolution_rules
                   SET support_count = ?, contradiction_count = ?,
                       confidence = ?, last_evaluated_at = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (total_support, total_contradictions, new_confidence,
                 now, now, rule_id),
            )

            # Deprecate if confidence dropped
            if new_confidence < confidence:
                await self.db.execute(
                    """UPDATE evolution_rules
                       SET status = 'deprecated', updated_at = ?
                       WHERE id = ?""",
                    (now, rule_id),
                )
                logger.info("rule_deprecated", rule_id=rule_id, confidence=new_confidence)

        await self.db.commit()

    async def optimize_strategies(self) -> dict[str, float]:
        """Adjust retrieval weights based on observed access patterns."""
        # Get recent search+access patterns
        weights = {
            "semantic_weight": 1.0,
            "keyword_weight": 0.5,
            "temporal_weight": 0.3,
            "default_importance": 0.5,
        }

        # Fetch recent access patterns and compute which strategies work best
        recent = await self.repo.list_recent(limit=100)
        if not recent:
            return weights

        # Memories with high access_count suggest their strategy was effective
        # Adjust weights based on average memory type distribution
        type_counts: dict[str, int] = {}
        for mem in recent:
            type_counts[mem.memory_type.value] = type_counts.get(mem.memory_type.value, 0) + 1

        # Simple heuristic: if semantic memories dominate, boost semantic search
        total = sum(type_counts.values())
        if total > 0:
            semantic_ratio = type_counts.get("semantic", 0) / total
            procedural_ratio = type_counts.get("procedural", 0) / total

            weights["semantic_weight"] = 1.0 + semantic_ratio * 0.5
            weights["keyword_weight"] = 0.5 + procedural_ratio * 0.5

        # Record the optimization as a metric
        await self.db.execute(
            """INSERT INTO evolution_metrics (id, metric_name, metric_value, recorded_at, dimension_json)
               VALUES (?, ?, ?, ?, ?)""",
            (str(uuid7()), "strategy_weights", weights.get("semantic_weight", 0.0),
             _now(), json.dumps(weights)),
        )
        await self.db.commit()

        return weights

    async def get_rules(
        self,
        rule_type: RuleType | None = None,
        min_confidence: float = 0.0,
        status: RuleStatus | None = None,
    ) -> list[EvolutionRule]:
        """Query learned rules."""
        conditions = ["1=1"]
        params: list[object] = []

        if rule_type:
            conditions.append("rule_type = ?")
            params.append(rule_type.value)
        if min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if status:
            conditions.append("status = ?")
            params.append(status.value)

        where = " AND ".join(conditions)
        rows = await self.db.fetch_all(
            f"SELECT * FROM evolution_rules WHERE {where} ORDER BY confidence DESC",
            params,
        )

        return [
            EvolutionRule(
                id=row["id"],
                rule_type=RuleType(row["rule_type"]),
                label=row["label"],
                condition=json.loads(row["condition_json"]),
                action=json.loads(row["action_json"]),
                confidence=row["confidence"],
                support_count=row["support_count"],
                contradiction_count=row["contradiction_count"],
                status=row["status"],
                last_fired_at=_parse_dt(row["last_fired_at"]),
                last_evaluated_at=_parse_dt(row["last_evaluated_at"]),
                created_at=_parse_dt(row["created_at"]),
                updated_at=_parse_dt(row["updated_at"]),
                superseded_by=row["superseded_by"],
            )
            for row in rows
        ]

    async def get_metrics(
        self,
        metric_names: list[str] | None = None,
        since: datetime | None = None,
    ) -> list[dict]:
        """Retrieve evolution fitness metrics."""
        conditions = ["1=1"]
        params: list[object] = []

        if metric_names:
            placeholders = ",".join(["?"] * len(metric_names))
            conditions.append(f"metric_name IN ({placeholders})")
            params.extend(metric_names)
        if since:
            conditions.append("recorded_at >= ?")
            params.append(since.isoformat())

        where = " AND ".join(conditions)
        rows = await self.db.fetch_all(
            f"SELECT * FROM evolution_metrics WHERE {where} ORDER BY recorded_at DESC LIMIT 100",
            params,
        )

        return [
            {
                "id": row["id"],
                "metric_name": row["metric_name"],
                "metric_value": row["metric_value"],
                "recorded_at": row["recorded_at"],
                "session_id": row["session_id"],
                "dimension": json.loads(row["dimension_json"] or "{}"),
            }
            for row in rows
        ]

    # ---- Helpers ----

    async def _persist_rules(self, rules: list[EvolutionRule]) -> None:
        """Save or update rules in the database."""
        now = _now()
        for rule in rules:
            await self.db.execute(
                """INSERT OR REPLACE INTO evolution_rules
                   (id, rule_type, label, condition_json, action_json,
                    confidence, support_count, contradiction_count,
                    last_evaluated_at, created_at, updated_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule.id,
                    rule.rule_type.value,
                    rule.label,
                    json.dumps(rule.condition, ensure_ascii=False),
                    json.dumps(rule.action, ensure_ascii=False),
                    rule.confidence,
                    rule.support_count,
                    rule.contradiction_count,
                    now,
                    now,
                    now,
                    rule.status,
                ),
            )
        await self.db.commit()

    async def _record_metrics(self, rules: list[EvolutionRule]) -> None:
        """Record evolution fitness metrics."""
        now = _now()
        # Query actual DB counts, not just newly-generated rules
        row = await self.db.fetch_one(
            "SELECT COUNT(*) as cnt, AVG(confidence) as avg_conf FROM evolution_rules WHERE status='active'"
        )
        total_rules = row["cnt"] if row else 0
        avg_confidence = row["avg_conf"] if row and row["avg_conf"] else 0.0

        metrics = [
            ("total_rules_active", total_rules),
            ("avg_rule_confidence", avg_confidence),
        ]

        corr_rules = [r for r in rules if r.rule_type == RuleType.CORRECTION_PATTERN]
        proc_rules = [r for r in rules if r.rule_type == RuleType.PROCEDURAL_TEMPLATE]

        metrics.append(("correction_patterns_discovered", len(corr_rules)))
        metrics.append(("procedural_templates_discovered", len(proc_rules)))

        for name, value in metrics:
            await self.db.execute(
                """INSERT INTO evolution_metrics (id, metric_name, metric_value, recorded_at)
                   VALUES (?, ?, ?, ?)""",
                (str(uuid7()), name, float(value), now),
            )
        await self.db.commit()

    @staticmethod
    def _extract_text(content: dict) -> str:
        """Extract text from content dict."""
        for key in ("text", "description", "summary", "error", "correction", "message"):
            if key in content and isinstance(content[key], str):
                return content[key]
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _extract_keywords(texts: list[str], n: int = 10) -> list[str]:
        """Extract common keywords from a list of texts."""
        import re
        from collections import Counter

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "to", "of", "in",
            "on", "at", "for", "with", "by", "from", "and", "or", "but",
            "not", "no", "this", "that", "it", "its",
        }

        words: Counter = Counter()
        for text in texts:
            found = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            for w in found:
                if w not in stopwords:
                    words[w] += 1

        return [w for w, _ in words.most_common(n)]
