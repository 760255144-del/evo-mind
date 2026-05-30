"""MUSE 分层记忆系统 — 模拟人类大脑的多层级经验抽象存储。

层级（从微观到宏观）：
  L0: 原始痕迹 (Raw Trace) — 单次交互的完整记录
  L1: 情节记忆 (Episodic) — 压缩后的事件摘要
  L2: 语义记忆 (Semantic) — 跨事件提炼的知识概念
  L3: SOP 记忆 (Procedural) — 可复用的标准操作程序
  L4: 策略记忆 (Strategic) — 元层面的进化策略

每一层向上抽象时：
  1. 压缩率 ~10:1
  2. 保留关键决策点
  3. 建立层间索引链接

参考: MUSE框架 (arXiv 2024), Google Reasoning Memory
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType, RelationType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 分层类型 ----

class MuseLevel(IntEnum):
    """MUSE 记忆的五个抽象层级"""
    RAW = 0        # 原始痕迹
    EPISODIC = 1   # 情节
    SEMANTIC = 2   # 语义知识
    PROCEDURAL = 3 # SOP 程序
    STRATEGIC = 4  # 元策略


LEVEL_TO_MEMORY_TYPE = {
    MuseLevel.RAW: MemoryType.EPISODIC,
    MuseLevel.EPISODIC: MemoryType.EPISODIC,
    MuseLevel.SEMANTIC: MemoryType.SEMANTIC,
    MuseLevel.PROCEDURAL: MemoryType.PROCEDURAL,
    MuseLevel.STRATEGIC: MemoryType.SEMANTIC,
}


@dataclass
class MuseExperience:
    """一条完整的经验，包含所有层级"""
    id: str = field(default_factory=uuid7)
    task: str = ""                     # 任务描述
    inputs: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    outcomes: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    trajectory: list[dict[str, Any]] = field(default_factory=list)  # 完整执行轨迹

    # 各层级的抽象
    level_0_raw: dict[str, Any] | None = None       # 原始记录
    level_1_episode: dict[str, Any] | None = None    # 情节摘要
    level_2_knowledge: dict[str, Any] | None = None   # 提炼知识
    level_3_sop: dict[str, Any] | None = None         # 标准操作程序
    level_4_strategy: dict[str, Any] | None = None    # 元策略

    # 关联
    related_experiences: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ---- MUSE 记忆管理器 ----

class MuseMemoryManager:
    """MUSE 分层记忆管理器 — 五级抽象存储与检索。

    核心流程:
      trace → abstract → link → retrieve → reuse

    每一级向上抽象时自动建立索引链接，
    确保高层策略可以追溯到原始经验。
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._experiences: dict[str, MuseExperience] = {}

    # ---- 经验记录 ----

    async def record_experience(self, exp: MuseExperience) -> str:
        """记录一条完整经验，自动进行五级抽象并存储。"""
        self._experiences[exp.id] = exp

        # L0: 原始痕迹
        exp.level_0_raw = {
            "task": exp.task,
            "inputs": exp.inputs,
            "actions": exp.actions,
            "outcomes": exp.outcomes,
            "trajectory": exp.trajectory,
        }
        l0_id = await self._store_level(exp, MuseLevel.RAW, exp.level_0_raw)

        # L1: 情节抽象 (压缩率 ~10:1)
        exp.level_1_episode = self._abstract_to_episode(exp)
        l1_id = await self._store_level(exp, MuseLevel.EPISODIC, exp.level_1_episode)

        # L2: 语义知识 (提炼关键概念)
        exp.level_2_knowledge = self._abstract_to_knowledge(exp)
        l2_id = await self._store_level(exp, MuseLevel.SEMANTIC, exp.level_2_knowledge)

        # L3: SOP (如果成功，提取可复用程序)
        if exp.success:
            exp.level_3_sop = self._extract_sop(exp)
            l3_id = await self._store_level(exp, MuseLevel.PROCEDURAL, exp.level_3_sop)

            # L4: 策略 (SOP 的元模式)
            exp.level_4_strategy = self._abstract_to_strategy(exp)
            await self._store_level(exp, MuseLevel.STRATEGIC, exp.level_4_strategy)

        logger.info("experience_recorded", id=exp.id[:12], task=exp.task[:60])
        return exp.id

    async def record_batch(self, experiences: list[MuseExperience]) -> list[str]:
        """批量记录经验，并交叉关联。"""
        ids = []
        for exp in experiences:
            eid = await self.record_experience(exp)
            ids.append(eid)

        # 建立经验间关联
        await self._link_experiences(experiences)

        return ids

    # ---- 五级抽象 ----

    def _abstract_to_episode(self, exp: MuseExperience) -> dict[str, Any]:
        """L0→L1: 原始痕迹 → 情节摘要 (压缩率 ~10:1)"""
        actions_summary = [
            {
                "step": i + 1,
                "action": a.get("action", "")[:100],
                "result": str(a.get("result", ""))[:200],
            }
            for i, a in enumerate(exp.actions[:20])
        ]

        return {
            "task": exp.task,
            "success": exp.success,
            "num_actions": len(exp.actions),
            "key_actions": actions_summary[:10],
            "outcome_summary": {
                k: str(v)[:200]
                for k, v in exp.outcomes.items()
                if k in ("result", "error", "summary", "score")
            },
            "duration_estimate": len(exp.trajectory),
            "failure_point": self._find_failure_point(exp),
            "success_pattern": self._find_success_pattern(exp) if exp.success else None,
        }

    def _abstract_to_knowledge(self, exp: MuseExperience) -> dict[str, Any]:
        """L1→L2: 情节 → 语义知识 (提炼概念和规则)"""
        # 从动作中提取关键概念
        concepts = set()
        rules_discovered = []

        for action in exp.actions:
            action_name = action.get("action", "")
            if "fix" in action_name.lower() or "repair" in action_name.lower():
                rules_discovered.append({
                    "trigger": action.get("trigger", ""),
                    "method": action_name,
                    "effectiveness": 1.0 if exp.success else 0.0,
                })
            # 提取技术概念
            for kw in ("pattern", "rule", "strategy", "algorithm", "optimize"):
                if kw in str(action).lower():
                    concepts.add(kw)

        return {
            "concepts": list(concepts),
            "rules_discovered": rules_discovered,
            "task_category": self._categorize_task(exp.task),
            "difficulty_estimate": "easy" if len(exp.actions) < 3 else "medium" if len(exp.actions) < 10 else "hard",
            "success_rate_context": 1.0 if exp.success else 0.0,
        }

    def _extract_sop(self, exp: MuseExperience) -> dict[str, Any]:
        """L2→L3: 提取标准操作程序 (SOP)"""
        steps = []
        for i, action in enumerate(exp.actions):
            steps.append({
                "order": i + 1,
                "instruction": action.get("action", ""),
                "expected_outcome": str(action.get("result", ""))[:200],
                "validation": action.get("validation", ""),
                "fallback": action.get("fallback", ""),
            })

        return {
            "sop_name": f"SOP: {exp.task[:80]}",
            "trigger_condition": exp.task[:200],
            "steps": steps[:15],
            "total_steps": len(steps),
            "success_rate": 1.0,  # 初次提取为 1.0，之后根据复用情况调整
            "reuse_count": 0,
            "tags": exp.tags,
            "created_from": exp.id,
        }

    def _abstract_to_strategy(self, exp: MuseExperience) -> dict[str, Any]:
        """L3→L4: SOP → 元策略"""
        sop = exp.level_3_sop or {}
        knowledge = exp.level_2_knowledge or {}

        return {
            "strategy_name": f"Strategy: {self._categorize_task(exp.task)}",
            "derived_from_sop_count": 1,
            "applicable_task_types": [self._categorize_task(exp.task)],
            "success_conditions": [
                "Preconditions verified" if exp.success else "Needs improvement",
                f"Category: {self._categorize_task(exp.task)}",
            ],
            "evolution_potential": 0.8 if exp.success else 0.3,
            "generalization": knowledge.get("concepts", []),
        }

    # ---- 检索与重用 (Reasoning Memory 积累→概括→重用) ----

    async def retrieve_relevant(
        self, task: str, level: MuseLevel = MuseLevel.PROCEDURAL, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """根据任务检索最相关的记忆 (Reasoning Memory 重用阶段)"""
        # 语义检索
        from evo_mind.core.models import SearchQuery
        query = SearchQuery(
            query_text=task,
            max_results=top_k * 2,
            semantic_weight=1.0,
            keyword_weight=0.5,
        )

        results = []
        try:
            retrieval = __import__('evo_mind.retrieval.engine', fromlist=['RetrievalEngine'])
            engine = retrieval.RetrievalEngine(self.store.db, self.store.vector_store, self.store.embedding)
            search_results = await engine.search(query)

            for sr in search_results:
                mem = sr.memory
                if "muse_level" in mem.metadata:
                    results.append({
                        "id": mem.id,
                        "content": mem.content,
                        "score": sr.score,
                        "level": mem.metadata.get("muse_level", 0),
                        "task": mem.content.get("task", ""),
                        "success": mem.content.get("success", False),
                    })
        except Exception:
            pass

        # 按层级和分数排序，优先返回高层策略
        results.sort(key=lambda r: (r["level"], r["score"]), reverse=True)
        return results[:top_k]

    async def generalize(self, task_category: str) -> dict[str, Any]:
        """跨经验概括 (Reasoning Memory 概括阶段)"""
        # 找到同类任务的所有 SOP
        sops = []
        for eid, exp in self._experiences.items():
            if self._categorize_task(exp.task) == task_category and exp.level_3_sop:
                sops.append(exp.level_3_sop)

        if not sops:
            return {"generalization": "insufficient_data", "task_category": task_category}

        # 合并 SOP 步骤
        common_steps = defaultdict(list)
        for sop in sops:
            for step in sop.get("steps", []):
                key = step["instruction"][:60]
                common_steps[key].append(step)

        # 只保留出现率 >= 50% 的步骤
        threshold = max(1, len(sops) // 2)
        generalized_steps = []
        for instruction, occurrences in common_steps.items():
            if len(occurrences) >= threshold:
                generalized_steps.append({
                    "instruction": instruction,
                    "frequency": len(occurrences) / len(sops),
                    "avg_order": sum(s["order"] for s in occurrences) / len(occurrences),
                })

        generalized_steps.sort(key=lambda s: s["avg_order"])

        return {
            "task_category": task_category,
            "source_sop_count": len(sops),
            "generalized_steps": generalized_steps[:10],
            "avg_success_rate": sum(s.get("success_rate", 0) for s in sops) / len(sops),
            "recommendation": "proceed" if len(sops) >= 3 else "collect_more_data",
        }

    # ---- 辅助方法 ----

    async def _store_level(
        self, exp: MuseExperience, level: MuseLevel, data: dict[str, Any]
    ) -> str:
        """存储某个抽象层级到 MemoryStore"""
        mem = await self.store.record(MemoryCreate(
            memory_type=LEVEL_TO_MEMORY_TYPE[level],
            content={
                **data,
                "experience_id": exp.id,
                "muse_level": int(level),
                "task": exp.task,
                "success": exp.success,
            },
            importance=0.3 + int(level) * 0.15,  # 层级越高越重要
            source="plugin",
            tags=[f"muse-L{int(level)}", *exp.tags],
            metadata={
                "muse_level": int(level),
                "experience_id": exp.id,
                "task_hash": str(hash(exp.task))[:12],
            },
        ))
        return mem.id

    async def _link_experiences(self, experiences: list[MuseExperience]) -> None:
        """建立经验间的关联链接 (相似任务、因果链等)"""
        for i, exp_a in enumerate(experiences):
            for j, exp_b in enumerate(experiences):
                if i >= j:
                    continue
                # 相同任务类别 → 关联
                if self._categorize_task(exp_a.task) == self._categorize_task(exp_b.task):
                    exp_a.related_experiences.append(exp_b.id)
                    exp_b.related_experiences.append(exp_a.id)

    @staticmethod
    def _categorize_task(task: str) -> str:
        """将任务分类"""
        task_lower = task.lower()
        if any(w in task_lower for w in ("fix", "bug", "repair", "debug")):
            return "bug_fixing"
        if any(w in task_lower for w in ("build", "implement", "create", "develop")):
            return "implementation"
        if any(w in task_lower for w in ("optimize", "improve", "enhance", "refactor")):
            return "optimization"
        if any(w in task_lower for w in ("analyze", "review", "understand", "explore")):
            return "analysis"
        if any(w in task_lower for w in ("learn", "train", "evolve", "adapt")):
            return "learning"
        return "general"

    @staticmethod
    def _find_failure_point(exp: MuseExperience) -> dict[str, Any] | None:
        """定位失败点"""
        if exp.success:
            return None
        for i, action in enumerate(reversed(exp.actions)):
            result = str(action.get("result", "")).lower()
            if any(w in result for w in ("error", "fail", "exception", "timeout")):
                return {"step": len(exp.actions) - i, "action": action.get("action", ""), "error": result[:200]}
        return {"step": len(exp.actions), "action": "unknown", "error": "unspecified failure"}

    @staticmethod
    def _find_success_pattern(exp: MuseExperience) -> dict[str, Any] | None:
        """提取成功模式"""
        if not exp.success or not exp.actions:
            return None
        # 找到决定性的成功步骤
        for i, action in enumerate(exp.actions):
            result = str(action.get("result", "")).lower()
            if any(w in result for w in ("success", "passed", "complete", "resolved")):
                return {
                    "critical_step": i + 1,
                    "action": action.get("action", ""),
                    "total_steps": len(exp.actions),
                }
        return {"critical_step": len(exp.actions), "action": exp.actions[-1].get("action", "")}
