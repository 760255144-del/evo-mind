"""经验驱动进化引擎 — 从执行到反思到进化的完整闭环。

三种核心机制:
1. SOP 提炼器 (Google Reasoning Memory 风格)
   - 积累(Accumulate): 收集成功和失败经验
   - 概括(Generalize): 跨案例提炼通用模式
   - 重用(Reuse): 将历史经验应用于新任务

2. 自我反思管线
   - 任务执行后主动复盘
   - 将成功轨迹提炼为可复用 SOP
   - 将失败案例转化为警示规则

3. 经验演化
   - SOP 版本管理: 每次复用后更新成功率
   - 冲突检测: 新旧 SOP 冲突时启动仲裁
   - 渐进增强: 成功率高的 SOP 自动升级为策略
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.training.muse_memory import MuseExperience, MuseLevel, MuseMemoryManager
from evo_mind.types import MemoryType, RelationType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 数据类型 ----

@dataclass
class SOP:
    """标准操作程序 — 可复用的经验模板"""
    id: str = field(default_factory=uuid7)
    name: str = ""
    trigger: str = ""                      # 触发条件
    steps: list[dict[str, Any]] = field(default_factory=list)
    success_rate: float = 1.0             # 使用成功率
    reuse_count: int = 0                   # 复用次数
    version: int = 1
    created_from: list[str] = field(default_factory=list)  # 来源经验ID
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass
class Reflection:
    """自我反思记录"""
    id: str = field(default_factory=uuid7)
    experience_id: str = ""
    task: str = ""
    success: bool = False
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    root_cause: str = ""                   # 根本原因分析
    lessons_learned: list[str] = field(default_factory=list)
    improvement_actions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: str = field(default_factory=_now)


@dataclass
class ReasoningMemoryEntry:
    """推理记忆条目 — 积累→概括→重用 的原子单位"""
    id: str = field(default_factory=uuid7)
    context: dict[str, Any] = field(default_factory=dict)    # 任务上下文
    action: dict[str, Any] = field(default_factory=dict)     # 采取的行动
    outcome: dict[str, Any] = field(default_factory=dict)    # 结果
    generalization: dict[str, Any] | None = None              # 跨案例概括
    reuse_count: int = 0
    last_reused_at: str | None = None


# ---- 经验驱动引擎 ----

class ExperienceDrivenEngine:
    """经验驱动的进化引擎。

    执行 SOP 驱动的持续改进:
      Task → Execute → Reflect → Extract SOP → Reuse → Evolve
    """

    def __init__(self, store: MemoryStore, muse: MuseMemoryManager | None = None) -> None:
        self.store = store
        self.muse = muse or MuseMemoryManager(store)
        self._sops: dict[str, SOP] = {}
        self._reasoning_memory: dict[str, ReasoningMemoryEntry] = {}
        self._reflections: dict[str, Reflection] = {}

    # ---- SOP 提炼 ----

    async def extract_sop_from_experience(self, exp: MuseExperience) -> SOP:
        """从一条经验中提炼标准操作程序"""
        steps = []
        for i, action in enumerate(exp.actions):
            steps.append({
                "order": i + 1,
                "instruction": action.get("action", ""),
                "expected_result": str(action.get("result", ""))[:200],
                "validation_check": self._generate_validation(action),
                "error_handling": action.get("error_handling", "Report and retry"),
            })

        sop = SOP(
            name=f"SOP: {exp.task[:60]}",
            trigger=exp.task[:200],
            steps=steps,
            success_rate=1.0 if exp.success else 0.0,
            created_from=[exp.id],
            tags=exp.tags,
        )

        self._sops[sop.id] = sop
        await self._store_sop(sop)
        return sop

    async def generalize_sops(self, task_category: str) -> SOP | None:
        """跨多个 SOP 概括出通用 SOP (Reasoning Memory: Generalize)"""
        category_sops = [
            s for s in self._sops.values()
            if any(t.startswith(task_category) for t in s.tags)
        ]

        if len(category_sops) < 2:
            return None

        # 找出公共步骤
        step_counter: dict[str, list[dict]] = defaultdict(list)
        for sop in category_sops:
            for step in sop.steps:
                key = step["instruction"][:60]
                step_counter[key].append(step)

        generalized_steps = []
        for instruction, occurrences in step_counter.items():
            if len(occurrences) >= max(2, len(category_sops) // 2):
                avg_order = sum(s["order"] for s in occurrences) / len(occurrences)
                generalized_steps.append({
                    "order": int(avg_order),
                    "instruction": instruction,
                    "frequency": len(occurrences) / len(category_sops),
                    "aggregated_from": len(occurrences),
                })

        if not generalized_steps:
            return None

        generalized_steps.sort(key=lambda s: s["order"])

        generalized = SOP(
            name=f"Generalized SOP: {task_category}",
            trigger=f"Task category: {task_category}",
            steps=generalized_steps,
            success_rate=sum(s.success_rate for s in category_sops) / len(category_sops),
            created_from=[s.id for s in category_sops],
            tags=[task_category, "generalized"],
            version=1,
        )

        self._sops[generalized.id] = generalized
        await self._store_sop(generalized)
        return generalized

    async def reuse_sop(self, sop_id: str, task: str, success: bool) -> SOP:
        """重用 SOP，更新其成功率 (Reasoning Memory: Reuse)"""
        sop = self._sops.get(sop_id)
        if not sop:
            # Try to load from store
            sop = await self._load_sop(sop_id)
            if not sop:
                raise ValueError(f"SOP not found: {sop_id}")

        # 指数移动平均更新成功率
        alpha = 0.1
        sop.reuse_count += 1
        sop.success_rate = sop.success_rate + alpha * ((1.0 if success else 0.0) - sop.success_rate)
        sop.updated_at = _now()

        self._sops[sop.id] = sop

        # 如果成功率降到阈值以下，标记为需要改进
        if sop.success_rate < 0.3 and sop.reuse_count >= 5:
            logger.warning("sop_degrading", sop=sop.name, success_rate=f"{sop.success_rate:.2f}")

        return sop

    # ---- 自我反思 ----

    async def reflect(self, exp: MuseExperience) -> Reflection:
        """对一次经验进行深度反思 (Self-Reflection)"""
        reflection = Reflection(
            experience_id=exp.id,
            task=exp.task,
            success=exp.success,
        )

        # 分析成功因素
        if exp.success:
            reflection.what_worked = self._analyze_success_factors(exp)
            reflection.lessons_learned = [
                f"Pattern confirmed: {factor}" for factor in reflection.what_worked[:3]
            ]
        else:
            reflection.what_failed = self._analyze_failure_factors(exp)
            reflection.root_cause = self._root_cause_analysis(exp)
            reflection.lessons_learned = [
                f"Avoid: {factor}" for factor in reflection.what_failed[:3]
            ]

        # 生成改进建议
        reflection.improvement_actions = self._generate_improvements(reflection)
        reflection.confidence = 0.8 if exp.success else 0.6

        self._reflections[reflection.id] = reflection

        # 存储为反馈记忆
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.FEEDBACK,
            content={
                "type": "self_reflection",
                "reflection_id": reflection.id,
                "task": reflection.task,
                "success": reflection.success,
                "what_worked": reflection.what_worked,
                "what_failed": reflection.what_failed,
                "root_cause": reflection.root_cause,
                "lessons_learned": reflection.lessons_learned,
                "improvements": reflection.improvement_actions,
            },
            importance=0.8,
            source="plugin",
            tags=["reflection", "learning", "success" if exp.success else "failure"],
        ))

        return reflection

    async def batch_reflect(self, experiences: list[MuseExperience]) -> list[Reflection]:
        """批量反思并交叉分析"""
        reflections = []
        for exp in experiences:
            r = await self.reflect(exp)
            reflections.append(r)

        # 跨案例根因分析
        await self._cross_case_analysis(reflections)

        return reflections

    # ---- Reasoning Memory (积累→概括→重用) ----

    async def accumulate(self, context: dict, action: dict, outcome: dict) -> str:
        """积累一条推理记忆"""
        entry = ReasoningMemoryEntry(
            context=context,
            action=action,
            outcome=outcome,
        )
        self._reasoning_memory[entry.id] = entry
        return entry.id

    async def generalize_from_memory(
        self, context_filter: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """从推理记忆中概括通用模式"""
        relevant = [
            e for e in self._reasoning_memory.values()
            if not context_filter or all(
                e.context.get(k) == v for k, v in context_filter.items()
            )
        ]

        if len(relevant) < 3:
            return {"status": "insufficient_data", "count": len(relevant)}

        # 提取成功模式
        successful = [e for e in relevant if e.outcome.get("success", False)]
        failed = [e for e in relevant if not e.outcome.get("success", False)]

        success_rate = len(successful) / len(relevant) if relevant else 0.0

        # 找出高频成功动作
        action_counter: dict[str, int] = defaultdict(int)
        for e in successful:
            action_key = str(e.action.get("type", e.action.get("name", "")))[:80]
            action_counter[action_key] += 1

        top_actions = sorted(action_counter.items(), key=lambda x: x[1], reverse=True)[:5]

        generalization = {
            "total_examples": len(relevant),
            "success_rate": success_rate,
            "top_successful_actions": [
                {"action": a, "count": c, "frequency": c / len(successful) if successful else 0}
                for a, c in top_actions
            ],
            "failure_patterns": self._extract_failure_patterns(failed),
            "confidence": min(0.9, len(relevant) / 20.0),
        }

        # 标记所有相关条目的概括
        for e in relevant:
            e.generalization = generalization

        return generalization

    # ---- 辅助方法 ----

    async def _store_sop(self, sop: SOP) -> None:
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.PROCEDURAL,
            content={
                "sop_id": sop.id,
                "name": sop.name,
                "trigger": sop.trigger,
                "steps": sop.steps,
                "success_rate": sop.success_rate,
                "reuse_count": sop.reuse_count,
                "version": sop.version,
            },
            importance=0.7,
            source="plugin",
            tags=["sop", *sop.tags],
        ))

    async def _load_sop(self, sop_id: str) -> SOP | None:
        # SOP is stored as a Memory with sop_id in content_json — query by content field
        row = await self.store.db.fetch_one(
            "SELECT * FROM memories WHERE json_extract(content_json, '$.sop_id') = ? AND deleted_at IS NULL",
            (sop_id,),
        )
        if not row:
            return None
        content = json.loads(row["content_json"])
        return SOP(
            id=sop_id,
            name=content.get("name", ""),
            trigger=content.get("trigger", ""),
            steps=content.get("steps", []),
            success_rate=content.get("success_rate", 1.0),
            reuse_count=content.get("reuse_count", 0),
            version=content.get("version", 1),
        )

    def _analyze_success_factors(self, exp: MuseExperience) -> list[str]:
        factors = []
        actions = exp.actions
        if len(actions) <= 3:
            factors.append("简洁高效的执行路径")
        if all(a.get("result", "").startswith("OK") or "success" in str(a.get("result", "")).lower()
               for a in actions if a.get("result")):
            factors.append("每个步骤都验证通过")
        if exp.tags:
            factors.append(f"相关经验标签: {', '.join(exp.tags[:3])}")
        return factors or ["任务成功完成"]

    def _analyze_failure_factors(self, exp: MuseExperience) -> list[str]:
        factors = []
        for i, action in enumerate(exp.actions):
            result = str(action.get("result", "")).lower()
            if "error" in result:
                factors.append(f"步骤 {i+1} 出错: {result[:80]}")
            elif "timeout" in result:
                factors.append(f"步骤 {i+1} 超时")
        if not factors:
            factors.append("未明确的失败原因")
        return factors

    def _root_cause_analysis(self, exp: MuseExperience) -> str:
        """简易根因分析 (5 Whys 方法)"""
        failure_point = self.muse._find_failure_point(exp) if hasattr(self.muse, '_find_failure_point') else None
        if failure_point:
            return f"Failure at step {failure_point.get('step', '?')}: {failure_point.get('error', 'unknown')}"
        return "Root cause not determined — need more data"

    def _generate_improvements(self, reflection: Reflection) -> list[str]:
        improvements = []
        if not reflection.success:
            improvements.append(f"Add validation before {reflection.root_cause[:60]}")
            improvements.append("Implement retry mechanism for failed steps")
            improvements.append("Add pre-condition checks")
        else:
            improvements.append("Generalize successful pattern into SOP")
            improvements.append("Increase confidence of related strategies")
        return improvements

    def _generate_validation(self, action: dict[str, Any]) -> str:
        action_name = action.get("action", "")
        if "fix" in action_name.lower():
            return "Verify fix passes tests"
        if "deploy" in action_name.lower():
            return "Verify deployment health check passes"
        if "build" in action_name.lower() or "create" in action_name.lower():
            return "Verify output meets specification"
        return "Verify step completed without errors"

    def _extract_failure_patterns(self, failed: list) -> list[dict]:
        patterns = []
        error_types = defaultdict(int)
        for e in failed:
            error = str(e.outcome.get("error", ""))[:80]
            if error:
                error_types[error] += 1
        return [
            {"error": e, "count": c, "frequency": c / len(failed) if failed else 0}
            for e, c in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    async def _cross_case_analysis(self, reflections: list[Reflection]) -> None:
        """跨案例根因分析"""
        failed = [r for r in reflections if not r.success]
        succeeded = [r for r in reflections if r.success]

        if failed:
            common_failures = defaultdict(int)
            for r in failed:
                for f in r.what_failed:
                    common_failures[f[:80]] += 1

            if common_failures:
                top_failure = max(common_failures, key=common_failures.get)
                logger.info("cross_case_pattern", top_failure=top_failure, count=common_failures[top_failure])
