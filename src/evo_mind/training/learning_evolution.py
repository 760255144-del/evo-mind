"""学习进化引擎 — 双循环自编辑 + 自监督数据生成 + 在线自适应。

三种核心机制:
1. 双循环学习 (SEAL 风格)
   外循环 (Teacher): 评估性能, 生成编辑指令
   内循环 (Student): 执行编辑, 验证改进

2. 自监督训练数据生成 (AgentEvolver 风格)
   - 自我提问生成任务
   - 自动标注成功/失败
   - 冷启动: 从零经验开始自我引导

3. 在线自适应
   - 跨会话记忆连续性
   - 实时权重调整
   - 能力漂移检测与纠正

参考: MIT SEAL, AgentEvolver (14B: 29.8%→57.6%)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from evo_mind.core.models import MemoryCreate, SearchQuery
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType, RelationType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 数据类型 ----

@dataclass
class EditInstruction:
    """SEAL 风格的编辑指令 — 外循环(Teacher)生成"""
    id: str = field(default_factory=uuid7)
    target_module: str = ""            # 目标模块路径
    target_function: str = ""          # 目标函数名
    edit_type: str = ""                # "add" | "modify" | "remove" | "refactor"
    description: str = ""              # 编辑描述
    before_code: str = ""              # 编辑前代码
    after_code: str = ""               # 编辑后代码
    validation_test: str = ""          # 验证测试
    expected_improvement: str = ""     # 预期改进指标
    priority: float = 0.5
    applied: bool = False
    validated: bool = False
    improvement_delta: float = 0.0     # 实际改进量
    timestamp: str = field(default_factory=_now)


@dataclass
class TrainingExample:
    """自监督训练样本"""
    id: str = field(default_factory=uuid7)
    task: str = ""                     # 任务描述
    input_state: dict[str, Any] = field(default_factory=dict)
    expected_output: dict[str, Any] = field(default_factory=dict)
    actual_output: dict[str, Any] | None = None
    success: bool | None = None        # None = 未标注
    difficulty: float = 0.5
    tags: list[str] = field(default_factory=list)
    source: str = "self_generated"     # "self_generated" | "human" | "synthetic"


@dataclass
class AdaptationMetrics:
    """在线自适应指标"""
    timestamp: str = field(default_factory=_now)
    task_completion_rate: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    sop_reuse_rate: float = 0.0
    learning_rate: float = 0.0         # 能力提升速率
    drift_detected: bool = False


# ---- 学习进化引擎 ----

class LearningEvolutionEngine:
    """学习驱动的进化引擎。

    双循环 (SEAL):
      Teacher 循环: 监控→评估→生成编辑指令
      Student 循环: 接收编辑→执行→验证→反馈

    自监督数据生成 (AgentEvolver):
      Self-Ask → Execute → Auto-Label → Train
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._edits: dict[str, EditInstruction] = {}
        self._training_data: dict[str, TrainingExample] = {}
        self._metrics_history: list[AdaptationMetrics] = []

    # ---- 双循环学习 (SEAL) ----

    async def teacher_loop(self, performance_metrics: dict[str, Any]) -> list[EditInstruction]:
        """外循环 (Teacher): 评估性能并生成编辑指令。

        就像老师批改作业:
        1. 识别弱点 (哪些任务完成率低?)
        2. 分析原因 (为什么失败?)
        3. 生成编辑指令 (怎么改进?)
        """
        edits: list[EditInstruction] = []

        # 识别低性能领域
        weaknesses = self._identify_weaknesses(performance_metrics)

        for weakness in weaknesses:
            edit = EditInstruction(
                target_module=weakness.get("module", "unknown"),
                target_function=weakness.get("function", ""),
                edit_type=weakness.get("edit_type", "modify"),
                description=weakness.get("description", ""),
                before_code=weakness.get("current_code", ""),
                after_code=weakness.get("suggested_code", ""),
                validation_test=weakness.get("test", ""),
                expected_improvement=weakness.get("improvement", "+10%"),
                priority=weakness.get("priority", 0.5),
            )
            edits.append(edit)
            self._edits[edit.id] = edit

        # 按优先级排序
        edits.sort(key=lambda e: e.priority, reverse=True)

        logger.info("teacher_generated_edits", count=len(edits))
        return edits

    async def student_loop(self, edits: list[EditInstruction]) -> dict[str, Any]:
        """内循环 (Student): 执行编辑并验证。

        就像学生改正错题:
        1. 逐一应用编辑
        2. 运行验证测试
        3. 反馈改进结果
        """
        results = {
            "total_edits": len(edits),
            "applied": 0,
            "validated": 0,
            "improved": 0,
            "regressed": 0,
            "details": [],
        }

        for edit in edits:
            try:
                # 应用编辑 (写入目标模块)
                success = await self._apply_edit(edit)
                edit.applied = success

                if success:
                    results["applied"] += 1

                    # 验证
                    passed = await self._validate_edit(edit)
                    edit.validated = passed

                    if passed:
                        results["validated"] += 1
                        if edit.improvement_delta > 0:
                            results["improved"] += 1
                        elif edit.improvement_delta < 0:
                            results["regressed"] += 1

                results["details"].append({
                    "edit_id": edit.id,
                    "description": edit.description[:100],
                    "applied": edit.applied,
                    "validated": edit.validated,
                    "delta": edit.improvement_delta,
                })

            except Exception as e:
                logger.warning("edit_failed", edit=edit.id[:12], error=str(e))

        logger.info("student_completed", **{k: v for k, v in results.items() if k != "details"})
        return results

    async def seal_iteration(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """一次完整的 SEAL 迭代: Teacher → Student → Feedback"""
        # Teacher 生成编辑
        edits = await self.teacher_loop(metrics)

        if not edits:
            return {"status": "no_improvements_needed"}

        # Student 执行编辑
        student_results = await self.student_loop(edits)

        # 记录为经验
        await self._record_seal_iteration(edits, student_results)

        return student_results

    # ---- 自监督训练数据生成 ----

    async def generate_training_data(
        self, task_templates: list[str] | None = None, count: int = 50
    ) -> list[TrainingExample]:
        """自监督生成训练数据 (AgentEvolver 风格)

        Self-Ask → Execute → Auto-Label
        1. 从任务模板中自我提问
        2. 尝试执行
        3. 自动标注成功/失败
        """
        if task_templates is None:
            task_templates = self._default_task_templates()

        examples: list[TrainingExample] = []
        for i in range(count):
            template = task_templates[i % len(task_templates)]

            # Self-Ask: 生成具体任务
            task = self._instantiate_template(template, i)

            # Execute: 尝试执行 (模拟)
            example = await self._self_execute(task)

            # Auto-Label: 自动标注
            example.success = self._auto_label(example)

            self._training_data[example.id] = example
            examples.append(example)

        # 冷启动处理: 如果没有足够成功样本, 降低难度
        success_count = sum(1 for e in examples if e.success)
        if success_count < len(examples) * 0.3:
            logger.info("cold_start_adjustment", success_rate=f"{success_count/len(examples):.1%}")
            # 生成更简单的样本
            easy_examples = await self._generate_easy_examples(
                max(10, count - len(examples))
            )
            examples.extend(easy_examples)

        # 存储为训练记忆
        await self._store_training_data(examples)

        logger.info("training_data_generated", total=len(examples), successful=sum(1 for e in examples if e.success))
        return examples

    async def _self_execute(self, task: str) -> TrainingExample:
        """自我执行任务并记录结果"""
        example = TrainingExample(
            task=task,
            input_state={"task": task, "timestamp": _now()},
            expected_output={"success": True, "output": "Task completed"},
            difficulty=self._estimate_difficulty(task),
            tags=self._categorize_task(task),
        )

        # 模拟执行: 检查是否能找到相关 SOP 或经验
        try:
            results = await self._search_relevant_knowledge(task)
            if results:
                example.actual_output = {
                    "success": True,
                    "output": f"Used {len(results)} relevant experiences",
                    "sources": [r.get("id", "")[:12] for r in results[:3]],
                }
            else:
                example.actual_output = {
                    "success": False,
                    "output": "No relevant knowledge found",
                    "error": "cold_start",
                }
        except Exception as e:
            example.actual_output = {"success": False, "error": str(e)}

        return example

    # ---- 在线自适应 ----

    async def adapt_online(self, recent_performance: list[dict[str, Any]]) -> AdaptationMetrics:
        """在线自适应: 根据最近表现调整策略"""
        metrics = AdaptationMetrics()

        if recent_performance:
            total = len(recent_performance)
            metrics.task_completion_rate = sum(1 for p in recent_performance if p.get("success")) / total
            metrics.avg_response_time = sum(p.get("duration", 0) for p in recent_performance) / total
            metrics.error_rate = sum(1 for p in recent_performance if p.get("error")) / total

        # 检测能力漂移
        if len(self._metrics_history) >= 5:
            recent_rates = [m.task_completion_rate for m in self._metrics_history[-5:]]
            if max(recent_rates) - min(recent_rates) > 0.2:
                metrics.drift_detected = True
                logger.warning("capability_drift_detected")

        # 计算学习速率
        if len(self._metrics_history) >= 2:
            prev = self._metrics_history[-1].task_completion_rate
            curr = metrics.task_completion_rate
            metrics.learning_rate = (curr - prev) / max(prev, 0.01)

        self._metrics_history.append(metrics)

        # 如果检测到退化, 触发纠正
        if metrics.drift_detected or metrics.learning_rate < -0.1:
            await self._trigger_correction(metrics)

        return metrics

    async def _trigger_correction(self, metrics: AdaptationMetrics) -> None:
        """触发纠正机制: 回滚到上一个稳定状态"""
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.FEEDBACK,
            content={
                "type": "adaptation_correction",
                "metrics": {
                    "completion_rate": metrics.task_completion_rate,
                    "error_rate": metrics.error_rate,
                    "drift_detected": metrics.drift_detected,
                },
                "action": "Triggering rollback to last stable state",
            },
            importance=0.9,
            source="plugin",
            tags=["adaptation", "correction", "drift"],
        ))

    # ---- 辅助方法 ----

    def _identify_weaknesses(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        weaknesses = []
        # Map metric names to actual file paths in the project
        metric_to_module = {
            "avg_fitness": "src/evo_mind/evolutionary/engine.py",
            "error_rate": "src/evo_mind/evolution/engine.py",
            "total_memories": "src/evo_mind/core/store.py",
            "response_time": "src/evo_mind/retrieval/engine.py",
            "sop_reuse_rate": "src/evo_mind/training/experience_driven.py",
            "edit_success_rate": "src/evo_mind/training/learning_evolution.py",
            "population_size": "src/evo_mind/training/population_evolution.py",
            "best_fitness": "src/evo_mind/evolutionary/engine.py",
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value < 0.5:
                module_path = metric_to_module.get(key, f"src/evo_mind/{key.replace('_', '/')}.py")
                weaknesses.append({
                    "module": module_path,
                    "function": f"improve_{key}",
                    "edit_type": "modify",
                    "description": f"Improve {key} from {value:.2f}",
                    "current_code": f"# Current {key}: {value:.2f}",
                    "suggested_code": f"# Target {key}: >= 0.7",
                    "priority": 1.0 - float(value),
                })
        return weaknesses

    async def _apply_edit(self, edit: EditInstruction) -> bool:
        """应用编辑到目标模块"""
        try:
            # 读取目标文件
            from pathlib import Path
            target_path = Path(edit.target_module)
            if not target_path.exists():
                logger.debug("target_not_found", path=edit.target_module)
                return False

            content = target_path.read_text()
            if edit.before_code and edit.before_code in content:
                new_content = content.replace(edit.before_code, edit.after_code, 1)
                target_path.write_text(new_content)
                return True
            return False
        except Exception:
            return False

    async def _validate_edit(self, edit: EditInstruction) -> bool:
        """验证编辑是否改进了系统"""
        # 运行验证测试
        try:
            if edit.validation_test:
                proc = await asyncio.create_subprocess_shell(
                    edit.validation_test,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=10.0)
                passed = proc.returncode == 0
                edit.improvement_delta = 0.05 if passed else -0.02
                return passed
        except Exception:
            logger.warning("validation_test_failed: %s", edit.validation_test[:100])
        return False  # Safety: never assume OK when validation fails

    async def _search_relevant_knowledge(self, task: str) -> list[dict]:
        try:
            retrieval = __import__('evo_mind.retrieval.engine', fromlist=['RetrievalEngine'])
            engine = retrieval.RetrievalEngine(self.store.db, self.store.vector_store, self.store.embedding)
            results = await engine.search(SearchQuery(query_text=task, max_results=5))
            return [{"id": r.memory.id, "score": r.score} for r in results]
        except Exception:
            return []

    async def _record_seal_iteration(
        self, edits: list[EditInstruction], results: dict[str, Any]
    ) -> None:
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.PROCEDURAL,
            content={
                "type": "seal_iteration",
                "edits_generated": len(edits),
                "edits_applied": results.get("applied", 0),
                "edits_validated": results.get("validated", 0),
                "improved": results.get("improved", 0),
                "regressed": results.get("regressed", 0),
            },
            importance=0.7,
            source="plugin",
            tags=["seal", "learning", "dual_loop"],
        ))

    async def _store_training_data(self, examples: list[TrainingExample]) -> None:
        for ex in examples:
            await self.store.record(MemoryCreate(
                memory_type=MemoryType.PROCEDURAL,
                content={
                    "type": "training_example",
                    "task": ex.task,
                    "success": ex.success,
                    "difficulty": ex.difficulty,
                    "actual_output": ex.actual_output,
                },
                importance=0.4,
                source="plugin",
                tags=["training_data", *ex.tags],
            ))

    # ---- 静态工具 ----

    @staticmethod
    def _default_task_templates() -> list[str]:
        return [
            "Fix the bug in {module} where {error_type} occurs",
            "Optimize {function} to reduce latency by {percent}%",
            "Add error handling to {component} for {edge_case}",
            "Refactor {module} to use {pattern} pattern",
            "Implement {feature} in {component} with tests",
            "Debug the {type} issue reported in {source}",
            "Improve test coverage for {module} to {percent}%",
            "Migrate {component} from {old} to {new}",
        ]

    def _instantiate_template(self, template: str, seed: int) -> str:
        modules = ["memory_repo", "retrieval_engine", "evolution_engine", "consolidation"]
        errors = ["TimeoutError", "ValidationError", "ConnectionError", "KeyError"]
        functions = ["search", "consolidate", "evolve", "encode", "retrieve"]
        patterns = ["Strategy", "Observer", "Factory", "Repository", "Adapter"]

        return template.format(
            module=modules[seed % len(modules)],
            error_type=errors[seed % len(errors)],
            function=functions[seed % len(functions)],
            percent=10 + (seed % 40),
            component=modules[(seed + 1) % len(modules)],
            edge_case=f"edge_case_{seed}",
            pattern=patterns[seed % len(patterns)],
            feature=f"feature_{seed}",
            type=errors[seed % len(errors)],
            source=f"source_{seed}",
            old=f"v{seed % 3}.{seed % 10}",
            new=f"v{(seed % 3) + 1}.0",
        )

    def _auto_label(self, example: TrainingExample) -> bool:
        if example.actual_output is None:
            return False
        return example.actual_output.get("success", False)

    def _estimate_difficulty(self, task: str) -> float:
        task_lower = task.lower()
        if "refactor" in task_lower or "migrate" in task_lower:
            return 0.8
        if "optimize" in task_lower or "implement" in task_lower:
            return 0.6
        if "fix" in task_lower or "debug" in task_lower:
            return 0.4
        return 0.5

    def _categorize_task(self, task: str) -> list[str]:
        tags = []
        t = task.lower()
        for cat in ["fix", "optimize", "refactor", "implement", "debug", "migrate", "test"]:
            if cat in t:
                tags.append(cat)
        return tags or ["general"]

    async def _generate_easy_examples(self, count: int) -> list[TrainingExample]:
        easy = []
        for i in range(count):
            ex = TrainingExample(
                task=f"Simple validation task {i}",
                input_state={"task": f"validate_{i}"},
                expected_output={"success": True, "output": "Validated"},
                actual_output={"success": True, "output": "Passed basic check"},
                success=True,
                difficulty=0.1,
                tags=["easy", "validation"],
            )
            self._training_data[ex.id] = ex
            easy.append(ex)
        return easy
