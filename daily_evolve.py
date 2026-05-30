#!/usr/bin/env python3
"""每日自主进化引擎 — 每天自动启动、自定目标、自完成。

工作流:
  1. 加载系统状态 (MUSE 记忆、适应度、规则)
  2. 内省弱点 (哪些指标差？哪些模块没进步？)
  3. 生成今日 SMART 目标
  4. 根据目标选择进化策略
  5. 执行训练 (经验驱动 + 学习进化 + 种群进化)
  6. 评估目标完成度
  7. 记录进化日志
  8. 报告进展

使用:
  python daily_evolve.py               # 运行今日进化
  python daily_evolve.py --report      # 查看历史进化报告
  python daily_evolve.py --goal "..."   # 手动设定目标
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---- 目标模型 ----

class GoalDifficulty(StrEnum):
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    AMBITIOUS = "ambitious"


class GoalCategory(StrEnum):
    MEMORY = "memory"
    CODE = "code"
    LEARNING = "learning"
    POPULATION = "population"
    META = "meta"


@dataclass
class DailyGoal:
    """一个 SMART 的每日进化目标"""
    id: str = ""
    date: str = field(default_factory=_today)
    category: GoalCategory = GoalCategory.MEMORY
    title: str = ""
    description: str = ""

    # SMART 指标
    target_metric: str = ""          # 要改进的指标名
    current_value: float = 0.0       # 当前值
    target_value: float = 0.0        # 目标值
    achieved_value: float = 0.0      # 实际达成值

    # 状态
    difficulty: GoalDifficulty = GoalDifficulty.MODERATE
    completed: bool = False
    progress_pct: float = 0.0        # 完成百分比

    # 执行的进化动作
    actions_planned: list[str] = field(default_factory=list)
    actions_executed: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)

    # 时间
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class DailyEvolutionLog:
    """每日进化日志"""
    date: str = field(default_factory=_today)
    goal: DailyGoal | None = None
    system_state_before: dict[str, Any] = field(default_factory=dict)
    system_state_after: dict[str, Any] = field(default_factory=dict)
    actions_taken: list[str] = field(default_factory=list)
    sops_extracted: int = 0
    rules_learned: int = 0
    edits_applied: int = 0
    fitness_delta: float = 0.0
    duration_seconds: float = 0.0
    summary: str = ""
    errors: list[str] = field(default_factory=list)


# ---- 目标生成器 ----

class GoalGenerator:
    """根据系统内省自动生成 SMART 目标。

    规则:
    - 每个目标必须是 Specific, Measurable, Achievable, Relevant, Time-bound
    - 目标基于系统弱点和历史趋势
    - 难度递增: 前日成功 → 今日加难; 前日失败 → 今日降难
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or Path.home() / ".evo_mind" / "daily_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.yesterday_log: DailyEvolutionLog | None = None

    def load_history(self) -> list[dict]:
        """加载历史进化日志 (返回原始 dict 避免反序列化嵌套 dataclass)"""
        logs = []
        for f in sorted(self.log_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                logs.append(data)
            except Exception:
                pass
        return logs

    def get_yesterday(self) -> DailyEvolutionLog | None:
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"{yesterday}.json"
        if log_file.exists():
            try:
                data = json.loads(log_file.read_text())
                return DailyEvolutionLog(**data)
            except Exception:
                pass
        return None

    def generate_goal(
        self,
        system_state: dict[str, Any],
        manual_goal: str | None = None,
    ) -> DailyGoal:
        """生成今日进化目标"""
        if manual_goal:
            return self._parse_manual_goal(manual_goal, system_state)

        self.yesterday_log = self.get_yesterday()

        # 分析弱点 → 生成候选目标 → 选择最佳
        candidates = self._generate_candidates(system_state)
        best = self._select_best(candidates, system_state)

        return best

    def _generate_candidates(self, state: dict[str, Any]) -> list[DailyGoal]:
        """根据系统状态生成候选目标"""
        candidates: list[DailyGoal] = []

        # Memory 目标: 提升整合率
        total = state.get("total_memories", 0)
        consolidated = state.get("consolidated_memories", 0)
        cons_ratio = consolidated / max(total, 1)
        if cons_ratio < 0.6 and total > 20:
            candidates.append(DailyGoal(
                category=GoalCategory.MEMORY,
                title="提升记忆整合率",
                description=f"通过合并和概括，将 {total} 条记忆的整合率从 {cons_ratio:.0%} 提升到 {min(cons_ratio+0.15, 0.7):.0%}",
                target_metric="consolidation_ratio",
                current_value=cons_ratio,
                target_value=min(cons_ratio + 0.15, 0.7),
                actions_planned=["run_consolidation", "extract_sops", "deduplicate"],
                difficulty=GoalDifficulty.MODERATE if cons_ratio > 0.3 else GoalDifficulty.HARD,
            ))

        # 规则目标
        rules_active = state.get("evolution_rules_active", 0)
        avg_conf = state.get("avg_rule_confidence", 0)
        if rules_active < 5 or avg_conf < 0.5:
            candidates.append(DailyGoal(
                category=GoalCategory.LEARNING,
                title="发现新的进化规则",
                description=f"从记忆中提炼至少 3 条新规则，平均置信度从 {avg_conf:.0%} 提升到 {min(avg_conf+0.2, 0.8):.0%}",
                target_metric="evolution_rules_active",
                current_value=rules_active,
                target_value=max(rules_active + 3, 8),
                actions_planned=["run_evolution", "detect_patterns", "evaluate_rules"],
                difficulty=GoalDifficulty.HARD if rules_active < 3 else GoalDifficulty.MODERATE,
            ))

        # 代码进化目标
        code_fixes = state.get("code_fixes_applied", 0)
        if code_fixes < 10:
            candidates.append(DailyGoal(
                category=GoalCategory.CODE,
                title="代码自我优化",
                description=f"扫描代码库，发现并修复至少 2 个问题",
                target_metric="code_fixes_succeeded",
                current_value=state.get("code_fixes_succeeded", 0),
                target_value=state.get("code_fixes_succeeded", 0) + 2,
                actions_planned=["run_code_evolution", "run_static_analysis", "apply_fixes"],
                difficulty=GoalDifficulty.EASY,
            ))

        # 种群适应度目标
        fitness = state.get("genetic_fitness", state.get("avg_fitness", 0.5))
        if fitness < 0.7:
            candidates.append(DailyGoal(
                category=GoalCategory.POPULATION,
                title="提升种群适应度",
                description=f"通过遗传进化将种群平均适应度从 {fitness:.2f} 提升到 {min(fitness+0.1, 0.8):.2f}",
                target_metric="avg_fitness",
                current_value=fitness,
                target_value=min(fitness + 0.1, 0.8),
                actions_planned=["run_genetic_algorithm", "evaluate_population", "natural_selection"],
                difficulty=GoalDifficulty.MODERATE,
            ))

        # 元目标: 如果一切正常，尝试自修改
        if cons_ratio > 0.5 and rules_active > 3 and fitness > 0.5:
            candidates.append(DailyGoal(
                category=GoalCategory.META,
                title="自我架构优化",
                description="尝试一次安全的自我修改，改进系统的某个模块",
                target_metric="self_modifications",
                current_value=state.get("self_modifications", 0),
                target_value=state.get("self_modifications", 0) + 1,
                actions_planned=["analyze_self", "generate_improvements", "safe_apply"],
                difficulty=GoalDifficulty.AMBITIOUS,
            ))

        return candidates

    def _select_best(self, candidates: list[DailyGoal], state: dict) -> DailyGoal:
        """选择最值得做的目标 (最大化 impact/difficulty 比)"""
        if not candidates:
            return DailyGoal(
                category=GoalCategory.MEMORY,
                title="基础进化维护",
                description="运行标准进化管线，维护系统健康",
                target_metric="system_health",
                current_value=state.get("system_health", 0.5),
                target_value=min(state.get("system_health", 0.5) + 0.05, 0.95),
                actions_planned=["run_consolidation", "run_evolution"],
            )

        # 如果昨天失败了，降低难度
        if self.yesterday_log and not self.yesterday_log.goal:
            candidates.sort(key=lambda g: g.difficulty == GoalDifficulty.EASY, reverse=True)
        elif self.yesterday_log and self.yesterday_log.goal:
            yesterday_goal = self.yesterday_log.goal
            if yesterday_goal.completed:
                # 昨天成功 → 今天挑战更难
                candidates.sort(key=lambda g: (
                    g.difficulty == GoalDifficulty.AMBITIOUS,
                    g.difficulty == GoalDifficulty.HARD,
                ), reverse=True)

        # 默认: 选影响最大的
        def impact_score(g: DailyGoal) -> float:
            improvement = g.target_value - g.current_value
            base = abs(improvement) / max(abs(g.current_value), 0.01)
            difficulty_weight = {
                GoalDifficulty.EASY: 1.0,
                GoalDifficulty.MODERATE: 0.7,
                GoalDifficulty.HARD: 0.4,
                GoalDifficulty.AMBITIOUS: 0.2,
            }
            return base * difficulty_weight.get(g.difficulty, 0.5)

        candidates.sort(key=impact_score, reverse=True)
        return candidates[0]

    def _parse_manual_goal(self, text: str, state: dict) -> DailyGoal:
        """解析手动指定的目标"""
        return DailyGoal(
            title=text,
            description=text,
            category=GoalCategory.META,
            target_metric="manual",
            current_value=0.0,
            target_value=1.0,
            actions_planned=["manual_execution"],
            difficulty=GoalDifficulty.MODERATE,
        )


# ---- 每日进化引擎 ----

class DailyEvolutionEngine:
    """每日自主进化引擎。

    每天:
    1. 醒来 → 检查系统状态
    2. 反思 → 昨天完成了什么？哪里没做好？
    3. 设定目标 → 今天要提升什么？
    4. 执行 → 选择策略、运行进化
    5. 评估 → 目标达成率？
    6. 记录 → 写入进化日志
    7. 休眠 → 等待明天
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or Path.home() / ".evo_mind" / "daily_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.goal_gen = GoalGenerator(self.log_dir)

    async def evolve(self, manual_goal: str | None = None) -> DailyEvolutionLog:
        """执行今日进化"""
        import logging
        logging.basicConfig(level=logging.WARNING)  # Reduce noise
        logger = logging.getLogger("daily_evolve")

        log = DailyEvolutionLog(date=_today())
        start_time = time.monotonic()

        print(f"\n{'='*60}")
        print(f"  🧬 每日自主进化 — {_today()}")
        print(f"{'='*60}\n")

        try:
            # 1. 初始化系统
            from evo_mind.persistence.database import Database
            from evo_mind.persistence.memory_repo import MemoryRepo
            from evo_mind.core.store import MemoryStore
            from evo_mind.training.orchestrator import TrainingOrchestrator, TrainingMode

            # CI 环境用项目目录，本地用 ~/.evo_mind
            import os as _os
            if _os.environ.get("CI") or _os.environ.get("GITHUB_ACTIONS"):
                data_dir = Path(__file__).parent / "data"
            else:
                data_dir = Path.home() / ".evo_mind" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            db = Database(data_dir / "evo_mind.db", pool_size=3)
            await db.initialize()

            # 2. 采集系统状态
            state = await self._collect_state(db)
            log.system_state_before = state
            print(f"  📊 系统状态: {state['total_memories']} 记忆, "
                  f"{state.get('evolution_rules_active',0)} 规则, "
                  f"健康度={state.get('system_health',0):.2f}")

            # 3. 如果数据库太小，先播种
            if state["total_memories"] < 10:
                await self._seed_initial_data(db)

            # 4. 生成今日目标
            goal = self.goal_gen.generate_goal(state, manual_goal)
            log.goal = goal
            goal.started_at = _now()

            print(f"\n  🎯 今日目标: {goal.title}")
            print(f"     类别: {goal.category.value} | 难度: {goal.difficulty.value}")
            print(f"     指标: {goal.target_metric}: {goal.current_value:.3f} → {goal.target_value:.3f}")
            print(f"     计划: {', '.join(goal.actions_planned)}")

            # 5. 初始化真实后端 (带降级)
            repo = MemoryRepo(db)

            # 嵌入引擎: 检查本地缓存，网络不可用时直接降级
            embedding = None
            import os as _os
            _os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1"
            _os.environ["HF_HUB_ENABLE_DOWNLOAD_RESUME"] = "0"
            try:
                from pathlib import Path as _Path
                cache_dir = _Path.home() / ".cache" / "huggingface" / "hub"
                model_files = list(cache_dir.glob("models--sentence-transformers--all-MiniLM-L6-v2/**/*.safetensors"))
                if model_files:
                    from evo_mind.embedding.local import LocalEmbeddingEngine
                    embedding = LocalEmbeddingEngine(model_name="all-MiniLM-L6-v2", device="cpu", cache_size=5000)
                    await asyncio.wait_for(embedding.initialize(), timeout=2.0)
                    print(f"  🧠 嵌入引擎: all-MiniLM-L6-v2 ({embedding.dimension}维)")
            except Exception:
                embedding = None
            if embedding is None:
                print(f"  🧠 嵌入引擎: 降级模式 (无网络)")

            # 尝试真实向量存储
            vector_store = None
            try:
                from evo_mind.persistence.vector_store import ChromaVectorStore
                chroma_path = Path.home() / ".evo_mind" / "data" / "chroma"
                chroma_path.mkdir(parents=True, exist_ok=True)
                vector_store = ChromaVectorStore(chroma_path)
                await vector_store.initialize()
                count = await vector_store.count()
                print(f"  📦 向量存储: ChromaDB ({count} 向量)")
            except Exception as e:
                vector_store = None  # Reset so fallback triggers
                print(f"  ⚠️ 向量存储不可用 ({e})，使用降级模式")

            # 降级: 如果真实后端不可用，使用 Mock
            if embedding is None:
                class _FallbackEmbedding:
                    @property
                    def dimension(self): return 384
                    async def encode(self, t):
                        import random as _r
                        _r.seed(hash(str(t)) & 0xFFFFFFFF)
                        return [_r.uniform(-1, 1) for _ in range(384)]
                    async def encode_batch(self, texts, b=32):
                        return [await self.encode(t) for t in texts]
                embedding = _FallbackEmbedding()

            if vector_store is None:
                class _FallbackVectorStore:
                    async def add(self, **kw): pass
                    async def query(self, query_embedding, n_results=10, where=None): return ([], [])
                    async def delete(self, i): pass
                    async def count(self): return 0
                    async def get_embeddings(self, i): return None
                vector_store = _FallbackVectorStore()

            store = MemoryStore(db, vector_store, embedding, repo)

            # 6. 数学训练 — 每日必修
            print(f"\n  📐 数学训练...")
            from evo_mind.math.trainer import MathTrainer
            math_trainer = MathTrainer(store)
            math_result = await math_trainer.train(problems_per_domain=2)
            math_improvements = await math_trainer.apply_knowledge()
            print(f"  📐 数学: {math_result.problems_correct}/{math_result.problems_attempted} 正确, "
                  f"弱项={math_result.weakest_domain}, 强项={math_result.strongest_domain}")
            if math_improvements:
                for imp in math_improvements:
                    print(f"     → {imp}")

            # 7. 沙盒探索 — 开放世界持续学习
            print(f"\n  🌍 沙盒探索...")
            from evo_mind.sandbox.explorer import SandboxExplorer
            explorer = SandboxExplorer(store)
            exploration_result = await explorer.explore(episodes=6, max_steps_per_episode=50)
            explorer_status = explorer.get_status()
            print(f"  🌍 探索: {exploration_result.episodes_completed} episodes, "
                  f"{exploration_result.skills_total} skills, {exploration_result.total_steps} steps")

            # 8. 量化优化 — 策略参数进化
            print(f"\n  📈 量化优化...")
            try:
                from evo_mind.quant.optimizer import QuantOptimizer
                quant = QuantOptimizer(store, population_size=20, max_generations=10)
                best_params = await quant.evolve()
                if best_params:
                    print(f"  📈 量化: 策略参数已优化 (胜率阈值={best_params.get('min_confidence',0):.2f}, "
                          f"止损={best_params.get('stop_loss_pct',0):.1f}%)")
            except Exception as e:
                print(f"  ⚠️ 量化优化跳过: {e}")

            # 9. 研究追踪 — 三大方向每日扫描
            print(f"\n  🔬 研究追踪...")
            from evo_mind.research.tracker import ResearchTracker
            research = ResearchTracker(store)
            research_result = await research.track()
            research_goals = await research._generate_research_goals(research_result)
            progress = research.get_research_progress()
            print(f"  🔬 研究: {research_result.papers_reviewed} 篇论文, "
                  f"{research_result.insights_extracted} 个洞察, {research_result.insights_applied} 个应用")
            print(f"  📚 进度: {progress['progress']}")
            if research_goals:
                for g in research_goals[:2]:
                    print(f"     🎯 {g[:80]}")

            # 8. 执行进化动作
            orch = TrainingOrchestrator(
                store,
                mode=TrainingMode.ROUND_ROBIN,
                population_size=6,
                max_rounds=5,
                target_fitness=0.85,
            )

            print(f"\n  ⚡ 执行进化...")
            train_state = await orch.train()
            goal.actions_executed = [
                f"rounds={train_state.rounds_completed}",
                f"experiences={train_state.total_experiences}",
                f"sops={train_state.total_sops}",
                f"edits={train_state.total_edits}",
            ]
            goal.progress_pct = min(1.0, train_state.best_fitness_ever / goal.target_value) if goal.target_value > 0 else 0.5
            log.sops_extracted = train_state.total_sops
            log.edits_applied = train_state.total_edits
            log.actions_taken = goal.actions_executed

            # 9. 记忆整合 + 规则学习
            try:
                from evo_mind.consolidation.engine import ConsolidationEngine
                from evo_mind.types import ConsolidationTrigger
                cons = ConsolidationEngine(store, None, store.embedding, store.vector_store, store.db,
                                           config={"min_candidates": 3, "similarity_threshold": 0.7})
                cons_result = await cons.consolidate(trigger=ConsolidationTrigger.MANUAL)
                print(f"  📦 整合: {cons_result.groups_formed} 组, {cons_result.summaries_generated} 摘要, {cons_result.memories_pruned} 修剪")
            except Exception as e:
                print(f"  ⚠️ 整合跳过: {e}")

            try:
                from evo_mind.evolution.engine import EvolutionEngine
                evo = EvolutionEngine(store, store.db)
                rules = await evo.evolve()
                print(f"  🧠 规则: {len(rules)} 条新规则")
            except Exception as e:
                print(f"  ⚠️ 规则跳过: {e}")

            # 10. 采集进化后状态
            state_after = await self._collect_state(db)
            log.system_state_after = state_after
            log.fitness_delta = state_after.get("system_health", 0) - state.get("system_health", 0)

            # 11. 评估目标完成度
            goal.completed = log.fitness_delta > 0.005
            goal.completed_at = _now()
            log.duration_seconds = time.monotonic() - start_time

            # 12. 生成摘要
            log.summary = (
                f"[{_today()}] {'✅ 完成' if goal.completed else '⚠️ 部分完成'}: "
                f"{goal.title} | "
                f"适应度变化: {log.fitness_delta:+.3f} | "
                f"SOP提取: {log.sops_extracted} | "
                f"耗时: {log.duration_seconds:.0f}s"
            )

            print(f"\n  📈 结果: {log.summary}")
            print(f"     适应度: {state.get('system_health',0):.3f} → {state_after.get('system_health',0):.3f} ({log.fitness_delta:+.3f})")

            await db.close()

        except Exception as e:
            log.errors.append(str(e))
            log.summary = f"[{_today()}] ❌ 失败: {e}"
            print(f"\n  ❌ 进化出错: {e}")

        finally:
            # 12. 保存进化日志
            self._save_log(log)

        return log

    async def _collect_state(self, db) -> dict[str, Any]:
        """采集系统状态"""
        state: dict[str, Any] = {
            "total_memories": 0,
            "consolidated_memories": 0,
            "evolution_rules_active": 0,
            "avg_rule_confidence": 0.0,
            "code_fixes_applied": 0,
            "code_fixes_succeeded": 0,
            "agents_spawned": 0,
            "system_health": 0.5,
            "genetic_fitness": 0.0,
            "self_modifications": 0,
        }

        try:
            r = await db.fetch_one("SELECT COUNT(*) as c FROM memories WHERE deleted_at IS NULL")
            if r: state["total_memories"] = r["c"]

            r = await db.fetch_one("SELECT COUNT(*) as c FROM memories WHERE status='consolidated' AND deleted_at IS NULL")
            if r: state["consolidated_memories"] = r["c"]

            r = await db.fetch_one("SELECT COUNT(*) as c, AVG(confidence) as a FROM evolution_rules WHERE status='active'")
            if r:
                state["evolution_rules_active"] = r["c"] or 0
                state["avg_rule_confidence"] = r["a"] or 0.0

            total = state["total_memories"]
            cons = state["consolidated_memories"]
            if total > 0:
                cons_ratio = cons / total
                rules_health = state["avg_rule_confidence"]
                state["system_health"] = (cons_ratio * 0.5 + rules_health * 0.5)
        except Exception:
            pass

        return state

    async def _seed_initial_data(self, db) -> None:
        """播种初始数据，让系统有东西可以进化"""
        from evo_mind.persistence.memory_repo import MemoryRepo
        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType
        import random

        repo = MemoryRepo(db)
        tasks = [
            "Fix bug in memory retrieval when query is empty",
            "Optimize consolidation clustering for large datasets",
            "Implement error handling in evolution engine",
            "Add logging to vector store operations",
            "Refactor the deduplication logic for clarity",
            "Improve embedding cache hit rate",
            "Debug race condition in async write operations",
            "Add validation to genome mutation ranges",
            "Enhance SOP extraction from failed experiences",
            "Optimize reciprocal rank fusion weights",
        ]

        for i, task in enumerate(tasks):
            success = random.random() > 0.3
            await repo.create(MemoryCreate(
                memory_type=MemoryType.EPISODIC if i % 2 == 0 else MemoryType.PROCEDURAL,
                content={
                    "text": task,
                    "description": f"Daily seed task {i}: {task}",
                    "action": random.choice(["fix", "optimize", "refactor", "implement"]),
                },
                importance=0.3 + random.random() * 0.5,
                tags=[random.choice(["bug", "optimization", "feature", "refactor"])],
            ))

    def _save_log(self, log: DailyEvolutionLog) -> None:
        """保存进化日志到文件"""
        log_file = self.log_dir / f"{log.date}.json"
        data = {
            "date": log.date,
            "goal": None,
            "system_state_before": log.system_state_before,
            "system_state_after": log.system_state_after,
            "actions_taken": log.actions_taken,
            "sops_extracted": log.sops_extracted,
            "rules_learned": log.rules_learned,
            "edits_applied": log.edits_applied,
            "fitness_delta": log.fitness_delta,
            "duration_seconds": log.duration_seconds,
            "summary": log.summary,
            "errors": log.errors,
        }
        if log.goal:
            data["goal"] = {
                "title": log.goal.title,
                "category": log.goal.category.value,
                "target_metric": log.goal.target_metric,
                "current_value": log.goal.current_value,
                "target_value": log.goal.target_value,
                "completed": log.goal.completed,
                "progress_pct": log.goal.progress_pct,
                "difficulty": log.goal.difficulty.value,
                "actions_executed": log.goal.actions_executed,
            }

        log_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def get_report(self, days: int = 7) -> str:
        """生成最近 N 天的进化报告"""
        logs = self.goal_gen.load_history()
        recent = logs[-days:]

        if not recent:
            return "暂无进化记录"

        lines = [
            f"\n{'='*60}",
            f"  📊 进化报告 (最近 {len(recent)} 天)",
            f"{'='*60}",
        ]

        # Work with dicts from JSON
        fitness_deltas = [l.get("fitness_delta", 0) for l in recent]
        avg_fitness_delta = sum(fitness_deltas) / len(fitness_deltas)
        total_sops = sum(l.get("sops_extracted", 0) for l in recent)
        total_edits = sum(l.get("edits_applied", 0) for l in recent)
        completed_days = sum(1 for l in recent if l.get("goal", {}).get("completed", False))

        lines.append(f"  ✅ 目标达成率: {completed_days}/{len(recent)} ({completed_days/len(recent):.0%})")
        lines.append(f"  📈 平均适应度变化: {avg_fitness_delta:+.4f}/天")
        lines.append(f"  📋 累计SOP提取: {total_sops}")
        lines.append(f"  🔧 累计代码编辑: {total_edits}")
        lines.append("")

        for log in recent:
            goal = log.get("goal", {})
            icon = "✅" if goal.get("completed") else "⚠️"
            lines.append(f"  {icon} {log.get('date', '?')}: {log.get('summary', '')}")

        lines.append("")
        return "\n".join(lines)


# ---- CLI ----

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="每日自主进化引擎")
    parser.add_argument("--goal", "-g", type=str, help="手动设定今日目标")
    parser.add_argument("--report", "-r", action="store_true", help="查看进化报告")
    parser.add_argument("--days", "-d", type=int, default=7, help="报告天数")
    args = parser.parse_args()

    engine = DailyEvolutionEngine()

    if args.report:
        print(engine.get_report(days=args.days))
    else:
        log = await engine.evolve(manual_goal=args.goal)
        print(f"\n  💾 日志已保存: {engine.log_dir / log.date}.json")


if __name__ == "__main__":
    asyncio.run(main())
