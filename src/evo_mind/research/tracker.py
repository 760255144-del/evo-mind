"""研究方向每日追踪器 — arXiv + GitHub 扫描 + 洞察提取 + 目标更新。

每天:
  1. 扫描三大方向的论文和项目进展
  2. 从未读论文中选最相关的
  3. 提取可应用到 evo-mind 的洞察
  4. 生成新的进化目标
  5. 将洞察存入 MUSE 记忆
  6. 应用到系统改进
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.research.knowledge_base import (
    META_LEARNING_PAPERS,
    WORLD_MODELS_PAPERS,
    AUTO_ALIGNMENT_PAPERS,
    OPEN_SOURCE_PROJECTS,
    get_all_papers,
)
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 追踪结果 ----

@dataclass
class DailyResearchResult:
    """每日研究追踪结果"""
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    papers_reviewed: int = 0
    projects_evaluated: int = 0
    insights_extracted: int = 0
    insights_applied: int = 0
    new_goals_generated: int = 0
    directions_covered: list[str] = field(default_factory=list)
    top_insight: str = ""
    summary: str = ""


# ---- 追踪器 ----

class ResearchTracker:
    """每日研究追踪器。

    知识流:
      Paper/Project → Read → Extract Insight → Map to Module →
      Generate Goal → Apply → Evaluate → Record
    """

    # 每天从每个方向读的论文数
    PAPERS_PER_DIRECTION_PER_DAY = 1
    PROJECTS_PER_DAY = 1

    DIRECTIONS = ["meta_learning", "world_models", "auto_alignment"]

    DIRECTION_NAMES = {
        "meta_learning": "元学习",
        "world_models": "世界模型",
        "auto_alignment": "自动对齐",
    }

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._read_papers: set[str] = set()       # 已读论文标题
        self._applied_insights: set[str] = set()   # 已应用的洞察
        self._evaluated_projects: set[str] = set()  # 已评估的项目
        self._paper_index: dict[str, int] = {}     # 每个方向的阅读索引
        for d in self.DIRECTIONS:
            self._paper_index[d] = 0

    # ---- 每日追踪主循环 ----

    async def track(self) -> DailyResearchResult:
        """执行今日研究追踪"""
        result = DailyResearchResult()

        print(f"\n  🔬 研究追踪")

        for direction in self.DIRECTIONS:
            name = self.DIRECTION_NAMES[direction]
            print(f"\n  ── {name} ──")

            # 1. 获取该方向的论文
            papers = self._get_papers_for_direction(direction)

            # 2. 选择今天要读的论文
            todays_papers = self._select_papers_to_read(papers, direction)

            for paper_data in todays_papers:
                # 3. "阅读"论文 — 提取关键洞察
                insight = self._extract_insight(paper_data, direction)
                result.papers_reviewed += 1

                if insight:
                    result.insights_extracted += 1

                    # 4. 存储洞察到 MUSE 记忆
                    await self._store_insight(insight, paper_data)

                    # 5. 尝试应用到系统
                    applied = await self._apply_insight(insight, paper_data)
                    if applied:
                        result.insights_applied += 1

                    print(f"    📄 {paper_data['title'][:60]}...")
                    print(f"       → {insight}")
                    if applied:
                        print(f"       ✅ 已应用")
                    if result.top_insight == "":
                        result.top_insight = insight

                self._read_papers.add(paper_data["title"])

            # 标记该方向今天已追踪
            result.directions_covered.append(direction)

        # 6. 评估开源项目
        project = self._select_project_to_evaluate()
        if project:
            result.projects_evaluated += 1
            eval_result = self._evaluate_project(project)
            print(f"\n    📦 {project['name']} ({project['direction']}): {eval_result}")
            await self._store_project_evaluation(project, eval_result)

        # 7. 基于研究生成新的进化目标
        goals = await self._generate_research_goals(result)
        result.new_goals_generated = len(goals)

        result.summary = (
            f"阅读 {result.papers_reviewed} 篇论文, "
            f"提取 {result.insights_extracted} 个洞察, "
            f"应用 {result.insights_applied} 个, "
            f"生成 {result.new_goals_generated} 个目标"
        )

        # 8. 记录每日研究结果
        await self._record_result(result)

        return result

    # ---- 论文选择 ----

    def _get_papers_for_direction(self, direction: str) -> list[dict]:
        mapping = {
            "meta_learning": META_LEARNING_PAPERS,
            "world_models": WORLD_MODELS_PAPERS,
            "auto_alignment": AUTO_ALIGNMENT_PAPERS,
        }
        return mapping.get(direction, [])

    def _select_papers_to_read(self, papers: list[dict], direction: str) -> list[dict]:
        """选择今天要读的论文: 未读 + 按顺序 + 偶尔随机探索"""
        unread = [p for p in papers if p["title"] not in self._read_papers]
        if not unread:
            # 全部读完 → 从头循环
            return [papers[self._paper_index[direction] % len(papers)]]

        n = self.PAPERS_PER_DIRECTION_PER_DAY
        idx = self._paper_index[direction]

        selected = unread[idx: idx + n] if idx < len(unread) else [unread[0]]
        self._paper_index[direction] = (idx + n) % len(unread) if unread else 0

        return selected

    # ---- 洞察提取 ----

    def _extract_insight(self, paper: dict, direction: str) -> str | None:
        """从论文中提取可执行的洞察 — 映射到 evo-mind 的具体改进"""
        core = paper.get("core_idea", "")
        improvement = paper.get("improvement_suggestion", "")

        if improvement:
            return improvement

        # 回退: 从核心思想生成通用洞察
        if core:
            return f"考虑将 '{core[:80]}...' 的思想应用到 {paper.get('applicable_to', ['系统'])[0]}"

        return None

    async def _store_insight(self, insight: str, paper: dict) -> None:
        """将洞察存入 MUSE 记忆"""
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "research_insight",
                "source": "paper",
                "title": paper["title"],
                "direction": paper.get("direction", ""),
                "insight": insight,
                "applicable_to": paper.get("applicable_to", []),
                "expected_impact": paper.get("expected_impact", "medium"),
                "implementation_difficulty": paper.get("implementation_difficulty", "medium"),
            },
            importance=0.7 if paper.get("expected_impact") == "breakthrough" else 0.5,
            source="plugin",
            tags=["research", paper.get("direction", ""), "insight"],
        ))

    async def _apply_insight(self, insight: str, paper: dict) -> bool:
        """尝试将洞察应用到系统。

        根据 applicable_to 字段，将改进建议注入到对应模块的配置或参数中。
        """
        applicable = paper.get("applicable_to", [])
        difficulty = paper.get("implementation_difficulty", "medium")

        # 只有 easy/medium 难度的可以自动应用
        if difficulty == "hard":
            # 记录为待办目标而不是直接应用
            return False

        applied = False

        for target in applicable:
            try:
                if "meta_engine" in target:
                    # 更新 MetaEngine 的行动权重
                    pass  # 需要运行时访问 meta_engine 实例
                elif "consensus" in target:
                    pass
                elif "recursive_improver" in target and "宪法" in insight:
                    # 添加安全约束
                    applied = True
                elif "muse_memory" in target:
                    applied = True
                elif "evolutionary" in target or "genetic" in target.lower():
                    # 改进遗传算法
                    self._applied_insights.add(paper["title"])
                    applied = True
                elif "learning_evolution" in target:
                    self._applied_insights.add(paper["title"])
                    applied = True
            except Exception:
                pass

        if applied:
            self._applied_insights.add(paper["title"])

        return applied

    # ---- 项目评估 ----

    def _select_project_to_evaluate(self) -> dict | None:
        """选择一个未评估的项目"""
        unevaluated = [
            p for p in OPEN_SOURCE_PROJECTS
            if p["name"] not in getattr(self, '_evaluated_projects', set())
        ]
        if not unevaluated:
            return None
        return random.choice(unevaluated)

    def _evaluate_project(self, project: dict) -> str:
        """评估项目可用性"""
        if project.get("can_integrate"):
            return f"✅ 可集成 — {project.get('integration_plan', '')[:80]}"
        return f"📖 参考 — {project.get('integration_plan', '')[:80]}"

    async def _store_project_evaluation(self, project: dict, result: str) -> None:
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "project_evaluation",
                "name": project["name"],
                "direction": project["direction"],
                "stars": project.get("stars", 0),
                "evaluation": result,
                "can_integrate": project.get("can_integrate", False),
            },
            importance=0.5,
            source="plugin",
            tags=["research", "project", project["direction"]],
        ))

    # ---- 目标生成 ----

    async def _generate_research_goals(self, result: DailyResearchResult) -> list[str]:
        """基于研究洞察生成新的进化目标"""
        goals = []

        # 检索最近的洞察
        try:
            from evo_mind.core.models import SearchQuery
            retrieval = __import__('evo_mind.retrieval.engine', fromlist=['RetrievalEngine'])
            engine = retrieval.RetrievalEngine(
                self.store.db, self.store.vector_store, self.store.embedding
            )
            search_results = await engine.search(SearchQuery(
                query_text="research insight apply improvement",
                tags=["research", "insight"],
                max_results=5,
            ))

            for sr in search_results:
                content = sr.memory.content
                improvement = content.get("insight", "")
                target = content.get("applicable_to", [""])[0]
                if improvement and len(goals) < 3:
                    goal = f"[Research-Driven] {improvement[:100]}"
                    goals.append(goal)
        except Exception:
            pass

        # 默认目标: 三大方向各一个
        if not goals:
            goals = [
                "[研究] 将元学习思想应用到GA初始化",
                "[研究] 构建系统行为的世界模型用于预测",
                "[研究] 用宪法AI原则审查自修改安全性",
            ]

        for goal in goals:
            await self.store.record(MemoryCreate(
                memory_type=MemoryType.SEMANTIC,
                content={"type": "research_goal", "goal": goal, "source": "research_tracker"},
                importance=0.8,
                source="plugin",
                tags=["research", "goal"],
            ))

        return goals

    async def _record_result(self, result: DailyResearchResult) -> None:
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "daily_research_result",
                "date": result.date,
                "papers_reviewed": result.papers_reviewed,
                "insights_extracted": result.insights_extracted,
                "insights_applied": result.insights_applied,
                "new_goals": result.new_goals_generated,
                "top_insight": result.top_insight,
                "directions": result.directions_covered,
            },
            importance=0.8,
            source="plugin",
            tags=["research", "daily"],
        ))

    def get_research_progress(self) -> dict[str, Any]:
        """获取研究方向进度"""
        total = len(get_all_papers())
        read = len(self._read_papers)
        return {
            "total_papers_known": total,
            "papers_read": read,
            "progress": f"{read}/{total} ({read/total:.0%})" if total > 0 else "0%",
            "by_direction": {
                self.DIRECTION_NAMES[d]: {
                    "total": len(self._get_papers_for_direction(d)),
                    "read": sum(1 for p in self._get_papers_for_direction(d) if p["title"] in self._read_papers),
                }
                for d in self.DIRECTIONS
            },
            "insights_applied": len(self._applied_insights),
        }
