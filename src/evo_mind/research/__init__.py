"""研究方向追踪 — 元学习 + 世界模型 + 自动对齐。

每天追踪 arXiv 论文和 GitHub 开源项目，
提取可执行洞察，驱动进化目标更新。
"""

from evo_mind.research.knowledge_base import (
    Paper,
    Project,
    ResearchInsight,
    META_LEARNING_PAPERS,
    WORLD_MODELS_PAPERS,
    AUTO_ALIGNMENT_PAPERS,
    OPEN_SOURCE_PROJECTS,
    get_all_papers,
)
from evo_mind.research.tracker import ResearchTracker, DailyResearchResult

__all__ = [
    "Paper",
    "Project",
    "ResearchInsight",
    "META_LEARNING_PAPERS",
    "WORLD_MODELS_PAPERS",
    "AUTO_ALIGNMENT_PAPERS",
    "OPEN_SOURCE_PROJECTS",
    "get_all_papers",
    "ResearchTracker",
    "DailyResearchResult",
]
