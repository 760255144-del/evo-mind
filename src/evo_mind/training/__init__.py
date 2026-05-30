"""训练进化系统 — 多智能体互相训练的完整框架。

基于四大进化方法:
1. 经验驱动进化 — MUSE分层记忆 + SOP提炼 + Reasoning Memory
2. 学习进化 — SEAL双循环 + 自监督数据生成 + 在线自适应
3. 架构重构 — Writable Runtime + 代码自修改
4. 种群演化 — DARWIN交叉修改 + 基因重组 + 自然选择

使用:
    from evo_mind.training import TrainingOrchestrator
    orchestrator = TrainingOrchestrator(store)
    await orchestrator.train()
"""

from evo_mind.training.muse_memory import (
    MuseMemoryManager,
    MuseExperience,
    MuseLevel,
)
from evo_mind.training.experience_driven import (
    ExperienceDrivenEngine,
    SOP,
    Reflection,
    ReasoningMemoryEntry,
)
from evo_mind.training.learning_evolution import (
    LearningEvolutionEngine,
    EditInstruction,
    TrainingExample,
    AdaptationMetrics,
)
from evo_mind.training.population_evolution import (
    DarwinEngine,
    Individual,
    Population,
)
from evo_mind.training.orchestrator import (
    TrainingOrchestrator,
    TrainingMode,
    TrainingRound,
    TrainingState,
)

__all__ = [
    # MUSE
    "MuseMemoryManager",
    "MuseExperience",
    "MuseLevel",
    # Experience Driven
    "ExperienceDrivenEngine",
    "SOP",
    "Reflection",
    "ReasoningMemoryEntry",
    # Learning Evolution
    "LearningEvolutionEngine",
    "EditInstruction",
    "TrainingExample",
    "AdaptationMetrics",
    # Population Evolution
    "DarwinEngine",
    "Individual",
    "Population",
    # Orchestrator
    "TrainingOrchestrator",
    "TrainingMode",
    "TrainingRound",
    "TrainingState",
]
