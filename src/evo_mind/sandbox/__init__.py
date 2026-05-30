"""沙盒环境 — 开放世界探索 + 好奇心驱动 + 终生学习。

每天在虚拟环境中持续交互、学习、积累技能。
好奇心驱动未知探索，MUSE记忆存储经验。
"""

from evo_mind.sandbox.environment import (
    BaseEnvironment,
    PuzzleEnvironment,
    OptimizationEnvironment,
    PredictionEnvironment,
    ExplorationEnvironment,
    EnvironmentType,
    create_environment,
    State,
    Action,
    StepResult,
)
from evo_mind.sandbox.curiosity import (
    CuriosityEngine,
    CuriositySignal,
    LifelongLearner,
    Skill,
)
from evo_mind.sandbox.explorer import (
    SandboxExplorer,
    ExplorationEpisode,
    DailyExplorationResult,
)

__all__ = [
    "BaseEnvironment",
    "PuzzleEnvironment",
    "OptimizationEnvironment",
    "PredictionEnvironment",
    "ExplorationEnvironment",
    "EnvironmentType",
    "create_environment",
    "State",
    "Action",
    "StepResult",
    "CuriosityEngine",
    "CuriositySignal",
    "LifelongLearner",
    "Skill",
    "SandboxExplorer",
    "ExplorationEpisode",
    "DailyExplorationResult",
]
