"""数学引擎 — 每日学习数学，记忆知识，应用到系统优化。

四大领域: 微积分 | 线性代数 | 概率论 | 最优化
闭环: 出题 → 解题 → 纠错 → 记忆 → 应用
"""

from evo_mind.math.engine import (
    CalculusEngine,
    LinearAlgebraEngine,
    ProbabilityEngine,
    OptimizationEngine,
    MathProblem,
    MathAttempt,
    MathKnowledge,
)
from evo_mind.math.trainer import MathTrainer, DailyMathResult

__all__ = [
    "CalculusEngine",
    "LinearAlgebraEngine",
    "ProbabilityEngine",
    "OptimizationEngine",
    "MathProblem",
    "MathAttempt",
    "MathKnowledge",
    "MathTrainer",
    "DailyMathResult",
]
