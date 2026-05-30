"""数学引擎 — 微积分、线性代数、概率、优化。

这个模块不仅解题，还将数学知识存储为 MUSE 记忆，
使系统能够用数学原理改进自身的进化算法。

四大领域:
  Calculus    — 导数、梯度、最值 (用于梯度下降优化)
  LinearAlg   — 矩阵、特征值、SVD (用于降维和嵌入)
  Probability — 分布、贝叶斯、期望 (用于不确定性推理)
  Optimization — 凸优化、拉格朗日、KKT (用于改进 GA 适应度)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from evo_mind.utils import uuid7


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 题目模型 ----

@dataclass
class MathProblem:
    """一道数学题"""
    id: str = field(default_factory=uuid7)
    domain: str = ""           # calculus | linear_algebra | probability | optimization
    topic: str = ""            # derivative | matrix | bayes | gradient_descent | ...
    question: str = ""
    difficulty: float = 0.5    # 0.0-1.0
    answer: float = 0.0
    solution_steps: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)


@dataclass
class MathAttempt:
    """一次解题尝试"""
    problem_id: str = ""
    student_answer: float = 0.0
    correct_answer: float = 0.0
    correct: bool = False
    error_margin: float = 0.0
    time_spent_seconds: float = 0.0
    timestamp: str = field(default_factory=_now)


@dataclass
class MathKnowledge:
    """一条数学知识 — 存入 MUSE 记忆"""
    domain: str = ""
    topic: str = ""
    formula: str = ""
    description: str = ""
    example: str = ""
    confidence: float = 1.0
    times_applied: int = 0
    times_succeeded: int = 0


# ---- 微积分引擎 ----

class CalculusEngine:
    """微积分: 导数、梯度、极值、泰勒展开"""

    @staticmethod
    def derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
        """数值导数 f'(x) = (f(x+h) - f(x-h)) / 2h (中心差分)"""
        return (f(x + h) - f(x - h)) / (2 * h)

    @staticmethod
    def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-4) -> float:
        """二阶导数 f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2"""
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)

    @staticmethod
    def gradient(f: Callable, point: list[float], h: float = 1e-6) -> list[float]:
        """多变量数值梯度 ∇f"""
        grad = []
        for i in range(len(point)):
            p_plus = point.copy()
            p_plus[i] += h
            p_minus = point.copy()
            p_minus[i] -= h
            grad.append((f(p_plus) - f(p_minus)) / (2 * h))
        return grad

    @staticmethod
    def gradient_descent(
        f: Callable,
        start: list[float],
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> tuple[list[float], float, int]:
        """梯度下降找极小值"""
        x = start.copy()
        for i in range(max_iter):
            grad = CalculusEngine.gradient(f, x)
            grad_norm = math.sqrt(sum(g * g for g in grad))
            if grad_norm < tol:
                return x, f(x), i + 1
            for j in range(len(x)):
                x[j] -= learning_rate * grad[j]
        return x, f(x), max_iter

    @staticmethod
    def find_minimum(
        f: Callable[[float], float],
        a: float, b: float,
        tol: float = 1e-6,
    ) -> tuple[float, float]:
        """黄金分割搜索一维函数极小值"""
        phi = (math.sqrt(5) - 1) / 2
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        while abs(b - a) > tol:
            if f(c) < f(d):
                b = d
            else:
                a = c
            c = b - phi * (b - a)
            d = a + phi * (b - a)
        x_min = (a + b) / 2
        return x_min, f(x_min)


# ---- 线性代数引擎 ----

class LinearAlgebraEngine:
    """线性代数: 矩阵运算、特征值、SVD 近似"""

    @staticmethod
    def dot(a: list[float], b: list[float]) -> float:
        """向量点积"""
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def norm(v: list[float]) -> float:
        """向量 L2 范数"""
        return math.sqrt(sum(x * x for x in v))

    @staticmethod
    def normalize(v: list[float]) -> list[float]:
        """向量归一化"""
        n = LinearAlgebraEngine.norm(v)
        if n == 0:
            return v.copy()
        return [x / n for x in v]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度 cos(θ) = a·b / (|a||b|)"""
        dot_ab = LinearAlgebraEngine.dot(a, b)
        norm_a = LinearAlgebraEngine.norm(a)
        norm_b = LinearAlgebraEngine.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_ab / (norm_a * norm_b)

    @staticmethod
    def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        """矩阵乘法 C = A @ B"""
        m, k = len(A), len(A[0])
        n = len(B[0])
        C = [[0.0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                C[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
        return C

    @staticmethod
    def power_iteration(
        A: list[list[float]],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[float, list[float]]:
        """幂迭代法求最大特征值和对应特征向量"""
        n = len(A)
        v = [random.random() for _ in range(n)]
        v = LinearAlgebraEngine.normalize(v)
        lambda_old = 0.0
        for _ in range(max_iter):
            Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
            v = LinearAlgebraEngine.normalize(Av)
            lambda_new = LinearAlgebraEngine.dot(
                [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)], v
            )
            if abs(lambda_new - lambda_old) < tol:
                return lambda_new, v
            lambda_old = lambda_new
        return lambda_old, v


# ---- 概率引擎 ----

class ProbabilityEngine:
    """概率论: 分布、贝叶斯、期望、方差"""

    @staticmethod
    def mean(data: list[float]) -> float:
        return sum(data) / len(data) if data else 0.0

    @staticmethod
    def variance(data: list[float], sample: bool = True) -> float:
        """方差 (默认样本方差, sample=False 为总体方差)"""
        if len(data) < 2:
            return 0.0
        m = ProbabilityEngine.mean(data)
        ss = sum((x - m) ** 2 for x in data)
        return ss / (len(data) - 1) if sample else ss / len(data)

    @staticmethod
    def std(data: list[float], sample: bool = True) -> float:
        return math.sqrt(ProbabilityEngine.variance(data, sample))

    @staticmethod
    def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """正态分布概率密度"""
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    @staticmethod
    def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """正态分布累积概率 (误差函数近似)"""
        return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

    @staticmethod
    def bayes_theorem(
        p_a: float, p_b_given_a: float, p_b_given_not_a: float
    ) -> float:
        """贝叶斯定理: P(A|B) = P(B|A)·P(A) / P(B)"""
        p_not_a = 1.0 - p_a
        p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
        if p_b == 0:
            return 0.0
        return (p_b_given_a * p_a) / p_b

    @staticmethod
    def entropy(probabilities: list[float]) -> float:
        """信息熵 H = -Σ p_i·log₂(p_i)"""
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    @staticmethod
    def expected_value(values: list[float], probs: list[float] | None = None) -> float:
        """期望 E[X]"""
        if probs is None:
            return ProbabilityEngine.mean(values)
        return sum(v * p for v, p in zip(values, probs))


# ---- 优化引擎 ----

class OptimizationEngine:
    """最优化: 牛顿法、拉格朗日乘子、凸优化判定"""

    @staticmethod
    def newton_method(
        f: Callable[[float], float],
        x0: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[float, int]:
        """牛顿法求根 f(x)=0"""
        x = x0
        for i in range(max_iter):
            fx = f(x)
            if abs(fx) < tol:
                return x, i + 1
            # 数值导数
            df = CalculusEngine.derivative(f, x)
            if abs(df) < 1e-12:
                break
            x = x - fx / df
        return x, max_iter

    @staticmethod
    def simplex_method(
        objective: list[float],  # 目标函数系数 c
        constraints: list[tuple[list[float], float]],  # A @ x <= b
        bounds: list[tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """线性规划单纯形法 (简化实现)

        minimize c^T x  subject to A x <= b, x >= 0
        """
        n = len(objective)
        m = len(constraints)

        # 简化: 对于2变量问题用顶点枚举
        if n == 2 and m <= 5:
            vertices = OptimizationEngine._enumerate_vertices(
                constraints, bounds
            )
            best_val = float("inf")
            best_x = [0.0, 0.0]
            for x1, x2 in vertices:
                val = objective[0] * x1 + objective[1] * x2
                if val < best_val:
                    best_val = val
                    best_x = [x1, x2]
            return {"x": best_x, "optimal_value": best_val, "method": "vertex_enumeration"}

        return {"x": [0.0] * n, "optimal_value": float("inf"), "method": "stub"}

    @staticmethod
    def _enumerate_vertices(
        constraints: list[tuple[list[float], float]],
        bounds: list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, float]]:
        """枚举线性约束的顶点"""
        vertices = []

        # 交点: 每对约束的交点
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                a1, b1 = constraints[i]
                a2, b2 = constraints[j]
                det = a1[0] * a2[1] - a1[1] * a2[0]
                if abs(det) < 1e-10:
                    continue
                x = (b1 * a2[1] - b2 * a1[1]) / det
                y = (a1[0] * b2 - a2[0] * b1) / det

                # 检查是否满足所有约束
                feasible = True
                for ak, bk in constraints:
                    if ak[0] * x + ak[1] * y > bk + 1e-9:
                        feasible = False
                        break
                if x >= -1e-9 and y >= -1e-9 and feasible:
                    vertices.append((x, y))

        return vertices

    @staticmethod
    def is_convex(f: Callable[[float], float], a: float, b: float, samples: int = 100) -> bool:
        """检测函数在 [a,b] 上是否凸 (f''(x) >= 0)"""
        for i in range(1, samples):
            x = a + (b - a) * i / samples
            d2 = CalculusEngine.second_derivative(f, x)
            if d2 < -1e-6:
                return False
        return True

    @staticmethod
    def simulated_annealing(
        f: Callable[[list[float]], float],
        bounds: list[tuple[float, float]],
        max_iter: int = 1000,
        T_start: float = 100.0,
        T_end: float = 0.01,
        cooling_rate: float = 0.95,
    ) -> tuple[list[float], float]:
        """模拟退火 — 全局优化"""
        n = len(bounds)
        x = [random.uniform(lo, hi) for lo, hi in bounds]
        fx = f(x)
        best_x, best_f = x.copy(), fx
        T = T_start

        while T > T_end and max_iter > 0:
            for _ in range(min(100, max_iter)):
                # 邻域扰动
                x_new = x.copy()
                i = random.randint(0, n - 1)
                lo, hi = bounds[i]
                x_new[i] += random.gauss(0, (hi - lo) * 0.1 * T / T_start)
                x_new[i] = max(lo, min(hi, x_new[i]))

                f_new = f(x_new)
                delta = f_new - fx

                if delta < 0 or random.random() < math.exp(-delta / T):
                    x, fx = x_new, f_new
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx

                max_iter -= 1

            T *= cooling_rate

        return best_x, best_f
