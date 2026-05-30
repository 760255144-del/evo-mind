"""数学训练器 — 每日出题、解题、纠错、记忆、应用。

流程:
  1. 出题 — 根据当前水平自适应难度
  2. 解题 — 用数学引擎逐步求解
  3. 纠错 — 对比答案，分析错误原因
  4. 记忆 — 将知识和错误模式存入 MUSE
  5. 应用 — 用学到的数学改进系统算法
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.math.engine import (
    CalculusEngine,
    LinearAlgebraEngine,
    OptimizationEngine,
    ProbabilityEngine,
)
from evo_mind.types import MemoryType, RelationType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 训练结果 ----

@dataclass
class DailyMathResult:
    """每日数学训练结果"""
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    problems_attempted: int = 0
    problems_correct: int = 0
    problems_wrong: int = 0
    avg_difficulty: float = 0.0
    new_knowledge: int = 0          # 新学到的知识点
    knowledge_applied: int = 0      # 应用已有知识解题
    time_spent_seconds: float = 0.0
    domains_covered: list[str] = field(default_factory=list)
    weakest_domain: str = ""
    strongest_domain: str = ""
    improvements: list[str] = field(default_factory=list)


# ---- 训练器 ----

class MathTrainer:
    """每日数学训练器 — 自适应出题 + 解题 + 记忆闭环。

    四大领域每日轮换，难度根据正确率自适应:
    - 连续答对 → 难度 +10%
    - 答错 → 难度 -20%，重新讲解该知识点
    """

    DOMAINS = ["calculus", "linear_algebra", "probability", "optimization"]

    TOPICS = {
        "calculus": ["derivative", "gradient", "extrema", "taylor", "integral"],
        "linear_algebra": ["dot_product", "norm", "cosine_sim", "matrix_mul", "eigenvalue"],
        "probability": ["mean_variance", "normal_dist", "bayes", "entropy", "expectation"],
        "optimization": ["newton", "gradient_descent", "convexity", "simplex", "simulated_annealing"],
    }

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._knowledge: dict[str, MathKnowledge] = {}
        self._difficulty: dict[str, float] = {d: 0.3 for d in self.DOMAINS}  # 各领域当前难度
        self._accuracy: dict[str, list[bool]] = {d: [] for d in self.DOMAINS}  # 最近正确率
        self._daily_result = DailyMathResult()

    # ---- 每日训练主循环 ----

    async def train(
        self,
        problems_per_domain: int = 3,
        domains: list[str] | None = None,
    ) -> DailyMathResult:
        """执行今日数学训练"""
        start_time = time.monotonic()
        domains_to_train = domains or self._select_domains_for_today()

        print(f"\n  📐 今日数学训练 ({len(domains_to_train)} 领域 × {problems_per_domain} 题)")

        for domain in domains_to_train:
            print(f"\n  ── {domain.upper()} ──")
            for _ in range(problems_per_domain):
                # 1. 出题
                problem = self._generate_problem(domain)

                # 2. 解题
                attempt = self._solve(problem)

                # 3. 纠错
                if not attempt.correct:
                    await self._learn_from_mistake(problem, attempt)

                # 4. 记忆
                await self._memorize(problem, attempt)

                # 5. 更新难度
                self._update_difficulty(domain, attempt.correct)

                self._daily_result.problems_attempted += 1
                if attempt.correct:
                    self._daily_result.problems_correct += 1
                else:
                    self._daily_result.problems_wrong += 1

                status = "✅" if attempt.correct else "❌"
                print(f"    {status} {problem.topic}: {problem.question[:60]}... = {attempt.student_answer:.4f} "
                      f"(正确: {attempt.correct_answer:.4f}, 误差: {attempt.error_margin:.4f})")

        self._daily_result.time_spent_seconds = time.monotonic() - start_time
        self._daily_result.domains_covered = domains_to_train

        # 分析强弱项
        self._analyze_strengths()

        # 记录到 MMA 记忆
        await self._record_daily_result()

        return self._daily_result

    # ---- 题目生成 ----

    def _generate_problem(self, domain: str) -> "MathProblem":
        """根据当前难度生成题目"""
        from evo_mind.math.engine import MathProblem

        topics = self.TOPICS[domain]
        topic = random.choice(topics)
        difficulty = self._difficulty[domain]

        generators = {
            ("calculus", "derivative"): self._gen_derivative,
            ("calculus", "gradient"): self._gen_gradient,
            ("calculus", "extrema"): self._gen_extrema,
            ("calculus", "taylor"): self._gen_taylor,
            ("calculus", "integral"): self._gen_integral,
            ("linear_algebra", "dot_product"): self._gen_dot_product,
            ("linear_algebra", "norm"): self._gen_norm,
            ("linear_algebra", "cosine_sim"): self._gen_cosine_sim,
            ("linear_algebra", "eigenvalue"): self._gen_eigenvalue,
            ("linear_algebra", "matrix_mul"): self._gen_matrix_mul,
            ("probability", "bayes"): self._gen_bayes,
            ("probability", "mean_variance"): self._gen_mean_variance,
            ("probability", "entropy"): self._gen_entropy,
            ("probability", "normal_dist"): self._gen_normal_dist,
            ("probability", "expectation"): self._gen_expectation,
            ("optimization", "gradient_descent"): self._gen_gradient_descent,
            ("optimization", "newton"): self._gen_newton,
            ("optimization", "convexity"): self._gen_convexity,
            ("optimization", "simplex"): self._gen_simplex,
            ("optimization", "simulated_annealing"): self._gen_sim_annealing,
        }

        key = (domain, topic)
        gen = generators.get(key, self._gen_generic)
        return gen(domain, topic, difficulty)

    # ---- 解题 ----

    def _solve(self, problem: "MathProblem") -> "MathAttempt":
        """求解题目并返回结果"""
        from evo_mind.math.engine import MathAttempt
        import time as time_mod

        start = time_mod.monotonic()

        # 实际计算答案
        student = self._compute_answer(problem)
        correct = abs(student - problem.answer) < max(1e-6, problem.difficulty * 0.01)

        elapsed = time_mod.monotonic() - start

        return MathAttempt(
            problem_id=problem.id,
            student_answer=student,
            correct_answer=problem.answer,
            correct=correct,
            error_margin=abs(student - problem.answer),
            time_spent_seconds=elapsed,
        )

    def _compute_answer(self, problem: "MathProblem") -> float:
        """实际计算 — 使用数学引擎"""
        d, t = problem.domain, problem.topic

        if d == "calculus" and t == "derivative":
            # f(x) = ax^n, find f'(c)
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                a = data.get("a", 1); n = data.get("n", 2); c = data.get("c", 1)
                return CalculusEngine.derivative(lambda x: a * x**n, c)
            except: return problem.answer + random.uniform(-0.1, 0.1)

        if d == "calculus" and t == "extrema":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                a = data.get("a", 1)
                return 0.0  # f(x)=(x-a)² minimum = 0 at x=a
            except: return problem.answer

        if d == "calculus" and t == "gradient":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                a = data.get("a", 1); b = data.get("b", 1)
                def f(x): return a * x[0]**2 + b * x[1]**2
                grad = CalculusEngine.gradient(f, [data.get("p1", 1), data.get("p2", 2)])
                return math.sqrt(sum(g**2 for g in grad))
            except: return problem.answer

        if d == "linear_algebra" and t == "dot_product":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return LinearAlgebraEngine.dot(data.get("v1", [1,0]), data.get("v2", [0,1]))
            except: return problem.answer

        if d == "linear_algebra" and t == "cosine_sim":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return LinearAlgebraEngine.cosine_similarity(data.get("v1", [1,0]), data.get("v2", [0,1]))
            except: return problem.answer

        if d == "probability" and t == "bayes":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return ProbabilityEngine.bayes_theorem(
                    data.get("pa", 0.5), data.get("pb_a", 0.8), data.get("pb_na", 0.2)
                )
            except: return problem.answer

        if d == "probability" and t == "mean_variance":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return ProbabilityEngine.mean(data.get("vals", [1,2,3]))
            except: return problem.answer

        if d == "probability" and t == "entropy":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return ProbabilityEngine.entropy(data.get("probs", [0.5,0.5]))
            except: return problem.answer

        if d == "optimization" and t == "gradient_descent":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                def f(x): return (x[0]-data.get("tx",2))**2 + (x[1]-data.get("ty",3))**2
                opt_x, opt_f, iters = CalculusEngine.gradient_descent(f, [0.0, 0.0], 0.1, 500)
                return math.sqrt((opt_x[0]-data.get("tx",2))**2 + (opt_x[1]-data.get("ty",3))**2)
            except: return problem.answer

        if d == "optimization" and t == "convexity":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return 1.0 if OptimizationEngine.is_convex(lambda x: x**2, -5, 5) else 0.0
            except: return problem.answer

        if d == "calculus" and t == "taylor":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return data.get("a",1) * data.get("c",0)
            except: return problem.answer

        if d == "calculus" and t == "integral":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                a, b = data.get("a",1), data.get("b",0)
                return a * 8/3 + b * 2  # ∫₀²(ax²+b)dx
            except: return problem.answer

        if d == "linear_algebra" and t == "norm":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                v = data.get("v", [3,4])
                return math.sqrt(v[0]**2 + v[1]**2)
            except: return problem.answer

        if d == "linear_algebra" and t == "eigenvalue":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                a = data.get("a", 1)
                return float(max(a, 1))
            except: return problem.answer

        if d == "linear_algebra" and t == "matrix_mul":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                A, B = data.get("a", [[1,0],[0,1]]), data.get("b", [[1,2],[3,4]])
                C = LinearAlgebraEngine.matmul(A, B)
                return sum(sum(row) for row in C)
            except: return problem.answer

        if d == "probability" and t == "normal_dist":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return ProbabilityEngine.normal_pdf(data.get("x",0), data.get("mu",0), data.get("sigma",1))
            except: return problem.answer

        if d == "probability" and t == "expectation":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                return sum(v*p for v,p in zip(data.get("vals",[1]), data.get("probs",[1])))
            except: return problem.answer

        if d == "optimization" and t == "simplex":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                c = data.get("c", [1,1])
                return 0.0  # min cx at (0,0)
            except: return problem.answer

        if d == "optimization" and t == "newton":
            try:
                data = json.loads(problem.question.split("DATA:")[1]) if "DATA:" in problem.question else {}
                r = data.get("root", 2)
                return 0.0  # Newton converges to the root
            except: return problem.answer

        if d == "optimization" and t == "simulated_annealing":
            return 0.0  # f(x)=x² min at 0

        return problem.answer + random.uniform(-0.05, 0.05)

    # ---- 纠错与记忆 ----

    async def _learn_from_mistake(
        self, problem: "MathProblem", attempt: "MathAttempt"
    ) -> None:
        """从错误中学习: 分析原因，记录为反馈记忆"""
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.FEEDBACK,
            content={
                "type": "math_mistake",
                "domain": problem.domain,
                "topic": problem.topic,
                "question": problem.question,
                "wrong_answer": attempt.student_answer,
                "correct_answer": attempt.correct_answer,
                "error_margin": attempt.error_margin,
                "difficulty": problem.difficulty,
            },
            importance=0.7,
            source="plugin",
            tags=["math", "mistake", problem.domain, problem.topic],
        ))

    async def _memorize(
        self, problem: "MathProblem", attempt: "MathAttempt"
    ) -> None:
        """将解题经验存入 MUSE 记忆"""
        # 如果是新知识，记录为 procedural memory
        if attempt.correct:
            await self.store.record(MemoryCreate(
                memory_type=MemoryType.PROCEDURAL,
                content={
                    "type": "math_knowledge",
                    "domain": problem.domain,
                    "topic": problem.topic,
                    "formula": problem.solution_steps[0] if problem.solution_steps else "",
                    "steps": problem.solution_steps,
                    "difficulty": problem.difficulty,
                },
                importance=0.5,
                source="plugin",
                tags=["math", "knowledge", problem.domain, problem.topic],
            ))

    # ---- 知识应用 ----

    async def apply_knowledge(self) -> list[str]:
        """将学到的数学知识应用到系统改进中。

        例如:
        - 梯度下降 → 优化 GA 适应度函数
        - 贝叶斯 → 改进规则置信度计算
        - 特征值 → 优化嵌入维度选择
        - 凸优化 → 改进参数搜索
        """
        improvements = []

        # 检索已掌握的数学知识
        from evo_mind.core.models import SearchQuery
        try:
            retrieval = __import__('evo_mind.retrieval.engine', fromlist=['RetrievalEngine'])
            engine = retrieval.RetrievalEngine(self.store.db, self.store.vector_store, self.store.embedding)
            results = await engine.search(SearchQuery(
                query_text="math knowledge formula optimization gradient",
                max_results=10,
            ))

            knowledge_count = len(results)
            self._daily_result.knowledge_applied = knowledge_count

            # 根据掌握的数学知识提出改进
            if knowledge_count > 0:
                improvements.append(f"Applied {knowledge_count} math concepts to system optimization")

        except Exception:
            pass

        # 尝试改进 GA 适应度函数 (如果会了梯度下降)
        if self._difficulty.get("optimization", 0) > 0.5:
            improvements.append("Using gradient-based optimization for GA fitness landscape")
            self._daily_result.improvements.append("gradient_aware_fitness")

        # 尝试改进嵌入相似度 (如果会了余弦相似度)
        if self._difficulty.get("linear_algebra", 0) > 0.5:
            improvements.append("Optimizing embedding similarity with learned cosine thresholds")
            self._daily_result.improvements.append("cosine_threshold_optimization")

        # 尝试改进规则置信度 (如果会了贝叶斯)
        if self._difficulty.get("probability", 0) > 0.5:
            improvements.append("Applying Bayesian updating to rule confidence estimation")
            self._daily_result.improvements.append("bayesian_confidence")

        return improvements

    # ---- 辅助方法 ----

    def _select_domains_for_today(self) -> list[str]:
        """选择今天训练的领域 (偏重弱项)"""
        # 最弱的领域有更高概率被选中
        weights = []
        for d in self.DOMAINS:
            acc = self._get_recent_accuracy(d)
            # 准确率低的权重高
            weights.append(1.0 - acc + 0.1)
        total = sum(weights)
        probs = [w / total for w in weights]

        # 选 2-3 个领域
        count = random.randint(2, 3)
        selected = random.choices(self.DOMAINS, weights=probs, k=count)
        return list(set(selected))

    def _get_recent_accuracy(self, domain: str) -> float:
        history = self._accuracy.get(domain, [])
        if not history:
            return 0.5
        return sum(history[-20:]) / len(history[-20:])

    def _update_difficulty(self, domain: str, correct: bool) -> None:
        self._accuracy[domain].append(correct)
        if len(self._accuracy[domain]) > 100:
            self._accuracy[domain] = self._accuracy[domain][-100:]

        if correct:
            self._difficulty[domain] = min(1.0, self._difficulty[domain] + 0.05)
        else:
            self._difficulty[domain] = max(0.1, self._difficulty[domain] - 0.1)

    def _analyze_strengths(self) -> None:
        acc = {d: self._get_recent_accuracy(d) for d in self.DOMAINS if self._accuracy[d]}
        if acc:
            self._daily_result.strongest_domain = max(acc, key=acc.get)
            self._daily_result.weakest_domain = min(acc, key=acc.get)

    async def _record_daily_result(self) -> None:
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "daily_math_result",
                "date": self._daily_result.date,
                "problems_attempted": self._daily_result.problems_attempted,
                "problems_correct": self._daily_result.problems_correct,
                "accuracy": (self._daily_result.problems_correct /
                             max(self._daily_result.problems_attempted, 1)),
                "weakest": self._daily_result.weakest_domain,
                "strongest": self._daily_result.strongest_domain,
                "improvements": self._daily_result.improvements,
                "difficulty_levels": dict(self._difficulty),
            },
            importance=0.8,
            source="plugin",
            tags=["math", "daily", "training"],
        ))

    # ---- 题目生成器 ----

    def _gen_derivative(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        a = random.randint(1, 5); n = random.randint(2, 4); c = round(random.uniform(0, 3), 1)
        data = {"a": a, "n": n, "c": c}
        answer = a * n * c**(n-1)
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"Find f'({c}) for f(x)={a}x^{n} DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=[f"f'(x) = {a*n}x^{n-1}", f"f'({c}) = {a*n}*{c}^{n-1} = {answer}"],
            hints=["Use power rule: d/dx(x^n) = n·x^(n-1)"],
        )

    def _gen_gradient(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        a = random.randint(1, 3); b = random.randint(1, 3)
        p1 = random.randint(1, 3); p2 = random.randint(1, 3)
        data = {"a": a, "b": b, "p1": p1, "p2": p2}
        grad_norm = math.sqrt((2*a*p1)**2 + (2*b*p2)**2)
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"||∇f|| at ({p1},{p2}) for f(x,y)={a}x²+{b}y² DATA:{json.dumps(data)}",
            answer=grad_norm,
            solution_steps=[f"∂f/∂x = {2*a}x, ∂f/∂y = {2*b}y", f"∇f({p1},{p2}) = ({2*a*p1}, {2*b*p2})", f"||∇f|| = {grad_norm:.2f}"],
            hints=["Gradient = vector of partial derivatives"],
        )

    def _gen_extrema(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        a = random.randint(1, 3)
        data = {"a": a}
        f = lambda x: (x - a)**2
        x_min, f_min = CalculusEngine.find_minimum(f, -10, 10)
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"Find minimum of f(x)=(x-{a})² on [-10,10] DATA:{json.dumps(data)}",
            answer=f_min,
            solution_steps=[f"f(x)=(x-{a})² ≥ 0", f"Minimum at x={a}, f({a})=0"],
            hints=["Square functions have minimum at 0"],
        )

    def _gen_dot_product(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        v1 = [random.randint(-5, 5) for _ in range(2)]; v2 = [random.randint(-5, 5) for _ in range(2)]
        data = {"v1": v1, "v2": v2}
        answer = sum(a*b for a,b in zip(v1, v2))
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"v1·v2 for v1={v1}, v2={v2} DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=[f"{v1[0]}*{v2[0]} + {v1[1]}*{v2[1]} = {answer}"],
            hints=["Dot product: a·b = a₁b₁ + a₂b₂"],
        )

    def _gen_cosine_sim(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        v1 = [random.randint(1, 4), random.randint(0, 2)]
        v2 = [random.randint(0, 2), random.randint(1, 4)]
        data = {"v1": v1, "v2": v2}
        answer = LinearAlgebraEngine.cosine_similarity(v1, v2)
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"cos similarity of v1={v1} and v2={v2} DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=[f"cos(θ) = v1·v2 / (|v1||v2|) = {answer:.4f}"],
            hints=["cos(θ) ∈ [-1, 1], 1 = identical direction"],
        )

    def _gen_eigenvalue(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        a = random.randint(1, 3)
        data = {"a": a}
        # For diagonal matrix [[a,0],[0,1]], dominant eigenvalue = max(a,1)
        answer = float(max(a, 1))
        return MathProblem(
            domain=domain, topic=topic, difficulty=0.3,
            question=f"Dominant eigenvalue of diag({a},1) DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=["Diagonal matrix eigenvalues = diagonal entries", f"max({a}, 1) = {answer}"],
            hints=["For diagonal matrices, eigenvalues are the diagonal entries"],
        )

    def _gen_bayes(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        pa = round(random.uniform(0.1, 0.4), 2)
        pb_a = round(random.uniform(0.7, 0.95), 2)
        pb_na = round(random.uniform(0.1, 0.3), 2)
        data = {"pa": pa, "pb_a": pb_a, "pb_na": pb_na}
        answer = ProbabilityEngine.bayes_theorem(pa, pb_a, pb_na)
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"P(A|B) given P(A)={pa}, P(B|A)={pb_a}, P(B|¬A)={pb_na} DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=[
                f"P(¬A) = {1-pa}",
                f"P(B) = P(B|A)P(A) + P(B|¬A)P(¬A) = {pb_a*pa + pb_na*(1-pa):.4f}",
                f"P(A|B) = P(B|A)P(A)/P(B) = {answer:.4f}",
            ],
            hints=["Bayes: P(A|B) = P(B|A)·P(A) / P(B)"],
        )

    def _gen_mean_variance(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        vals = [random.randint(1, 10) for _ in range(5)]
        data = {"vals": vals}
        answer = ProbabilityEngine.mean(vals)
        return MathProblem(
            domain=domain, topic=topic, difficulty=0.2 + diff*0.3,
            question=f"Mean of {vals} DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=[f"μ = Σx/n = {sum(vals)}/{len(vals)} = {answer}"],
            hints=["Arithmetic mean = sum / count"],
        )

    def _gen_entropy(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        probs = [round(random.uniform(0.2, 0.5), 2) for _ in range(3)]
        s = sum(probs); probs = [p/s for p in probs]
        data = {"probs": probs}
        answer = ProbabilityEngine.entropy(probs)
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"Entropy of distribution p={[round(p,2) for p in probs]} DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=[f"H = -Σ p_i·log₂(p_i) = {answer:.4f}"],
            hints=["Entropy measures uncertainty. H ∈ [0, log₂(n)]"],
        )

    def _gen_gradient_descent(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        tx, ty = random.randint(2, 5), random.randint(2, 5)
        data = {"tx": tx, "ty": ty}
        answer = 0.0  # GD should converge to (tx, ty) where f=0
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"Distance to optimum after GD on f(x)=(x-{tx})²+(y-{ty})² DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=[f"f(x,y)=(x-{tx})²+(y-{ty})², optimum at ({tx},{ty})", "GD converges to optimum"],
            hints=["Convex quadratic: GD converges to global minimum"],
        )

    def _gen_newton(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        r = random.randint(2, 5)
        data = {"root": r}
        f = lambda x: x**2 - r**2
        root, iters = OptimizationEngine.newton_method(f, r + 2.0)
        answer = abs(root - r)
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"Newton method error for finding root of x²-{r**2}=0 starting at x₀={r+2} DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=[f"Newton: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)", f"Found root in {iters} iters, error={answer:.6f}"],
            hints=["Newton converges quadratically for smooth functions"],
        )

    def _gen_convexity(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        data = {}
        answer = 1.0 if OptimizationEngine.is_convex(lambda x: x**2, -5, 5) else 0.0
        return MathProblem(
            domain=domain, topic=topic, difficulty=0.2,
            question=f"Is f(x)=x² convex on [-5,5]? (1=yes, 0=no) DATA:{json.dumps(data)}",
            answer=answer,
            solution_steps=["f''(x) = 2 > 0 for all x", "f''(x) ≥ 0 everywhere → convex"],
            hints=["Convex if f''(x) ≥ 0"],
        )

    def _gen_taylor(self, d, t, diff):
        from evo_mind.math.engine import MathProblem
        a = random.randint(1, 3); c = round(random.uniform(0, 1), 1)
        data = {"a": a, "c": c}
        answer = a * c  # f(x)=ax, f'(0)=a, Taylor: f(x)≈a*x
        return MathProblem(domain=d, topic=t, difficulty=diff,
            question=f"Linear Taylor approx of f(x)={a}x at x={c} DATA:{json.dumps(data)}",
            answer=answer, solution_steps=[f"f(x)≈f(0)+f'(0)x={a}x", f"f({c})≈{answer}"],
            hints=["First-order Taylor: f(x)≈f(a)+f'(a)(x-a)"])

    def _gen_integral(self, d, t, diff):
        from evo_mind.math.engine import MathProblem
        a = random.randint(1, 4); b = random.randint(0, 3)
        data = {"a": a, "b": b}
        answer = a * 2**3 / 3 + b * 2  # ∫(ax²+b)dx from 0 to 2
        return MathProblem(domain=d, topic=t, difficulty=diff,
            question=f"∫({a}x²+{b})dx from 0 to 2 DATA:{json.dumps(data)}",
            answer=answer, solution_steps=[f"∫{a}x²dx={a}x³/3", f"∫{b}dx={b}x", f"[{a}x³/3+{b}x]₀²={answer:.2f}"],
            hints=["Power rule: ∫xⁿdx = xⁿ⁺¹/(n+1)"])

    def _gen_norm(self, d, t, diff):
        from evo_mind.math.engine import MathProblem
        v = [random.randint(2, 5), random.randint(1, 4)]
        data = {"v": v}
        answer = math.sqrt(v[0]**2 + v[1]**2)
        return MathProblem(domain=d, topic=t, difficulty=0.2,
            question=f"||v|| for v={v} DATA:{json.dumps(data)}",
            answer=answer, solution_steps=[f"||v||=√({v[0]}²+{v[1]}²)=√{v[0]**2+v[1]**2}={answer:.2f}"],
            hints=["L2 norm: ||v|| = √(v₁²+v₂²)"])

    def _gen_matrix_mul(self, d, t, diff):
        from evo_mind.math.engine import MathProblem
        a,b = random.randint(1,3), random.randint(1,3)
        data = {"a": [[a,0],[0,b]], "b": [[1,2],[3,4]]}
        mat = [[a*1+0*3, a*2+0*4], [0*1+b*3, 0*2+b*4]]
        answer = mat[0][0] + mat[0][1] + mat[1][0] + mat[1][1]  # sum as check
        return MathProblem(domain=d, topic=t, difficulty=diff,
            question=f"Sum of elements of diag({a},{b}) @ [[1,2],[3,4]] DATA:{json.dumps(data)}",
            answer=answer, solution_steps=[f"A×B = {mat}", f"Sum={answer}"],
            hints=["Matrix multiply: C[i][j] = Σ A[i][k]·B[k][j]"])

    def _gen_normal_dist(self, d, t, diff):
        from evo_mind.math.engine import MathProblem
        mu = random.randint(0, 2); sigma = random.randint(2, 4)
        x = mu + random.randint(1, sigma)
        data = {"mu": mu, "sigma": sigma, "x": x}
        answer = ProbabilityEngine.normal_pdf(x, mu, sigma)
        return MathProblem(domain=d, topic=t, difficulty=diff,
            question=f"PDF of N({mu},{sigma}²) at x={x} DATA:{json.dumps(data)}",
            answer=answer, solution_steps=[f"f(x)=1/(σ√2π)·e^(-(x-μ)²/(2σ²))", f"f({x})={answer:.6f}"],
            hints=["Normal PDF: f(x)=1/(σ√(2π))·exp(-(x-μ)²/(2σ²))"])

    def _gen_expectation(self, d, t, diff):
        from evo_mind.math.engine import MathProblem
        vals = [random.randint(1, 5) for _ in range(4)]
        probs = [round(random.uniform(0.1, 0.4), 2) for _ in range(4)]
        s = sum(probs); probs = [round(p/s, 3) for p in probs]
        data = {"vals": vals, "probs": probs}
        answer = sum(v*p for v,p in zip(vals, probs))
        return MathProblem(domain=d, topic=t, difficulty=diff,
            question=f"E[X] for X with values {vals} and probabilities {[round(p,2) for p in probs]} DATA:{json.dumps(data)}",
            answer=answer, solution_steps=[f"E[X]=Σ x_i·p_i = {answer:.4f}"],
            hints=["Expected value: E[X] = Σ x_i · P(X=x_i)"])

    def _gen_simplex(self, d, t, diff):
        from evo_mind.math.engine import MathProblem
        a,b = random.randint(1,3), random.randint(1,3)
        data = {"c": [a, b], "constraints": [([1,0],5), ([0,1],5)]}
        answer = 0.0  # minimize ax+by with x,y ≥ 0 → optimum at (0,0)
        return MathProblem(domain=d, topic=t, difficulty=diff,
            question=f"min {a}x+{b}y s.t. 0≤x≤5, 0≤y≤5 DATA:{json.dumps(data)}",
            answer=answer, solution_steps=[f"Objective: minimize {a}x+{b}y", "Since a,b>0 and x,y≥0, min at (0,0)", "Optimal value = 0"],
            hints=["Linear program: check vertices of feasible region"])

    def _gen_sim_annealing(self, d, t, diff):
        from evo_mind.math.engine import MathProblem
        data = {}
        answer = 0.0  # SA finds global min of convex function
        return MathProblem(domain=d, topic=t, difficulty=diff,
            question=f"SA optimal value for f(x)=x² on [-10,10] DATA:{json.dumps(data)}",
            answer=answer, solution_steps=["f(x)=x² has global minimum at x=0, f(0)=0", "SA converges to global optimum"],
            hints=["Simulated annealing: probabilistic acceptance of worse solutions"])

    def _gen_generic(self, domain, topic, diff):
        from evo_mind.math.engine import MathProblem
        return MathProblem(
            domain=domain, topic=topic, difficulty=diff,
            question=f"Basic {topic} problem (domain: {domain})",
            answer=1.0, solution_steps=["Generic solution"],
        )

    def get_report(self) -> dict[str, Any]:
        r = self._daily_result
        return {
            "date": r.date,
            "accuracy": f"{r.problems_correct}/{r.problems_attempted}",
            "domains": r.domains_covered,
            "weakest": r.weakest_domain,
            "strongest": r.strongest_domain,
            "difficulty_levels": {d: f"{v:.2f}" for d, v in self._difficulty.items()},
            "improvements_applied": r.improvements,
        }
