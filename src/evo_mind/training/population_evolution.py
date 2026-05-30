"""种群进化引擎 — DARWIN 风格的交叉修改 + 基因重组 + 自然选择。

核心机制:
1. DARWIN 交叉修改引擎
   - 多个独立 Agent 模型互相修改对方的训练代码
   - 每个 Agent 既是"编辑者"也是"被编辑者"
   - 通过竞争与合作实现能力跃迁

2. 基因重组引擎
   - 将算法逻辑编码为"基因序列"
   - 交叉: 两个父代交换基因片段
   - 变异: 随机扰动基因
   - 选择: 适应度高的基因进入下一代

3. 种群管理
   - 种群规模动态调整
   - 生态位分化: 不同 Agent 专精不同任务
   - 灭绝与重生: 低适应度个体被淘汰, 新个体从最优基因诞生

参考: DARWIN (arXiv 2024), 遗传算法, NEAT
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 种群个体 ----

@dataclass
class Individual:
    """种群中的一个个体 — 代表一个 Agent 实例或策略"""
    id: str = field(default_factory=uuid7)
    name: str = "agent"
    generation: int = 0

    # 基因 (可演化的参数和代码)
    genome: dict[str, Any] = field(default_factory=dict)

    # 适应度
    fitness: float = 0.0
    fitness_history: list[float] = field(default_factory=list)

    # 生态位 (专精领域)
    niche: str = "general"             # "bug_fixing" | "optimization" | "implementation" | ...
    niche_fitness: dict[str, float] = field(default_factory=dict)

    # DARWIN: 代码修改记录
    modifications_made: int = 0         # 对其他个体的修改次数
    modifications_received: int = 0     # 被其他个体修改的次数
    modification_success_rate: float = 0.5

    # 血缘
    parent_ids: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)

    # 状态
    alive: bool = True
    age: int = 0                        # 经历的代数
    max_age: int = 20                   # 最大寿命

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id[:12],
            "name": self.name,
            "gen": self.generation,
            "fitness": round(self.fitness, 4),
            "niche": self.niche,
            "alive": self.alive,
            "age": self.age,
        }


# ---- 种群 ----

@dataclass
class Population:
    """进化种群"""
    id: str = field(default_factory=uuid7)
    individuals: dict[str, Individual] = field(default_factory=dict)
    generation: int = 0
    max_size: int = 20
    min_size: int = 5

    # 统计
    avg_fitness_history: list[float] = field(default_factory=list)
    best_fitness_history: list[float] = field(default_factory=list)
    diversity_history: list[float] = field(default_factory=list)

    def get_alive(self) -> list[Individual]:
        return [i for i in self.individuals.values() if i.alive]

    def get_by_niche(self, niche: str) -> list[Individual]:
        return [i for i in self.get_alive() if i.niche == niche]


# ---- DARWIN 交叉修改引擎 ----

class DarwinEngine:
    """DARWIN 引擎: 让 Agent 像程序员一样互相修改代码。

    每个 Agent 可以:
    1. 读取其他 Agent 的"代码" (genome)
    2. 提出修改建议
    3. 验证修改效果
    4. 好的修改被采纳, 坏的被拒绝

    这实现了代码层面的自然选择。
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.population = Population()
        self._modification_history: list[dict[str, Any]] = []

    async def create_initial_population(self, size: int = 10) -> Population:
        """创建初始种群, 每个个体有不同的初始基因"""
        niches = ["bug_fixing", "optimization", "implementation", "analysis", "general"]
        genomes = self._seed_genomes(size)

        for i in range(size):
            ind = Individual(
                name=f"agent-{i:03d}",
                generation=0,
                genome=genomes[i],
                niche=niches[i % len(niches)],
                niche_fitness={n: random.uniform(0.3, 0.7) for n in niches},
            )
            self.population.individuals[ind.id] = ind

        self.population.max_size = size * 2
        self.population.min_size = max(3, size // 3)

        logger.info("population_created", size=size)
        return self.population

    async def darwin_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Individual:
        """DARWIN 交叉: 两个父代互相修改对方的基因, 产生子代

        过程:
        1. Parent1 审查 Parent2 的基因并提出修改
        2. Parent2 审查 Parent1 的基因并提出修改
        3. 合并修改生成子代
        4. 子代继承双方的优点
        """
        child = Individual(
            name=f"child-of-{parent1.name}-{parent2.name}",
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
            niche=parent1.niche if parent1.fitness >= parent2.fitness else parent2.niche,
        )

        # Parent1 修改 Parent2
        p1_modifications = await self._propose_modifications(parent1, parent2)

        # Parent2 修改 Parent1
        p2_modifications = await self._propose_modifications(parent2, parent1)

        # 合并基因组: 取双方最优基因
        merged_genome = {}
        all_keys = set(parent1.genome.keys()) | set(parent2.genome.keys())
        for key in all_keys:
            v1 = parent1.genome.get(key, 0)
            v2 = parent2.genome.get(key, 0)
            # 取适应度高的父代的基因, 加一点随机混合
            if parent1.fitness >= parent2.fitness:
                merged_genome[key] = v1 * 0.7 + v2 * 0.3
            else:
                merged_genome[key] = v2 * 0.7 + v1 * 0.3

        # 应用修改
        for mod in p1_modifications + p2_modifications:
            if mod.get("accepted", False):
                key = mod.get("gene_key", "")
                if key in merged_genome:
                    merged_genome[key] = mod.get("new_value", merged_genome[key])

        child.genome = merged_genome

        # 记录修改
        parent1.modifications_made += len(p1_modifications)
        parent2.modifications_made += len(p2_modifications)

        self.population.individuals[child.id] = child
        parent1.children_ids.append(child.id)
        parent2.children_ids.append(child.id)

        return child

    async def _propose_modifications(
        self, editor: Individual, target: Individual
    ) -> list[dict[str, Any]]:
        """一个 Agent (editor) 审查并修改另一个 Agent (target) 的基因"""
        modifications = []

        for gene_key, gene_value in target.genome.items():
            editor_value = editor.genome.get(gene_key)

            # 如果 editor 在这个基因上表现更好, 建议 target 采纳
            if editor_value is not None and editor.fitness > target.fitness:
                # 计算改进建议 (向 editor 的值靠拢)
                new_value = gene_value + 0.3 * (editor_value - gene_value)

                # 验证: 改进方向是否合理
                accepted = editor.fitness > target.fitness + 0.05

                modifications.append({
                    "gene_key": gene_key,
                    "old_value": gene_value,
                    "new_value": new_value,
                    "editor_id": editor.id,
                    "editor_fitness": editor.fitness,
                    "target_fitness": target.fitness,
                    "accepted": accepted,
                })

        target.modifications_received += len(modifications)
        return modifications

    # ---- 基因重组 ----

    async def genetic_recombination(self) -> list[Individual]:
        """对整个种群进行基因重组: 选择→交叉→变异→选择"""
        alive = self.population.get_alive()
        if len(alive) < 2:
            return []

        new_generation: list[Individual] = []

        # 精英保留: 前 20% 直接进入下一代
        sorted_alive = sorted(alive, key=lambda i: i.fitness, reverse=True)
        elite_count = max(1, len(sorted_alive) // 5)
        elites = [deepcopy(ind) for ind in sorted_alive[:elite_count]]
        for e in elites:
            e.id = uuid7()
            e.age += 1
        new_generation.extend(elites)

        # 锦标赛选择 + 交叉产生子代
        while len(new_generation) < self.population.max_size:
            # Safety: maximum attempts to prevent infinite loop
            max_attempts = self.population.max_size * 3
            for _ in range(max_attempts):
                if len(new_generation) >= self.population.max_size:
                    break
                p1 = self._tournament_select(alive, tournament_size=3)
                p2 = self._tournament_select(alive, tournament_size=3)
                if p1.id == p2.id:
                    continue
                child = await self.darwin_crossover(p1, p2)
                child = self._mutate(child)
                child.fitness = min(1.0, (p1.fitness + p2.fitness) / 2 * random.uniform(0.8, 1.2))
                child.age = 0
                new_generation.append(child)
            break  # Exit after max_attempts even if not full

        # 淘汰: 超过 max_size 的个体
        new_generation = new_generation[:self.population.max_size]

        # 年龄管理: 太老的个体死亡
        for ind in new_generation:
            if ind.age >= ind.max_age:
                ind.alive = False

        # 替换种群
        old_ids = set(self.population.individuals.keys())
        self.population.individuals = {ind.id: ind for ind in new_generation}
        self.population.generation += 1

        # 记录统计
        fitnesses = [i.fitness for i in new_generation]
        self.population.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        self.population.best_fitness_history.append(max(fitnesses))
        self.population.diversity_history.append(self._compute_diversity(new_generation))

        logger.info(
            "generation_complete",
            gen=self.population.generation,
            pop_size=len(new_generation),
            avg_fitness=f"{self.population.avg_fitness_history[-1]:.3f}",
            best_fitness=f"{self.population.best_fitness_history[-1]:.3f}",
        )

        return new_generation

    # ---- 自然选择 ----

    async def natural_selection(self) -> int:
        """自然选择: 淘汰低适应度个体, 保留优秀个体"""
        alive = self.population.get_alive()
        if len(alive) <= self.population.min_size:
            return 0

        # 按适应度排序
        sorted_individuals = sorted(alive, key=lambda i: i.fitness)

        # 淘汰最低的 20%
        cull_count = max(1, len(sorted_individuals) // 5)
        culled = 0

        for ind in sorted_individuals[:cull_count]:
            # 如果适应度显著低于平均值, 淘汰
            avg_fitness = sum(i.fitness for i in alive) / len(alive)
            if ind.fitness < avg_fitness * 0.5:
                ind.alive = False
                culled += 1
                logger.debug("individual_culled", name=ind.name, fitness=f"{ind.fitness:.3f}")

        # 生态位平衡: 确保每个生态位都有代表
        niches_present = {ind.niche for ind in self.population.get_alive()}
        for niche in {"bug_fixing", "optimization", "implementation", "analysis", "general"}:
            if niche not in niches_present:
                # 从最佳个体克隆并变异出一个新生态位代表
                best = sorted(alive, key=lambda i: i.fitness, reverse=True)[0]
                new_ind = deepcopy(best)
                new_ind.id = uuid7()
                new_ind.name = f"{niche}-specialist"
                new_ind.niche = niche
                new_ind.genome = self._mutate(new_ind).genome
                new_ind.fitness = best.fitness * 0.7
                new_ind.age = 0
                self.population.individuals[new_ind.id] = new_ind
                logger.info("niche_repopulated", niche=niche)

        return culled

    # ---- 适应度评估 ----

    async def evaluate_population(self) -> None:
        """评估整个种群的适应度"""
        for ind in self.population.get_alive():
            fitness = await self._evaluate_individual(ind)
            ind.fitness_history.append(fitness)
            # 指数移动平均
            ind.fitness = ind.fitness * 0.7 + fitness * 0.3
            ind.age += 1

            # 生态位适应度
            ind.niche_fitness[ind.niche] = fitness

    async def _evaluate_individual(self, ind: Individual) -> float:
        """评估一个个体的适应度"""
        score = 0.0
        genome = ind.genome

        # 检查基因组完整性
        expected_genes = [
            "semantic_weight", "keyword_weight", "temporal_weight",
            "importance_default", "consolidation_threshold",
            "dedup_threshold", "prune_age_days", "max_memories",
            "learning_rate", "exploration_rate",
        ]
        completeness = sum(1 for g in expected_genes if g in genome) / len(expected_genes)
        score += completeness * 0.3

        # 检查基因值是否在合理范围
        valid_ranges = {
            "semantic_weight": (0.0, 3.0),
            "keyword_weight": (0.0, 2.0),
            "temporal_weight": (0.0, 1.5),
            "importance_default": (0.0, 1.0),
            "consolidation_threshold": (0.5, 0.95),
            "dedup_threshold": (0.8, 1.0),
            "prune_age_days": (7.0, 365.0),
            "max_memories": (1000.0, 1000000.0),
            "learning_rate": (0.001, 0.5),
            "exploration_rate": (0.01, 0.5),
        }
        valid_count = 0
        for key, (lo, hi) in valid_ranges.items():
            val = genome.get(key)
            if val is not None and lo <= val <= hi:
                valid_count += 1
        score += (valid_count / max(len(valid_ranges), 1)) * 0.3

        # DARWIN 分数: 修改成功率
        score += ind.modification_success_rate * 0.2

        # 生态位专精度: 在自己的生态位中表现越好, 分数越高
        niche_fitness = ind.niche_fitness.get(ind.niche, 0.5)
        score += niche_fitness * 0.2

        return min(1.0, max(0.0, score))

    # ---- 变异 ----

    def _mutate(self, ind: Individual, mutation_strength: float = 0.1) -> Individual:
        """对个体的基因进行随机变异"""
        mutant = deepcopy(ind)
        for key in mutant.genome:
            if random.random() < 0.1:  # 10% 变异率
                delta = random.gauss(0, mutation_strength)
                if isinstance(mutant.genome[key], (int, float)):
                    mutant.genome[key] += delta
                    # 确保在合理范围
                    mutant.genome[key] = max(0.0, min(1e6, mutant.genome[key]))
        return mutant

    # ---- 辅助方法 ----

    def _tournament_select(
        self, population: list[Individual], tournament_size: int = 3
    ) -> Individual:
        """锦标赛选择"""
        candidates = random.sample(
            population, min(tournament_size, len(population))
        )
        return max(candidates, key=lambda i: i.fitness)

    def _compute_diversity(self, population: list[Individual]) -> float:
        """计算种群基因多样性"""
        if len(population) < 2:
            return 0.0
        keys = list(population[0].genome.keys())
        if not keys:
            return 0.0
        total_var = 0.0
        for key in keys:
            values = [ind.genome.get(key, 0.0) for ind in population]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            total_var += variance
        return total_var / len(keys)

    @staticmethod
    def _seed_genomes(size: int) -> list[dict[str, Any]]:
        """生成多样化的初始基因组"""
        genomes = []
        for i in range(size):
            g = {
                "semantic_weight": random.uniform(0.5, 2.0),
                "keyword_weight": random.uniform(0.2, 1.5),
                "temporal_weight": random.uniform(0.1, 1.0),
                "importance_default": random.uniform(0.3, 0.8),
                "consolidation_threshold": random.uniform(0.6, 0.9),
                "dedup_threshold": random.uniform(0.85, 0.99),
                "prune_age_days": random.uniform(30.0, 180.0),
                "max_memories": random.uniform(10000, 200000),
                "learning_rate": random.uniform(0.01, 0.3),
                "exploration_rate": random.uniform(0.05, 0.3),
            }
            genomes.append(g)
        return genomes

    def get_population_stats(self) -> dict[str, Any]:
        alive = self.population.get_alive()
        if not alive:
            return {"status": "extinct"}

        return {
            "generation": self.population.generation,
            "population_size": len(alive),
            "avg_fitness": sum(i.fitness for i in alive) / len(alive),
            "best_fitness": max(i.fitness for i in alive),
            "diversity": self._compute_diversity(alive),
            "niches": {
                niche: len(self.population.get_by_niche(niche))
                for niche in {"bug_fixing", "optimization", "implementation", "analysis", "general"}
            },
            "top_performers": [
                {"name": i.name, "fitness": round(i.fitness, 4), "niche": i.niche}
                for i in sorted(alive, key=lambda x: x.fitness, reverse=True)[:5]
            ],
        }
