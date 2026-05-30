"""量化策略进化优化器 — 用遗传算法调优交易参数。

将每个策略的参数编码为基因组，通过 P&L 反馈驱动进化。
与 evo-mind 的 EvolutionaryEngine 集成。

优化维度:
  - 入场阈值 (RSI 超买/超卖, MACD 交叉, ...)
  - 止损/止盈比例
  - 仓位大小
  - 时间框架权重
  - 信号置信度阈值
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.evolutionary.genome import Gene, Genome
from evo_mind.evolutionary.operators import (
    blend_crossover,
    elitist_selection,
    gaussian_mutation,
    population_diversity,
    tournament_selection,
)
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 策略基因组 ----

@dataclass
class StrategyGenome(Genome):
    """交易策略的可进化基因组"""

    @staticmethod
    def _default_genes() -> dict[str, Gene]:
        return {
            # 入场参数
            "rsi_oversold": Gene("rsi_oversold", 30.0, 10.0, 40.0, 0.05, 2.0),
            "rsi_overbought": Gene("rsi_overbought", 70.0, 60.0, 90.0, 0.05, 2.0),
            "macd_threshold": Gene("macd_threshold", 0.0, -5.0, 5.0, 0.1, 0.5),
            # 风控参数
            "stop_loss_pct": Gene("stop_loss_pct", 2.0, 0.5, 10.0, 0.1, 0.5),
            "take_profit_pct": Gene("take_profit_pct", 4.0, 1.0, 20.0, 0.1, 1.0),
            "risk_reward_ratio": Gene("risk_reward_ratio", 2.0, 1.0, 5.0, 0.1, 0.2),
            "max_position_pct": Gene("max_position_pct", 10.0, 1.0, 50.0, 0.1, 2.0),
            # 信号过滤
            "min_confidence": Gene("min_confidence", 0.6, 0.3, 0.9, 0.05, 0.05),
            "signal_cooldown_hours": Gene("signal_cooldown_hours", 4.0, 1.0, 48.0, 0.1, 2.0),
            # 时间框架权重
            "weight_1h": Gene("weight_1h", 0.3, 0.0, 1.0, 0.1, 0.1),
            "weight_4h": Gene("weight_4h", 0.4, 0.0, 1.0, 0.1, 0.1),
            "weight_1d": Gene("weight_1d", 0.3, 0.0, 1.0, 0.1, 0.1),
        }


# ---- 优化器 ----

class QuantOptimizer:
    """量化策略进化优化器。

    用历史交易 P&L 作为适应度函数，
    进化最佳策略参数。

    流程:
      P&L 数据 → 适应度评估 → 遗传进化 → 最优参数
    """

    def __init__(
        self,
        store: MemoryStore,
        population_size: int = 30,
        max_generations: int = 20,
        elite_count: int = 3,
    ):
        self.store = store
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_count = elite_count

        self._population: list[StrategyGenome] = []
        self._best: StrategyGenome | None = None
        self._fitness_history: list[float] = []
        self._trade_history: list[dict[str, Any]] = []

    async def evolve(self) -> dict[str, float]:
        """运行遗传进化，返回最优参数"""
        # 加载历史交易数据作为适应度基础
        await self._load_trade_history()

        if not self._trade_history:
            logger.warning("no_trade_history_for_evolution")
            return StrategyGenome().get_params()

        # 初始化种群
        self._population = [StrategyGenome() for _ in range(self.population_size)]
        for g in self._population:
            for gene in g.genes.values():
                gene.value = random.uniform(gene.min_value, gene.max_value)
            g.clamp()

        best_fitness = -float("inf")
        stale = 0

        for generation in range(self.max_generations):
            # 评估适应度
            for genome in self._population:
                genome.fitness = self._evaluate_fitness(genome)

            # 排序
            self._population.sort(key=lambda g: g.fitness, reverse=True)
            current_best = self._population[0]
            avg_fit = sum(g.fitness for g in self._population) / len(self._population)

            self._fitness_history.append(current_best.fitness)

            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                self._best = current_best
                stale = 0
            else:
                stale += 1

            if stale >= 5:
                break

            # 创建下一代
            new_pop = elitist_selection(self._population, self.elite_count)
            new_pop = [StrategyGenome() for _ in range(self.elite_count)]
            for i in range(self.elite_count):
                new_pop[i].genes = {
                    k: Gene(k, self._population[i].genes[k].value,
                            self._population[i].genes[k].min_value,
                            self._population[i].genes[k].max_value)
                    for k in self._population[i].genes
                }

            while len(new_pop) < self.population_size:
                p1 = tournament_selection(self._population, 3)
                p2 = tournament_selection(self._population, 3)
                if hasattr(p1, 'id') and hasattr(p2, 'id') and p1.id == p2.id:
                    continue
                c1, c2 = blend_crossover(p1, p2)
                c1 = gaussian_mutation(c1)
                c2 = gaussian_mutation(c2)
                new_pop.append(c1)
                if len(new_pop) < self.population_size:
                    new_pop.append(c2)

            self._population = new_pop[:self.population_size]

        # 记录最优
        if self._best:
            await self._record_result(self._best)
            return self._best.get_params()
        return StrategyGenome().get_params()

    def _evaluate_fitness(self, genome: StrategyGenome) -> float:
        """根据历史交易评估策略适应度。

        模拟: 用这个策略的参数去过滤历史信号，
        看哪些交易会被执行，计算总 P&L。
        """
        params = genome.get_params()
        total_pnl = 0.0
        trades_taken = 0
        wins = 0

        for trade in self._trade_history:
            # 置信度过滤
            if trade.get("confidence", 0) < params.get("min_confidence", 0.5):
                continue

            # RSI 范围检查
            rsi = trade.get("indicators", {}).get("rsi", 50)
            if trade.get("direction") == "long" and rsi > params.get("rsi_overbought", 70):
                continue

            trades_taken += 1
            pnl = trade.get("pnl_pct", 0)
            total_pnl += pnl
            if pnl > 0:
                wins += 1

        if trades_taken == 0:
            return 0.0

        win_rate = wins / trades_taken
        # 适应度 = 总收益 × 胜率 × 对数(交易数)
        fitness = total_pnl * (0.5 + 0.5 * win_rate) * math.log(trades_taken + 1)
        return fitness

    async def _load_trade_history(self) -> None:
        """从 MUSE 记忆加载历史交易"""
        try:
            rows = await self.store.db.fetch_all(
                """SELECT content_json FROM memories
                   WHERE json_extract(content_json, '$.type') = 'trade_outcome'
                     AND deleted_at IS NULL
                   ORDER BY created_at DESC LIMIT 500"""
            )
            for row in rows:
                content = json.loads(row["content_json"])
                self._trade_history.append(content)
        except Exception as e:
            logger.warning("trade_history_load_failed: %s", e)

    async def _record_result(self, best: StrategyGenome) -> None:
        """记录进化结果"""
        params = best.get_params()
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "quant_optimization_result",
                "best_params": {k: round(v, 4) for k, v in params.items()},
                "best_fitness": round(best.fitness, 4),
                "generations": len(self._fitness_history),
                "fitness_trajectory": [round(f, 4) for f in self._fitness_history[-10:]],
            },
            importance=0.9,
            source="plugin",
            tags=["quant", "optimization", "strategy"],
        ))

    def get_best_params(self) -> dict[str, float]:
        if self._best:
            return self._best.get_params()
        return StrategyGenome().get_params()


import json  # noqa: E402 — needed at runtime for _load_trade_history
