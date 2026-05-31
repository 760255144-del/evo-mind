#!/usr/bin/env python3
"""模拟盘交易训练 — GA进化参数直到胜率80%

进化闭环:
  回测 → 评估 → GA变异参数 → 再回测 → 优胜劣汰 → 胜率提升
"""

import asyncio, math, random, sys, json, os
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _now():
    return datetime.now(timezone.utc).isoformat()


class SimulatedMarket:
    """模拟市场 — 不同行情模式"""

    @staticmethod
    def trending(seed: int, length: int = 500) -> list[float]:
        """趋势行情"""
        rng = random.Random(seed)
        price = 65000.0
        trend = rng.uniform(-0.0003, 0.0003)
        prices = []
        for _ in range(length):
            price *= math.exp(rng.gauss(trend, 0.015))
            prices.append(price)
        return prices

    @staticmethod
    def ranging(seed: int, length: int = 500) -> list[float]:
        """震荡行情"""
        rng = random.Random(seed)
        price = 65000.0
        center = price
        prices = []
        for _ in range(length):
            price = center + math.sin(len(prices) * 0.05) * 3000 + rng.gauss(0, 200)
            prices.append(price)
        return prices

    @staticmethod
    def volatile(seed: int, length: int = 500) -> list[float]:
        """高波动行情"""
        rng = random.Random(seed)
        price = 65000.0
        prices = []
        for _ in range(length):
            price *= math.exp(rng.gauss(0, 0.03))
            prices.append(price)
        return prices

    @staticmethod
    def bull(seed: int, length: int = 500) -> list[float]:
        """牛市"""
        rng = random.Random(seed)
        price = 65000.0
        prices = []
        for _ in range(length):
            price *= math.exp(rng.gauss(0.0005, 0.015))
            prices.append(price)
        return prices


class Strategy:
    """可进化的交易策略"""

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.fast = int(p.get("fast", random.randint(3, 15)))
        self.slow = int(p.get("slow", random.randint(16, 50)))
        self.stop_loss = p.get("stop_loss", random.uniform(1.0, 8.0))
        self.take_profit = p.get("take_profit", random.uniform(2.0, 15.0))
        self.min_volume_ratio = p.get("min_volume_ratio", random.uniform(0.5, 2.0))
        self.entry_delay = int(p.get("entry_delay", random.randint(0, 3)))
        self.trailing_stop = p.get("trailing_stop", random.uniform(0, 3.0))
        self.vol_filter = p.get("vol_filter", random.uniform(1.5, 4.0))
        self.fitness = 0.0

    def to_dict(self) -> dict:
        return {
            "fast": self.fast, "slow": self.slow,
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "min_volume_ratio": round(self.min_volume_ratio, 2),
            "entry_delay": self.entry_delay,
            "trailing_stop": round(self.trailing_stop, 2),
            "vol_filter": round(self.vol_filter, 2),
        }

    def mutate(self, rate: float = 0.2) -> "Strategy":
        """随机变异"""
        new = Strategy(self.to_dict())
        if random.random() < rate:
            new.fast = max(2, new.fast + random.randint(-3, 3))
        if random.random() < rate:
            new.slow = max(new.fast + 3, new.slow + random.randint(-8, 8))
        if random.random() < rate:
            new.stop_loss = max(0.5, min(10.0, new.stop_loss + random.gauss(0, 1.0)))
        if random.random() < rate:
            new.take_profit = max(1.0, min(20.0, new.take_profit + random.gauss(0, 2.0)))
        if random.random() < rate:
            new.entry_delay = max(0, min(5, new.entry_delay + random.choice([-1, 0, 1])))
        if random.random() < rate:
            new.trailing_stop = max(0, min(5.0, new.trailing_stop + random.gauss(0, 0.5)))
        if random.random() < rate:
            new.vol_filter = max(0.5, min(6.0, new.vol_filter + random.gauss(0, 0.3)))
        return new

    def crossover(self, other: "Strategy") -> "Strategy":
        """交叉"""
        child = Strategy()
        child.fast = random.choice([self.fast, other.fast])
        child.slow = random.choice([self.slow, other.slow])
        child.stop_loss = random.choice([self.stop_loss, other.stop_loss])
        child.take_profit = random.choice([self.take_profit, other.take_profit])
        child.entry_delay = random.choice([self.entry_delay, other.entry_delay])
        child.trailing_stop = random.choice([self.trailing_stop, other.trailing_stop])
        child.vol_filter = random.choice([self.vol_filter, other.vol_filter])
        return child


def backtest_strategy(strategy: Strategy, prices: list[float]) -> dict:
    """回测单个策略"""
    if len(prices) < strategy.slow + 5:
        return {"win_rate": 0, "trades": 0, "pnl_pct": 0}

    # 计算均线
    fast_ma = [0.0] * len(prices)
    slow_ma = [0.0] * len(prices)
    for i in range(strategy.fast - 1, len(prices)):
        fast_ma[i] = sum(prices[i - strategy.fast + 1:i + 1]) / strategy.fast
    for i in range(strategy.slow - 1, len(prices)):
        slow_ma[i] = sum(prices[i - strategy.slow + 1:i + 1]) / strategy.slow

    position = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    trades = []

    for i in range(strategy.slow, len(prices) - 1):
        price_now = prices[i + 1]

        # 波动率过滤
        if i >= 20:
            returns = [abs(math.log(prices[j] / prices[j-1])) for j in range(i-19, i+1)]
            volatility = sum(returns) / 20 * 100
        else:
            volatility = 2.0
        if volatility > strategy.vol_filter:
            continue

        # 入场信号
        if position == 0 and fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
            # 延迟入场
            entry_idx = min(i + 1 + strategy.entry_delay, len(prices) - 1)
            position = 1
            entry_price = prices[entry_idx]
            highest_since_entry = entry_price

        elif position == 1:
            highest_since_entry = max(highest_since_entry, price_now)
            pnl_pct = (price_now / entry_price - 1) * 100

            # 移动止损
            trail_stop = highest_since_entry * (1 - strategy.trailing_stop / 100)

            # 出场信号
            exit_signal = (
                (fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]) or
                pnl_pct <= -strategy.stop_loss or
                pnl_pct >= strategy.take_profit or
                (strategy.trailing_stop > 0 and price_now < trail_stop)
            )

            if exit_signal:
                trades.append({"pnl_pct": pnl_pct, "win": pnl_pct > 0})
                position = 0

    if not trades:
        return {"win_rate": 0, "trades": 0, "pnl_pct": 0, "max_dd": 0}

    wins = sum(1 for t in trades if t["win"])
    win_rate = wins / len(trades)
    total_pnl = sum(t["pnl_pct"] for t in trades)
    avg_win = sum(t["pnl_pct"] for t in trades if t["win"]) / wins if wins > 0 else 0
    avg_loss = sum(t["pnl_pct"] for t in trades if not t["win"]) / (len(trades) - wins) if len(trades) > wins else 0

    # 回撤
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    for t in trades:
        equity *= (1 + t["pnl_pct"] / 100)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)

    # 综合适应度
    fitness = (win_rate * 0.4 + min(total_pnl / 50, 1.0) * 0.3 +
               min((avg_win / max(abs(avg_loss), 0.01)) / 3, 1.0) * 0.2 +
               (1.0 - min(max_dd / 30, 1.0)) * 0.1)

    return {
        "win_rate": round(win_rate, 4),
        "trades": len(trades),
        "pnl_pct": round(total_pnl, 2),
        "max_dd": round(max_dd, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "fitness": round(fitness, 4),
    }


async def evolve_to_target(target_win_rate: float = 0.80, max_generations: int = 50):
    """进化到目标胜率"""
    print(f"🎯 目标胜率: {target_win_rate:.0%}")
    print(f"🧬 种群大小: 30 · 最大代数: {max_generations}")
    print()

    # 多行情训练
    markets = {
        "trending": SimulatedMarket.trending(1, 500),
        "ranging": SimulatedMarket.ranging(2, 500),
        "volatile": SimulatedMarket.volatile(3, 500),
        "bull": SimulatedMarket.bull(4, 500),
        "trending2": SimulatedMarket.trending(5, 500),
        "ranging2": SimulatedMarket.ranging(6, 500),
    }

    # 初始种群
    population = [Strategy() for _ in range(30)]
    best_ever = None
    best_ever_score = 0
    history = []

    for gen in range(max_generations):
        # 评估
        for s in population:
            scores = []
            for name, prices in markets.items():
                r = backtest_strategy(s, prices)
                scores.append(r.get("fitness", 0))
            s.fitness = sum(scores) / len(scores)

        # 排序
        population.sort(key=lambda s: s.fitness, reverse=True)
        best = population[0]

        # 在单一市场详细评估最佳
        best_detail = backtest_strategy(best, markets["trending"])
        history.append(best_detail)

        if best.fitness > best_ever_score:
            best_ever = Strategy(best.to_dict())
            best_ever_score = best.fitness
            best_ever.fitness = best.fitness

        wr = best_detail["win_rate"]
        print(f"  第{gen+1:2d}代: 胜率={wr:.1%} PnL={best_detail['pnl_pct']:+.1f}% "
              f"交易={best_detail['trades']}笔 回撤={best_detail['max_dd']:.1f}% "
              f"fast={best.fast} slow={best.slow} sl={best.stop_loss:.1f} tp={best.take_profit:.1f}")

        if wr >= target_win_rate:
            print(f"\n🏆 第{gen+1}代达到目标胜率 {wr:.1%}!")
            break

        # 精英保留 + 交叉 + 变异
        elite_count = 5
        new_pop = [Strategy(s.to_dict()) for s in population[:elite_count]]

        while len(new_pop) < 30:
            p1 = random.choice(population[:15])  # 从前半选
            p2 = random.choice(population[:15])
            child = p1.crossover(p2)
            child = child.mutate(0.3)
            new_pop.append(child)

        population = new_pop

    # 最终评估
    print(f"\n📊 最终结果:")
    best = best_ever or population[0]
    print(f"  参数: fast={best.fast} slow={best.slow} sl={best.stop_loss:.1f}% tp={best.take_profit:.1f}%")

    total_win_rate = 0
    for name, prices in markets.items():
        r = backtest_strategy(best, prices)
        total_win_rate += r["win_rate"]
        print(f"  {name:12s}: 胜率={r['win_rate']:.1%} PnL={r['pnl_pct']:+.1f}% 交易={r['trades']}")

    avg_wr = total_win_rate / len(markets)
    print(f"\n  平均胜率: {avg_wr:.1%}")
    print(f"  最佳参数: {json.dumps(best.to_dict())}")

    return best, avg_wr, history


if __name__ == "__main__":
    best, wr, history = asyncio.run(evolve_to_target(0.80, 50))

    # 保存结果
    result = {
        "timestamp": _now(),
        "target_win_rate": 0.80,
        "achieved_win_rate": wr,
        "best_params": best.to_dict(),
        "history": history[-5:],
    }

    Path("evolution_data").mkdir(exist_ok=True)
    json.dump(result, open("evolution_data/simulate_result.json", "w"), indent=2)
    print(f"\n💾 结果已保存到 evolution_data/simulate_result.json")
