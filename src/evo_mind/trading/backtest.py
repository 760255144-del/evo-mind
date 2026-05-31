"""回测引擎 — 用历史K线验证策略。

回测 → 计算指标 → 存入MUSE → 进化优化参数
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.trading.exchange import Kline
from evo_mind.types import MemoryType

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    params: dict[str, float] = field(default_factory=dict)


async def backtest_ma_cross(
    store: MemoryStore,
    klines: list[Kline],
    fast_period: int = 7,
    slow_period: int = 25,
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 6.0,
) -> BacktestResult:
    """双均线交叉策略回测。

    BUY:  快线上穿慢线
    SELL: 快线下穿慢线
    """
    if len(klines) < slow_period + 1:
        return BacktestResult()

    # 计算均线
    closes = [k.close for k in klines]
    fast_ma = _sma(closes, fast_period)
    slow_ma = _sma(closes, slow_period)

    position = 0  # 0=空仓, 1=持仓
    entry_price = 0.0
    trades: list[dict] = []
    equity = 100.0  # 百分比计价
    peak = 100.0
    max_dd = 0.0

    for i in range(slow_period, len(klines) - 1):
        if position == 0 and fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
            # 金叉 → 买入
            position = 1
            entry_price = klines[i+1].open  # 下一根K线开盘价入场

        elif position == 1 and fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
            # 死叉 → 卖出
            exit_price = klines[i+1].open
            pnl_pct = (exit_price / entry_price - 1) * 100
            trades.append({"entry": entry_price, "exit": exit_price, "pnl_pct": pnl_pct})
            equity *= (1 + pnl_pct / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
            position = 0

        # 止损止盈
        if position == 1:
            current = klines[i+1].open
            pnl = (current / entry_price - 1) * 100
            if pnl <= -stop_loss_pct or pnl >= take_profit_pct:
                trades.append({"entry": entry_price, "exit": current, "pnl_pct": pnl})
                equity *= (1 + pnl / 100)
                position = 0

    if not trades:
        return BacktestResult(symbol="", total_trades=0)

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    total_pnl = sum(t["pnl_pct"] for t in trades)
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 0

    # 简化的夏普比率
    returns = [t["pnl_pct"] for t in trades]
    avg_ret = sum(returns) / len(returns)
    std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5
    sharpe = (avg_ret / std_ret) if std_ret > 0 else 0.0

    result = BacktestResult(
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        total_pnl_pct=total_pnl,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        profit_factor=(sum(t["pnl_pct"] for t in wins) / abs(sum(t["pnl_pct"] for t in losses)))
        if losses else float("inf"),
        params={"fast": fast_period, "slow": slow_period, "stop": stop_loss_pct, "tp": take_profit_pct},
    )

    # 存储到 MUSE
    await store.record(MemoryCreate(
        memory_type=MemoryType.PROCEDURAL,
        content={
            "type": "backtest_result",
            "strategy": "ma_cross",
            "win_rate": round(win_rate, 4),
            "total_pnl_pct": round(total_pnl, 2),
            "sharpe": round(sharpe, 4),
            "trades": len(trades),
            "params": result.params,
        },
        importance=0.7 if sharpe > 0 else 0.3,
        source="plugin",
        tags=["backtest", "strategy", "ma_cross"],
    ))

    return result


def _sma(data: list[float], period: int) -> list[float]:
    result = [0.0] * len(data)
    for i in range(period - 1, len(data)):
        result[i] = sum(data[i - period + 1:i + 1]) / period
    return result
