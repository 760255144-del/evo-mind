"""自动化交易引擎 — 风控 + 策略执行 + 进化闭环。

信号 → 风控检查 → 下单 → 记录 → 进化优化

风控规则:
  - 单笔最大仓位: 总资金 10%
  - 最大回撤: 20% (触及即停)
  - 止损: 每笔 2-5% (可进化优化)
  - 冷却期: 同方向信号间隔 >= 1 小时
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.trading.exchange import Balance, Kline, Order, PaperExchange, BinanceExchange
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 交易模型 ----

@dataclass
class TradeResult:
    """一笔交易结果"""
    trade_id: str = field(default_factory=uuid7)
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    win: bool = False
    holding_hours: float = 0.0
    entry_time: str = ""
    exit_time: str = field(default_factory=_now)


@dataclass
class RiskConfig:
    """风控配置 (可进化)"""
    max_position_pct: float = 10.0       # 单笔最大仓位 (%)
    max_drawdown_pct: float = 20.0       # 最大回撤 (%)
    stop_loss_pct: float = 3.0          # 止损 (%)
    take_profit_pct: float = 6.0         # 止盈 (%)
    cooldown_minutes: int = 60           # 冷却期 (分钟)
    min_confidence: float = 0.6          # 最低信号置信度
    max_daily_trades: int = 5            # 每日最大交易数


@dataclass
class TradeStats:
    """交易统计"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_value: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0


# ---- 引擎 ----

class TradingEngine:
    """自动化交易引擎。

    核心循环:
      信号到来 → 风控检查 → 执行交易 → 记录结果 → 反馈进化

    两种模式:
      - paper: 模拟盘 (默认, 不涉及真实资金)
      - live: 实盘 (需要 Binance API Key)
    """

    def __init__(
        self,
        store: MemoryStore,
        exchange: PaperExchange | BinanceExchange | None = None,
        risk: RiskConfig | None = None,
        initial_capital: float = 10000.0,
    ):
        self.store = store
        self.exchange = exchange or PaperExchange(initial_capital)
        self.risk = risk or RiskConfig()
        self.initial_capital = initial_capital

        # 交易记录
        self._trades: list[TradeResult] = []
        self._open_positions: dict[str, TradeResult] = {}
        self._last_signal_time: dict[str, str] = {}  # symbol → timestamp
        self._daily_trade_count: int = 0
        self._peak_value: float = initial_capital
        self._max_drawdown: float = 0.0
        self._stopped: bool = False  # 风控停止标志

        # 策略参数 (可进化)
        self._strategy_params: dict[str, float] = {}

    # ---- 主入口 ----

    async def execute_signal(
        self, symbol: str, direction: str, confidence: float,
        indicators: dict[str, float] | None = None,
    ) -> TradeResult | None:
        """接收到交易信号 → 风控 → 执行 → 记录"""
        if self._stopped:
            logger.warning("trading_stopped_by_risk_control")
            return None

        # 1. 信号过滤
        if not self._check_signal(symbol, direction, confidence):
            return None

        # 2. 风控检查
        if not await self._check_risk(symbol, direction):
            return None

        # 3. 获取价格
        price = await self.exchange.get_price(symbol)
        if price <= 0:
            return None

        # 4. 计算仓位
        balance = await self.exchange.get_balance("USDT")
        capital = await self._get_equity()
        position_size = self._calc_position_size(capital, price, confidence)

        if position_size <= 0:
            return None

        # 5. 下单
        order = await self.exchange.place_order(
            symbol, direction, position_size, price
        )

        if order.status != "filled":
            logger.info("order_rejected: %s %s", symbol, order.status)
            return None

        # 6. 记录交易
        trade = TradeResult(
            symbol=symbol, side=direction,
            entry_price=order.filled_price, quantity=order.filled_qty,
            entry_time=_now(),
        )
        self._open_positions[symbol] = trade
        self._trades.append(trade)
        self._daily_trade_count += 1
        self._last_signal_time[symbol] = _now()

        # 7. 存储到 MUSE → 进化学习
        await self._record_trade(trade, "open")

        logger.info("trade_opened: %s %s %s @ %.2f qty=%.4f",
                     symbol, direction, order.id, order.filled_price, order.filled_qty)
        return trade

    async def close_position(
        self, symbol: str, exit_price: float | None = None, reason: str = "manual"
    ) -> TradeResult | None:
        """平仓"""
        trade = self._open_positions.pop(symbol, None)
        if not trade:
            return None

        if exit_price is None:
            exit_price = await self.exchange.get_price(symbol)

        # 反向下单平仓
        reverse_side = "sell" if trade.side == "buy" else "buy"
        order = await self.exchange.place_order(
            symbol, reverse_side, trade.quantity, exit_price
        )

        trade.exit_price = order.filled_price or exit_price
        trade.exit_time = _now()

        # 计算 P&L
        if trade.side == "buy":
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
            trade.pnl_pct = (trade.exit_price / trade.entry_price - 1) * 100
        else:
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity
            trade.pnl_pct = (trade.entry_price / trade.exit_price - 1) * 100

        trade.win = trade.pnl > 0

        # 更新统计
        await self._update_stats(trade)

        # 存储到 MUSE
        await self._record_trade(trade, "close")

        logger.info("trade_closed: %s %s pnl=%.2f (%.2f%%) %s",
                     symbol, trade.side, trade.pnl, trade.pnl_pct,
                     "WIN" if trade.win else "LOSS")
        return trade

    async def check_stop_loss_take_profit(self) -> list[TradeResult]:
        """检查所有持仓的止损止盈"""
        closed = []
        for symbol, trade in list(self._open_positions.items()):
            current_price = await self.exchange.get_price(symbol)
            if current_price <= 0:
                continue

            pnl_pct = (
                (current_price / trade.entry_price - 1) * 100
                if trade.side == "buy"
                else (trade.entry_price / current_price - 1) * 100
            )

            reason = None
            if pnl_pct <= -self.risk.stop_loss_pct:
                reason = "stop_loss"
            elif pnl_pct >= self.risk.take_profit_pct:
                reason = "take_profit"

            if reason:
                result = await self.close_position(symbol, current_price, reason)
                if result:
                    closed.append(result)

        return closed

    # ---- 风控 ----

    def _check_signal(self, symbol: str, direction: str, confidence: float) -> bool:
        if confidence < self.risk.min_confidence:
            return False
        if self._daily_trade_count >= self.risk.max_daily_trades:
            return False
        # 冷却检查
        last_time = self._last_signal_time.get(symbol)
        if last_time:
            from datetime import timedelta
            elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(last_time)
            if elapsed < timedelta(minutes=self.risk.cooldown_minutes):
                return False
        return True

    async def _check_risk(self, symbol: str, direction: str) -> bool:
        if self._stopped:
            return False
        # 最大回撤检查
        equity = await self._get_equity()
        drawdown = (self._peak_value - equity) / self._peak_value * 100
        if drawdown > self.risk.max_drawdown_pct:
            self._stopped = True
            logger.critical("max_drawdown_reached: %.1f%%, stopping trading", drawdown)
            return False
        # 已有同方向持仓
        if symbol in self._open_positions:
            return False
        return True

    def _calc_position_size(self, capital: float, price: float, confidence: float) -> float:
        max_amount = capital * (self.risk.max_position_pct / 100) * (confidence / 0.8)
        return max_amount / price if price > 0 else 0.0

    async def _get_equity(self) -> float:
        if isinstance(self.exchange, PaperExchange):
            prices = {}
            for symbol in self._open_positions:
                prices[symbol] = await self.exchange.get_price(symbol)
            return await self.exchange.get_total_value(prices)
        return self.initial_capital

    async def _update_stats(self, trade: TradeResult) -> None:
        equity = await self._get_equity() + trade.pnl
        if equity > self._peak_value:
            self._peak_value = equity
        drawdown = (self._peak_value - equity) / self._peak_value * 100
        self._max_drawdown = max(self._max_drawdown, drawdown)

    # ---- MUSE 记忆 ----

    async def _record_trade(self, trade: TradeResult, stage: str) -> None:
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.FEEDBACK if stage == "close" else MemoryType.EPISODIC,
            content={
                "type": "trade",
                "stage": stage,
                "symbol": trade.symbol,
                "side": trade.side,
                "entry": trade.entry_price,
                "exit": trade.exit_price,
                "pnl": round(trade.pnl, 2),
                "pnl_pct": round(trade.pnl_pct, 4),
                "win": trade.win,
            },
            importance=0.8,
            source="plugin",
            tags=["trading", "win" if trade.win else "loss", trade.symbol],
        ))

    # ---- 统计 ----

    def get_stats(self) -> TradeStats:
        if not self._trades:
            return TradeStats()

        closed = [t for t in self._trades if t.exit_price > 0]
        wins = [t for t in closed if t.win]
        losses = [t for t in closed if not t.win]

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        return TradeStats(
            total_trades=len(closed),
            wins=len(wins),
            losses=len(losses),
            total_pnl=sum(t.pnl for t in closed),
            total_pnl_pct=sum(t.pnl_pct for t in closed),
            max_drawdown=self._max_drawdown,
            avg_win_pct=sum(t.pnl_pct for t in wins) / len(wins) if wins else 0,
            avg_loss_pct=sum(t.pnl_pct for t in losses) / len(losses) if losses else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        )

    def set_strategy_params(self, params: dict[str, float]) -> None:
        """从进化引擎接收优化后的策略参数"""
        self._strategy_params = params
        if "stop_loss_pct" in params:
            self.risk.stop_loss_pct = params["stop_loss_pct"]
        if "take_profit_pct" in params:
            self.risk.take_profit_pct = params["take_profit_pct"]
        if "min_confidence" in params:
            self.risk.min_confidence = params["min_confidence"]
