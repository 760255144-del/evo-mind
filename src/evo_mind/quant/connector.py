"""BeeQuant/TradingView 量化信号连接器。

接入方式:
  1. Webhook: TradingView alert → POST to local endpoint
  2. CSV import: 从 BeeQuant 导出的历史信号
  3. Manual: 通过 CLI 手动录入信号

每条信号存储为 MUSE 记忆，用于进化优化。
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- 信号模型 ----

@dataclass
class TradingSignal:
    """一条量化交易信号 (BeeQuant/TradingView 格式)"""
    id: str = field(default_factory=uuid7)
    symbol: str = ""             # BTCUSDT, ETHUSDT, ...
    timeframe: str = "1h"        # 1m, 5m, 15m, 1h, 4h, 1d
    direction: str = "long"      # long | short | neutral
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.5      # 0.0-1.0
    strategy_name: str = ""      # 策略名称 (Hive Bars, MA Cross, ...)
    indicators: dict[str, float] = field(default_factory=dict)  # RSI, MACD, ...
    timestamp: str = field(default_factory=_now)
    source: str = "manual"       # webhook | csv | manual


@dataclass
class TradeOutcome:
    """交易结果"""
    signal_id: str = ""
    symbol: str = ""
    entry: float = 0.0
    exit: float = 0.0
    pnl_pct: float = 0.0
    pnl_absolute: float = 0.0
    win: bool = False
    holding_time_hours: float = 0.0
    exit_reason: str = ""        # take_profit | stop_loss | manual | timeout
    timestamp: str = field(default_factory=_now)


# ---- 连接器 ----

class QuantConnector:
    """BeeQuant/TradingView 信号连接器。

    将外部量化信号接入 evo-mind 进化系统:
      信号 → MUSE记忆 → 进化优化 → 改进策略参数
    """

    # TradingView → evo-mind 字段映射
    TV_FIELD_MAP = {
        "ticker": "symbol",
        "exchange": "symbol",
        "price": "entry_price",
        "stop": "stop_loss",
        "limit": "take_profit",
        "interval": "timeframe",
    }

    def __init__(self, store: MemoryStore):
        self.store = store
        self._signals: dict[str, TradingSignal] = {}
        self._outcomes: dict[str, TradeOutcome] = {}

    # ---- 信号接入 ----

    async def ingest_webhook(self, payload: dict[str, Any]) -> TradingSignal:
        """从 TradingView Webhook JSON 接入信号"""
        signal = TradingSignal(
            symbol=payload.get("ticker", payload.get("symbol", "")),
            timeframe=payload.get("interval", payload.get("timeframe", "1h")),
            direction=payload.get("direction", "long"),
            entry_price=float(payload.get("price", payload.get("entry_price", 0))),
            stop_loss=float(payload.get("stop", payload.get("stop_loss", 0))),
            take_profit=float(payload.get("limit", payload.get("take_profit", 0))),
            confidence=float(payload.get("confidence", 0.5)),
            strategy_name=payload.get("strategy_name", payload.get("strategy", "beequant")),
            indicators={
                k: float(v) for k, v in payload.items()
                if k.lower() in ("rsi", "macd", "ema", "sma", "volume", "bb_upper", "bb_lower")
                and isinstance(v, (int, float, str))
            },
            source="webhook",
        )
        await self._store_signal(signal)
        return signal

    async def import_csv(self, path: str | Path) -> list[TradingSignal]:
        """从 CSV 文件批量导入历史信号"""
        signals = []
        path = Path(path)

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                signal = TradingSignal(
                    symbol=row.get("symbol", row.get("ticker", "")),
                    timeframe=row.get("timeframe", "1h"),
                    direction=row.get("direction", "long"),
                    entry_price=float(row.get("entry_price", 0)),
                    stop_loss=float(row.get("stop_loss", 0)),
                    take_profit=float(row.get("take_profit", 0)),
                    confidence=float(row.get("confidence", 0.5)),
                    strategy_name=row.get("strategy_name", "beequant"),
                    source="csv",
                )
                signals.append(signal)

        # 批量存储
        for signal in signals:
            await self._store_signal(signal)

        logger.info("csv_imported", count=len(signals), path=str(path))
        return signals

    async def add_manual(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop: float = 0,
        target: float = 0,
        confidence: float = 0.5,
        strategy: str = "manual",
    ) -> TradingSignal:
        """手动录入信号"""
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            confidence=confidence,
            strategy_name=strategy,
            source="manual",
        )
        await self._store_signal(signal)
        return signal

    # ---- 交易结果记录 ----

    async def record_outcome(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: str = "manual",
        holding_hours: float = 0,
    ) -> TradeOutcome | None:
        """记录交易结果，用于反馈学习"""
        signal = self._signals.get(signal_id)
        if not signal:
            return None

        pnl_pct = (exit_price - signal.entry_price) / signal.entry_price * 100
        if signal.direction == "short":
            pnl_pct = -pnl_pct

        outcome = TradeOutcome(
            signal_id=signal_id,
            symbol=signal.symbol,
            entry=signal.entry_price,
            exit=exit_price,
            pnl_pct=pnl_pct,
            pnl_absolute=pnl_pct * signal.entry_price / 100,
            win=pnl_pct > 0,
            holding_time_hours=holding_hours,
            exit_reason=exit_reason,
        )
        self._outcomes[signal_id] = outcome

        # 存储为反馈记忆 → 进化学习
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.FEEDBACK,
            content={
                "type": "trade_outcome",
                "symbol": signal.symbol,
                "strategy": signal.strategy_name,
                "direction": signal.direction,
                "entry": signal.entry_price,
                "exit": exit_price,
                "pnl_pct": round(pnl_pct, 4),
                "win": outcome.win,
                "confidence": signal.confidence,
                "indicators": signal.indicators,
            },
            importance=0.8 if outcome.win else 0.4,
            source="plugin",
            tags=["quant", "trade", "win" if outcome.win else "loss", signal.strategy_name],
        ))

        return outcome

    # ---- 统计 ----

    def get_stats(self) -> dict[str, Any]:
        """获取交易统计"""
        if not self._outcomes:
            return {"trades": 0}

        wins = sum(1 for o in self._outcomes.values() if o.win)
        total = len(self._outcomes)
        pnls = [o.pnl_pct for o in self._outcomes.values()]
        avg_pnl = sum(pnls) / len(pnls)

        by_strategy: dict[str, dict] = {}
        for o in self._outcomes.values():
            sig = self._signals.get(o.signal_id)
            strat = sig.strategy_name if sig else "unknown"
            if strat not in by_strategy:
                by_strategy[strat] = {"wins": 0, "total": 0, "pnl_sum": 0.0}
            by_strategy[strat]["total"] += 1
            if o.win:
                by_strategy[strat]["wins"] += 1
            by_strategy[strat]["pnl_sum"] += o.pnl_pct

        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": f"{wins/total:.1%}" if total > 0 else "N/A",
            "avg_pnl_pct": round(avg_pnl, 4),
            "total_pnl_pct": round(sum(pnls), 4),
            "by_strategy": {
                name: {
                    "win_rate": f"{s['wins']/s['total']:.1%}",
                    "avg_pnl": round(s["pnl_sum"]/s["total"], 4),
                    "trades": s["total"],
                }
                for name, s in by_strategy.items()
            },
        }

    async def _store_signal(self, signal: TradingSignal) -> None:
        """将信号存储为 MUSE 记忆"""
        self._signals[signal.id] = signal

        await self.store.record(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={
                "type": "trading_signal",
                "signal_id": signal.id,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "strategy": signal.strategy_name,
                "indicators": signal.indicators,
            },
            importance=signal.confidence,
            source="plugin",
            tags=["quant", "signal", signal.strategy_name, signal.symbol],
        ))
