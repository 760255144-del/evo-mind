"""自动化交易 — 信号→风控→执行→回测→进化优化→挣钱"""

from evo_mind.trading.exchange import PaperExchange, BinanceExchange, Order, Balance, Kline
from evo_mind.trading.engine import TradingEngine, TradeResult, RiskConfig, TradeStats
from evo_mind.trading.backtest import backtest_ma_cross, BacktestResult

__all__ = [
    "PaperExchange", "BinanceExchange", "Order", "Balance", "Kline",
    "TradingEngine", "TradeResult", "RiskConfig", "TradeStats",
    "backtest_ma_cross", "BacktestResult",
]
