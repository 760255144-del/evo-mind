"""量化交易进化 — BeeQuant/TradingView 信号接入 + 策略优化。

信号 → MUSE记忆 → 进化优化 → 改进参数 → 更高胜率
"""

from evo_mind.quant.connector import QuantConnector, TradingSignal, TradeOutcome
from evo_mind.quant.optimizer import QuantOptimizer, StrategyGenome

__all__ = [
    "QuantConnector",
    "TradingSignal",
    "TradeOutcome",
    "QuantOptimizer",
    "StrategyGenome",
]
