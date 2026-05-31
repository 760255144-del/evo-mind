"""交易所连接器 — Binance/OKX API 接入。

支持:
  - 实盘: Binance REST API
  - 模拟: Paper trading (默认, 安全)
  - 数据: K线获取, 余额查询
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Order:
    """订单"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = "BTCUSDT"
    side: str = "buy"           # buy | sell
    order_type: str = "market"  # market | limit
    quantity: float = 0.0
    price: float = 0.0
    status: str = "pending"     # pending | filled | cancelled | rejected
    filled_price: float = 0.0
    filled_qty: float = 0.0
    timestamp: str = field(default_factory=_now)


@dataclass
class Balance:
    """账户余额"""
    asset: str = ""
    free: float = 0.0
    locked: float = 0.0


@dataclass
class Kline:
    """K线数据"""
    open_time: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


class PaperExchange:
    """模拟交易所 — 用于安全测试。

    特点:
    - 模拟订单成交 (市场价即时成交)
    - 虚拟余额
    - 完全无风险
    """

    def __init__(self, initial_balance: float = 10000.0):
        self._balances: dict[str, float] = {"USDT": initial_balance}
        self._orders: dict[str, Order] = {}
        self._prices: dict[str, float] = {}  # 模拟价格

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    async def get_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    async def get_balance(self, asset: str = "USDT") -> Balance:
        return Balance(asset=asset, free=self._balances.get(asset, 0.0))

    async def place_order(
        self, symbol: str, side: str, quantity: float, price: float = 0.0
    ) -> Order:
        """下单 (模拟即时成交)"""
        order = Order(symbol=symbol, side=side, quantity=quantity, price=price)

        base, quote = self._split_symbol(symbol)
        current_price = await self.get_price(symbol) or price

        if side == "buy":
            cost = quantity * current_price
            if self._balances.get(quote, 0) < cost:
                order.status = "rejected"
                self._orders[order.id] = order
                return order
            self._balances[quote] = self._balances.get(quote, 0) - cost
            self._balances[base] = self._balances.get(base, 0) + quantity
        else:
            if self._balances.get(base, 0) < quantity:
                order.status = "rejected"
                self._orders[order.id] = order
                return order
            self._balances[base] = self._balances.get(base, 0) - quantity
            self._balances[quote] = self._balances.get(quote, 0) + quantity * current_price

        order.status = "filled"
        order.filled_price = current_price
        order.filled_qty = quantity
        self._orders[order.id] = order
        return order

    async def get_total_value(self, prices: dict[str, float] | None = None) -> float:
        """计算账户总价值 (USDT)"""
        prices = prices or self._prices
        total = self._balances.get("USDT", 0.0)
        for asset, amount in self._balances.items():
            if asset != "USDT" and amount > 0:
                total += amount * prices.get(f"{asset}USDT", 0)
        return total

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str]:
        for quote in ["USDT", "USDC", "BUSD", "BTC", "ETH"]:
            if symbol.endswith(quote):
                return symbol[:-len(quote)], quote
        return symbol[:-4], symbol[-4:]


class BinanceExchange:
    """Binance 实盘交易所。

    需要 API Key + Secret (从 Binance 账户获取)。
    """

    BASE_URL = "https://api.binance.com"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self._paper = PaperExchange()  # fallback for when no API keys

    @property
    def is_live(self) -> bool:
        return bool(self.api_key and self.api_secret)

    async def get_price(self, symbol: str) -> float:
        """获取实时价格"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.BASE_URL}/api/v3/ticker/price?symbol={symbol}"
                async with session.get(url) as resp:
                    data = await resp.json()
                    return float(data.get("price", 0))
        except Exception:
            return self._paper.get_price(symbol)

    async def get_klines(
        self, symbol: str, interval: str = "1h", limit: int = 100
    ) -> list[Kline]:
        """获取K线数据"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.BASE_URL}/api/v3/klines"
                params = {"symbol": symbol, "interval": interval, "limit": limit}
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                    return [
                        Kline(
                            open_time=k[0], open=float(k[1]), high=float(k[2]),
                            low=float(k[3]), close=float(k[4]), volume=float(k[5]),
                        )
                        for k in data
                    ]
        except Exception as e:
            logger.warning("klines_fetch_failed: %s", e)
            return []

    async def get_balance(self, asset: str = "USDT") -> Balance:
        """查询余额"""
        if not self.is_live:
            return self._paper.get_balance(asset)

        import aiohttp
        try:
            params = {"timestamp": int(time.time() * 1000)}
            params["signature"] = self._sign(params)
            headers = {"X-MBX-APIKEY": self.api_key}

            async with aiohttp.ClientSession() as session:
                url = f"{self.BASE_URL}/api/v3/account"
                async with session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    for b in data.get("balances", []):
                        if b["asset"] == asset:
                            return Balance(
                                asset=asset,
                                free=float(b["free"]),
                                locked=float(b["locked"]),
                            )
        except Exception as e:
            logger.warning("balance_fetch_failed: %s", e)
        return Balance(asset=asset)

    async def place_order(
        self, symbol: str, side: str, quantity: float, price: float = 0.0
    ) -> Order:
        """下单"""
        if not self.is_live:
            return await self._paper.place_order(symbol, side, quantity, price)

        import aiohttp
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
            "timestamp": int(time.time() * 1000),
        }
        params["signature"] = self._sign(params)

        try:
            headers = {"X-MBX-APIKEY": self.api_key}
            async with aiohttp.ClientSession() as session:
                url = f"{self.BASE_URL}/api/v3/order"
                async with session.post(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    order = Order(
                        id=str(data.get("orderId", "")),
                        symbol=symbol, side=side, quantity=quantity,
                        status=data.get("status", "rejected"),
                        filled_price=float(data.get("fills", [{}])[0].get("price", 0)),
                        filled_qty=float(data.get("executedQty", 0)),
                    )
                    return order
        except Exception as e:
            logger.error("order_failed: %s", e)
            order = Order(symbol=symbol, side=side, quantity=quantity, status="rejected")
            return order

    def _sign(self, params: dict) -> str:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
