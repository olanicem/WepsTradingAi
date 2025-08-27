#!/usr/bin/env python3
# ==============================================================
# ⚡️ WEPS MT5 Live Trade Executor — Final Production Version
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Places real MT5 trades based on reflex policy decision
#   - Spiral-aware, with SL/TP, volume, phase tagging, and logging
# ==============================================================

import MetaTrader5 as mt5
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("WEPS.MT5TradeExecutor")

# CONFIG
DEFAULT_VOLUME = 0.5  # Lot size (for FTMO 10k)
MAGIC_NUMBER = 309001
DEVIATION = 5  # Max price slippage

class MT5TradeExecutor:
    def __init__(self, symbol: str = "EURUSD", volume: float = DEFAULT_VOLUME):
        self.symbol = symbol
        self.volume = volume
        self._connect_mt5()

    def _connect_mt5(self):
        if not mt5.initialize():
            raise RuntimeError(f"[MT5] ❌ Initialization failed: {mt5.last_error()}")
        logger.info("[MT5] ✅ Terminal connected.")

    def _get_symbol_info(self):
        info = mt5.symbol_info(self.symbol)
        if info is None or not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                raise RuntimeError(f"[MT5] ❌ Failed to select symbol: {self.symbol}")
        return info

    def place_trade(self, decision: Dict[str, Any]):
        action = decision.get("action", "hold")
        if action == "hold":
            logger.info("[MT5] 🚫 No trade: Reflex action is HOLD.")
            return {"status": "no_trade", "reason": "hold"}

        self._get_symbol_info()
        price = mt5.symbol_info_tick(self.symbol).ask if action == "buy" else mt5.symbol_info_tick(self.symbol).bid

        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        sl = decision.get("sl")
        tp = decision.get("tp")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": f"WEPS-{decision.get('phase')}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[MT5] ❌ Trade failed: {result.retcode} | {result.comment}")
            return {
                "status": "failed",
                "retcode": result.retcode,
                "comment": result.comment
            }

        logger.info(f"[MT5] ✅ Trade Executed: {action.upper()} {self.symbol} | Entry: {price} | SL: {sl} | TP: {tp}")
        return {
            "status": "success",
            "order_id": result.order,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "symbol": self.symbol,
            "action": action,
            "timestamp": str(datetime.utcnow())
        }

    def shutdown(self):
        mt5.shutdown()
        logger.info("[MT5] 🔌 Disconnected from terminal.")


__all__ = ["MT5TradeExecutor"]
