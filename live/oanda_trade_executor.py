#!/usr/bin/env python3
# ================================================================
# OANDA Trade Executor — REST v20 Live Order Execution
# Author: Ola Bode (WEPS Creator)
# ================================================================

import logging
import requests
import uuid

OANDA_API_URL = "https://api-fxtrade.oanda.com/v3"
ACCESS_TOKEN = "f5e3c467dedd6fa4ed9782451f7f7a38-b29498e233dafaf188e82e579a6be7ac"
ACCOUNT_ID = "101-004-31642123-001"

logger = logging.getLogger("WEPS.OANDA")

def execute_oanda_trade(symbol, action, lot_size, tp=None, sl=None):
    """
    Executes a live trade on OANDA using REST v20 API.

    Args:
        symbol (str): e.g., "EURUSD"
        action (str): "buy" or "sell"
        lot_size (float): Number of units in lots (1.0 = 100,000)
        tp (float): Optional take-profit price
        sl (float): Optional stop-loss price

    Returns:
        dict: OANDA trade response or error
    """
    try:
        headers = {
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }

        instrument = symbol.upper().replace("USD", "_USD").replace("GBP", "GBP_").replace("EUR", "EUR_") \
                                   .replace("JPY", "JPY_").replace("BTC", "BTC_").replace("ETH", "ETH_")
        instrument = instrument.replace("__", "_").strip("_")

        units = int(lot_size * 100000)
        if action.lower() == "sell":
            units = -units

        order_data = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }

        # Add TP/SL if provided
        if tp:
            order_data["order"]["takeProfitOnFill"] = {"price": str(tp)}
        if sl:
            order_data["order"]["stopLossOnFill"] = {"price": str(sl)}

        url = f"{OANDA_API_URL}/accounts/{ACCOUNT_ID}/orders"
        response = requests.post(url, json=order_data, headers=headers)
        data = response.json()

        if response.status_code == 201:
            logger.info(f"✅ OANDA trade executed: {action.upper()} {symbol} @ {units} units")
            return {"status": "success", "order": data}
        else:
            logger.warning(f"❌ OANDA order failed: {data}")
            return {"status": "error", "response": data}

    except Exception as e:
        logger.error(f"[{symbol}] ❌ Exception executing OANDA trade: {e}", exc_info=True)
        return {"status": "exception", "message": str(e)}
