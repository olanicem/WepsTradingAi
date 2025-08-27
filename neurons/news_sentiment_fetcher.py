#!/usr/bin/env python3
# ==========================================================
# 🧬 WEPS NewsSentimentFetcher — Genome-Aware Reflex Intelligence
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Fetches news from GNews & FMP APIs
#   - Matches genome keywords with intelligent fuzzy logic
#   - Converts TextBlob polarity to spiral-aware risk signal
# ==========================================================

import os
import re
import requests
import logging
from typing import List
from textblob import TextBlob
from difflib import get_close_matches

from weps.genome.genome_hub import load_sentiment_genome

logger = logging.getLogger("WEPS.NewsSentimentFetcher")
logger.setLevel(logging.INFO)


class NewsSentimentFetcher:
    def __init__(self, symbol: str, reaction_bias: float = None):
        self.symbol = symbol.upper()
        self.api_key = os.getenv("NEWSAPI_KEY")
        self.fmp_key = os.getenv("FMP_API_KEY")

        if not self.api_key:
            raise EnvironmentError("❌ NEWSAPI_KEY is not set.")

        try:
            genome = load_sentiment_genome(self.symbol)
            base_keywords = genome.get("sensitivity_keywords", [])
            self.asset_keywords = self._expand_keywords(base_keywords)
            self.reaction_bias = reaction_bias if reaction_bias else genome.get("reaction_bias", 1.0)
            logger.info(f"✅ NewsSentimentFetcher initialized for [{self.symbol}] "
                        f"with keywords={self.asset_keywords} and reaction_bias={self.reaction_bias:.2f}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load sentiment genome for {self.symbol}: {e}")

    def _expand_keywords(self, keywords: List[str]) -> List[str]:
        expanded = set()
        for kw in keywords:
            kw_clean = kw.lower().strip()
            expanded.add(kw_clean)
            expanded.add(kw_clean.replace(" ", ""))
            if " " in kw_clean:
                expanded.update(kw_clean.split())
        return list(expanded)

    def _fuzzy_match(self, title: str) -> bool:
        title_words = re.sub(r"[^\w\s]", "", title.lower()).split()
        for kw in self.asset_keywords:
            if kw in title.lower():
                return True
            if get_close_matches(kw, title_words, cutoff=0.9):
                return True
        return False

    def fetch_news_gnews(self) -> List[str]:
        try:
            url = f"https://gnews.io/api/v4/top-headlines?token={self.api_key}&lang=en&max=100"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            articles = res.json().get("articles", [])
            headlines = [a.get("title", "") for a in articles if a.get("title")]
            logger.info(f"📰 GNews fetched {len(headlines)} headlines.")
            return headlines
        except Exception as e:
            logger.warning(f"⚠️ GNews API failed: {e}")
            return []

    def fetch_news_fmp(self) -> List[str]:
        try:
            if not self.fmp_key:
                return []
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={self.symbol}&limit=50&apikey={self.fmp_key}"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data = res.json()
            headlines = [item.get("title", "") for item in data if item.get("title")]
            logger.info(f"📰 FMP fetched {len(headlines)} headlines.")
            return headlines
        except Exception as e:
            logger.warning(f"⚠️ FMP News fallback failed: {e}")
            return []

    def analyze_sentiment(self, headlines: List[str]) -> float:
        if not headlines:
            logger.warning("⚠️ No headlines available.")
            return 0.0

        filtered = [h for h in headlines if self._fuzzy_match(h)]
        if not filtered:
            logger.warning("⚠️ No headlines matched keywords.")
            return 0.0

        total = sum(TextBlob(h).sentiment.polarity for h in filtered)
        avg = total / len(filtered)
        logger.info(f"🧠 Avg sentiment polarity: {avg:.4f} from {len(filtered)} relevant headlines.")
        return avg

    def compute_signal(self, avg_polarity: float, elliot_direction: str, phase: str) -> dict:
        signal, adjustment = "neutral", 0.0
        if avg_polarity > 0.1:
            signal = "risk-on" if elliot_direction == "impulse" else "risk-off"
            adjustment = 0.02 if signal == "risk-on" else -0.02
        elif avg_polarity < -0.1:
            signal = "risk-off" if elliot_direction == "impulse" else "risk-on"
            adjustment = -0.02 if signal == "risk-off" else 0.02

        phase_weight = {"rebirth": 1.15, "growth": 1.0, "decay": 0.85, "death": 0.7}
        adjustment *= phase_weight.get(phase, 1.0)
        adjustment *= self.reaction_bias

        result = {
            "sentiment_signal": signal,
            "sentiment_adjustment": round(adjustment, 4),
            "average_polarity": round(avg_polarity, 4),
            "phase": phase,
            "elliot_direction": elliot_direction
        }
        logger.info(f"✅ Final sentiment computed: {result}")
        return result

    def run(self, elliot_direction: str = "impulse", phase: str = "growth") -> dict:
        try:
            headlines = self.fetch_news_gnews()
            if not headlines:
                headlines = self.fetch_news_fmp()
            polarity = self.analyze_sentiment(headlines)
            return self.compute_signal(polarity, elliot_direction, phase)
        except Exception as e:
            logger.error(f"❌ Sentiment pipeline failed for {self.symbol}: {e}", exc_info=True)
            return {
                "sentiment_signal": "neutral",
                "sentiment_adjustment": 0.0,
                "average_polarity": 0.0,
                "phase": phase,
                "elliot_direction": elliot_direction,
                "fallback": True,
            }
