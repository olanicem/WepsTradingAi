# ===============================================================
# WEPS DNA BUILD SCRIPT — Full Spiral-Aware Vector Constructor
# File: weps/dna/build_all_organism_dna.py
# Author: Ola | WEPS Creator
# Description: Generates and stores full 40D FESI-compliant DNA
#              vectors for each organism using live OHLCV data.
# ===============================================================

import os
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from weps.zoo.organism_registry import organism_registry
from weps.utils.fmp_fetcher import fetch_ohlcv_data
from weps.dna.dna_encoder import encode_dna_vector
from weps.utils.log_utils import log_event

FMP_API_KEY = "oemCdaq9J01EUH6VkoGrAisCdTYZLXfI"
OUTPUT_DIR = "weps/dna/data"
START_DATE = "2009-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_dna_for_organism(symbol: str, config: dict):
    log_event(f"[DNA_BUILD] • {symbol:<8} | 🧬 Starting DNA vector generation...")

    try:
        df = fetch_ohlcv_data(symbol, FMP_API_KEY, start_date=START_DATE, end_date=END_DATE)
        if df is None or df.empty:
            log_event(f"[DNA_BUILD] • {symbol:<8} | ❌ No OHLCV data returned.")
            return

        df.sort_values("date", inplace=True)
        df.dropna(inplace=True)

        dna_records = []
        for i in range(7, len(df)):
            try:
                dna_vector = encode_dna_vector(df.iloc[i-7:i+1].copy(), config)
                row = {"date": df.iloc[i]["date"], **{f"f{i+1}": v for i, v in enumerate(dna_vector)}}
                dna_records.append(row)
            except Exception as e:
                log_event(f"[DNA_BUILD] • {symbol:<8} | ⚠️ Encoding error on {df.iloc[i]['date']}: {str(e)}")
                continue

        if dna_records:
            output_df = pd.DataFrame(dna_records)
            output_file = os.path.join(OUTPUT_DIR, f"{symbol}_dna.csv")
            output_df.to_csv(output_file, index=False)
            log_event(f"[DNA_BUILD] • {symbol:<8} | ✅ DNA complete: {len(output_df)} rows saved.")
        else:
            log_event(f"[DNA_BUILD] • {symbol:<8} | ⚠️ No valid DNA vectors built.")

    except Exception as ex:
        log_event(f"[DNA_BUILD] • {symbol:<8} | ❌ Critical failure: {str(ex)}")

    time.sleep(1.1)  # Rate limiting

def build_dna_for_all():
    log_event(f"[DNA_BUILD] • GLOBAL   | 🚀 Initiating full organism DNA generation for {len(organism_registry)} organisms...")
    for organism in tqdm(organism_registry, desc="🧠 Building Spiral DNA"):
        build_dna_for_organism(organism["symbol"], organism)
    log_event(f"[DNA_BUILD] • GLOBAL   | ✅ All organism DNA complete.")

if __name__ == "__main__":
    build_dna_for_all()
