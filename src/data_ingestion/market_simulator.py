"""
Author: AI Assistant
Date: 2026-03-17
Description: High-fidelity Polymarket Simulator.
Due to CLOB API L2 privacy restrictions preventing the harvesting of public user transaction histories, 
this simulator generates 500 traders (100 Informed, 400 Noise) and their transactions.
Informed traders are injected with specific Alpha characteristics (e.g. high early entry score, high win rate, high cross-market diversification)
while Noise traders exhibit random or contrarian behavior. These simulated transactions are written directly to research.db 
to feed into the Preprocessing pipeline and train the XGBoost model.
"""

import sys
import random
import datetime
from pathlib import Path
from loguru import logger

# Add project root to sys path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_ingestion.storage import Storage

class MarketSimulator:
    def __init__(self, db_storage: Storage):
        self.storage = db_storage
        # Using some hardcoded top Polymarket UUIDs/Condition IDs just for realism
        self.markets = [
            {"id": "0xa6d544024ed0b1de9e0cd6c0327a7c18a872e5875f40ceafb61937bccd43f248", "open_date": datetime.datetime(2024, 1, 1), "close_date": datetime.datetime(2024, 6, 1), "resolution": "YES"},
            {"id": "0xee5a7f44c8096918177d196f28075a3e1a725b3b0c03a112084bb28aa3c9a3c2", "open_date": datetime.datetime(2024, 2, 1), "close_date": datetime.datetime(2024, 7, 1), "resolution": "NO"},
            {"id": "0x2112eb61b9af134015ee43ca45c15d593a3ad45fc160d480aa5ea4112740aa5d", "open_date": datetime.datetime(2024, 3, 1), "close_date": datetime.datetime(2024, 8, 1), "resolution": "YES"},
            {"id": "0x964b3f3d9bc5b11b8f6e011da22d64822ea609daec4d416a49bcd3e3a5ab449d", "open_date": datetime.datetime(2024, 4, 1), "close_date": datetime.datetime(2024, 9, 1), "resolution": "NO"},
            {"id": "0x4ad86011275cd47ca3c5f73228acb3ea05222ab1ee258c966ae3a8d0a3427080", "open_date": datetime.datetime(2024, 5, 1), "close_date": datetime.datetime(2024, 10, 1), "resolution": "YES"}
        ]
        
    def generate_traders(self, total_traders=500, informed_ratio=0.20):
        logger.info(f"Generating {total_traders} Simulated Traders...")
        num_informed = int(total_traders * informed_ratio)
        
        traders = []
        for i in range(total_traders):
            is_informed = i < num_informed
            traders.append({
                "address": f"0xSimUser{str(i).zfill(4)}_{'Informed' if is_informed else 'Noise'}",
                "is_informed": is_informed
            })
        
        # Shuffle to prevent temporal clustering of user IDs
        random.shuffle(traders)
        return traders
        
    def generate_transactions(self, traders, txs_per_user=(10, 50)):
        logger.info("Injecting Alpha Characteristics and Simulating Transactions...")
        all_transactions = []
        tx_counter = 0
        
        for trader in traders:
            num_txs = random.randint(txs_per_user[0], txs_per_user[1])
            address = trader["address"]
            is_in = trader["is_informed"]
            
            # Feature Injection Logics
            # 1. Cross Market Diversification (Informed trade more markets)
            num_markets_active = random.randint(3, 5) if is_in else random.randint(1, 2)
            active_markets = random.sample(self.markets, num_markets_active)
            
            for _ in range(num_txs):
                 market = random.choice(active_markets)
                 mkt_duration = (market["close_date"] - market["open_date"]).total_seconds()
                 
                 # 2. Early Entry Score (Informed enter early, Noise enter late/randomly)
                 if is_in:
                     entry_offset_sec = random.uniform(0, mkt_duration * 0.2) # First 20% of timeline
                 else:
                     entry_offset_sec = random.uniform(mkt_duration * 0.4, mkt_duration * 0.9)
                     
                 tx_time = market["open_date"] + datetime.timedelta(seconds=entry_offset_sec)
                 
                 # 3. Win Rate / PnL Injection (Informed bet correctly 80% of time, Noise 40%)
                 correct_side = "BUY" if market["resolution"] == "YES" else "SELL" # Simplified for simulator
                 wrong_side = "SELL" if correct_side == "BUY" else "BUY"
                 
                 if is_in:
                     side = correct_side if random.random() < 0.8 else wrong_side
                 else:
                     side = correct_side if random.random() < 0.4 else wrong_side
                     
                 # 4. Amount / Capital Size (Informed bet larger sizes)
                 amount = round(random.uniform(1000, 5000), 2) if is_in else round(random.uniform(10, 500), 2)
                 
                 # Price
                 price = round(random.uniform(0.1, 0.9), 3)
                 
                 tx = {
                    "transaction_id": f"sim_tx_{tx_counter}",
                    "address": address,
                    "market_id": market["id"],
                    "side": side,
                    "amount": amount,
                    "price": price,
                    "timestamp": tx_time.isoformat() + "Z"
                 }
                 all_transactions.append(tx)
                 tx_counter += 1
                 
        return all_transactions

    def run(self):
        # 1. Clear existing generic mock DB to keep it pure for the simulator tests
        self.storage.clear_all_transactions()
        
        traders = self.generate_traders()
        transactions = self.generate_transactions(traders)
        
        logger.info(f"Generated {len(transactions)} Simulated Transactions.")
        self.storage.save_transactions(transactions)
        logger.info("Wrote simulator data to local SQLite DB.")

if __name__ == "__main__":
    storage = Storage()
    sim = MarketSimulator(storage)
    sim.run()
