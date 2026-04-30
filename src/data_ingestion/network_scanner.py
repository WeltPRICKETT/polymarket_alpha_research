"""
Author: AI Assistant
Date: 2026-03-17
Description: Scans top Polymarket markets to harvest a list of real active wallet addresses.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import List

# Add project root to sys path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_ingestion.polymarket_client import PolymarketClient
from py_clob_client.clob_types import TradeParams

class NetworkScanner:
    def __init__(self):
        self.client = PolymarketClient()

    def get_active_wallets(self, num_markets: int = 50, target_wallets: int = 500) -> List[str]:
        # We need to scan open markets because CLOB only returns trades for open markets effectively
        top_markets = self.client.get_top_markets(limit=num_markets, closed=False)
        wallets = set()
        
        logger.info(f"Scanning {len(top_markets)} top markets for active wallets...")
        
        clob = self.client._get_active_clob_client()
        if not clob:
            logger.error("No CLOB client available to scan trades.")
            return []

        for mkt_id in top_markets:
            if len(wallets) >= target_wallets:
                break
                
            try:
                # get_trades requires L2 auth
                res = clob.get_trades(TradeParams(market=mkt_id))
                if res and 'data' in res:
                    for trade in res['data']:
                        maker = trade.get('maker_address')
                        taker = trade.get('taker_address')
                        if maker: wallets.add(maker)
                        if taker: wallets.add(taker)
                logger.info(f"Market {mkt_id}: Found {len(wallets)} unique wallets so far.")
            except Exception as e:
                logger.warning(f"Failed to fetch trades for market {mkt_id}: {e}")
                
        return list(wallets)[:target_wallets]

if __name__ == "__main__":
    scanner = NetworkScanner()
    wallets = scanner.get_active_wallets(num_markets=10, target_wallets=50)
    print(wallets)
