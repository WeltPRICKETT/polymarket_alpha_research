"""
Author: AI Assistant
Date: 2026-03-18
Description: Trading strategies designed to run within the BacktestEngine.
"""

from typing import Any, List, Set
from loguru import logger
import random
import pandas as pd

class BaseStrategy:
    """Base strategy interface."""
    def __init__(self):
        self.engine = None
        
    def bind_engine(self, engine: Any):
        self.engine = engine
        
    def on_market_trade(self, trade_event: pd.Series):
        """Called for every historical trade event."""
        pass


class ShadowWhaleStrategy(BaseStrategy):
    """
    Subscribes to trades made by specifically identified top traders (whales).
    When an informed trader makes a BUY order, we copy it with a proportional or fixed size.
    """
    def __init__(self, informed_addresses: List[str], max_position_size: float = 500.0,
                 copy_fraction: float = 0.05):
        """
        informed_addresses: exact wallet addresses to copy
        max_position_size: max USDC to allocate per copied trade
        copy_fraction: fraction of the whale's trade size to copy
        """
        super().__init__()
        self.target_wallets = set(informed_addresses)
        self.max_position_size = max_position_size
        self.copy_fraction = copy_fraction
        
    def on_market_trade(self, trade_event: pd.Series):
        wallet = trade_event.get('address', '')
        if wallet in self.target_wallets:
            side = trade_event.get('side', '').upper()
            if side == 'BUY':
                whale_amount = trade_event.get('amount', 0.0)
                price = trade_event.get('price', 0.0)
                outcome = trade_event.get('_outcome', 'Yes') # Default to Yes if missing
                market_id = trade_event.get('market_id', '')
                
                # Determine our trade size
                our_amount = whale_amount * self.copy_fraction
                our_amount = min(our_amount, self.max_position_size)
                
                if our_amount > 1.0 and price > 0 and price < 1.0:
                    self.engine.execute_trade(
                        market_id=market_id,
                        outcome=outcome,
                        side='BUY',
                        amount=our_amount,
                        price=price
                    )


class RandomBaselineStrategy(BaseStrategy):
    """
    Randomly enters trades to serve as a baseline to prove Alpha.
    """
    def __init__(self, trade_probability: float = 0.01, fixed_trade_size: float = 50.0):
        super().__init__()
        self.trade_prob = trade_probability
        self.trade_size = fixed_trade_size
        
    def on_market_trade(self, trade_event: pd.Series):
        if random.random() < self.trade_prob:
            side = trade_event.get('side', '').upper()
            if side == 'BUY':
                price = trade_event.get('price', 0.0)
                outcome = trade_event.get('_outcome', 'Yes')
                market_id = trade_event.get('market_id', '')
                
                if price > 0 and price < 1.0:
                    self.engine.execute_trade(
                        market_id=market_id,
                        outcome=outcome,
                        side='BUY',
                        amount=self.trade_size,
                        price=price
                    )
