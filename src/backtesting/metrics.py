"""
Author: AI Assistant
Date: 2026-03-18
Description: Performance metrics calculator for backtest equity curves.
"""

import pandas as pd
import numpy as np

def calculate_metrics(equity_df: pd.DataFrame, initial_capital: float) -> dict:
    """
    Calculates primary financial metrics from an equity curve DataFrame.
    equity_df must contain 'timestamp' and 'total_equity' columns.
    """
    if equity_df.empty or 'total_equity' not in equity_df.columns:
        return {}
        
    final_equity = equity_df['total_equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Calculate Drawdown
    equity_df['peak'] = equity_df['total_equity'].cummax()
    equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()
    
    # Needs chronological timestamps to calculate Sharpe properly. 
    # If not enough variance or time, return simplified Sharpe.
    # Assuming daily returns proxy from event to event
    equity_df['returns'] = equity_df['total_equity'].pct_change().fillna(0)
    
    # Annualized Sharpe (assuming roughly 252 trading days equivalents, but Polymarket is 24/7 so 365)
    # This is a very rough proxy because timestamps are unevenly spaced.
    mean_return = equity_df['returns'].mean()
    std_return = equity_df['returns'].std()
    
    if std_return == 0 or np.isnan(std_return):
        sharpe_ratio = 0.0
    else:
        # Assuming the events are roughly daily to scale, if they are intraday, the scale factor differs. 
        # Using a generic un-annualized event Sharpe for simplicity, or scale by sqrt(N)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(365)
        
    return {
        'Initial Capital': initial_capital,
        'Final Equity': final_equity,
        'Total Return (%)': total_return * 100,
        'Max Drawdown (%)': max_drawdown * 100,
        'Sharpe Ratio': sharpe_ratio
    }
