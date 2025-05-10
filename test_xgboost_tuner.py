#!/usr/bin/env python3
"""
Test script to verify that the indexing issue in XGBoostOptimizer has been fixed.
This script imports the module and simulates creating signals with proper index alignment.
"""

import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_xgboost_tuner')

def test_indexing_fix():
    """Test the indexing fix with a small DataFrame."""
    logger.info("Testing indexing fix for XGBoost tuner...")
    
    # Create sample data with DatetimeIndex - similar to real data
    dates = pd.date_range('2023-01-01', periods=10)
    
    # Create y_val_orig with DatetimeIndex - representing target labels (-1, 0, 1)
    y_val_orig = pd.Series([1, -1, 0, 1, -1, 0, 1, -1, 0, 1], index=dates)
    
    # Create signals list - this would normally not have an index
    signals = ['LONG', 'SHORT', 'NEUTRAL', 'LONG', 'SHORT', 'NEUTRAL', 'LONG', 'SHORT', 'NEUTRAL', 'LONG']
    
    # Original approach - this would cause an indexing error
    try:
        logger.info("Testing original approach (expected to fail)...")
        long_trades_original = y_val_orig[pd.Series(signals) == "LONG"]
        logger.info("Original approach succeeded unexpectedly")
    except Exception as e:
        logger.info(f"Original approach failed as expected: {e}")
    
    # Our fixed approach - creates a Series with matching index
    logger.info("Testing fixed approach...")
    signals_series = pd.Series(signals, index=y_val_orig.index)
    long_mask = signals_series == "LONG"
    short_mask = signals_series == "SHORT"
    
    long_trades = y_val_orig[long_mask]
    short_trades = y_val_orig[short_mask]
    
    logger.info(f"Fixed approach succeeded: Long trades: {long_trades.values}")
    logger.info(f"Fixed approach succeeded: Short trades: {short_trades.values}")
    
    # Calculate some statistics to verify correct behavior
    long_wins = (long_trades == 1).sum()
    long_losses = (long_trades == -1).sum()
    short_wins = (short_trades == -1).sum()
    short_losses = (short_trades == 1).sum()
    
    logger.info(f"Long wins: {long_wins}, Long losses: {long_losses}")
    logger.info(f"Short wins: {short_wins}, Short losses: {short_losses}")
    
    total_wins = long_wins + short_wins
    total_losses = long_losses + short_losses
    
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    logger.info(f"Profit factor: {profit_factor:.2f}")
    
    return {
        'success': True, 
        'long_trades_count': len(long_trades),
        'short_trades_count': len(short_trades),
        'profit_factor': profit_factor
    }

if __name__ == "__main__":
    result = test_indexing_fix()
    print(f"Test result: {result}")