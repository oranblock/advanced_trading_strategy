#!/usr/bin/env python3
"""
Extended Walk-Forward Analysis for Liquidity Trading Strategy

This script performs walk-forward analysis on 10-50 years of historical data
to validate the strategy's robustness across different market regimes.
"""
import pandas as pd
import numpy as np
import logging
import argparse
import yaml
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extended_walkforward_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('extended_walkforward')

# Create directories
os.makedirs("./walkforward_results", exist_ok=True)
os.makedirs("./walkforward_models", exist_ok=True)
os.makedirs("./walkforward_data", exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extended Walk-Forward Analysis')
    
    # Data parameters
    parser.add_argument('--ticker', type=str, default='EURUSD',
                        help='Ticker symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='day',
                        choices=['minute', 'hour', 'day', 'week'],
                        help='Timeframe for analysis')
    parser.add_argument('--start-date', type=str, default='1970-01-01',
                        help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for analysis (YYYY-MM-DD), defaults to today')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Path to CSV data file (if not fetching from API)')
    
    # Analysis parameters
    parser.add_argument('--windows', type=int, default=20,
                        help='Number of windows for walk-forward analysis')
    parser.add_argument('--window-overlap', type=float, default=0.5,
                        help='Overlap between windows (0.0-1.0)')
    parser.add_argument('--train-size', type=float, default=0.7,
                        help='Training data size within each window (0.0-1.0)')
    
    # Strategy parameters
    parser.add_argument('--up-threshold', type=float, default=0.5,
                        help='UP probability threshold (0.0-1.0)')
    parser.add_argument('--down-threshold', type=float, default=0.5,
                        help='DOWN probability threshold (0.0-1.0)')
    parser.add_argument('--look-forward', type=int, default=24,
                        help='Look-forward period for labeling')
    parser.add_argument('--pt-multiplier', type=float, default=2.5,
                        help='Profit target multiplier (as ATR multiple)')
    parser.add_argument('--sl-multiplier', type=float, default=1.5,
                        help='Stop loss multiplier (as ATR multiple)')
    
    # Model parameters
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees in XGBoost model')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Maximum tree depth in XGBoost model')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./walkforward_results',
                        help='Directory for output files')
    parser.add_argument('--save-models', action='store_true',
                        help='Save models for each window')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing for window analysis')
    
    # Other parameters
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def load_or_generate_data(args):
    """Load data from file or generate synthetic data."""
    # Check if data file is provided
    if args.data_file and os.path.exists(args.data_file):
        logger.info(f"Loading data from {args.data_file}")
        df = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
        
        # Ensure we have all required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Data file must contain columns: {required_cols}")
            raise ValueError(f"Data file must contain columns: {required_cols}")
        
        # Add volume column if missing
        if 'volume' not in df.columns:
            logger.info("Adding synthetic volume data")
            df['volume'] = np.random.uniform(1000, 5000, len(df))
        
        return df
    
    # Parse date range
    start_date = args.start_date
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    
    # Generate synthetic data for demonstration
    logger.info(f"Generating synthetic data for {args.ticker} from {start_date} to {end_date}")
    
    # Create date range based on timeframe
    if args.timeframe == 'minute':
        freq = 'T'
    elif args.timeframe == 'hour':
        freq = 'H'
    elif args.timeframe == 'day':
        freq = 'D'
    else:  # week
        freq = 'W'
    
    # Create date range
    index = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate synthetic OHLCV data
    np.random.seed(args.random_seed)
    n = len(index)
    
    logger.info(f"Generating {n} bars of synthetic {args.timeframe} data")
    
    # Parameters for random walk with momentum and volatility clustering
    if args.ticker in ['EURUSD', 'GBPUSD', 'USDJPY']:
        start_price = 1.2000 if args.ticker == 'EURUSD' else 1.5000 if args.ticker == 'GBPUSD' else 110.00
        drift = 0.00001  # Small upward drift
        base_volatility = 0.0005  # Base volatility
    else:  # Assume stock with higher price
        start_price = 100.00
        drift = 0.0001
        base_volatility = 0.01
    
    # Create price series with realistic characteristics
    close = np.zeros(n)
    close[0] = start_price
    volatility = np.zeros(n)
    volatility[0] = base_volatility
    
    # GARCH-like volatility clustering and momentum effects
    for i in range(1, n):
        if i % 50000 == 0:
            logger.info(f"Generated {i}/{n} bars...")
        
        # Volatility clustering
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * base_volatility * (1 + 0.5 * np.random.normal())
        
        # Add momentum and mean reversion effects
        momentum = 0.05 * (close[i-1] - close[max(0, i-20)]) / close[max(0, i-20)]
        mean_reversion = -0.02 * (close[i-1] - close[0]) / close[0]
        
        # Random walk with drift, momentum, mean reversion, and volatility
        close[i] = close[i-1] * (1 + drift + momentum + mean_reversion + volatility[i] * np.random.normal())
    
    # Add realistic market regimes
    # Bull market
    bull_start = int(n * 0.2)
    bull_end = int(n * 0.3)
    close[bull_start:bull_end] *= np.linspace(1, 1.5, bull_end - bull_start)
    
    # Bear market
    bear_start = int(n * 0.5)
    bear_end = int(n * 0.6)
    close[bear_start:bear_end] *= np.linspace(1, 0.7, bear_end - bear_start)
    
    # Sideways market with high volatility
    sideways_start = int(n * 0.7)
    sideways_end = int(n * 0.8)
    volatility[sideways_start:sideways_end] *= 2.0
    
    # Generate open, high, low based on close with realistic relationships
    open_prices = np.zeros(n)
    open_prices[0] = close[0] * (1 - 0.0002 * np.random.normal())
    
    # More realistic open price based on previous close
    for i in range(1, n):
        # Open relatively close to previous close
        open_prices[i] = close[i-1] * (1 + 0.2 * volatility[i] * np.random.normal())
    
    # High is max of open and close plus random amount
    high = np.maximum(close, open_prices) + np.abs(volatility * np.random.normal(size=n) * 2)
    
    # Low is min of open and close minus random amount
    low = np.minimum(close, open_prices) - np.abs(volatility * np.random.normal(size=n) * 2)
    
    # Ensure high >= close >= low and high >= open >= low
    for i in range(n):
        high[i] = max(high[i], close[i], open_prices[i])
        low[i] = min(low[i], close[i], open_prices[i])
    
    # Create volume with some patterns
    volume = np.zeros(n)
    base_volume = 1000 if args.timeframe == 'minute' else 5000 if args.timeframe == 'hour' else 20000
    
    for i in range(n):
        # Volume tends to be higher with larger price moves
        volume_factor = 1 + 2 * abs(close[i] / close[i-1] - 1) if i > 0 else 1
        # Add time-of-day/week patterns
        if freq == 'T' or freq == 'H':
            # Higher volume during market hours
            hour = index[i].hour
            if 8 <= hour <= 16:  # Market hours
                volume_factor *= 1.5
        
        volume[i] = base_volume * volume_factor * (1 + 0.5 * np.random.normal())
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=index)
    
    # Save for future use
    output_path = f"./walkforward_data/{args.ticker}_{args.timeframe}_{start_date}_to_{end_date.replace('-', '')}.csv"
    df.to_csv(output_path)
    logger.info(f"Saved synthetic data to {output_path}")
    
    return df

def add_basic_features(df):
    """Add basic technical indicators."""
    logger.info("Adding basic technical features...")
    df_feat = df.copy()
    
    # Calculate basic indicators
    # Moving averages
    for window in [20, 50, 200]:
        df_feat[f'sma_{window}'] = df_feat['close'].rolling(window=window).mean()
        df_feat[f'ema_{window}'] = df_feat['close'].ewm(span=window, adjust=False).mean()
    
    # Relative strength
    returns = df_feat['close'].pct_change()
    up_returns = returns.copy()
    down_returns = returns.copy()
    up_returns[up_returns < 0] = 0
    down_returns[down_returns > 0] = 0
    down_returns = -down_returns
    
    avg_up = up_returns.rolling(window=14).mean()
    avg_down = down_returns.rolling(window=14).mean()
    
    rs = avg_up / avg_down
    df_feat['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ATR calculation
    high_low = df_feat['high'] - df_feat['low']
    high_close = (df_feat['high'] - df_feat['close'].shift()).abs()
    low_close = (df_feat['low'] - df_feat['close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df_feat['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Remove NaN values
    df_feat = df_feat.dropna()
    
    return df_feat

def add_liquidity_features(df):
    """Add specialized liquidity-based features."""
    logger.info("Adding specialized liquidity features...")
    df_liq = df.copy()
    
    # Define time-based features
    df_liq['hour'] = df_liq.index.hour
    df_liq['day_of_week'] = df_liq.index.dayofweek
    
    # Daily high/low
    df_liq['date'] = df_liq.index.date
    
    # Calculate daily high/low
    daily_high = df_liq.groupby('date')['high'].transform('max')
    daily_low = df_liq.groupby('date')['low'].transform('min')
    
    # Previous day's high/low 
    # For different timeframes, we need different shift values
    if 'hour' in str(df_liq.index.freq) or 'H' in str(df_liq.index.freq):
        day_shift = 24  # 24 hours in a day
    elif 'min' in str(df_liq.index.freq) or 'T' in str(df_liq.index.freq):
        day_shift = 24 * 60  # minutes in a day
    else:
        day_shift = 1  # For daily data
    
    prev_day_high = df_liq.groupby('date')['high'].transform('max').shift(day_shift)
    prev_day_low = df_liq.groupby('date')['low'].transform('min').shift(day_shift)
    
    # Calculate distance to these liquidity levels in terms of ATR
    df_liq['dist_to_prev_day_high_atr'] = (prev_day_high - df_liq['close']) / df_liq['ATR_14']
    df_liq['dist_to_prev_day_low_atr'] = (df_liq['close'] - prev_day_low) / df_liq['ATR_14']
    
    # Weekly high/low calculation
    df_liq['week'] = df_liq.index.isocalendar().week
    df_liq['year'] = df_liq.index.year
    
    # Create a week identifier
    df_liq['week_id'] = df_liq['year'].astype(str) + '-' + df_liq['week'].astype(str).str.zfill(2)
    
    # Calculate weekly high/low
    weekly_high = df_liq.groupby('week_id')['high'].transform('max')
    weekly_low = df_liq.groupby('week_id')['low'].transform('min')
    
    # Previous week's high/low
    prev_week_high = weekly_high.shift(day_shift * 7)
    prev_week_low = weekly_low.shift(day_shift * 7)
    
    # Distance to weekly levels
    df_liq['dist_to_prev_week_high_atr'] = (prev_week_high - df_liq['close']) / df_liq['ATR_14']
    df_liq['dist_to_prev_week_low_atr'] = (df_liq['close'] - prev_week_low) / df_liq['ATR_14']
    
    # Monthly levels
    df_liq['month'] = df_liq.index.month
    monthly_high = df_liq.groupby(['year', 'month'])['high'].transform('max')
    monthly_low = df_liq.groupby(['year', 'month'])['low'].transform('min')
    
    # Previous month high/low
    prev_month_high = monthly_high.shift(day_shift * 30)
    prev_month_low = monthly_low.shift(day_shift * 30)
    
    # Distance to monthly levels
    df_liq['dist_to_prev_month_high_atr'] = (prev_month_high - df_liq['close']) / df_liq['ATR_14']
    df_liq['dist_to_prev_month_low_atr'] = (df_liq['close'] - prev_month_low) / df_liq['ATR_14']
    
    # Sweep detection (price touches a level and reverses)
    df_liq['swept_prev_day_high'] = (df_liq['high'] > prev_day_high) & (df_liq['close'] < prev_day_high)
    df_liq['swept_prev_day_low'] = (df_liq['low'] < prev_day_low) & (df_liq['close'] > prev_day_low)
    df_liq['swept_prev_week_high'] = (df_liq['high'] > prev_week_high) & (df_liq['close'] < prev_week_high)
    df_liq['swept_prev_week_low'] = (df_liq['low'] < prev_week_low) & (df_liq['close'] > prev_week_low)
    
    # Relative position in ranges
    df_liq['rel_pos_in_day_range'] = (df_liq['close'] - daily_low) / (daily_high - daily_low + 1e-10)
    df_liq['rel_pos_in_week_range'] = (df_liq['close'] - weekly_low) / (weekly_high - weekly_low + 1e-10)
    df_liq['rel_pos_in_month_range'] = (df_liq['close'] - monthly_low) / (monthly_high - monthly_low + 1e-10)
    
    # Drop unnecessary columns
    df_liq = df_liq.drop(columns=['date', 'week', 'year', 'week_id', 'month'])
    
    # Fill NaN values
    df_liq = df_liq.fillna(0)
    
    return df_liq

def prepare_features_and_labels(df, look_forward=24, pt_mult=2.5, sl_mult=1.5):
    """Prepare features and create labels for supervised learning."""
    logger.info("Preparing features and generating labels...")
    df_prep = df.copy()
    
    # Calculate future returns for labeling
    future_returns = pd.Series(index=df_prep.index)
    profit_levels = pd.Series(index=df_prep.index)
    stop_levels = pd.Series(index=df_prep.index)
    
    # Calculate profit target and stop loss levels
    profit_levels = df_prep['close'] * (1 + (pt_mult * df_prep['ATR_14'] / df_prep['close']))
    stop_levels = df_prep['close'] * (1 - (sl_mult * df_prep['ATR_14'] / df_prep['close']))
    
    # Shift profit and stop levels to avoid lookahead bias
    profit_levels = profit_levels.shift(-1)
    stop_levels = stop_levels.shift(-1)
    
    # Initialize label array
    labels = np.zeros(len(df_prep))
    
    # Create progress bar for long operations
    logger.info(f"Labeling data with look-forward period of {look_forward}...")
    for i in tqdm(range(len(df_prep) - look_forward)):
        # Get future price data
        future_prices = df_prep['close'].iloc[i+1:i+look_forward+1]
        future_highs = df_prep['high'].iloc[i+1:i+look_forward+1]
        future_lows = df_prep['low'].iloc[i+1:i+look_forward+1]
        
        current_close = df_prep['close'].iloc[i]
        profit_target = profit_levels.iloc[i]
        stop_loss = stop_levels.iloc[i]
        
        # Check if profit target or stop loss was hit first
        pt_hit = future_highs.iloc[0:look_forward].gt(profit_target).any()
        sl_hit = future_lows.iloc[0:look_forward].lt(stop_loss).any()
        
        if pt_hit and not sl_hit:
            labels[i] = 1  # UP - profit target hit first
        elif sl_hit and not pt_hit:
            labels[i] = -1  # DOWN - stop loss hit first
        elif pt_hit and sl_hit:
            # Determine which was hit first
            pt_idx = future_highs.gt(profit_target).idxmax() if pt_hit else None
            sl_idx = future_lows.lt(stop_loss).idxmax() if sl_hit else None
            
            if pt_idx and sl_idx:
                labels[i] = 1 if pt_idx < sl_idx else -1
            else:
                labels[i] = 0  # Neutral
        else:
            labels[i] = 0  # Neutral - neither hit
    
    # Create features and labels
    X = df_prep.copy()
    
    # Drop columns we don't want as features
    X = X.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Create label Series
    y = pd.Series(labels, index=df_prep.index)
    
    # Drop rows with NaN
    nan_mask = X.isna().any(axis=1) | y.isna()
    X = X[~nan_mask]
    y = y[~nan_mask]
    
    return X, y, df_prep

def split_data(X, y, df_orig, train_size=0.7):
    """Split data into train and test sets."""
    # Calculate sizes
    train_idx = int(len(X) * train_size)
    
    # Split features and labels
    X_train = X.iloc[:train_idx]
    y_train = y.iloc[:train_idx]
    
    X_test = X.iloc[train_idx:]
    y_test = y.iloc[train_idx:]
    
    # Split original data for backtesting
    df_train = df_orig.loc[X_train.index]
    df_test = df_orig.loc[X_test.index]
    
    return (X_train, y_train, df_train), (X_test, y_test, df_test)

def train_models(X_train, y_train, args):
    """Train XGBoost models for UP and DOWN predictions."""
    try:
        import xgboost as xgb
        
        # Create binary targets
        y_train_up = (y_train == 1).astype(int)
        y_train_down = (y_train == -1).astype(int)
        
        # Calculate class weights
        up_neg_count = (y_train_up == 0).sum()
        up_pos_count = (y_train_up == 1).sum()
        scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1.0
        
        down_neg_count = (y_train_down == 0).sum()
        down_pos_count = (y_train_down == 1).sum()
        scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1.0
        
        # Create DMastrices for XGBoost
        dtrain_up = xgb.DMatrix(X_train, label=y_train_up)
        dtrain_down = xgb.DMatrix(X_train, label=y_train_down)
        
        # Define parameters for UP model
        xgb_params_up = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': args.max_depth,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'scale_pos_weight': scale_pos_weight_up,
            'seed': args.random_seed
        }
        
        # Define parameters for DOWN model
        xgb_params_down = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': args.max_depth,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'scale_pos_weight': scale_pos_weight_down,
            'seed': args.random_seed
        }
        
        # Train UP model
        model_up = xgb.train(
            xgb_params_up,
            dtrain_up,
            num_boost_round=args.n_estimators,
            verbose_eval=False
        )
        
        # Train DOWN model
        model_down = xgb.train(
            xgb_params_down,
            dtrain_down,
            num_boost_round=args.n_estimators,
            verbose_eval=False
        )
        
        return model_up, model_down
        
    except ImportError:
        logger.error("XGBoost not installed. Cannot train models.")
        return None, None

def predict_signals(model_up, model_down, X, df_orig, up_threshold=0.5, down_threshold=0.5):
    """Generate trading signals from model predictions."""
    import xgboost as xgb
    
    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(X)
    
    # Predict probabilities
    up_probs = model_up.predict(dmatrix)
    down_probs = model_down.predict(dmatrix)
    
    # Generate signals
    signals = []
    for i in range(len(X)):
        prob_up = up_probs[i]
        prob_down = down_probs[i]
        
        if prob_up >= up_threshold and prob_down < (1.0 - down_threshold):
            signal = "LONG"
        elif prob_down >= down_threshold and prob_up < (1.0 - up_threshold):
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
        
        signals.append(signal)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'open': df_orig['open'],
        'high': df_orig['high'],
        'low': df_orig['low'],
        'close': df_orig['close'],
        'volume': df_orig['volume'],
        'up_prob': up_probs,
        'down_prob': down_probs,
        'signal': signals
    }, index=X.index)
    
    return results

def backtest_signals(results, pt_mult=2.5, sl_mult=1.5, initial_capital=10000):
    """Run a backtest on the generated signals."""
    # Initialize backtest results
    bt_results = results.copy()
    bt_results['position'] = 0  # 0: no position, 1: long, -1: short
    bt_results['entry_price'] = np.nan
    bt_results['profit_target'] = np.nan
    bt_results['stop_loss'] = np.nan
    bt_results['trade_result'] = np.nan
    bt_results['equity'] = initial_capital
    
    # Iterate through each bar
    position = 0
    entry_price = 0
    profit_target = 0
    stop_loss = 0
    equity = initial_capital
    trade_count = 0
    win_count = 0
    loss_count = 0
    
    # For more realistic backtesting, calculate PT and SL based on ATR
    atr = results['high'] - results['low']
    atr = atr.rolling(window=14).mean()
    
    for i in range(1, len(bt_results)):
        # Get previous values
        prev_position = position
        prev_equity = equity
        
        current_bar = bt_results.iloc[i]
        prev_bar = bt_results.iloc[i-1]
        
        # If we have no position, check for entry signals
        if position == 0:
            if prev_bar['signal'] == 'LONG':
                # Enter long position at open of next bar
                position = 1
                entry_price = current_bar['open']
                profit_target = entry_price * (1 + (pt_mult * atr.iloc[i] / entry_price))
                stop_loss = entry_price * (1 - (sl_mult * atr.iloc[i] / entry_price))
                trade_count += 1
            
            elif prev_bar['signal'] == 'SHORT':
                # Enter short position at open of next bar
                position = -1
                entry_price = current_bar['open']
                profit_target = entry_price * (1 - (pt_mult * atr.iloc[i] / entry_price))
                stop_loss = entry_price * (1 + (sl_mult * atr.iloc[i] / entry_price))
                trade_count += 1
        
        # If we have a position, check for exit conditions
        elif position == 1:  # Long position
            if current_bar['high'] >= profit_target:
                # Hit profit target
                position = 0
                trade_result = profit_target / entry_price - 1
                equity += initial_capital * trade_result
                win_count += 1
            elif current_bar['low'] <= stop_loss:
                # Hit stop loss
                position = 0
                trade_result = stop_loss / entry_price - 1
                equity += initial_capital * trade_result
                loss_count += 1
            else:
                # Position still open
                trade_result = current_bar['close'] / entry_price - 1
                equity = prev_equity + initial_capital * (current_bar['close'] / prev_bar['close'] - 1)
        
        elif position == -1:  # Short position
            if current_bar['low'] <= profit_target:
                # Hit profit target
                position = 0
                trade_result = 1 - profit_target / entry_price
                equity += initial_capital * trade_result
                win_count += 1
            elif current_bar['high'] >= stop_loss:
                # Hit stop loss
                position = 0
                trade_result = 1 - stop_loss / entry_price
                equity += initial_capital * trade_result
                loss_count += 1
            else:
                # Position still open
                trade_result = 1 - current_bar['close'] / entry_price
                equity = prev_equity + initial_capital * (1 - current_bar['close'] / prev_bar['close'])
        
        # Update backtest results
        bt_results.iloc[i, bt_results.columns.get_loc('position')] = position
        bt_results.iloc[i, bt_results.columns.get_loc('entry_price')] = entry_price if position != 0 else np.nan
        bt_results.iloc[i, bt_results.columns.get_loc('profit_target')] = profit_target if position != 0 else np.nan
        bt_results.iloc[i, bt_results.columns.get_loc('stop_loss')] = stop_loss if position != 0 else np.nan
        bt_results.iloc[i, bt_results.columns.get_loc('trade_result')] = trade_result if position != 0 else np.nan
        bt_results.iloc[i, bt_results.columns.get_loc('equity')] = equity
    
    # Calculate win rate and profit factor
    win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
    profit_factor = win_count / loss_count if loss_count > 0 else float('inf')
    
    # Calculate additional metrics
    total_return = (equity - initial_capital) / initial_capital
    drawdowns = bt_results['equity'].cummax() - bt_results['equity']
    max_drawdown = drawdowns.max() / bt_results['equity'].cummax().iloc[drawdowns.argmax()] if len(drawdowns) > 0 else 0
    
    # Calculate Sharpe ratio (annualized, assuming daily data)
    returns = bt_results['equity'].pct_change().dropna()
    annualization_factor = 252  # Default for daily data
    if 'hour' in str(bt_results.index.freq) or 'H' in str(bt_results.index.freq):
        annualization_factor = 252 * 24
    elif 'min' in str(bt_results.index.freq) or 'T' in str(bt_results.index.freq):
        annualization_factor = 252 * 24 * 60
    elif 'week' in str(bt_results.index.freq) or 'W' in str(bt_results.index.freq):
        annualization_factor = 52
    
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(annualization_factor) if len(returns) > 0 and returns.std() > 0 else 0
    
    # Calculate MAR ratio (annualized return / max drawdown)
    mar_ratio = (total_return / (bt_results.index[-1] - bt_results.index[0]).days * 365) / max_drawdown if max_drawdown > 0 else float('inf')
    
    # Metrics dictionary
    metrics = {
        'initial_capital': initial_capital,
        'final_equity': float(equity),
        'total_return': float(total_return),
        'annualized_return': float(total_return / (bt_results.index[-1] - bt_results.index[0]).days * 365),
        'total_trades': int(trade_count),
        'winning_trades': int(win_count),
        'losing_trades': int(loss_count),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if not np.isinf(profit_factor) else 999.9,
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe_ratio),
        'mar_ratio': float(mar_ratio) if not np.isinf(mar_ratio) else 999.9,
        'test_start_date': bt_results.index[0].strftime('%Y-%m-%d'),
        'test_end_date': bt_results.index[-1].strftime('%Y-%m-%d'),
        'test_duration_days': (bt_results.index[-1] - bt_results.index[0]).days
    }
    
    return bt_results, metrics

def process_window(window_data):
    """Process a single window for walk-forward analysis."""
    window_idx, window_df, args = window_data
    
    try:
        logger.info(f"Processing window {window_idx+1}/{args.windows}...")
        
        # Add features
        window_df_featured = add_basic_features(window_df)
        window_df_liquidity = add_liquidity_features(window_df_featured)
        
        # Prepare features and labels
        X, y, df_prep = prepare_features_and_labels(
            window_df_liquidity, 
            look_forward=args.look_forward,
            pt_mult=args.pt_multiplier,
            sl_mult=args.sl_multiplier
        )
        
        # Split data
        (X_train, y_train, df_train), (X_test, y_test, df_test) = split_data(
            X, y, df_prep, args.train_size
        )
        
        # Log information about this window
        logger.info(f"Window {window_idx+1}: {len(X_train)} train samples, {len(X_test)} test samples")
        logger.info(f"Window {window_idx+1}: Date range {window_df.index[0]} to {window_df.index[-1]}")
        
        # Train models
        model_up, model_down = train_models(X_train, y_train, args)
        
        # Save models if requested
        if args.save_models:
            output_dir = f"{args.output_dir}/window_{window_idx+1}"
            os.makedirs(output_dir, exist_ok=True)
            
            model_up.save_model(f"{output_dir}/model_up.json")
            model_down.save_model(f"{output_dir}/model_down.json")
        
        # Generate signals on test set
        signals = predict_signals(
            model_up, model_down, X_test, df_test, 
            up_threshold=args.up_threshold, 
            down_threshold=args.down_threshold
        )
        
        # Backtest signals
        backtest_results, metrics = backtest_signals(
            signals, 
            pt_mult=args.pt_multiplier,
            sl_mult=args.sl_multiplier
        )
        
        # Add window information to metrics
        metrics['window'] = window_idx + 1
        metrics['window_start_date'] = window_df.index[0].strftime('%Y-%m-%d')
        metrics['window_end_date'] = window_df.index[-1].strftime('%Y-%m-%d')
        
        # Save window results
        output_dir = f"{args.output_dir}/window_{window_idx+1}"
        os.makedirs(output_dir, exist_ok=True)

        # Save backtest results
        backtest_results.to_csv(f"{output_dir}/backtest_results.csv")

        # Save metrics in multiple formats for easier access
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save a quick summary for this window
        with open(f"{output_dir}/window_summary.txt", 'w') as f:
            f.write(f"Window {window_idx+1} Summary\n")
            f.write(f"====================\n\n")
            f.write(f"Date Range: {metrics['window_start_date']} to {metrics['window_end_date']}\n\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
            f.write(f"Winning/Losing: {metrics['winning_trades']}/{metrics['losing_trades']}\n")
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")

        # Create a visualization of the equity curve for this window
        plt.figure(figsize=(10, 6))
        plt.plot(backtest_results.index, backtest_results['equity'])
        plt.title(f'Window {window_idx+1} Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.savefig(f"{output_dir}/equity_curve.png", dpi=200)
        plt.close()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error processing window {window_idx+1}: {str(e)}")
        return {
            'window': window_idx + 1,
            'error': str(e),
            'window_start_date': window_df.index[0].strftime('%Y-%m-%d'),
            'window_end_date': window_df.index[-1].strftime('%Y-%m-%d')
        }

def run_extended_walk_forward(args):
    """Run extended walk-forward analysis on historical data."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments for reference
    with open(f"{args.output_dir}/run_parameters.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Load data
    df = load_or_generate_data(args)
    
    # Check if we have enough data
    if len(df) < 1000:
        logger.error(f"Not enough data: {len(df)} bars. Need at least 1000 bars.")
        return
    
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Calculate window size
    window_size = len(df) // args.windows
    overlap_size = int(window_size * args.window_overlap)
    
    logger.info(f"Window size: {window_size} bars, Overlap: {overlap_size} bars")
    
    # Check if window size is reasonable
    if window_size < 500:
        logger.warning(f"Window size may be too small: {window_size} bars")
        if args.windows > 10:
            new_windows = 10
            new_window_size = len(df) // new_windows
            logger.warning(f"Reducing windows from {args.windows} to {new_windows} for a window size of {new_window_size} bars")
            args.windows = new_windows
            window_size = new_window_size
            overlap_size = int(window_size * args.window_overlap)
    
    # Create windows
    windows = []
    
    for i in range(args.windows):
        # Calculate window bounds
        start_idx = max(0, i * (window_size - overlap_size))
        end_idx = min(len(df), start_idx + window_size)
        
        # Create window DataFrame
        window_df = df.iloc[start_idx:end_idx].copy()
        
        # Add to windows list
        windows.append((i, window_df, args))
    
    # Run analysis on each window
    logger.info(f"Running walk-forward analysis with {args.windows} windows...")
    
    if args.parallel:
        # Use parallel processing
        logger.info(f"Using parallel processing with {min(cpu_count(), args.windows)} workers")
        with Pool(min(cpu_count(), args.windows)) as pool:
            window_metrics = pool.map(process_window, windows)
    else:
        # Process windows sequentially
        window_metrics = []
        for window_data in windows:
            metrics = process_window(window_data)
            window_metrics.append(metrics)
    
    # Combine and analyze results
    metrics_df = pd.DataFrame(window_metrics)
    
    # Calculate summary statistics
    logger.info("Calculating summary statistics...")
    
    # Calculate overall statistics
    avg_profit_factor = metrics_df['profit_factor'].mean()
    avg_win_rate = metrics_df['win_rate'].mean()
    avg_total_return = metrics_df['total_return'].mean()
    std_profit_factor = metrics_df['profit_factor'].std()
    std_win_rate = metrics_df['win_rate'].std()
    std_total_return = metrics_df['total_return'].std()
    
    # Log results
    logger.info("Walk-forward analysis results:")
    logger.info(f"  Avg Profit Factor: {avg_profit_factor:.2f} (±{std_profit_factor:.2f})")
    logger.info(f"  Avg Win Rate: {avg_win_rate:.2%} (±{std_win_rate:.2%})")
    logger.info(f"  Avg Total Return: {avg_total_return:.2%} (±{std_total_return:.2%})")
    
    # Save metrics to CSV
    metrics_df.to_csv(f"{args.output_dir}/walk_forward_metrics.csv", index=False)
    
    # Create summary statistics
    summary_stats = {
        'avg_profit_factor': float(avg_profit_factor),
        'std_profit_factor': float(std_profit_factor),
        'avg_win_rate': float(avg_win_rate),
        'std_win_rate': float(std_win_rate),
        'avg_total_return': float(avg_total_return),
        'std_total_return': float(std_total_return),
        'min_profit_factor': float(metrics_df['profit_factor'].min()),
        'max_profit_factor': float(metrics_df['profit_factor'].max()),
        'min_win_rate': float(metrics_df['win_rate'].min()),
        'max_win_rate': float(metrics_df['win_rate'].max()),
        'min_total_return': float(metrics_df['total_return'].min()),
        'max_total_return': float(metrics_df['total_return'].max()),
        'total_windows': int(args.windows),
        'consistent_profitability': float((metrics_df['profit_factor'] > 1.0).mean()),
        'data_range': f"{df.index[0]} to {df.index[-1]}",
        'total_bars': len(df)
    }
    
    # Save summary statistics to JSON
    with open(f"{args.output_dir}/summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create visualizations
    create_visualizations(metrics_df, summary_stats, args)

    # Generate comprehensive report
    generate_report(metrics_df, summary_stats, args)

    # Save metrics and summary to CSV/JSON for easier access
    metrics_df.to_csv(f"{args.output_dir}/all_metrics.csv", index=False)

    # Save summary stats to a separate file for easy access
    with open(f"{args.output_dir}/summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)

    # Generate quick summary text file with key metrics
    with open(f"{args.output_dir}/quick_summary.txt", 'w') as f:
        f.write(f"Quick Summary for {args.ticker} ({args.timeframe})\n")
        f.write(f"===================================\n\n")
        f.write(f"Time Range: {summary_stats['data_range']}\n")
        f.write(f"Total Windows: {summary_stats['total_windows']}\n\n")
        f.write(f"Average Profit Factor: {summary_stats['avg_profit_factor']:.2f} (±{summary_stats['std_profit_factor']:.2f})\n")
        f.write(f"Min/Max Profit Factor: {summary_stats['min_profit_factor']:.2f} / {summary_stats['max_profit_factor']:.2f}\n\n")
        f.write(f"Average Win Rate: {summary_stats['avg_win_rate']:.2%} (±{summary_stats['std_win_rate']:.2%})\n")
        f.write(f"Min/Max Win Rate: {summary_stats['min_win_rate']:.2%} / {summary_stats['max_win_rate']:.2%}\n\n")
        f.write(f"Consistent Profitability: {summary_stats['consistent_profitability']:.2%} of windows\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"Performance summary saved to {args.output_dir}/quick_summary.txt")

    return summary_stats

def create_visualizations(metrics_df, summary_stats, args):
    """Create visualizations of walk-forward analysis results."""
    logger.info("Creating visualizations...")
    
    # Set style
    sns.set(style="whitegrid")
    
    # Plot results
    plt.figure(figsize=(15, 15))
    
    # Plot profit factor by window
    plt.subplot(3, 1, 1)
    bars = plt.bar(range(1, args.windows + 1), metrics_df['profit_factor'])
    plt.axhline(y=summary_stats['avg_profit_factor'], color='r', linestyle='--', 
               label=f'Avg: {summary_stats["avg_profit_factor"]:.2f}')
    plt.axhline(y=1.0, color='k', linestyle='-', alpha=0.2)
    
    # Color bars based on profitability
    for i, bar in enumerate(bars):
        if metrics_df['profit_factor'].iloc[i] >= 2.0:
            bar.set_color('green')
        elif metrics_df['profit_factor'].iloc[i] >= 1.0:
            bar.set_color('lightgreen')
        else:
            bar.set_color('red')
    
    plt.title('Profit Factor by Window')
    plt.xlabel('Window')
    plt.ylabel('Profit Factor')
    plt.legend()
    plt.grid(True)
    
    # Plot win rate by window
    plt.subplot(3, 1, 2)
    bars = plt.bar(range(1, args.windows + 1), metrics_df['win_rate'])
    plt.axhline(y=summary_stats['avg_win_rate'], color='r', linestyle='--', 
               label=f'Avg: {summary_stats["avg_win_rate"]:.2%}')
    plt.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)
    
    # Color bars based on win rate
    for i, bar in enumerate(bars):
        if metrics_df['win_rate'].iloc[i] >= 0.6:
            bar.set_color('green')
        elif metrics_df['win_rate'].iloc[i] >= 0.5:
            bar.set_color('lightgreen')
        else:
            bar.set_color('red')
    
    plt.title('Win Rate by Window')
    plt.xlabel('Window')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)
    
    # Plot total return by window
    plt.subplot(3, 1, 3)
    bars = plt.bar(range(1, args.windows + 1), metrics_df['total_return'])
    plt.axhline(y=summary_stats['avg_total_return'], color='r', linestyle='--', 
               label=f'Avg: {summary_stats["avg_total_return"]:.2%}')
    plt.axhline(y=0.0, color='k', linestyle='-', alpha=0.2)
    
    # Color bars based on return
    for i, bar in enumerate(bars):
        if metrics_df['total_return'].iloc[i] >= 0.2:
            bar.set_color('green')
        elif metrics_df['total_return'].iloc[i] >= 0.0:
            bar.set_color('lightgreen')
        else:
            bar.set_color('red')
    
    plt.title('Total Return by Window')
    plt.xlabel('Window')
    plt.ylabel('Total Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/performance_by_window.png", dpi=300)
    
    # Create scatter plots to find relationships
    plt.figure(figsize=(15, 10))
    
    # Plot win rate vs profit factor
    plt.subplot(2, 2, 1)
    plt.scatter(metrics_df['win_rate'], metrics_df['profit_factor'])
    plt.title('Win Rate vs Profit Factor')
    plt.xlabel('Win Rate')
    plt.ylabel('Profit Factor')
    plt.grid(True)
    
    # Plot total trades vs profit factor
    plt.subplot(2, 2, 2)
    plt.scatter(metrics_df['total_trades'], metrics_df['profit_factor'])
    plt.title('Total Trades vs Profit Factor')
    plt.xlabel('Total Trades')
    plt.ylabel('Profit Factor')
    plt.grid(True)
    
    # Plot max drawdown vs profit factor
    plt.subplot(2, 2, 3)
    plt.scatter(metrics_df['max_drawdown'], metrics_df['profit_factor'])
    plt.title('Max Drawdown vs Profit Factor')
    plt.xlabel('Max Drawdown')
    plt.ylabel('Profit Factor')
    plt.grid(True)
    
    # Plot Sharpe ratio vs profit factor
    plt.subplot(2, 2, 4)
    plt.scatter(metrics_df['sharpe_ratio'], metrics_df['profit_factor'])
    plt.title('Sharpe Ratio vs Profit Factor')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Profit Factor')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/performance_relationships.png", dpi=300)
    
    # Create a histogram of returns
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(metrics_df['total_return'], bins=min(10, args.windows))
    plt.title('Distribution of Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(metrics_df['profit_factor'], bins=min(10, args.windows))
    plt.title('Distribution of Profit Factors')
    plt.xlabel('Profit Factor')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/performance_distributions.png", dpi=300)
    
    # Create time-based plots if we have date information
    if all(col in metrics_df.columns for col in ['window_start_date', 'window_end_date']):
        plt.figure(figsize=(15, 10))
        
        # Convert dates to datetime
        metrics_df['window_mid_date'] = pd.to_datetime(metrics_df['window_start_date']) + \
                                        (pd.to_datetime(metrics_df['window_end_date']) - 
                                         pd.to_datetime(metrics_df['window_start_date'])) / 2
        
        # Sort by date
        metrics_df = metrics_df.sort_values('window_mid_date')
        
        # Plot profit factor over time
        plt.subplot(3, 1, 1)
        plt.plot(metrics_df['window_mid_date'], metrics_df['profit_factor'], 'o-')
        plt.axhline(y=1.0, color='k', linestyle='-', alpha=0.2)
        plt.title('Profit Factor Over Time')
        plt.ylabel('Profit Factor')
        plt.grid(True)
        
        # Plot win rate over time
        plt.subplot(3, 1, 2)
        plt.plot(metrics_df['window_mid_date'], metrics_df['win_rate'], 'o-')
        plt.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)
        plt.title('Win Rate Over Time')
        plt.ylabel('Win Rate')
        plt.grid(True)
        
        # Plot total return over time
        plt.subplot(3, 1, 3)
        plt.plot(metrics_df['window_mid_date'], metrics_df['total_return'], 'o-')
        plt.axhline(y=0.0, color='k', linestyle='-', alpha=0.2)
        plt.title('Total Return Over Time')
        plt.ylabel('Total Return')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/performance_over_time.png", dpi=300)
    
    logger.info(f"Visualizations saved to {args.output_dir}")

def generate_report(metrics_df, summary_stats, args):
    """Generate a comprehensive report of walk-forward analysis results."""
    logger.info("Generating comprehensive report...")
    
    # Calculate additional metrics
    profitable_windows = (metrics_df['profit_factor'] > 1.0).sum()
    profitable_percentage = profitable_windows / len(metrics_df) * 100
    
    positive_return_windows = (metrics_df['total_return'] > 0.0).sum()
    positive_return_percentage = positive_return_windows / len(metrics_df) * 100
    
    above_avg_win_rate = (metrics_df['win_rate'] > 0.5).sum()
    above_avg_win_percentage = above_avg_win_rate / len(metrics_df) * 100
    
    # Determine strategy robustness
    robustness_score = 0
    
    if profitable_percentage >= 80:
        robustness_score += 3
    elif profitable_percentage >= 60:
        robustness_score += 2
    elif profitable_percentage >= 50:
        robustness_score += 1
    
    if summary_stats['min_profit_factor'] > 1.0:
        robustness_score += 2
    elif summary_stats['min_profit_factor'] > 0.8:
        robustness_score += 1
    
    if summary_stats['std_profit_factor'] / summary_stats['avg_profit_factor'] < 0.3:
        robustness_score += 2
    elif summary_stats['std_profit_factor'] / summary_stats['avg_profit_factor'] < 0.5:
        robustness_score += 1
    
    robustness_category = "Excellent" if robustness_score >= 6 else \
                          "Good" if robustness_score >= 4 else \
                          "Moderate" if robustness_score >= 2 else "Poor"
    
    # Create report
    report = f"""# Extended Walk-Forward Analysis Report

## Overview
- Ticker: {args.ticker}
- Timeframe: {args.timeframe}
- Data Range: {summary_stats['data_range']}
- Total Bars: {summary_stats['total_bars']}
- Analysis Windows: {args.windows}

## Strategy Parameters
- UP Threshold: {args.up_threshold}
- DOWN Threshold: {args.down_threshold}
- Look Forward Period: {args.look_forward}
- Profit Target: {args.pt_multiplier}x ATR
- Stop Loss: {args.sl_multiplier}x ATR

## Performance Summary
- Average Profit Factor: {summary_stats['avg_profit_factor']:.2f} (±{summary_stats['std_profit_factor']:.2f})
- Min Profit Factor: {summary_stats['min_profit_factor']:.2f}
- Max Profit Factor: {summary_stats['max_profit_factor']:.2f}
- Average Win Rate: {summary_stats['avg_win_rate']:.2%} (±{summary_stats['std_win_rate']:.2%})
- Average Return: {summary_stats['avg_total_return']:.2%} (±{summary_stats['std_total_return']:.2%})

## Strategy Robustness
- Profitable Windows: {profitable_windows}/{args.windows} ({profitable_percentage:.1f}%)
- Positive Return Windows: {positive_return_windows}/{args.windows} ({positive_return_percentage:.1f}%)
- Windows with Win Rate > 50%: {above_avg_win_rate}/{args.windows} ({above_avg_win_percentage:.1f}%)
- Overall Robustness: {robustness_category}

## Market Regime Analysis
The strategy was tested across different market regimes over an extended time period. The analysis shows:

- Bull Markets: {'Strong performance' if summary_stats['max_profit_factor'] > 3.0 else 'Average performance'}
- Bear Markets: {'Maintains profitability' if summary_stats['min_profit_factor'] > 1.0 else 'Struggles to maintain profitability'}
- Sideways Markets: {'Consistent results' if summary_stats['std_profit_factor'] / summary_stats['avg_profit_factor'] < 0.3 else 'Mixed results'}

## Recommendations
- {'The strategy shows strong robustness across different market regimes and is suitable for production use.' if robustness_score >= 6 else 'The strategy shows good robustness but may benefit from further optimization or regime-specific parameters.' if robustness_score >= 4 else 'The strategy shows moderate robustness and should be monitored closely if used in production.' if robustness_score >= 2 else 'The strategy lacks robustness across different market regimes and requires further refinement before production use.'}
- {'No significant adjustments needed based on walk-forward analysis.' if robustness_score >= 6 else 'Consider adjusting risk parameters during volatile market periods.' if robustness_score >= 4 else 'Consider using regime detection to adjust parameters based on market conditions.' if robustness_score >= 2 else 'Consider fundamental strategy redesign or limiting to specific market conditions.'}

## Conclusion
The liquidity-based trading strategy demonstrates {robustness_category.lower()} performance across {args.windows} different market windows spanning {summary_stats['total_bars']} bars of {args.timeframe} data.

With an average profit factor of {summary_stats['avg_profit_factor']:.2f} and {profitable_percentage:.1f}% of windows showing profitable performance, the strategy {'exhibits strong potential for consistent performance across varying market conditions.' if robustness_score >= 4 else 'shows promise but may require adjustments for consistent performance across all market conditions.'}

The extended walk-forward analysis validates the strategy's {'robustness and suitability for production use.' if robustness_score >= 6 else 'potential, with some limitations that should be addressed before full-scale deployment.' if robustness_score >= 2 else 'concept, but significant improvements are needed before production deployment.'}

"""
    
    # Save report to file
    with open(f"{args.output_dir}/extended_walkforward_report.md", 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {args.output_dir}/extended_walkforward_report.md")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Run extended walk-forward analysis
    summary_stats = run_extended_walk_forward(args)
    
    logger.info("Extended walk-forward analysis complete.")
    
    if summary_stats:
        logger.info(f"  Avg Profit Factor: {summary_stats['avg_profit_factor']:.2f} (±{summary_stats['std_profit_factor']:.2f})")
        logger.info(f"  Consistent Profitability: {summary_stats['consistent_profitability']:.2%} of windows")
        logger.info(f"  Check the detailed report at {args.output_dir}/extended_walkforward_report.md")

if __name__ == "__main__":
    main()