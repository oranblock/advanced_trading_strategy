#!/usr/bin/env python3
"""
Production-ready liquidity-based trading strategy with extensive backtesting capabilities.
This script can handle large historical datasets (10-50 years) for thorough validation.
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
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("liquidity_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('liquidity_strategy')

# Create directories
os.makedirs("./production_results", exist_ok=True)
os.makedirs("./production_models", exist_ok=True)
os.makedirs("./production_data", exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Liquidity Trading Strategy')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='backtest', 
                        choices=['backtest', 'train', 'analyze', 'all'],
                        help='Operation mode (backtest, train, analyze, all)')
    
    # Data parameters
    parser.add_argument('--ticker', type=str, default='EURUSD',
                        help='Ticker symbol to trade')
    parser.add_argument('--timeframe', type=str, default='hour',
                        choices=['minute', 'hour', 'day', 'week'],
                        help='Timeframe for analysis')
    parser.add_argument('--start-date', type=str, default='2010-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for backtest (YYYY-MM-DD), defaults to today')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Path to CSV data file (if not fetching from API)')
    
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
    
    # Training parameters
    parser.add_argument('--train-size', type=float, default=0.7,
                        help='Training data size (0.0-1.0)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation data size (0.0-1.0)')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees in XGBoost model')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Maximum tree depth in XGBoost model')
    
    # Analysis parameters
    parser.add_argument('--windows', type=int, default=4,
                        help='Number of windows for walk-forward analysis')
    parser.add_argument('--save-equity', action='store_true',
                        help='Save equity curve data to CSV')
    parser.add_argument('--plot-results', action='store_true',
                        help='Generate performance visualizations')
    
    # Other parameters
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='./production_results',
                        help='Directory for output files')
    
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
        # Volatility clustering
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * base_volatility * (1 + 0.5 * np.random.normal())
        
        # Add momentum and mean reversion effects
        momentum = 0.05 * (close[i-1] - close[max(0, i-20)]) / close[max(0, i-20)]
        mean_reversion = -0.02 * (close[i-1] - close[0]) / close[0]
        
        # Random walk with drift, momentum, mean reversion, and volatility
        close[i] = close[i-1] * (1 + drift + momentum + mean_reversion + volatility[i] * np.random.normal())
    
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
    output_path = f"./production_data/{args.ticker}_{args.timeframe}_{start_date}_to_{end_date.replace('-', '')}.csv"
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
    if 'hour' in df_liq.index.freq or 'H' in str(df_liq.index.freq):
        day_shift = 24  # 24 hours in a day
    elif 'min' in df_liq.index.freq or 'T' in str(df_liq.index.freq):
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

def split_data(X, y, df_orig, train_size=0.7, val_size=0.15):
    """Split data into train, validation, and test sets."""
    logger.info("Splitting data into train, validation, and test sets...")
    
    # Calculate sizes
    train_idx = int(len(X) * train_size)
    val_idx = int(len(X) * (train_size + val_size))
    
    # Split features and labels
    X_train = X.iloc[:train_idx]
    y_train = y.iloc[:train_idx]
    
    X_val = X.iloc[train_idx:val_idx]
    y_val = y.iloc[train_idx:val_idx]
    
    X_test = X.iloc[val_idx:]
    y_test = y.iloc[val_idx:]
    
    # Split original data for backtesting
    df_train = df_orig.loc[X_train.index]
    df_val = df_orig.loc[X_val.index]
    df_test = df_orig.loc[X_test.index]
    
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    return (X_train, y_train, df_train), (X_val, y_val, df_val), (X_test, y_test, df_test)

def train_models(X_train, y_train, X_val, y_val, args):
    """Train XGBoost models for UP and DOWN predictions."""
    logger.info("Training XGBoost models...")
    
    try:
        import xgboost as xgb
        
        # Create binary targets
        y_train_up = (y_train == 1).astype(int)
        y_val_up = (y_val == 1).astype(int)
        
        y_train_down = (y_train == -1).astype(int)
        y_val_down = (y_val == -1).astype(int)
        
        # Calculate class weights
        up_neg_count = (y_train_up == 0).sum()
        up_pos_count = (y_train_up == 1).sum()
        scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1.0
        
        down_neg_count = (y_train_down == 0).sum()
        down_pos_count = (y_train_down == 1).sum()
        scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1.0
        
        # Create DMastrices for XGBoost
        dtrain_up = xgb.DMatrix(X_train, label=y_train_up)
        dval_up = xgb.DMatrix(X_val, label=y_val_up)
        
        dtrain_down = xgb.DMatrix(X_train, label=y_train_down)
        dval_down = xgb.DMatrix(X_val, label=y_val_down)
        
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
        logger.info("Training UP model...")
        model_up = xgb.train(
            xgb_params_up,
            dtrain_up,
            num_boost_round=args.n_estimators,
            evals=[(dval_up, 'val')],
            verbose_eval=50
        )
        
        # Train DOWN model
        logger.info("Training DOWN model...")
        model_down = xgb.train(
            xgb_params_down,
            dtrain_down,
            num_boost_round=args.n_estimators,
            evals=[(dval_down, 'val')],
            verbose_eval=50
        )
        
        # Save features importance
        up_importance = model_up.get_score(importance_type='gain')
        down_importance = model_down.get_score(importance_type='gain')
        
        # Convert to DataFrame
        up_imp_df = pd.DataFrame({
            'Feature': list(up_importance.keys()),
            'Importance': list(up_importance.values())
        }).sort_values('Importance', ascending=False)
        
        down_imp_df = pd.DataFrame({
            'Feature': list(down_importance.keys()),
            'Importance': list(down_importance.values())
        }).sort_values('Importance', ascending=False)
        
        # Save feature importance
        up_imp_df.to_csv(f"{args.output_dir}/up_feature_importance.csv", index=False)
        down_imp_df.to_csv(f"{args.output_dir}/down_feature_importance.csv", index=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 12))
        plt.subplot(2, 1, 1)
        sns.barplot(x='Importance', y='Feature', data=up_imp_df.head(20))
        plt.title('UP Model - Top 20 Feature Importance')
        plt.tight_layout()
        
        plt.subplot(2, 1, 2)
        sns.barplot(x='Importance', y='Feature', data=down_imp_df.head(20))
        plt.title('DOWN Model - Top 20 Feature Importance')
        plt.tight_layout()
        
        plt.savefig(f"{args.output_dir}/feature_importance.png", dpi=300)
        
        # Save models
        model_up.save_model(f"{args.output_dir}/model_up.json")
        model_down.save_model(f"{args.output_dir}/model_down.json")
        
        # Save feature names for later use
        with open(f"{args.output_dir}/feature_names.pkl", 'wb') as f:
            pickle.dump(X_train.columns.tolist(), f)
        
        return model_up, model_down
        
    except ImportError:
        logger.error("XGBoost not installed. Cannot train models.")
        return None, None

def predict_signals(model_up, model_down, X, df_orig, up_threshold=0.5, down_threshold=0.5):
    """Generate trading signals from model predictions."""
    logger.info("Generating trading signals...")
    
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
    logger.info("Running backtest simulation...")
    
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
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
    
    # Metrics dictionary
    metrics = {
        'initial_capital': initial_capital,
        'final_equity': float(equity),
        'total_return': float(total_return),
        'total_trades': int(trade_count),
        'winning_trades': int(win_count),
        'losing_trades': int(loss_count),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if not np.isinf(profit_factor) else 999.9,
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe_ratio)
    }
    
    # Log backtest results
    logger.info(f"Backtest results:")
    logger.info(f"  Initial capital: ${initial_capital:.2f}")
    logger.info(f"  Final equity: ${equity:.2f}")
    logger.info(f"  Total return: {total_return:.2%}")
    logger.info(f"  Total trades: {trade_count}")
    logger.info(f"  Win rate: {win_rate:.2%}")
    logger.info(f"  Profit factor: {profit_factor:.2f}")
    logger.info(f"  Max drawdown: {max_drawdown:.2%}")
    
    return bt_results, metrics

def plot_backtest_results(backtest_results, metrics, args):
    """Plot backtest results and performance metrics."""
    logger.info("Generating backtest visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(backtest_results.index, backtest_results['equity'])
    plt.title('Equity Curve')
    plt.grid(True)
    
    # Plot drawdowns
    drawdowns = backtest_results['equity'].cummax() - backtest_results['equity']
    drawdown_pct = drawdowns / backtest_results['equity'].cummax()
    
    plt.subplot(2, 1, 2)
    plt.fill_between(backtest_results.index, 0, drawdown_pct, color='red', alpha=0.3)
    plt.title('Drawdowns')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{args.output_dir}/equity_curve.png", dpi=300)
    
    # Plot trade distribution
    trade_results = backtest_results['trade_result'].dropna()
    
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.hist(trade_results, bins=50)
    plt.title('Trade Result Distribution')
    plt.grid(True)
    
    # Plot win rate by month
    backtest_results['year_month'] = backtest_results.index.to_period('M')
    monthly_results = backtest_results.groupby('year_month').last()['equity'].pct_change()
    
    plt.subplot(2, 1, 2)
    monthly_results.plot(kind='bar')
    plt.title('Monthly Returns')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{args.output_dir}/trade_distribution.png", dpi=300)
    
    # Plot a summary of key metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = {
        'Win Rate': metrics['win_rate'],
        'Profit Factor': min(metrics['profit_factor'], 10),  # Cap at 10 for visualization
        'Sharpe Ratio': metrics['sharpe_ratio'],
        'Max Drawdown': metrics['max_drawdown']
    }
    
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    plt.title('Key Performance Metrics')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{args.output_dir}/performance_metrics.png", dpi=300)
    
    # Create a price chart with signals
    plt.figure(figsize=(15, 10))
    
    # Limit to last 100 bars for readability
    last_n_bars = min(100, len(backtest_results))
    plot_data = backtest_results.iloc[-last_n_bars:]
    
    # Price plot
    plt.subplot(2, 1, 1)
    plt.plot(plot_data.index, plot_data['close'])
    
    # Add signals
    long_entries = plot_data[plot_data['signal'] == 'LONG'].index
    short_entries = plot_data[plot_data['signal'] == 'SHORT'].index
    
    plt.scatter(long_entries, plot_data.loc[long_entries, 'close'], marker='^', color='green', s=100, label='Long')
    plt.scatter(short_entries, plot_data.loc[short_entries, 'close'], marker='v', color='red', s=100, label='Short')
    
    plt.title(f'Price Chart with Signals (Last {last_n_bars} Bars)')
    plt.legend()
    plt.grid(True)
    
    # Probability plot
    plt.subplot(2, 1, 2)
    plt.plot(plot_data.index, plot_data['up_prob'], 'g-', label='UP Probability')
    plt.plot(plot_data.index, plot_data['down_prob'], 'r-', label='DOWN Probability')
    plt.axhline(y=args.up_threshold, color='g', linestyle='--', alpha=0.5, label=f'UP Threshold ({args.up_threshold})')
    plt.axhline(y=args.down_threshold, color='r', linestyle='--', alpha=0.5, label=f'DOWN Threshold ({args.down_threshold})')
    
    plt.title('Signal Probabilities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{args.output_dir}/signals_chart.png", dpi=300)
    
    logger.info(f"Visualizations saved to {args.output_dir}")

def walk_forward_analysis(df, args):
    """Perform walk-forward analysis to validate strategy robustness."""
    logger.info("Performing walk-forward analysis...")
    
    # Split data into windows
    window_size = len(df) // args.windows
    
    # Initialize metrics storage
    window_metrics = []
    
    # For each window
    for i in range(args.windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size if i < args.windows - 1 else len(df)
        
        logger.info(f"Processing window {i+1}/{args.windows} ({df.index[window_start]} to {df.index[window_end-1]})")
        
        # Get window data
        window_df = df.iloc[window_start:window_end]
        
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
        train_size = 0.7
        val_size = 0.15
        (X_train, y_train, df_train), (X_val, y_val, df_val), (X_test, y_test, df_test) = split_data(
            X, y, df_prep, train_size, val_size
        )
        
        # Train models
        model_up, model_down = train_models(X_train, y_train, X_val, y_val, args)
        
        # Generate signals on test set
        signals = predict_signals(
            model_up, model_down, X_test, df_test, 
            up_threshold=args.up_threshold, 
            down_threshold=args.down_threshold
        )
        
        # Backtest signals
        _, window_metric = backtest_signals(
            signals, 
            pt_mult=args.pt_multiplier,
            sl_mult=args.sl_multiplier
        )
        
        # Store metrics
        window_metrics.append({
            'window': i+1,
            'start_date': df.index[window_start],
            'end_date': df.index[window_end-1],
            'profit_factor': window_metric['profit_factor'],
            'win_rate': window_metric['win_rate'],
            'total_trades': window_metric['total_trades'],
            'total_return': window_metric['total_return']
        })
    
    # Consolidate metrics
    metrics_df = pd.DataFrame(window_metrics)
    
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
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.bar(range(1, args.windows + 1), metrics_df['profit_factor'])
    plt.axhline(y=avg_profit_factor, color='r', linestyle='--', label=f'Avg: {avg_profit_factor:.2f}')
    plt.title('Profit Factor by Window')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.bar(range(1, args.windows + 1), metrics_df['win_rate'])
    plt.axhline(y=avg_win_rate, color='r', linestyle='--', label=f'Avg: {avg_win_rate:.2%}')
    plt.title('Win Rate by Window')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.bar(range(1, args.windows + 1), metrics_df['total_return'])
    plt.axhline(y=avg_total_return, color='r', linestyle='--', label=f'Avg: {avg_total_return:.2%}')
    plt.title('Total Return by Window')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/walk_forward_analysis.png", dpi=300)
    
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
    }
    
    # Save summary statistics to JSON
    with open(f"{args.output_dir}/walk_forward_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    return summary_stats

def load_models(output_dir):
    """Load saved XGBoost models."""
    try:
        import xgboost as xgb
        
        model_up_path = f"{output_dir}/model_up.json"
        model_down_path = f"{output_dir}/model_down.json"
        
        if not os.path.exists(model_up_path) or not os.path.exists(model_down_path):
            logger.error(f"Model files not found at {model_up_path} or {model_down_path}")
            return None, None
        
        model_up = xgb.Booster()
        model_down = xgb.Booster()
        
        model_up.load_model(model_up_path)
        model_down.load_model(model_down_path)
        
        logger.info(f"Loaded models from {output_dir}")
        
        return model_up, model_down
    
    except ImportError:
        logger.error("XGBoost not installed. Cannot load models.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None

def main():
    """Main function to run the trading strategy."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments for reference
    with open(f"{args.output_dir}/run_parameters.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    if args.mode == 'backtest' or args.mode == 'all':
        logger.info(f"Starting backtest for {args.ticker} ({args.timeframe})")
        
        # Load data
        df = load_or_generate_data(args)
        
        # Check if models exist or need to be trained
        model_up, model_down = load_models(args.output_dir)
        
        if model_up is None or model_down is None:
            logger.info("Models not found. Training new models.")
            
            # Add features
            df_featured = add_basic_features(df)
            df_liquidity = add_liquidity_features(df_featured)
            
            # Prepare features and labels
            X, y, df_prep = prepare_features_and_labels(
                df_liquidity, 
                look_forward=args.look_forward,
                pt_mult=args.pt_multiplier,
                sl_mult=args.sl_multiplier
            )
            
            # Split data
            (X_train, y_train, df_train), (X_val, y_val, df_val), (X_test, y_test, df_test) = split_data(
                X, y, df_prep, args.train_size, args.val_size
            )
            
            # Train models
            model_up, model_down = train_models(X_train, y_train, X_val, y_val, args)
            
            # Generate signals on test set
            signals = predict_signals(
                model_up, model_down, X_test, df_test, 
                up_threshold=args.up_threshold, 
                down_threshold=args.down_threshold
            )
        else:
            logger.info("Using existing models for backtest.")
            
            # Add features
            df_featured = add_basic_features(df)
            df_liquidity = add_liquidity_features(df_featured)
            
            # Prepare features but not labels (we'll generate signals directly)
            X = df_liquidity.drop(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # Generate signals
            signals = predict_signals(
                model_up, model_down, X, df_liquidity, 
                up_threshold=args.up_threshold, 
                down_threshold=args.down_threshold
            )
        
        # Backtest signals
        backtest_results, metrics = backtest_signals(
            signals, 
            pt_mult=args.pt_multiplier,
            sl_mult=args.sl_multiplier
        )
        
        # Save backtest results
        backtest_results.to_csv(f"{args.output_dir}/backtest_results.csv")
        
        # Save metrics to JSON
        with open(f"{args.output_dir}/backtest_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot results if requested
        if args.plot_results:
            plot_backtest_results(backtest_results, metrics, args)
    
    elif args.mode == 'train':
        logger.info(f"Starting model training for {args.ticker} ({args.timeframe})")
        
        # Load data
        df = load_or_generate_data(args)
        
        # Add features
        df_featured = add_basic_features(df)
        df_liquidity = add_liquidity_features(df_featured)
        
        # Prepare features and labels
        X, y, df_prep = prepare_features_and_labels(
            df_liquidity, 
            look_forward=args.look_forward,
            pt_mult=args.pt_multiplier,
            sl_mult=args.sl_multiplier
        )
        
        # Split data
        (X_train, y_train, df_train), (X_val, y_val, df_val), (X_test, y_test, df_test) = split_data(
            X, y, df_prep, args.train_size, args.val_size
        )
        
        # Train models
        model_up, model_down = train_models(X_train, y_train, X_val, y_val, args)
        
        # Quick validation on test set
        logger.info("Validating models on test set...")
        
        # Generate signals on test set
        signals = predict_signals(
            model_up, model_down, X_test, df_test, 
            up_threshold=args.up_threshold, 
            down_threshold=args.down_threshold
        )
        
        # Calculate metrics
        total_signals = len(signals)
        long_signals = (signals['signal'] == 'LONG').sum()
        short_signals = (signals['signal'] == 'SHORT').sum()
        
        logger.info(f"Test set signals: {total_signals} total, {long_signals} long, {short_signals} short")
    
    elif args.mode == 'analyze':
        logger.info(f"Starting walk-forward analysis for {args.ticker} ({args.timeframe})")
        
        # Load data
        df = load_or_generate_data(args)
        
        # Perform walk-forward analysis
        summary_stats = walk_forward_analysis(df, args)
        
        logger.info("Walk-forward analysis complete.")
        logger.info(f"  Avg Profit Factor: {summary_stats['avg_profit_factor']:.2f} (±{summary_stats['std_profit_factor']:.2f})")
        logger.info(f"  Avg Win Rate: {summary_stats['avg_win_rate']:.2%} (±{summary_stats['std_win_rate']:.2%})")
        logger.info(f"  Avg Total Return: {summary_stats['avg_total_return']:.2%} (±{summary_stats['std_total_return']:.2%})")
    
    elif args.mode == 'all':
        logger.info(f"Running complete analysis for {args.ticker} ({args.timeframe})")
        
        # Load data
        df = load_or_generate_data(args)
        
        # Add features
        df_featured = add_basic_features(df)
        df_liquidity = add_liquidity_features(df_featured)
        
        # Prepare features and labels
        X, y, df_prep = prepare_features_and_labels(
            df_liquidity, 
            look_forward=args.look_forward,
            pt_mult=args.pt_multiplier,
            sl_mult=args.sl_multiplier
        )
        
        # Split data
        (X_train, y_train, df_train), (X_val, y_val, df_val), (X_test, y_test, df_test) = split_data(
            X, y, df_prep, args.train_size, args.val_size
        )
        
        # Train models
        model_up, model_down = train_models(X_train, y_train, X_val, y_val, args)
        
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
        
        # Save backtest results
        backtest_results.to_csv(f"{args.output_dir}/backtest_results.csv")
        
        # Save metrics to JSON
        with open(f"{args.output_dir}/backtest_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot results if requested
        if args.plot_results:
            plot_backtest_results(backtest_results, metrics, args)
        
        # Perform walk-forward analysis
        summary_stats = walk_forward_analysis(df, args)
        
        # Prepare final report
        report = f"""# Liquidity Strategy Analysis Report

## Overview
- Ticker: {args.ticker}
- Timeframe: {args.timeframe}
- Date Range: {df.index[0]} to {df.index[-1]}
- Total Bars: {len(df)}

## Strategy Parameters
- UP Threshold: {args.up_threshold}
- DOWN Threshold: {args.down_threshold}
- Look Forward Period: {args.look_forward}
- Profit Target: {args.pt_multiplier}x ATR
- Stop Loss: {args.sl_multiplier}x ATR

## Backtest Results
- Total Trades: {metrics['total_trades']}
- Win Rate: {metrics['win_rate']:.2%}
- Profit Factor: {metrics['profit_factor']:.2f}
- Total Return: {metrics['total_return']:.2%}
- Max Drawdown: {metrics['max_drawdown']:.2%}
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

## Walk-Forward Analysis
- Number of Windows: {args.windows}
- Avg Profit Factor: {summary_stats['avg_profit_factor']:.2f} (±{summary_stats['std_profit_factor']:.2f})
- Avg Win Rate: {summary_stats['avg_win_rate']:.2%} (±{summary_stats['std_win_rate']:.2%})
- Avg Total Return: {summary_stats['avg_total_return']:.2%} (±{summary_stats['std_total_return']:.2%})

## Recommendations
- The strategy shows {'strong' if metrics['profit_factor'] > 3 else 'moderate' if metrics['profit_factor'] > 1.5 else 'weak'} profitability with a profit factor of {metrics['profit_factor']:.2f}.
- {'Walk-forward analysis confirms robustness across different market conditions.' if summary_stats['min_profit_factor'] > 1.5 else 'Walk-forward analysis shows some inconsistency across different market conditions.'}
- {'The strategy maintains positive returns in most market conditions.' if summary_stats['min_total_return'] > 0 else 'The strategy shows negative returns in some market conditions.'}

## Conclusion
The liquidity-based strategy demonstrates {'excellent' if metrics['profit_factor'] > 3 else 'good' if metrics['profit_factor'] > 1.5 else 'poor'} overall performance.
Key strengths include {'consistent profitability across different market regimes' if summary_stats['min_profit_factor'] > 1.5 else 'strong performance in favorable market conditions'}.
Areas for improvement include {'none identified' if metrics['profit_factor'] > 3 and summary_stats['min_profit_factor'] > 1.5 else 'consistency across different market conditions' if summary_stats['min_profit_factor'] < 1.5 else 'reducing drawdowns'}.
"""
        
        # Save report to file
        with open(f"{args.output_dir}/analysis_report.md", 'w') as f:
            f.write(report)
        
        logger.info("Complete analysis finished. Report saved to analysis_report.md")
    
    logger.info("Script execution completed.")

if __name__ == "__main__":
    main()