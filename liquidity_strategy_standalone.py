#!/usr/bin/env python3
"""
Standalone implementation of the liquidity-based trading strategy
that doesn't depend on external data sources or complex imports.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('liquidity_strategy')

# Create directories
os.makedirs("./final_liquidity_strategy", exist_ok=True)
os.makedirs("./data", exist_ok=True)

def load_config():
    """Load the strategy configuration."""
    config_path = Path("./config/liquidity_settings.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return {
            'UP_PROB_THRESHOLD': 0.5,
            'DOWN_PROB_THRESHOLD': 0.5,
            'LOOK_FORWARD_CANDLES': 24,
            'PROFIT_TAKE_ATR': 2.5,
            'STOP_LOSS_ATR': 1.5
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def generate_synthetic_data(n_samples=8760):  # 1 year of hourly data
    """Generate synthetic price data for demonstration purposes."""
    logger.info("Generating synthetic market data...")
    
    # Create date range for 1 year of hourly data
    index = pd.date_range(start='2021-01-01', end='2022-01-01', freq='H')[:n_samples]
    
    # Generate synthetic OHLCV data
    np.random.seed(42)  # For reproducibility
    n = len(index)
    
    # Create trending price series with realistic volatility
    close = np.zeros(n)
    close[0] = 1.2000  # Starting price for EUR/USD
    
    # Parameters for random walk
    drift = 0.00001  # Small upward drift
    volatility = 0.0005  # Typical hourly volatility for EUR/USD
    
    for i in range(1, n):
        # Random walk with drift and volatility
        close[i] = close[i-1] + drift + volatility * np.random.normal()
    
    # Generate open, high, low based on close
    open_prices = close[:-1].copy()
    open_prices = np.append([close[0] - 0.0002], open_prices)  # First open
    
    # High is max of open and close plus random amount
    high = np.maximum(close, open_prices) + np.random.uniform(0, volatility*2, n)
    
    # Low is min of open and close minus random amount
    low = np.minimum(close, open_prices) - np.random.uniform(0, volatility*2, n)
    
    # Create volume with some patterns
    volume = np.random.uniform(1000, 5000, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=index)
    
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
    
    # Daily high/low
    df_liq['date'] = df_liq.index.date
    
    # Calculate daily high/low
    daily_high = df_liq.groupby('date')['high'].transform('max')
    daily_low = df_liq.groupby('date')['low'].transform('min')
    
    # Previous day's high/low (using shift to get previous day's values)
    prev_day_high = df_liq.groupby('date')['high'].transform('max').shift(24)
    prev_day_low = df_liq.groupby('date')['low'].transform('min').shift(24)
    
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
    prev_week_id = df_liq['week_id'].shift(24*7)  # Approximate - not perfect but works for synthetic data
    prev_weekly_high = df_liq.groupby('week_id')['high'].transform('max').shift(24*7)
    prev_weekly_low = df_liq.groupby('week_id')['low'].transform('min').shift(24*7)
    
    # Distance to weekly levels
    df_liq['dist_to_prev_week_high_atr'] = (prev_weekly_high - df_liq['close']) / df_liq['ATR_14']
    df_liq['dist_to_prev_week_low_atr'] = (df_liq['close'] - prev_weekly_low) / df_liq['ATR_14']
    
    # Sweep detection (price touches a level and reverses)
    df_liq['swept_prev_day_high'] = (df_liq['high'] > prev_day_high) & (df_liq['close'] < prev_day_high)
    df_liq['swept_prev_day_low'] = (df_liq['low'] < prev_day_low) & (df_liq['close'] > prev_day_low)
    df_liq['swept_prev_week_high'] = (df_liq['high'] > prev_weekly_high) & (df_liq['close'] < prev_weekly_high)
    df_liq['swept_prev_week_low'] = (df_liq['low'] < prev_weekly_low) & (df_liq['close'] > prev_weekly_low)
    
    # Relative position in range
    df_liq['rel_pos_in_day_range'] = (df_liq['close'] - daily_low) / (daily_high - daily_low)
    
    # Drop unnecessary columns
    df_liq = df_liq.drop(columns=['date', 'week', 'year', 'week_id'])
    
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
    
    # Loop through each candle to determine outcome
    for i in range(len(df_prep) - look_forward):
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
    
    return X, y

def train_and_evaluate_strategy():
    """Train and evaluate the liquidity-focused trading strategy."""
    # Load configuration
    config = load_config()
    
    # Generate synthetic data
    raw_df = generate_synthetic_data()
    
    # Add basic features
    df_basic = add_basic_features(raw_df)
    
    # Add liquidity features
    df_liquidity = add_liquidity_features(df_basic)
    
    # Save the prepared dataset
    df_liquidity.to_csv("./data/liquidity_featured_data.csv")
    
    # Prepare features and labels
    X_all, y_all = prepare_features_and_labels(
        df_liquidity,
        look_forward=config['LOOK_FORWARD_CANDLES'],
        pt_mult=config['PROFIT_TAKE_ATR'],
        sl_mult=config['STOP_LOSS_ATR']
    )
    
    # Check for data issues
    if len(X_all) < 100:
        logger.error("Not enough data for meaningful training.")
        return
    
    logger.info(f"Features shape: {X_all.shape}, Labels shape: {y_all.shape}")
    logger.info(f"Label distribution: {y_all.value_counts(normalize=True)}")
    
    # Split into train/test sets
    train_size = int(len(X_all) * 0.8)
    X_train = X_all.iloc[:train_size]
    y_train = y_all.iloc[:train_size]
    X_test = X_all.iloc[train_size:]
    y_test = y_all.iloc[train_size:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train models
    logger.info("Training XGBoost models...")
    try:
        import xgboost as xgb
        
        # Train UP model
        y_train_up = (y_train == 1).astype(int)
        y_test_up = (y_test == 1).astype(int)
        
        # Define parameters for UP model
        xgb_params_up = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }
        
        # Create scale_pos_weight
        up_neg_count = (y_train_up == 0).sum()
        up_pos_count = (y_train_up == 1).sum()
        scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1.0
        xgb_params_up['scale_pos_weight'] = scale_pos_weight_up
        
        # Create DMatrix
        dtrain_up = xgb.DMatrix(X_train, label=y_train_up)
        dtest_up = xgb.DMatrix(X_test, label=y_test_up)
        
        # Train UP model
        model_up = xgb.train(
            xgb_params_up,
            dtrain_up,
            num_boost_round=100,
            evals=[(dtest_up, 'test')],
            verbose_eval=False
        )
        
        # Train DOWN model
        y_train_down = (y_train == -1).astype(int)
        y_test_down = (y_test == -1).astype(int)
        
        # Define parameters for DOWN model
        xgb_params_down = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }
        
        # Create scale_pos_weight
        down_neg_count = (y_train_down == 0).sum()
        down_pos_count = (y_train_down == 1).sum()
        scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1.0
        xgb_params_down['scale_pos_weight'] = scale_pos_weight_down
        
        # Create DMatrix
        dtrain_down = xgb.DMatrix(X_train, label=y_train_down)
        dtest_down = xgb.DMatrix(X_test, label=y_test_down)
        
        # Train DOWN model
        model_down = xgb.train(
            xgb_params_down,
            dtrain_down,
            num_boost_round=100,
            evals=[(dtest_down, 'test')],
            verbose_eval=False
        )
        
        # Save models
        model_up.save_model("./final_liquidity_strategy/model_up.json")
        model_down.save_model("./final_liquidity_strategy/model_down.json")
        
        # Evaluate strategy
        logger.info("Evaluating strategy...")
        
        # Get thresholds from config
        up_threshold = config.get('UP_PROB_THRESHOLD', 0.5)
        down_threshold = config.get('DOWN_PROB_THRESHOLD', 0.5)
        
        # Make predictions
        dtest = xgb.DMatrix(X_test)
        up_probs = model_up.predict(dtest)
        down_probs = model_down.predict(dtest)
        
        # Generate signals
        signals = []
        for i in range(len(X_test)):
            signal = "NEUTRAL"
            prob_up = up_probs[i]
            prob_down = down_probs[i]
            
            if prob_up >= up_threshold and prob_down < (1.0 - down_threshold):
                signal = "LONG"
            elif prob_down >= down_threshold and prob_up < (1.0 - up_threshold):
                signal = "SHORT"
            
            signals.append(signal)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Actual_Outcome': y_test,
            'Signal': signals,
            'UP_Prob': up_probs,
            'DOWN_Prob': down_probs
        }, index=X_test.index)
        
        # Save results
        results_df.to_csv("./final_liquidity_strategy/test_results.csv")
        
        # Calculate performance metrics
        long_trades = results_df[results_df['Signal'] == 'LONG']
        short_trades = results_df[results_df['Signal'] == 'SHORT']
        
        long_win_rate = (long_trades['Actual_Outcome'] == 1).mean() if len(long_trades) > 0 else 0
        short_win_rate = (short_trades['Actual_Outcome'] == -1).mean() if len(short_trades) > 0 else 0
        
        long_wins = (long_trades['Actual_Outcome'] == 1).sum()
        long_losses = (long_trades['Actual_Outcome'] == -1).sum()
        short_wins = (short_trades['Actual_Outcome'] == -1).sum()
        short_losses = (short_trades['Actual_Outcome'] == 1).sum()
        
        total_wins = long_wins + short_wins
        total_losses = long_losses + short_losses
        
        # Calculate profit factor (>=1 is profitable)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Print performance
        logger.info(f"Strategy performance (UP={up_threshold}, DOWN={down_threshold}):")
        logger.info(f"Total trades: {len(long_trades) + len(short_trades)} out of {len(X_test)} opportunities")
        logger.info(f"Long trades: {len(long_trades)}, Short trades: {len(short_trades)}")
        logger.info(f"Long win rate: {long_win_rate:.4f}, Short win rate: {short_win_rate:.4f}")
        logger.info(f"Total wins: {total_wins}, Total losses: {total_losses}")
        logger.info(f"Profit factor: {profit_factor:.4f}")
        
        # Create visualizations
        # Trade distribution
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.bar(['Long Trades', 'Short Trades'], [len(long_trades), len(short_trades)])
        plt.title('Trade Distribution')
        
        # Win rates
        plt.subplot(2, 1, 2)
        plt.bar(['Long Win Rate', 'Short Win Rate'], [long_win_rate, short_win_rate])
        plt.title('Win Rates')
        plt.tight_layout()
        plt.savefig("./final_liquidity_strategy/performance_summary.png")
        
        # Profit factor
        plt.figure(figsize=(10, 6))
        plt.bar(['Profit Factor'], [profit_factor])
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
        plt.title('Profit Factor (>1.0 is profitable)')
        plt.tight_layout()
        plt.savefig("./final_liquidity_strategy/profit_factor.png")
        
        # Save metrics to JSON
        metrics = {
            'profit_factor': float(profit_factor) if not np.isinf(profit_factor) else 999.9,
            'total_trades': int(len(long_trades) + len(short_trades)),
            'long_trades': int(len(long_trades)),
            'short_trades': int(len(short_trades)),
            'long_win_rate': float(long_win_rate),
            'short_win_rate': float(short_win_rate),
            'total_wins': int(total_wins),
            'total_losses': int(total_losses)
        }
        
        with open("./final_liquidity_strategy/performance_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
        
    except ImportError:
        logger.error("XGBoost not installed. Cannot train models.")
        return None
        
if __name__ == "__main__":
    train_and_evaluate_strategy()