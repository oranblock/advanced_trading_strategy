#!/bin/bash

# Script to implement the best liquidity-based strategy based on our findings
# Uses the already identified good settings (PF=4.17 with UP=0.5, DOWN=0.5)

BASE_DIR="/home/clouduser/advanced_trading_strategy"
RESULTS_DIR="$BASE_DIR/final_liquidity_strategy"
mkdir -p $RESULTS_DIR
mkdir -p $BASE_DIR/data

echo "Implementing optimal liquidity-based trading strategy..."

# 1. Create a specialized configuration file with optimal settings
cat > $BASE_DIR/config/liquidity_settings.yaml << EOL
# Optimal settings for liquidity-based trading strategy
DOWN_PROB_THRESHOLD: 0.5
LOOK_FORWARD_CANDLES: 24
MULTIPLIER: 1
PROFIT_TAKE_ATR: 2.5
START_DATE: '2021-01-01'
STOP_LOSS_ATR: 1.5
SWING_POINT_ORDER: 10
TICKER: C:EURUSD
TIMEFRAME: hour
UP_PROB_THRESHOLD: 0.5
# XGBoost parameters from previous tuning
XGB_PARAMS:
  alpha: 0.00019840517943508243
  booster: gbtree
  colsample_bytree: 0.9338766053096792
  eta: 0.021958567151236525
  eval_metric: logloss
  gamma: 2.7453993032043625e-06
  grow_policy: lossguide
  lambda: 2.9356380909005304e-07
  max_depth: 6
  min_child_weight: 10
  n_estimators: 252
  objective: binary:logistic
  subsample: 0.9253513647285841
XGB_PARAMS_DOWN:
  alpha: 0.2919022269103127
  booster: gbtree
  colsample_bytree: 0.8771031075694039
  eta: 0.04529902238302772
  eval_metric: logloss
  gamma: 0.0009062997479241217
  grow_policy: depthwise
  lambda: 4.7263546788993326e-07
  max_depth: 5
  min_child_weight: 9
  n_estimators: 334
  objective: binary:logistic
  subsample: 0.98394478746366
EOL

echo "Created optimized configuration file"

# 2. Create a final implementation of the liquidity strategy
cat > $BASE_DIR/final_liquidity_strategy.py << 'EOL'
#!/usr/bin/env python3
"""
Final implementation of the liquidity-based trading strategy
which showed best performance in our testing.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import os

# Import core components
from src.data_handler import fetch_polygon_data
from src.feature_generator import add_basic_ta_features, prepare_features_and_labels
from src.advanced_features import add_external_liquidity_features
from src.config_loader import load_app_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('liquidity_strategy')

def load_liquidity_config():
    """Load the specialized liquidity strategy config."""
    config_path = Path("./config/liquidity_settings.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def prepare_liquidity_features(raw_df, config):
    """
    Prepare a dataset focused on external liquidity features which showed best performance.
    
    Args:
        raw_df: Raw OHLCV DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with liquidity features
    """
    logger.info("Preparing specialized liquidity features dataset...")
    
    # Add basic TA features first
    df_with_basic_ta = add_basic_ta_features(raw_df.copy())
    
    # Add specialized external liquidity features
    df_with_liquidity = add_external_liquidity_features(df_with_basic_ta.copy(), config)
    
    return df_with_liquidity

def train_liquidity_strategy(save_model=True):
    """
    Train the specialized liquidity-based trading strategy.
    
    Args:
        save_model: Whether to save the trained model
        
    Returns:
        tuple: (model_up, model_down, X_test, y_test) - Models and test data
    """
    # Load specialized config
    config = load_liquidity_config()
    
    # Fetch market data
    logger.info("Fetching market data...")
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    raw_df = fetch_polygon_data(
        api_key=config.get('POLYGON_API_KEY', os.environ.get('POLYGON_API_KEY', '')),
        ticker=config['TICKER'],
        timespan=config['TIMEFRAME'],
        multiplier=config['MULTIPLIER'],
        start_date_str=config['START_DATE'],
        end_date_str=end_date_str
    )
    
    if raw_df.empty:
        logger.error("No data fetched. Exiting.")
        return None, None, None, None
    
    logger.info(f"Raw data shape: {raw_df.shape}")
    
    # Prepare specialized dataset with liquidity features
    df_featured = prepare_liquidity_features(raw_df, config)
    
    # Save this specialized dataset for future use
    os.makedirs("./data", exist_ok=True)
    df_featured.to_csv("./data/liquidity_featured_data.csv")
    logger.info("Saved specialized liquidity dataset")
    
    # Prepare features and labels
    X_all, y_all = prepare_features_and_labels(
        df_featured.copy(),
        look_forward_candles=config['LOOK_FORWARD_CANDLES'],
        pt_atr_multiplier=config['PROFIT_TAKE_ATR'],
        sl_atr_multiplier=config['STOP_LOSS_ATR']
    )
    
    logger.info(f"Prepared features: {X_all.shape}, labels: {y_all.shape}")
    
    # Split into train/test
    train_size = int(len(X_all) * 0.8)
    X_train = X_all.iloc[:train_size]
    y_train = y_all.iloc[:train_size]
    X_test = X_all.iloc[train_size:]
    y_test = y_all.iloc[train_size:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train UP model
    logger.info("Training UP model...")
    import xgboost as xgb
    
    # Create binary targets
    y_train_up = (y_train == 1).astype(int)
    y_test_up = (y_test == 1).astype(int)
    
    # Get parameters from config
    xgb_params_up = config.get('XGB_PARAMS', {})
    
    # Calculate class weights
    up_neg_count = (y_train_up == 0).sum()
    up_pos_count = (y_train_up == 1).sum()
    scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1.0
    
    # Create DMatrix for faster training
    dtrain_up = xgb.DMatrix(X_train, label=y_train_up)
    dtest_up = xgb.DMatrix(X_test, label=y_test_up)
    
    # Add scale_pos_weight to parameters
    params_up = xgb_params_up.copy()
    params_up['scale_pos_weight'] = scale_pos_weight_up
    
    # Train model
    n_estimators_up = params_up.pop('n_estimators', 100)
    model_up = xgb.train(
        params_up,
        dtrain_up,
        num_boost_round=n_estimators_up,
        evals=[(dtest_up, 'test')],
        verbose_eval=False
    )
    
    # Train DOWN model
    logger.info("Training DOWN model...")
    
    # Create binary targets
    y_train_down = (y_train == -1).astype(int)
    y_test_down = (y_test == -1).astype(int)
    
    # Get parameters from config
    xgb_params_down = config.get('XGB_PARAMS_DOWN', config.get('XGB_PARAMS', {}))
    
    # Calculate class weights
    down_neg_count = (y_train_down == 0).sum()
    down_pos_count = (y_train_down == 1).sum()
    scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1.0
    
    # Create DMatrix for faster training
    dtrain_down = xgb.DMatrix(X_train, label=y_train_down)
    dtest_down = xgb.DMatrix(X_test, label=y_test_down)
    
    # Add scale_pos_weight to parameters
    params_down = xgb_params_down.copy()
    params_down['scale_pos_weight'] = scale_pos_weight_down
    
    # Train model
    n_estimators_down = params_down.pop('n_estimators', 100)
    model_down = xgb.train(
        params_down,
        dtrain_down,
        num_boost_round=n_estimators_down,
        evals=[(dtest_down, 'test')],
        verbose_eval=False
    )
    
    # Save models if requested
    if save_model:
        model_up.save_model(f"./final_liquidity_strategy/model_up.json")
        model_down.save_model(f"./final_liquidity_strategy/model_down.json")
        
        # Save feature names
        with open(f"./final_liquidity_strategy/feature_names.pkl", 'wb') as f:
            pickle.dump(X_train.columns.tolist(), f)
            
        logger.info("Saved models and feature names")
    
    # Evaluate on test set
    evaluate_strategy(model_up, model_down, X_test, y_test, config)
    
    return model_up, model_down, X_test, y_test

def evaluate_strategy(model_up, model_down, X_test, y_test, config):
    """
    Evaluate the strategy on test data.
    
    Args:
        model_up: Trained UP model
        model_down: Trained DOWN model
        X_test: Test features
        y_test: Test labels
        config: Configuration dictionary
    
    Returns:
        dict: Performance metrics
    """
    logger.info("Evaluating strategy on test data...")
    
    # Get probability thresholds from config
    up_threshold = float(config['UP_PROB_THRESHOLD'])
    down_threshold = float(config['DOWN_PROB_THRESHOLD'])
    
    # Make predictions
    import xgboost as xgb
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
    
    # Calculate performance metrics
    signal_counts = results_df['Signal'].value_counts()
    long_trades = results_df[results_df['Signal'] == 'LONG']
    short_trades = results_df[results_df['Signal'] == 'SHORT']
    
    # Calculate win rates
    long_win_rate = (long_trades['Actual_Outcome'] == 1).mean() if len(long_trades) > 0 else 0
    short_win_rate = (short_trades['Actual_Outcome'] == -1).mean() if len(short_trades) > 0 else 0
    
    # Calculate trade metrics
    long_wins = (long_trades['Actual_Outcome'] == 1).sum()
    long_losses = (long_trades['Actual_Outcome'] == -1).sum()
    short_wins = (short_trades['Actual_Outcome'] == -1).sum()
    short_losses = (short_trades['Actual_Outcome'] == 1).sum()
    
    total_wins = long_wins + short_wins
    total_losses = long_losses + short_losses
    
    profit_factor = total_wins / total_losses if total_losses > 0 else 0 
    
    # Calculate more detailed metrics
    avg_win = len(long_trades[long_trades['Actual_Outcome'] == 1]) / len(y_test) * 100 if len(y_test) > 0 else 0
    avg_loss = len(long_trades[long_trades['Actual_Outcome'] == -1]) / len(y_test) * 100 if len(y_test) > 0 else 0
    
    # Print results
    logger.info(f"Testing results with UP threshold = {up_threshold}, DOWN threshold = {down_threshold}:")
    logger.info(f"Total trades: {len(long_trades) + len(short_trades)} out of {len(X_test)} opportunities")
    logger.info(f"Long trades: {len(long_trades)}, Short trades: {len(short_trades)}")
    logger.info(f"Long win rate: {long_win_rate:.4f}, Short win rate: {short_win_rate:.4f}")
    logger.info(f"Total wins: {total_wins}, Total losses: {total_losses}")
    logger.info(f"Profit factor: {profit_factor:.4f}")
    
    # Save detailed results
    results_df.to_csv(f"./final_liquidity_strategy/test_results.csv")
    
    # Create visualization of the test results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.bar(['Long Trades', 'Short Trades'], [len(long_trades), len(short_trades)])
    plt.title('Trade Distribution')
    
    plt.subplot(2, 1, 2)
    plt.bar(['Long Win Rate', 'Short Win Rate'], [long_win_rate, short_win_rate])
    plt.title('Win Rates')
    plt.tight_layout()
    plt.savefig(f"./final_liquidity_strategy/performance_summary.png")
    
    # Create profit factor chart
    plt.figure(figsize=(10, 6))
    plt.bar(['Profit Factor'], [profit_factor])
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
    plt.title('Profit Factor (>1.0 is profitable)')
    plt.tight_layout()
    plt.savefig(f"./final_liquidity_strategy/profit_factor.png")
    
    # Save metrics to JSON
    import json
    metrics = {
        'profit_factor': float(profit_factor),
        'total_trades': int(len(long_trades) + len(short_trades)),
        'long_trades': int(len(long_trades)),
        'short_trades': int(len(short_trades)),
        'long_win_rate': float(long_win_rate),
        'short_win_rate': float(short_win_rate),
        'total_wins': int(total_wins),
        'total_losses': int(total_losses)
    }
    
    with open(f"./final_liquidity_strategy/performance_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    # Train and evaluate strategy
    train_liquidity_strategy(save_model=True)
EOL

echo "Created final liquidity strategy implementation"

# Make the script executable
chmod +x $BASE_DIR/final_liquidity_strategy.py

# Run the final strategy
echo "Running the final liquidity strategy..."
cd $BASE_DIR
python final_liquidity_strategy.py > $RESULTS_DIR/execution.log 2>&1

# Check the results
echo "Checking liquidity strategy results..."
if [ -f "$RESULTS_DIR/performance_metrics.json" ]; then
    echo "Strategy execution completed. Performance metrics:"
    cat $RESULTS_DIR/performance_metrics.json
else
    echo "Strategy metrics file not found. Checking execution log..."
    tail -n 20 $RESULTS_DIR/execution.log
fi

echo "Final liquidity strategy process complete."
echo "Check the detailed results in $RESULTS_DIR directory"