#!/bin/bash

# Script to create and optimize a trading strategy focusing on external liquidity features
# which showed the best performance in feature testing

BASE_DIR="/home/clouduser/advanced_trading_strategy"
LOGS_DIR="$BASE_DIR/liquidity_optimization"
mkdir -p $LOGS_DIR

echo "Starting optimization focusing on external liquidity features..."

# Create specialized feature file that focuses on external liquidity
cat > $BASE_DIR/src/liquidity_strategy.py << 'EOL'
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from .data_handler import fetch_polygon_data
from .feature_generator import add_basic_ta_features
from .advanced_features import add_external_liquidity_features
from .model_trainer import train_binary_xgb_model
from .config_loader import load_app_config

logger = logging.getLogger('liquidity_strategy')

def prepare_liquidity_features(raw_df, config):
    """
    Prepare a dataset focused on external liquidity features which showed best performance.
    """
    logger.info("Preparing specialized liquidity features dataset...")
    
    # Add basic TA features first
    df_with_basic_ta = add_basic_ta_features(raw_df.copy())
    
    # Add specialized external liquidity features
    df_with_liquidity = add_external_liquidity_features(df_with_basic_ta.copy(), config)
    
    return df_with_liquidity

def run_liquidity_optimization():
    """
    Run a specialized optimization focusing on external liquidity features.
    """
    config = load_app_config()
    
    # Fetch market data
    logger.info("Fetching market data...")
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    raw_df = fetch_polygon_data(
        api_key=config['POLYGON_API_KEY'],
        ticker=config['TICKER'],
        timespan=config['TIMEFRAME'],
        multiplier=config['MULTIPLIER'],
        start_date_str=config['START_DATE'],
        end_date_str=end_date_str
    )
    
    if raw_df.empty:
        logger.error("No data fetched. Exiting.")
        return
    
    # Prepare specialized dataset with liquidity features
    df_featured = prepare_liquidity_features(raw_df, config)
    
    # Save this specialized dataset for future use
    df_featured.to_csv("./data/liquidity_featured_data.csv")
    logger.info("Saved specialized liquidity dataset")
    
    logger.info("Liquidity optimization setup complete. Use this dataset for optimized training.")
    
    # Here you could add code to run specific optimizations on this dataset
    
    return df_featured

if __name__ == "__main__":
    run_liquidity_optimization()
EOL

echo "Created specialized liquidity feature strategy"

# Create a script to run threshold tuning on the liquidity strategy
cat > $BASE_DIR/optimize_thresholds_liquidity.py << 'EOL'
#!/usr/bin/env python3
"""
Script to optimize probability thresholds specifically for the liquidity-focused strategy.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.config_loader import load_app_config
from src.data_handler import fetch_polygon_data
from src.feature_generator import prepare_features_and_labels
from src.validation_framework import StrategyValidator
from src.liquidity_strategy import prepare_liquidity_features
from src.threshold_tuner import visualize_threshold_results, update_config_with_thresholds

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('liquidity_threshold_tuner')

def run_liquidity_threshold_tuning():
    """Run threshold tuning specifically for liquidity features."""
    # Load configuration
    config = load_app_config()
    
    # Fetch and prepare data
    logger.info("Fetching and preparing data...")
    
    # 1. Fetch Market Data
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    raw_df = fetch_polygon_data(
        api_key=config['POLYGON_API_KEY'],
        ticker=config['TICKER'],
        timespan=config['TIMEFRAME'],
        multiplier=config['MULTIPLIER'],
        start_date_str=config['START_DATE'],
        end_date_str=end_date_str
    )
    
    if raw_df.empty:
        logger.error(f"No market data fetched for {config['TICKER']}. Exiting.")
        return
    
    logger.info(f"Raw market data shape: {raw_df.shape}")
    
    # 2. Prepare specialized liquidity features
    logger.info("Preparing specialized liquidity features...")
    df_featured = prepare_liquidity_features(raw_df, config)
    logger.info(f"Data shape after liquidity features: {df_featured.shape}")
    
    # 3. Labeling Data
    logger.info("Labeling data...")
    X_all, y_all = prepare_features_and_labels(
        df_featured.copy(),
        look_forward_candles=config['LOOK_FORWARD_CANDLES'],
        pt_atr_multiplier=config['PROFIT_TAKE_ATR'],
        sl_atr_multiplier=config['STOP_LOSS_ATR']
    )
    
    if X_all.empty or y_all.empty:
        logger.error("Not enough data after feature engineering and labeling. Exiting.")
        return
    
    logger.info(f"Shape of X_all: {X_all.shape}, Shape of y_all: {y_all.shape}")
    logger.info(f"Label distribution (y_all):\n{y_all.value_counts(normalize=True)}")
    
    # 4. Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./validation_results/liquidity_thresholds_{timestamp}")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 5. Create validator
    validator = StrategyValidator(X_all, y_all, config, results_dir=results_dir)
    
    # 6. Define larger threshold grid for more thorough search
    up_thresholds = np.linspace(0.5, 0.9, 9)  # 0.5, 0.55, 0.6, ..., 0.9
    down_thresholds = np.linspace(0.5, 0.9, 9)  # 0.5, 0.55, 0.6, ..., 0.9
    
    # 7. Initialize result grid
    threshold_results = []
    best_profit_factor = 0
    best_up_threshold = 0.55
    best_down_threshold = 0.55
    
    # 8. Run grid search with more cross-validation folds
    n_splits = 3  # Increased from 2
    test_size = 0.2
    val_size = 0.15
    
    logger.info(f"Running threshold grid search with {len(up_thresholds)}x{len(down_thresholds)} combinations")
    
    for up_threshold in up_thresholds:
        for down_threshold in down_thresholds:
            up_threshold_rounded = round(up_threshold, 2)
            down_threshold_rounded = round(down_threshold, 2)
            
            param_config = {
                'UP_PROB_THRESHOLD': up_threshold_rounded,
                'DOWN_PROB_THRESHOLD': down_threshold_rounded
            }
            
            logger.info(f"Testing thresholds: UP={up_threshold_rounded}, DOWN={down_threshold_rounded}")
            
            # Evaluate parameters
            avg_metrics = validator.evaluate_params(param_config, n_splits, test_size, val_size)
            
            result = {
                'UP_PROB_THRESHOLD': up_threshold_rounded,
                'DOWN_PROB_THRESHOLD': down_threshold_rounded,
                'profit_factor': avg_metrics['profit_factor'],
                'total_trades': avg_metrics['total_trades'],
                'long_win_rate': avg_metrics['long_win_rate'],
                'short_win_rate': avg_metrics['short_win_rate'],
                'neutral_count': avg_metrics['neutral_count']
            }
            threshold_results.append(result)
            
            # Update best parameters
            if avg_metrics['profit_factor'] > best_profit_factor:
                best_profit_factor = avg_metrics['profit_factor']
                best_up_threshold = up_threshold_rounded
                best_down_threshold = down_threshold_rounded
                logger.info(f"New best: UP={up_threshold_rounded}, DOWN={down_threshold_rounded}, PF={best_profit_factor:.2f}")
    
    # 9. Convert results to DataFrame
    results_df = pd.DataFrame(threshold_results)
    
    # 10. Filter out combinations with too few trades for more reliable results
    min_trades = 10
    filtered_results = results_df[results_df['total_trades'] >= min_trades]
    
    # If all combinations have fewer than min_trades, keep all results
    if filtered_results.empty:
        filtered_results = results_df
    
    # 11. Save results
    results_df.to_csv(results_dir / 'threshold_grid_results.csv', index=False)
    
    # 12. Find overall best result (after filtering)
    if not filtered_results.empty:
        best_row = filtered_results.loc[filtered_results['profit_factor'].idxmax()]
        best_up_threshold = best_row['UP_PROB_THRESHOLD']
        best_down_threshold = best_row['DOWN_PROB_THRESHOLD']
        best_profit_factor = best_row['profit_factor']
    
    logger.info(f"Best thresholds after filtering: UP={best_up_threshold}, DOWN={best_down_threshold}, PF={best_profit_factor:.2f}")
    
    # 13. Visualize results
    visualize_threshold_results(results_df, results_dir)
    
    best_thresholds = {
        'UP_PROB_THRESHOLD': best_up_threshold,
        'DOWN_PROB_THRESHOLD': best_down_threshold
    }
    
    # 14. Save best thresholds
    with open(results_dir / 'best_thresholds.json', 'w') as f:
        json.dump(best_thresholds, f)
    
    # 15. Update configuration (optional - uncomment if you want automatic update)
    if best_profit_factor > 1.5:  # Only update if significantly better
        update_config_with_thresholds(best_thresholds)
        logger.info("Updated configuration with best thresholds due to significant improvement")
    
    logger.info(f"Threshold tuning complete. Best thresholds: UP={best_up_threshold}, DOWN={best_down_threshold}")
    logger.info(f"Best profit factor: {best_profit_factor:.2f}")
    
    return best_thresholds, best_profit_factor

if __name__ == "__main__":
    run_liquidity_threshold_tuning()
EOL

echo "Created specialized threshold tuning script"

# Run the specialized optimization
echo "Running liquidity feature extraction..."
cd $BASE_DIR
python -c "from src.liquidity_strategy import run_liquidity_optimization; run_liquidity_optimization()" > $LOGS_DIR/liquidity_features.log 2>&1

echo "Running specialized threshold tuning..."
cd $BASE_DIR
python optimize_thresholds_liquidity.py > $LOGS_DIR/liquidity_thresholds.log 2>&1

echo "Optimization complete. Check logs in $LOGS_DIR directory"

# Optionally, we could extract the best parameters and display them
echo "Best threshold parameters:"
grep "Best thresholds after filtering" $LOGS_DIR/liquidity_thresholds.log | tail -1

echo "You can now push these results to GitHub with ./push_results_to_github.sh"