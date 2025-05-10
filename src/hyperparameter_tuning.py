import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from pathlib import Path

from .config_loader import load_app_config
from .data_handler import fetch_polygon_data, get_economic_calendar_from_api
from .feature_generator import add_basic_ta_features, add_ict_features_v2, prepare_features_and_labels
from .validation_framework import StrategyValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('hyperparameter_tuning')

def load_data_for_tuning():
    """Load and prepare data for hyperparameter tuning."""
    # 1. Load Configuration
    logger.info("[1. Loading Configuration]")
    config = load_app_config()

    # 2. Fetch Market Data
    logger.info("[2. Fetching Market Data]")
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
        return None, None, None
    
    logger.info(f"Raw market data shape: {raw_df.shape}")

    # 3. Feature Engineering
    logger.info("[3. Performing Feature Engineering]")
    df_with_basic_ta = add_basic_ta_features(raw_df.copy())
    df_featured = add_ict_features_v2(df_with_basic_ta.copy(), swing_order=config['SWING_POINT_ORDER'])
    logger.info(f"Data shape after TA and ICT features: {df_featured.shape}")

    # 4. Labeling Data
    logger.info("[4. Labeling Data]")
    X_all, y_all = prepare_features_and_labels(
        df_featured.copy(),
        look_forward_candles=config['LOOK_FORWARD_CANDLES'],
        pt_atr_multiplier=config['PROFIT_TAKE_ATR'],
        sl_atr_multiplier=config['STOP_LOSS_ATR']
    )
    
    if X_all.empty or y_all.empty:
        logger.error("Not enough data after feature engineering and labeling. Exiting.")
        return None, None, None
    
    logger.info(f"Shape of X_all: {X_all.shape}, Shape of y_all: {y_all.shape}")
    logger.info(f"Label distribution (y_all):\n{y_all.value_counts(normalize=True)}")
    
    return X_all, y_all, config

def tune_probability_thresholds(X_all, y_all, config):
    """Find optimal probability thresholds for UP and DOWN models."""
    logger.info("Starting probability threshold tuning...")
    
    # Set up validation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./validation_results/prob_thresholds_{timestamp}")
    
    # Create validator
    validator = StrategyValidator(X_all, y_all, config, results_dir=results_dir)
    
    # Find best thresholds
    best_config = validator.find_best_probability_thresholds(
        threshold_range=(0.5, 0.8),
        step=0.05,
        n_splits=3,
        test_size=0.2,
        val_size=0.15
    )
    
    logger.info(f"Best probability thresholds: {best_config}")
    
    # Evaluate final model with best thresholds
    test_metrics = validator.evaluate_final_model(best_config)
    logger.info(f"Test metrics with best thresholds: Profit Factor={test_metrics['profit_factor']:.2f}")
    
    # Save best config
    with open(results_dir / "best_prob_thresholds.json", 'w') as f:
        json.dump(best_config, f, indent=4)
    
    return best_config, test_metrics

def tune_xgboost_parameters(X_all, y_all, config, fixed_prob_thresholds=None):
    """Optimize XGBoost hyperparameters."""
    logger.info("Starting XGBoost parameter tuning...")
    
    # Set up validation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./validation_results/xgb_params_{timestamp}")
    
    # Create validator
    validator = StrategyValidator(X_all, y_all, config, results_dir=results_dir)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5]
    }
    
    # Create fixed parameters dict if probability thresholds are provided
    fixed_params = {}
    if fixed_prob_thresholds:
        fixed_params.update(fixed_prob_thresholds)
    
    # Find best XGBoost parameters
    best_config = validator.optimize_xgboost_params(
        param_grid=param_grid,
        fixed_params=fixed_params,
        n_splits=3,
        test_size=0.2,
        val_size=0.15
    )
    
    logger.info(f"Best XGBoost parameters: {best_config['XGB_PARAMS']}")
    
    # Evaluate final model with best parameters
    test_metrics = validator.evaluate_final_model(best_config)
    logger.info(f"Test metrics with best parameters: Profit Factor={test_metrics['profit_factor']:.2f}")
    
    # Save best config
    with open(results_dir / "best_xgb_params.json", 'w') as f:
        json.dump(best_config, f, indent=4)
    
    return best_config, test_metrics

def update_config_with_best_params(best_prob_thresholds=None, best_xgb_params=None):
    """Update the configuration file with the best parameters."""
    import yaml

    config_path = Path("./config/settings.yaml")

    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update with best parameters
    if best_prob_thresholds:
        for key, value in best_prob_thresholds.items():
            config[key] = value

    if best_xgb_params:
        # Check if we have model-specific parameters
        if 'XGB_PARAMS_UP' in best_xgb_params and 'XGB_PARAMS_DOWN' in best_xgb_params:
            # Store separate parameters for UP and DOWN models
            config['XGB_PARAMS_UP'] = best_xgb_params['XGB_PARAMS_UP']
            config['XGB_PARAMS_DOWN'] = best_xgb_params['XGB_PARAMS_DOWN']
            # We'll still keep a common XGB_PARAMS for compatibility
            config['XGB_PARAMS'] = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,
            }
        elif 'XGB_PARAMS' in best_xgb_params:
            # Store common parameters
            config['XGB_PARAMS'] = best_xgb_params['XGB_PARAMS']

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Updated configuration file with best parameters at {config_path}")

def run_hyperparameter_tuning():
    """Run the full hyperparameter tuning process."""
    # Load data
    X_all, y_all, config = load_data_for_tuning()
    if X_all is None or y_all is None or config is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # 1. Tune probability thresholds
    best_prob_thresholds, prob_threshold_metrics = tune_probability_thresholds(X_all, y_all, config)
    
    # 2. Tune XGBoost parameters with the best probability thresholds
    best_xgb_params, xgb_metrics = tune_xgboost_parameters(X_all, y_all, config, best_prob_thresholds)
    
    # Update configuration with best parameters
    update_config_with_best_params(best_prob_thresholds, best_xgb_params)
    
    # Log final results
    logger.info("==== Hyperparameter Tuning Summary ====")
    logger.info(f"Best Probability Thresholds: {best_prob_thresholds}")
    logger.info(f"  - Profit Factor: {prob_threshold_metrics['profit_factor']:.2f}")
    logger.info(f"  - Total Trades: {prob_threshold_metrics['total_trades']}")
    logger.info(f"  - Win Rate (Long): {prob_threshold_metrics['long_win_rate']*100:.1f}%")
    logger.info(f"  - Win Rate (Short): {prob_threshold_metrics['short_win_rate']*100:.1f}%")
    
    logger.info(f"Best XGBoost Parameters: {best_xgb_params['XGB_PARAMS']}")
    logger.info(f"  - Profit Factor: {xgb_metrics['profit_factor']:.2f}")
    logger.info(f"  - Total Trades: {xgb_metrics['total_trades']}")
    logger.info(f"  - Win Rate (Long): {xgb_metrics['long_win_rate']*100:.1f}%")
    logger.info(f"  - Win Rate (Short): {xgb_metrics['short_win_rate']*100:.1f}%")
    
    logger.info("Configuration file updated with best parameters.")
    logger.info("==== Hyperparameter Tuning Complete ====")

if __name__ == "__main__":
    run_hyperparameter_tuning()