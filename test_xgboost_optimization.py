#!/usr/bin/env python3
"""
Test script to verify the XGBoostOptimizer with mock data.
This script creates mock data and runs a simplified version of the optimization process.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import optuna
from pathlib import Path

from src.xgboost_tuner import XGBoostOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_xgboost_optimization')

def create_mock_data(size=1000):
    """Create mock data for testing."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2023-01-01', periods=size)
    
    # Create features
    features = {}
    for i in range(10):  # Create 10 features
        features[f'feature_{i}'] = np.random.normal(0, 1, size)
    
    X = pd.DataFrame(features, index=dates)
    
    # Create target labels (-1, 0, 1)
    # We'll make them somewhat predictable based on features
    probs = 0.5 + 0.3 * (X['feature_0'] + X['feature_1'] - X['feature_2'])
    y_values = []
    
    for p in probs:
        if p > 0.65:
            y_values.append(1)  # UP
        elif p < 0.35:
            y_values.append(-1)  # DOWN
        else:
            y_values.append(0)  # NEUTRAL
    
    y = pd.Series(y_values, index=dates)
    
    logger.info(f"Created mock data: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"Label distribution: {y.value_counts()}")
    
    return X, y

def run_mock_optimization():
    """Run a mock optimization process."""
    logger.info("Starting mock XGBoost optimization...")
    
    # Create mock data
    X_all, y_all = create_mock_data(size=1000)
    
    # Create mock config
    config = {
        'UP_PROB_THRESHOLD': 0.55,
        'DOWN_PROB_THRESHOLD': 0.55,
        'XGB_PARAMS': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'grow_policy': 'depthwise',
        }
    }
    
    # Create temporary results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./test_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create and run optimizer with very few trials
    optimizer = XGBoostOptimizer(X_all, y_all, config, results_dir=results_dir)
    
    # Run only 2 trials to test functionality while keeping runtime short
    try:
        # Optimize UP model
        logger.info("Optimizing UP model...")
        best_params_up = optimizer.optimize_up_model(n_trials=2)
        
        # Optimize DOWN model
        logger.info("Optimizing DOWN model...")
        best_params_down = optimizer.optimize_down_model(n_trials=2)
        
        logger.info("Optimization successful!")
        logger.info(f"Best UP parameters: {best_params_up}")
        logger.info(f"Best DOWN parameters: {best_params_down}")
        
        return True
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return False

if __name__ == "__main__":
    success = run_mock_optimization()
    print(f"Optimization test {'succeeded' if success else 'failed'}")