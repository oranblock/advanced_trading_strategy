#!/usr/bin/env python3
"""
Test script to verify that the _evaluate_on_test_set method works properly.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.xgboost_tuner import XGBoostOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_evaluation')

def create_mock_data(size=200):
    """Create mock data for testing."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2023-01-01', periods=size)
    
    # Create features
    features = {}
    for i in range(5):
        features[f'feature_{i}'] = np.random.normal(0, 1, size)
    
    X = pd.DataFrame(features, index=dates)
    
    # Create target labels
    y_values = np.random.choice([-1, 0, 1], size=size, p=[0.3, 0.4, 0.3])
    y = pd.Series(y_values, index=dates)
    
    return X, y

def test_evaluation_method():
    """Test the _evaluate_on_test_set method."""
    logger.info("Creating test data...")
    X_all, y_all = create_mock_data()
    
    config = {
        'UP_PROB_THRESHOLD': 0.55,
        'DOWN_PROB_THRESHOLD': 0.55,
        'XGB_PARAMS': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'eta': 0.1,
        },
        'XGB_PARAMS_UP': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'eta': 0.1,
            'n_estimators': 50,
        },
        'XGB_PARAMS_DOWN': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'eta': 0.1,
            'n_estimators': 50,
        }
    }
    
    # Create optimizer
    results_dir = f"./test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    optimizer = XGBoostOptimizer(X_all, y_all, config, results_dir=results_dir)
    
    # Manually set best params
    optimizer.best_params_up = config['XGB_PARAMS_UP']
    optimizer.best_params_down = config['XGB_PARAMS_DOWN']
    
    try:
        # Test the evaluation method
        logger.info("Testing _evaluate_on_test_set method...")
        test_metrics = optimizer._evaluate_on_test_set(config)
        
        logger.info(f"Test completed successfully.")
        logger.info(f"Test metrics: {test_metrics}")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_evaluation_method()
    print(f"Evaluation test {'succeeded' if success else 'failed'}")