#!/usr/bin/env python3
"""
Script for running trading strategy hyperparameter tuning.
"""

import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tune_strategy')

def main():
    """Process command line arguments and run the appropriate tuning script."""
    parser = argparse.ArgumentParser(description='Trading Strategy Hyperparameter Tuning')
    
    parser.add_argument('--tune-all', action='store_true',
                        help='Run all tuning processes (thresholds, XGBoost, etc.)')
    
    parser.add_argument('--tune-thresholds', action='store_true',
                        help='Tune model probability thresholds only')

    parser.add_argument('--tune-xgboost', action='store_true',
                        help='Tune XGBoost parameters only')

    parser.add_argument('--advanced-threshold-tuning', action='store_true',
                        help='Run advanced probability threshold tuning with visualization')

    parser.add_argument('--advanced-xgboost-tuning', action='store_true',
                        help='Run advanced XGBoost parameter tuning with Optuna')

    parser.add_argument('--tune-labeling', action='store_true',
                        help='Tune labeling parameters (look-forward, PT/SL)')

    parser.add_argument('--experiment-features', action='store_true',
                        help='Run experiments on advanced ICT features')

    parser.add_argument('--trials', type=int, default=30,
                        help='Number of trials for Optuna-based tuning (default: 30)')

    parser.add_argument('--quick', action='store_true',
                        help='Run a quick tuning with fewer parameter combinations')
    
    args = parser.parse_args()
    
    # Create validation_results directory if it doesn't exist
    Path("./validation_results").mkdir(exist_ok=True)
    
    if args.advanced_threshold_tuning:
        logger.info("Running advanced threshold tuning with visualization")
        from src.threshold_tuner import run_threshold_tuning
        run_threshold_tuning()
    elif args.advanced_xgboost_tuning:
        logger.info(f"Running advanced XGBoost tuning with Optuna ({args.trials} trials)")
        from src.xgboost_tuner import run_xgboost_tuning
        run_xgboost_tuning(n_trials=args.trials)
    elif args.tune_labeling:
        logger.info("Running labeling parameter tuning")
        from src.labeling_tuner import run_labeling_tuning
        run_labeling_tuning()
    elif args.experiment_features:
        logger.info("Running advanced ICT feature experiments")
        from src.feature_experiment import run_feature_experiments
        if args.quick:
            logger.info("Using quick mode for feature experiments")
            run_feature_experiments(quick_mode=True)
        else:
            run_feature_experiments(quick_mode=False)
    elif args.tune_all or (not args.tune_thresholds and not args.tune_xgboost
                          and not args.advanced_threshold_tuning and not args.advanced_xgboost_tuning
                          and not args.tune_labeling and not args.experiment_features):
        # Default to running all tuning if no specific option is selected
        logger.info("Running all hyperparameter tuning processes")
        from src.hyperparameter_tuning import run_hyperparameter_tuning
        run_hyperparameter_tuning()
    elif args.tune_thresholds:
        logger.info("Running probability threshold tuning only")
        from src.hyperparameter_tuning import load_data_for_tuning, tune_probability_thresholds, update_config_with_best_params

        X_all, y_all, config = load_data_for_tuning()
        if X_all is not None and y_all is not None and config is not None:
            best_thresholds, metrics = tune_probability_thresholds(X_all, y_all, config)
            update_config_with_best_params(best_prob_thresholds=best_thresholds)
    elif args.tune_xgboost:
        logger.info("Running XGBoost parameter tuning only")
        from src.hyperparameter_tuning import load_data_for_tuning, tune_xgboost_parameters, update_config_with_best_params

        X_all, y_all, config = load_data_for_tuning()
        if X_all is not None and y_all is not None and config is not None:
            # When tuning XGB parameters only, use current probability thresholds
            fixed_probs = {
                'UP_PROB_THRESHOLD': config.get('UP_PROB_THRESHOLD', 0.55),
                'DOWN_PROB_THRESHOLD': config.get('DOWN_PROB_THRESHOLD', 0.55)
            }
            best_params, metrics = tune_xgboost_parameters(X_all, y_all, config, fixed_probs)
            update_config_with_best_params(best_xgb_params=best_params)

if __name__ == "__main__":
    main()