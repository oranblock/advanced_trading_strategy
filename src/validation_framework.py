import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict
import xgboost as xgb
import json
from pathlib import Path
import logging

from .json_utils import save_json, convert_numpy_types

from .model_trainer import train_binary_xgb_model, evaluate_model, get_feature_importances

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('validation_framework')

class StrategyValidator:
    """Framework for validating and tuning trading strategy parameters."""
    
    def __init__(self, X_all, y_all, config, results_dir="./validation_results"):
        """
        Initialize the validator with data and config.
        
        Args:
            X_all: DataFrame with features
            y_all: Series with target labels (-1, 0, 1)
            config: Configuration dictionary
            results_dir: Directory to save validation results
        """
        self.X_all = X_all
        self.y_all = y_all
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Store results from runs
        self.run_results = defaultdict(list)
        
    def create_time_series_splits(self, n_splits=5, test_size=0.2, val_size=0.2):
        """
        Create time-series train/validation/test splits.
        
        Args:
            n_splits: Number of time series folds
            test_size: Proportion of data for final testing
            val_size: Proportion of training data for validation
            
        Returns:
            List of (train_idx, val_idx, test_idx) for each fold
        """
        # First, reserve the last test_size percentage for final testing
        n_samples = len(self.X_all)
        test_size_absolute = int(n_samples * test_size)
        train_val_size = n_samples - test_size_absolute
        
        # Create indices for the non-test data
        train_val_indices = np.arange(train_val_size)
        test_indices = np.arange(train_val_size, n_samples)
        
        # Create time series splits for the non-test data
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        splits = []
        for train_idx, val_idx in tscv.split(self.X_all.iloc[train_val_indices]):
            # Convert back to original indices
            actual_train_idx = train_val_indices[train_idx]
            actual_val_idx = train_val_indices[val_idx]
            
            splits.append((actual_train_idx, actual_val_idx, test_indices))
            
        return splits
    
    def evaluate_params(self, param_config, n_splits=5, test_size=0.2, val_size=0.2):
        """
        Evaluate a specific parameter configuration.
        
        Args:
            param_config: Dictionary with parameters to evaluate
                - Should contain keys like UP_PROB_THRESHOLD, DOWN_PROB_THRESHOLD, etc.
            n_splits: Number of time series folds
            test_size: Proportion of data for final testing
            val_size: Proportion of training data for validation
            
        Returns:
            avg_metrics: Dictionary with average metrics across all folds
        """
        # Merge base config with param_config
        eval_config = self.config.copy()
        eval_config.update(param_config)
        
        # Create time series splits
        splits = self.create_time_series_splits(n_splits, test_size, val_size)
        
        # Track metrics for each fold
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx, _) in enumerate(splits):
            X_train, X_val = self.X_all.iloc[train_idx], self.X_all.iloc[val_idx]
            y_train_orig, y_val_orig = self.y_all.iloc[train_idx], self.y_all.iloc[val_idx]
            
            fold_result = self._evaluate_single_fold(
                fold_idx, X_train, y_train_orig, X_val, y_val_orig, eval_config
            )
            fold_metrics.append(fold_result)
            
        # Calculate average metrics across folds
        avg_metrics = self._average_fold_metrics(fold_metrics)
        
        # Store results
        run_id = f"run_{len(self.run_results) + 1}"
        self.run_results[run_id] = {
            "config": param_config,
            "avg_metrics": avg_metrics,
            "fold_metrics": fold_metrics
        }
        
        # Save results
        self._save_results(run_id)
        
        return avg_metrics
    
    def _evaluate_single_fold(self, fold_idx, X_train, y_train_orig, X_val, y_val_orig, config):
        """Evaluate a single fold with the given parameters."""
        logger.info(f"Evaluating fold {fold_idx+1}")
        
        # Model UP training
        y_train_up = (y_train_orig == 1).astype(int)
        y_val_up = (y_val_orig == 1).astype(int)
        up_neg_count = (y_train_up == 0).sum()
        up_pos_count = (y_train_up == 1).sum()
        scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1
        
        model_up = train_binary_xgb_model(X_train, y_train_up, config['XGB_PARAMS'], scale_pos_weight_up)
        probs_up_val, _ = evaluate_model(
            model_up, X_val, y_val_up, f"Fold {fold_idx+1} - Model UP", 
            config['UP_PROB_THRESHOLD'], verbose=False
        )
        
        # Model DOWN training
        y_train_down = (y_train_orig == -1).astype(int)
        y_val_down = (y_val_orig == -1).astype(int)
        down_neg_count = (y_train_down == 0).sum()
        down_pos_count = (y_train_down == 1).sum()
        scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1
        
        model_down = train_binary_xgb_model(X_train, y_train_down, config['XGB_PARAMS'], scale_pos_weight_down)
        probs_down_val, _ = evaluate_model(
            model_down, X_val, y_val_down, f"Fold {fold_idx+1} - Model DOWN", 
            config['DOWN_PROB_THRESHOLD'], verbose=False
        )
        
        # Signal Generation Logic
        final_signals = []
        for i in range(len(X_val)):
            signal = "NEUTRAL"
            prob_up = probs_up_val[i]
            prob_down = probs_down_val[i]
            
            up_thresh = float(config['UP_PROB_THRESHOLD'])
            down_thresh = float(config['DOWN_PROB_THRESHOLD'])
            
            if prob_up >= up_thresh and prob_down < (1.0 - down_thresh):
                signal = "LONG"
            elif prob_down >= down_thresh and prob_up < (1.0 - up_thresh):
                signal = "SHORT"
            final_signals.append(signal)
        
        results_df = pd.DataFrame({
            'Actual_Outcome': y_val_orig,
            'Prob_UP': probs_up_val,
            'Prob_DOWN': probs_down_val,
            'Signal': final_signals
        }, index=X_val.index)
        
        # Calculate metrics
        signal_counts = results_df['Signal'].value_counts()
        long_trades = results_df[results_df['Signal'] == 'LONG']
        short_trades = results_df[results_df['Signal'] == 'SHORT']
        
        long_win_rate = 0
        if not long_trades.empty:
            long_win_rate = (long_trades['Actual_Outcome'] == 1).mean()
        
        short_win_rate = 0
        if not short_trades.empty:
            short_win_rate = (short_trades['Actual_Outcome'] == -1).mean()
        
        total_trades = len(long_trades) + len(short_trades)
        
        # Calculate profit factor (basic version)
        # Assumes 1:1 reward for now, can be updated with actual R values
        long_wins = (long_trades['Actual_Outcome'] == 1).sum()
        long_losses = (long_trades['Actual_Outcome'] == -1).sum()
        short_wins = (short_trades['Actual_Outcome'] == -1).sum()
        short_losses = (short_trades['Actual_Outcome'] == 1).sum()
        
        total_wins = long_wins + short_wins
        total_losses = long_losses + short_losses
        
        profit_factor = 0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        
        # Store metrics
        metrics = {
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "total_trades": total_trades,
            "profit_factor": profit_factor,
            "long_wins": long_wins,
            "long_losses": long_losses,
            "short_wins": short_wins,
            "short_losses": short_losses,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "neutral_count": signal_counts.get("NEUTRAL", 0)
        }
        
        return metrics
    
    def _average_fold_metrics(self, fold_metrics):
        """Calculate average metrics across folds."""
        if not fold_metrics:
            return {}
        
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            avg_metrics[key] = np.mean([fm[key] for fm in fold_metrics])
            
        return avg_metrics
    
    def _save_results(self, run_id):
        """Save results to disk."""
        run_data = self.run_results[run_id]

        # Save to disk using utility function
        save_json(run_data, self.results_dir / f"{run_id}.json")

        logger.info(f"Saved results for {run_id} to {self.results_dir / f'{run_id}.json'}")
    
    def find_best_probability_thresholds(self, threshold_range=(0.5, 0.8), step=0.05, 
                                        n_splits=5, test_size=0.2, val_size=0.2):
        """
        Find optimal probability thresholds for UP and DOWN models.
        
        Args:
            threshold_range: Tuple of (min, max) for probability threshold
            step: Step size for threshold values
            n_splits, test_size, val_size: Parameters for evaluate_params
            
        Returns:
            best_config: Dictionary with best UP_PROB_THRESHOLD and DOWN_PROB_THRESHOLD
        """
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        best_profit_factor = 0
        best_config = {}
        
        for up_threshold in thresholds:
            for down_threshold in thresholds:
                param_config = {
                    'UP_PROB_THRESHOLD': float(up_threshold),
                    'DOWN_PROB_THRESHOLD': float(down_threshold)
                }
                
                logger.info(f"Testing thresholds: UP={up_threshold}, DOWN={down_threshold}")
                avg_metrics = self.evaluate_params(param_config, n_splits, test_size, val_size)
                
                if avg_metrics['profit_factor'] > best_profit_factor:
                    best_profit_factor = avg_metrics['profit_factor']
                    best_config = param_config.copy()
                    logger.info(f"New best: UP={up_threshold}, DOWN={down_threshold}, PF={best_profit_factor:.2f}")
        
        return best_config
    
    def optimize_xgboost_params(self, param_grid, fixed_params=None, 
                              n_splits=5, test_size=0.2, val_size=0.2):
        """
        Grid search over XGBoost parameters.
        
        Args:
            param_grid: Dictionary where keys are XGBoost parameters and values are lists of values to try
            fixed_params: Dictionary of non-XGBoost parameters to keep fixed
            n_splits, test_size, val_size: Parameters for evaluate_params
            
        Returns:
            best_config: Dictionary with best XGBoost parameters
        """
        from itertools import product
        
        fixed_params = fixed_params or {}
        best_profit_factor = 0
        best_config = {}
        
        # Get all combinations of parameter values
        param_names = list(param_grid.keys())
        param_values = list(product(*[param_grid[name] for name in param_names]))
        
        for values in param_values:
            # Create XGBoost parameters dictionary
            xgb_params = {name: value for name, value in zip(param_names, values)}
            
            # Update full configuration
            param_config = fixed_params.copy()
            param_config['XGB_PARAMS'] = {**self.config['XGB_PARAMS'], **xgb_params}
            
            logger.info(f"Testing XGBoost params: {xgb_params}")
            avg_metrics = self.evaluate_params(param_config, n_splits, test_size, val_size)
            
            if avg_metrics['profit_factor'] > best_profit_factor:
                best_profit_factor = avg_metrics['profit_factor']
                best_config = param_config.copy()
                logger.info(f"New best: {xgb_params}, PF={best_profit_factor:.2f}")
        
        return best_config
    
    def optimize_labeling_params(self, look_forward_range, pt_atr_range, sl_atr_range, 
                              fixed_params=None, n_splits=5, test_size=0.2, val_size=0.2):
        """
        Grid search over labeling parameters.
        
        Args:
            look_forward_range: List of LOOK_FORWARD_CANDLES values to try
            pt_atr_range: List of PROFIT_TAKE_ATR values to try
            sl_atr_range: List of STOP_LOSS_ATR values to try
            fixed_params: Dictionary of non-labeling parameters to keep fixed
            n_splits, test_size, val_size: Parameters for evaluate_params
            
        Note:
            This function requires regenerating labels for each combination,
            which is not implemented here. The actual implementation would
            need to use prepare_features_and_labels for each parameter set.
            
        Returns:
            best_config: Dictionary with best labeling parameters
        """
        # This is a placeholder - actual implementation would require
        # regenerating labels for each combination of parameters
        logger.warning("optimize_labeling_params is a placeholder and not fully implemented")
        
        return {}
    
    def evaluate_final_model(self, best_config):
        """
        Evaluate the best configuration on the test set.

        Args:
            best_config: Dictionary with the best parameters

        Returns:
            test_metrics: Dictionary with metrics on the test set
        """
        # Merge base config with best_config
        eval_config = self.config.copy()
        eval_config.update(best_config)

        # Create a manual train/test split (last 20% as test)
        test_size = int(len(self.X_all) * 0.2)
        train_size = len(self.X_all) - test_size

        train_idx = np.arange(train_size)
        test_idx = np.arange(train_size, len(self.X_all))

        X_train, X_test = self.X_all.iloc[train_idx], self.X_all.iloc[test_idx]
        y_train_orig, y_test_orig = self.y_all.iloc[train_idx], self.y_all.iloc[test_idx]
        
        # Train and evaluate the final model
        logger.info("Training and evaluating final model with best configuration")
        
        # Model UP
        y_train_up = (y_train_orig == 1).astype(int)
        y_test_up = (y_test_orig == 1).astype(int)
        up_neg_count = (y_train_up == 0).sum()
        up_pos_count = (y_train_up == 1).sum()
        scale_pos_weight_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1
        
        model_up = train_binary_xgb_model(X_train, y_train_up, eval_config['XGB_PARAMS'], scale_pos_weight_up)
        probs_up_test, _ = evaluate_model(
            model_up, X_test, y_test_up, "Final Model UP", 
            eval_config['UP_PROB_THRESHOLD'], verbose=True
        )
        
        # Model DOWN
        y_train_down = (y_train_orig == -1).astype(int)
        y_test_down = (y_test_orig == -1).astype(int)
        down_neg_count = (y_train_down == 0).sum()
        down_pos_count = (y_train_down == 1).sum()
        scale_pos_weight_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1
        
        model_down = train_binary_xgb_model(X_train, y_train_down, eval_config['XGB_PARAMS'], scale_pos_weight_down)
        probs_down_test, _ = evaluate_model(
            model_down, X_test, y_test_down, "Final Model DOWN", 
            eval_config['DOWN_PROB_THRESHOLD'], verbose=True
        )
        
        # Feature importances
        fi_up = get_feature_importances(model_up, X_train.columns)
        fi_down = get_feature_importances(model_down, X_train.columns)
        
        # Signal Generation
        final_signals = []
        for i in range(len(X_test)):
            signal = "NEUTRAL"
            prob_up = probs_up_test[i]
            prob_down = probs_down_test[i]
            
            up_thresh = float(eval_config['UP_PROB_THRESHOLD'])
            down_thresh = float(eval_config['DOWN_PROB_THRESHOLD'])
            
            if prob_up >= up_thresh and prob_down < (1.0 - down_thresh):
                signal = "LONG"
            elif prob_down >= down_thresh and prob_up < (1.0 - up_thresh):
                signal = "SHORT"
            final_signals.append(signal)
        
        results_df = pd.DataFrame({
            'Actual_Outcome': y_test_orig,
            'Prob_UP': probs_up_test,
            'Prob_DOWN': probs_down_test,
            'Signal': final_signals
        }, index=X_test.index)
        
        # Calculate metrics
        signal_counts = results_df['Signal'].value_counts()
        long_trades = results_df[results_df['Signal'] == 'LONG']
        short_trades = results_df[results_df['Signal'] == 'SHORT']
        
        long_win_rate = 0
        if not long_trades.empty:
            long_win_rate = (long_trades['Actual_Outcome'] == 1).mean()
        
        short_win_rate = 0
        if not short_trades.empty:
            short_win_rate = (short_trades['Actual_Outcome'] == -1).mean()
        
        total_trades = len(long_trades) + len(short_trades)
        
        # Calculate profit factor
        long_wins = (long_trades['Actual_Outcome'] == 1).sum()
        long_losses = (long_trades['Actual_Outcome'] == -1).sum()
        short_wins = (short_trades['Actual_Outcome'] == -1).sum()
        short_losses = (short_trades['Actual_Outcome'] == 1).sum()
        
        total_wins = long_wins + short_wins
        total_losses = long_losses + short_losses
        
        profit_factor = 0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        
        # Test metrics
        test_metrics = {
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "total_trades": total_trades,
            "profit_factor": profit_factor,
            "long_wins": long_wins,
            "long_losses": long_losses,
            "short_wins": short_wins,
            "short_losses": short_losses,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "neutral_count": signal_counts.get("NEUTRAL", 0),
            "feature_importances_up": fi_up.to_dict() if fi_up is not None else {},
            "feature_importances_down": fi_down.to_dict() if fi_down is not None else {}
        }
        
        # Save detailed results
        results_df.to_csv(self.results_dir / "final_model_signals.csv")
        
        # Save test metrics using utility function
        save_json(test_metrics, self.results_dir / "final_model_metrics.json")
            
        logger.info(f"Saved final model metrics to {self.results_dir / 'final_model_metrics.json'}")
        
        return test_metrics

# Extend model_trainer.py's evaluate_model to support silent mode
def evaluate_model(model, X_test, y_test_binary, model_name="Model", prob_threshold=0.5, verbose=True):
    """
    Evaluates the binary classifier and returns predictions and probabilities.
    
    Args:
        model: Trained classifier model
        X_test: Test features
        y_test_binary: Binary test labels
        model_name: Name to display in output
        prob_threshold: Threshold for positive prediction
        verbose: Whether to print evaluation metrics
        
    Returns:
        probs, binary_preds: Probability estimates and binary predictions
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    X_test_processed = X_test.astype(float)
    
    probs = model.predict_proba(X_test_processed)[:, 1]
    binary_preds = (probs >= prob_threshold).astype(int)
    
    if verbose:
        print(f"\n--- {model_name} - Test Set Performance (Threshold: {prob_threshold}) ---")
        print(classification_report(y_test_binary, binary_preds, target_names=['Negative', 'Positive'], zero_division=0))
        print(f"{model_name} - Confusion Matrix:")
        print(confusion_matrix(y_test_binary, binary_preds))
    
    return probs, binary_preds