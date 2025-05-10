import pandas as pd
import numpy as np
from numpy import integer as np_int
from numpy import floating as np_float
import logging
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .json_utils import save_json

from .config_loader import load_app_config
from .data_handler import fetch_polygon_data
from .feature_generator import add_basic_ta_features, add_ict_features_v2, prepare_features_and_labels
from .validation_framework import StrategyValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xgboost_tuner')

class XGBoostOptimizer:
    """Advanced XGBoost parameter optimization using Optuna."""
    
    def __init__(self, X_all, y_all, config, results_dir="./xgboost_tuning_results"):
        """
        Initialize the XGBoost optimizer.
        
        Args:
            X_all: DataFrame with features
            y_all: Series with target labels (-1, 0, 1)
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        self.X_all = X_all
        self.y_all = y_all
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Fixed probability thresholds (from config or supplied)
        self.up_threshold = float(config.get('UP_PROB_THRESHOLD', 0.55))
        self.down_threshold = float(config.get('DOWN_PROB_THRESHOLD', 0.55))
        
        # Create validator for evaluating models
        self.validator = StrategyValidator(X_all, y_all, config, results_dir=results_dir)
        
        # Optuna study for UP model
        self.study_up = None
        
        # Optuna study for DOWN model
        self.study_down = None
        
        # Best parameters
        self.best_params_up = None
        self.best_params_down = None
        
        # Create time-series folds for consistent evaluation
        self.splits = self.validator.create_time_series_splits(n_splits=3, test_size=0.2)
    
    def _create_binary_target(self, y, is_up_model):
        """Create binary target variable for UP or DOWN model."""
        if is_up_model:
            return (y == 1).astype(int)  # UP model: 1 for UP, 0 otherwise
        else:
            return (y == -1).astype(int)  # DOWN model: 1 for DOWN, 0 otherwise
    
    def _objective_up(self, trial):
        """Objective function for UP model optimization."""
        # Define hyperparameters to optimize
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            # tree-specific parameters
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        }
        
        # Number of boosting rounds
        num_boost_round = trial.suggest_int('num_boost_round', 50, 500)
        
        # We'll compute the average profit factor across folds
        fold_profit_factors = []
        fold_f1_scores = []
        
        # Iterate through folds
        for fold_idx, (train_idx, val_idx, _) in enumerate(self.splits):
            X_train, X_val = self.X_all.iloc[train_idx], self.X_all.iloc[val_idx]
            y_train_orig, y_val_orig = self.y_all.iloc[train_idx], self.y_all.iloc[val_idx]
            
            # Create binary targets
            y_train_up = self._create_binary_target(y_train_orig, is_up_model=True)
            y_val_up = self._create_binary_target(y_val_orig, is_up_model=True)
            
            # Calculate scale_pos_weight
            up_neg_count = (y_train_up == 0).sum()
            up_pos_count = (y_train_up == 1).sum()
            scale_pos_weight = up_neg_count / up_pos_count if up_pos_count > 0 else 1.0
            param['scale_pos_weight'] = scale_pos_weight
            
            # Create DMatrix for faster training
            dtrain = xgb.DMatrix(X_train, label=y_train_up)
            dval = xgb.DMatrix(X_val, label=y_val_up)
            
            # Train model
            model = xgb.train(
                param,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, 'val')],
                verbose_eval=False
            )
            
            # Make predictions
            y_pred_proba = model.predict(dval)
            y_pred = (y_pred_proba >= self.up_threshold).astype(int)
            
            # Calculate F1 score for this fold
            f1 = f1_score(y_val_up, y_pred, zero_division=0)
            fold_f1_scores.append(f1)
            
            # For profit factor, we need to run the whole signal generation logic
            # First, we need the DOWN model predictions too - use a simple default model for now
            # In a real scenario, you'd use the best DOWN model found so far
            default_down_params = self.config.get('XGB_PARAMS', {}).copy()
            y_train_down = self._create_binary_target(y_train_orig, is_up_model=False)
            dtrain_down = xgb.DMatrix(X_train, label=y_train_down)
            
            # Calculate scale_pos_weight for DOWN model
            down_neg_count = (y_train_down == 0).sum()
            down_pos_count = (y_train_down == 1).sum()
            default_down_params['scale_pos_weight'] = down_neg_count / down_pos_count if down_pos_count > 0 else 1.0
            
            # Train default DOWN model
            model_down = xgb.train(
                default_down_params,
                dtrain_down,
                num_boost_round=150,
                verbose_eval=False
            )
            
            # Predict with DOWN model
            dval_down = xgb.DMatrix(X_val)
            down_probs = model_down.predict(dval_down)
            
            # Generate signals
            signals = []
            for i in range(len(X_val)):
                signal = "NEUTRAL"
                prob_up = y_pred_proba[i]
                prob_down = down_probs[i]

                if prob_up >= self.up_threshold and prob_down < (1.0 - self.down_threshold):
                    signal = "LONG"
                elif prob_down >= self.down_threshold and prob_up < (1.0 - self.up_threshold):
                    signal = "SHORT"

                signals.append(signal)

            # Calculate profit factor
            # Create signals Series with matching index to avoid indexing errors
            signals_series = pd.Series(signals, index=y_val_orig.index)
            long_mask = signals_series == "LONG"
            short_mask = signals_series == "SHORT"

            long_trades = y_val_orig[long_mask]
            short_trades = y_val_orig[short_mask]
            
            long_wins = (long_trades == 1).sum()
            long_losses = (long_trades == -1).sum()
            short_wins = (short_trades == -1).sum()
            short_losses = (short_trades == 1).sum()
            
            total_wins = long_wins + short_wins
            total_losses = long_losses + short_losses
            
            profit_factor = 1.0  # Default/neutral value
            if total_losses > 0:
                profit_factor = total_wins / total_losses
            
            fold_profit_factors.append(profit_factor)
        
        # Calculate average metrics across folds
        avg_profit_factor = np.mean(fold_profit_factors)
        avg_f1 = np.mean(fold_f1_scores)
        
        # Optuna maximizes the objective value
        # We'll use a weighted combination of profit factor and F1 score
        # Profit factor is more important but F1 helps ensure model quality
        return avg_profit_factor * 0.7 + avg_f1 * 0.3
    
    def _objective_down(self, trial):
        """Objective function for DOWN model optimization."""
        # Define hyperparameters to optimize
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            # tree-specific parameters
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        }
        
        # Number of boosting rounds
        num_boost_round = trial.suggest_int('num_boost_round', 50, 500)
        
        # We'll compute the average profit factor across folds
        fold_profit_factors = []
        fold_f1_scores = []
        
        # Iterate through folds
        for fold_idx, (train_idx, val_idx, _) in enumerate(self.splits):
            X_train, X_val = self.X_all.iloc[train_idx], self.X_all.iloc[val_idx]
            y_train_orig, y_val_orig = self.y_all.iloc[train_idx], self.y_all.iloc[val_idx]
            
            # Create binary targets
            y_train_down = self._create_binary_target(y_train_orig, is_up_model=False)
            y_val_down = self._create_binary_target(y_val_orig, is_up_model=False)
            
            # Calculate scale_pos_weight
            down_neg_count = (y_train_down == 0).sum()
            down_pos_count = (y_train_down == 1).sum()
            scale_pos_weight = down_neg_count / down_pos_count if down_pos_count > 0 else 1.0
            param['scale_pos_weight'] = scale_pos_weight
            
            # Create DMatrix for faster training
            dtrain = xgb.DMatrix(X_train, label=y_train_down)
            dval = xgb.DMatrix(X_val, label=y_val_down)
            
            # Train model
            model = xgb.train(
                param,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, 'val')],
                verbose_eval=False
            )
            
            # Make predictions
            y_pred_proba = model.predict(dval)
            y_pred = (y_pred_proba >= self.down_threshold).astype(int)
            
            # Calculate F1 score for this fold
            f1 = f1_score(y_val_down, y_pred, zero_division=0)
            fold_f1_scores.append(f1)
            
            # For profit factor, we need to run the whole signal generation logic
            # Use the best UP model found so far or a default model
            default_up_params = self.config.get('XGB_PARAMS', {}).copy()
            y_train_up = self._create_binary_target(y_train_orig, is_up_model=True)
            dtrain_up = xgb.DMatrix(X_train, label=y_train_up)
            
            # Calculate scale_pos_weight for UP model
            up_neg_count = (y_train_up == 0).sum()
            up_pos_count = (y_train_up == 1).sum()
            default_up_params['scale_pos_weight'] = up_neg_count / up_pos_count if up_pos_count > 0 else 1.0
            
            # Train default UP model
            model_up = xgb.train(
                default_up_params,
                dtrain_up,
                num_boost_round=150,
                verbose_eval=False
            )
            
            # Predict with UP model
            dval_up = xgb.DMatrix(X_val)
            up_probs = model_up.predict(dval_up)
            
            # Generate signals
            signals = []
            for i in range(len(X_val)):
                signal = "NEUTRAL"
                prob_up = up_probs[i]
                prob_down = y_pred_proba[i]

                if prob_up >= self.up_threshold and prob_down < (1.0 - self.down_threshold):
                    signal = "LONG"
                elif prob_down >= self.down_threshold and prob_up < (1.0 - self.up_threshold):
                    signal = "SHORT"

                signals.append(signal)

            # Calculate profit factor
            # Create signals Series with matching index to avoid indexing errors
            signals_series = pd.Series(signals, index=y_val_orig.index)
            long_mask = signals_series == "LONG"
            short_mask = signals_series == "SHORT"

            long_trades = y_val_orig[long_mask]
            short_trades = y_val_orig[short_mask]
            
            long_wins = (long_trades == 1).sum()
            long_losses = (long_trades == -1).sum()
            short_wins = (short_trades == -1).sum()
            short_losses = (short_trades == 1).sum()
            
            total_wins = long_wins + short_wins
            total_losses = long_losses + short_losses
            
            profit_factor = 1.0  # Default/neutral value
            if total_losses > 0:
                profit_factor = total_wins / total_losses
            
            fold_profit_factors.append(profit_factor)
        
        # Calculate average metrics across folds
        avg_profit_factor = np.mean(fold_profit_factors)
        avg_f1 = np.mean(fold_f1_scores)
        
        # Optuna maximizes the objective value
        # We'll use a weighted combination of profit factor and F1 score
        return avg_profit_factor * 0.7 + avg_f1 * 0.3
    
    def optimize_up_model(self, n_trials=50):
        """
        Optimize UP model parameters using Optuna.
        
        Args:
            n_trials: Number of Optuna trials
            
        Returns:
            best_params: Dictionary with best parameters
        """
        logger.info(f"Starting UP model optimization with {n_trials} trials")
        
        # Create Optuna study
        self.study_up = optuna.create_study(
            direction="maximize",
            study_name="xgboost_up_optimization"
        )
        
        # Optimize
        self.study_up.optimize(self._objective_up, n_trials=n_trials)
        
        # Get best parameters
        best_trial = self.study_up.best_trial
        
        # Convert to XGBoost parameters
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': best_trial.params.get('booster', 'gbtree'),
            'lambda': best_trial.params.get('lambda', 1.0),
            'alpha': best_trial.params.get('alpha', 0.0),
            'max_depth': best_trial.params.get('max_depth', 6),
            'eta': best_trial.params.get('eta', 0.1),
            'gamma': best_trial.params.get('gamma', 0.0),
            'subsample': best_trial.params.get('subsample', 0.8),
            'colsample_bytree': best_trial.params.get('colsample_bytree', 0.8),
            'min_child_weight': best_trial.params.get('min_child_weight', 1),
            'grow_policy': best_trial.params.get('grow_policy', 'depthwise'),
        }
        
        # Add n_estimators (num_boost_round in XGBoost API)
        n_estimators = best_trial.params.get('num_boost_round', 100)
        param['n_estimators'] = n_estimators
        
        # Save best parameters
        self.best_params_up = param.copy()
        
        logger.info(f"Best UP model parameters: {param}")
        logger.info(f"Best UP model value: {best_trial.value:.4f}")
        
        # Save visualization
        self._save_optuna_visualizations(self.study_up, "up_model")
        
        return param
    
    def optimize_down_model(self, n_trials=50):
        """
        Optimize DOWN model parameters using Optuna.
        
        Args:
            n_trials: Number of Optuna trials
            
        Returns:
            best_params: Dictionary with best parameters
        """
        logger.info(f"Starting DOWN model optimization with {n_trials} trials")
        
        # Create Optuna study
        self.study_down = optuna.create_study(
            direction="maximize",
            study_name="xgboost_down_optimization"
        )
        
        # Optimize
        self.study_down.optimize(self._objective_down, n_trials=n_trials)
        
        # Get best parameters
        best_trial = self.study_down.best_trial
        
        # Convert to XGBoost parameters
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': best_trial.params.get('booster', 'gbtree'),
            'lambda': best_trial.params.get('lambda', 1.0),
            'alpha': best_trial.params.get('alpha', 0.0),
            'max_depth': best_trial.params.get('max_depth', 6),
            'eta': best_trial.params.get('eta', 0.1),
            'gamma': best_trial.params.get('gamma', 0.0),
            'subsample': best_trial.params.get('subsample', 0.8),
            'colsample_bytree': best_trial.params.get('colsample_bytree', 0.8),
            'min_child_weight': best_trial.params.get('min_child_weight', 1),
            'grow_policy': best_trial.params.get('grow_policy', 'depthwise'),
        }
        
        # Add n_estimators (num_boost_round in XGBoost API)
        n_estimators = best_trial.params.get('num_boost_round', 100)
        param['n_estimators'] = n_estimators
        
        # Save best parameters
        self.best_params_down = param.copy()
        
        logger.info(f"Best DOWN model parameters: {param}")
        logger.info(f"Best DOWN model value: {best_trial.value:.4f}")
        
        # Save visualization
        self._save_optuna_visualizations(self.study_down, "down_model")
        
        return param
    
    def _save_optuna_visualizations(self, study, model_name):
        """Save Optuna visualization plots."""
        try:
            # Create optimization history plot
            fig1 = plot_optimization_history(study)
            fig1.write_image(str(self.results_dir / f"{model_name}_optimization_history.png"))

            # Create parameter importance plot
            fig2 = plot_param_importances(study)
            fig2.write_image(str(self.results_dir / f"{model_name}_param_importances.png"))

            logger.info(f"Saved {model_name} optimization visualizations")
        except Exception as e:
            logger.warning(f"Could not create Optuna visualizations: {e}")
            logger.info("This is non-critical. If you want visualizations, install kaleido: pip install -U kaleido")
    
    def evaluate_best_models(self):
        """
        Evaluate the best models on the test set.
        
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        if self.best_params_up is None or self.best_params_down is None:
            logger.error("Models have not been optimized yet")
            return None
        
        # Create a new configuration with best parameters
        best_config = self.config.copy()
        
        # Create a combined XGB_PARAMS dictionary
        # We'll keep common parameters and specify model-specific parameters later
        common_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
        }
        
        # Save best parameters
        save_json(self.best_params_up, self.results_dir / "best_up_params.json")
        save_json(self.best_params_down, self.results_dir / "best_down_params.json")
        
        # Evaluate final models
        best_config['XGB_PARAMS_UP'] = self.best_params_up
        best_config['XGB_PARAMS_DOWN'] = self.best_params_down
        best_config['XGB_PARAMS'] = common_params
        
        # Evaluate on test set
        test_metrics = self._evaluate_on_test_set(best_config)
        
        # Save test metrics
        save_json(test_metrics, self.results_dir / "final_model_metrics.json")
        
        logger.info(f"Test set profit factor: {test_metrics['profit_factor']:.2f}")
        logger.info(f"Test set total trades: {test_metrics['total_trades']}")
        
        return test_metrics
    
    def _evaluate_on_test_set(self, best_config):
        """Evaluate the best models on the test set."""
        # Get the test set
        _, _, test_idx = self.splits[0]

        # Create masks for train/test split instead of using drop
        train_mask = ~self.X_all.index.isin(self.X_all.iloc[test_idx].index)

        X_train = self.X_all[train_mask]
        y_train_orig = self.y_all[train_mask]

        X_test = self.X_all.iloc[test_idx]
        y_test_orig = self.y_all.iloc[test_idx]
        
        # Prepare binary targets
        y_train_up = self._create_binary_target(y_train_orig, is_up_model=True)
        y_test_up = self._create_binary_target(y_test_orig, is_up_model=True)
        
        y_train_down = self._create_binary_target(y_train_orig, is_up_model=False)
        y_test_down = self._create_binary_target(y_test_orig, is_up_model=False)
        
        # Train UP model
        up_params = best_config['XGB_PARAMS_UP'].copy()
        up_neg_count = (y_train_up == 0).sum()
        up_pos_count = (y_train_up == 1).sum()
        up_params['scale_pos_weight'] = up_neg_count / up_pos_count if up_pos_count > 0 else 1.0
        
        dtrain_up = xgb.DMatrix(X_train, label=y_train_up)
        n_estimators_up = up_params.pop('n_estimators', 100)
        model_up = xgb.train(up_params, dtrain_up, num_boost_round=n_estimators_up)
        
        # Train DOWN model
        down_params = best_config['XGB_PARAMS_DOWN'].copy()
        down_neg_count = (y_train_down == 0).sum()
        down_pos_count = (y_train_down == 1).sum()
        down_params['scale_pos_weight'] = down_neg_count / down_pos_count if down_pos_count > 0 else 1.0
        
        dtrain_down = xgb.DMatrix(X_train, label=y_train_down)
        n_estimators_down = down_params.pop('n_estimators', 100)
        model_down = xgb.train(down_params, dtrain_down, num_boost_round=n_estimators_down)
        
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
            
            if prob_up >= self.up_threshold and prob_down < (1.0 - self.down_threshold):
                signal = "LONG"
            elif prob_down >= self.down_threshold and prob_up < (1.0 - self.up_threshold):
                signal = "SHORT"
            
            signals.append(signal)
        
        results_df = pd.DataFrame({
            'Actual_Outcome': y_test_orig,
            'Prob_UP': up_probs,
            'Prob_DOWN': down_probs,
            'Signal': signals
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
        
        # Calculate model metrics
        up_preds = (up_probs >= self.up_threshold).astype(int)
        down_preds = (down_probs >= self.down_threshold).astype(int)
        
        up_accuracy = accuracy_score(y_test_up, up_preds)
        up_precision = precision_score(y_test_up, up_preds, zero_division=0)
        up_recall = recall_score(y_test_up, up_preds, zero_division=0)
        up_f1 = f1_score(y_test_up, up_preds, zero_division=0)
        
        down_accuracy = accuracy_score(y_test_down, down_preds)
        down_precision = precision_score(y_test_down, down_preds, zero_division=0)
        down_recall = recall_score(y_test_down, down_preds, zero_division=0)
        down_f1 = f1_score(y_test_down, down_preds, zero_division=0)
        
        # Calculate trade metrics
        long_wins = (long_trades['Actual_Outcome'] == 1).sum()
        long_losses = (long_trades['Actual_Outcome'] == -1).sum()
        short_wins = (short_trades['Actual_Outcome'] == -1).sum()
        short_losses = (short_trades['Actual_Outcome'] == 1).sum()
        
        total_wins = long_wins + short_wins
        total_losses = long_losses + short_losses
        
        profit_factor = 0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        
        metrics = {
            # Trade metrics
            'long_win_rate': float(long_win_rate),
            'short_win_rate': float(short_win_rate),
            'long_trades': int(len(long_trades)),
            'short_trades': int(len(short_trades)),
            'total_trades': int(len(long_trades) + len(short_trades)),
            'profit_factor': float(profit_factor),
            'long_wins': int(long_wins),
            'long_losses': int(long_losses),
            'short_wins': int(short_wins),
            'short_losses': int(short_losses),
            'total_wins': int(total_wins),
            'total_losses': int(total_losses),
            'neutral_count': int(signal_counts.get('NEUTRAL', 0)),
            
            # Model metrics
            'up_accuracy': float(up_accuracy),
            'up_precision': float(up_precision),
            'up_recall': float(up_recall),
            'up_f1': float(up_f1),
            'down_accuracy': float(down_accuracy),
            'down_precision': float(down_precision),
            'down_recall': float(down_recall),
            'down_f1': float(down_f1)
        }
        
        # Save detailed results
        results_df.to_csv(self.results_dir / "final_model_signals.csv")
        
        return metrics
    
    def get_optimized_config(self):
        """
        Get a configuration dictionary with optimized parameters.
        
        Returns:
            optimized_config: Dictionary with optimized parameters
        """
        if self.best_params_up is None or self.best_params_down is None:
            logger.error("Models have not been optimized yet")
            return None
        
        # Start with the original config
        optimized_config = self.config.copy()
        
        # Create a common XGB_PARAMS dictionary with shared parameters
        common_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
        }
        
        # Set the common parameters
        optimized_config['XGB_PARAMS'] = common_params
        
        # Set the model-specific parameters
        optimized_config['XGB_PARAMS_UP'] = self.best_params_up
        optimized_config['XGB_PARAMS_DOWN'] = self.best_params_down
        
        return optimized_config

def update_config_with_xgb_params(best_params_up, best_params_down):
    """
    Update settings.yaml with the best XGBoost parameters.

    Args:
        best_params_up: Dictionary with best UP model parameters
        best_params_down: Dictionary with best DOWN model parameters
    """
    import yaml

    config_path = Path("./config/settings.yaml")

    # Read existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert parameters to standard Python types
    def convert_to_standard_types(params):
        result = {}
        for key, value in params.items():
            if isinstance(value, (np.integer, np.int64)):
                result[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                result[key] = float(value)
            elif isinstance(value, dict):
                result[key] = convert_to_standard_types(value)
            else:
                result[key] = value
        return result

    # Update XGBoost parameters
    # We'll update the common XGB_PARAMS with the UP model parameters
    # and add model-specific parameters
    config['XGB_PARAMS'] = convert_to_standard_types(best_params_up)
    config['XGB_PARAMS_DOWN'] = convert_to_standard_types(best_params_down)

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Updated configuration file with best XGBoost parameters at {config_path}")

def run_xgboost_tuning(n_trials=30):
    """Run XGBoost parameter optimization."""
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
    
    # 2. Feature Engineering
    logger.info("Performing feature engineering...")
    df_with_basic_ta = add_basic_ta_features(raw_df.copy())
    df_featured = add_ict_features_v2(df_with_basic_ta.copy(), swing_order=config['SWING_POINT_ORDER'])
    logger.info(f"Data shape after TA and ICT features: {df_featured.shape}")
    
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
    
    # 4. Create XGBoostOptimizer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./validation_results/xgboost_tuning_{timestamp}"
    optimizer = XGBoostOptimizer(X_all, y_all, config, results_dir=results_dir)
    
    # 5. Optimize UP model
    logger.info("Optimizing UP model...")
    best_params_up = optimizer.optimize_up_model(n_trials=n_trials)
    
    # 6. Optimize DOWN model
    logger.info("Optimizing DOWN model...")
    best_params_down = optimizer.optimize_down_model(n_trials=n_trials)
    
    # 7. Evaluate best models
    logger.info("Evaluating best models...")
    test_metrics = optimizer.evaluate_best_models()
    
    # 8. Update configuration
    logger.info("Updating configuration with best parameters...")
    update_config_with_xgb_params(best_params_up, best_params_down)
    
    # 9. Log results
    logger.info("==== XGBoost Tuning Results ====")
    logger.info(f"UP model parameters: {best_params_up}")
    logger.info(f"DOWN model parameters: {best_params_down}")
    logger.info(f"Test profit factor: {test_metrics['profit_factor']:.2f}")
    logger.info(f"Test total trades: {test_metrics['total_trades']}")
    logger.info(f"UP model F1 score: {test_metrics['up_f1']:.4f}")
    logger.info(f"DOWN model F1 score: {test_metrics['down_f1']:.4f}")
    logger.info("==== XGBoost Tuning Complete ====")
    
    return best_params_up, best_params_down, test_metrics

if __name__ == "__main__":
    run_xgboost_tuning()