import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

def train_binary_xgb_model(X_train, y_train_binary, xgb_params, scale_pos_weight=1):
    """Trains a binary XGBoost classifier."""
    model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight)
    
    # XGBoost can handle NaNs if you configure it (missing=np.nan), 
    # but it's often better to preprocess them.
    # Forcing X_train to be float to avoid potential type issues with XGBoost if columns are int/object
    X_train_processed = X_train.astype(float)

    model.fit(X_train_processed, y_train_binary)
    return model

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
    X_test_processed = X_test.astype(float)

    probs = model.predict_proba(X_test_processed)[:, 1]
    binary_preds = (probs >= prob_threshold).astype(int)

    if verbose:
        print(f"\n--- {model_name} - Test Set Performance (Threshold: {prob_threshold}) ---")
        print(classification_report(y_test_binary, binary_preds, target_names=['Negative', 'Positive'], zero_division=0))
        print(f"{model_name} - Confusion Matrix:")
        print(confusion_matrix(y_test_binary, binary_preds))

    return probs, binary_preds

def get_feature_importances(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        return importances.sort_values('importance', ascending=False)
    return None

if __name__ == '__main__':
    # Create dummy data for testing model training
    from sklearn.datasets import make_classification
    X, y_orig = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                    n_classes=3, weights=[0.7, 0.15, 0.15], random_state=42) # 0=Neutral, 1=Up, 2=Down
    
    # Convert y_orig to our -1, 0, 1 format
    # Let's map class 0 -> 0 (Neutral), class 1 -> 1 (Up), class 2 -> -1 (Down)
    y = np.select([y_orig == 0, y_orig == 1, y_orig == 2], [0, 1, -1], default=0)

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    # TimeSeriesSplit example
    tscv = TimeSeriesSplit(n_splits=2) # Simplified for test
    train_idx, test_idx = list(tscv.split(X_df, y_series))[-1]
    
    X_train_sample, X_test_sample = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train_orig_sample, y_test_orig_sample = y_series.iloc[train_idx], y_series.iloc[test_idx]

    sample_xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 
        'use_label_encoder': False, 'random_state': 42, 'n_estimators': 50
    }

    # Test Model UP
    y_train_up_sample = (y_train_orig_sample == 1).astype(int)
    y_test_up_sample = (y_test_orig_sample == 1).astype(int)
    up_neg_count = (y_train_up_sample == 0).sum()
    up_pos_count = (y_train_up_sample == 1).sum()
    spw_up = up_neg_count / up_pos_count if up_pos_count > 0 else 1
    
    print(f"Training dummy Model UP (Positives: {up_pos_count}, Negatives: {up_neg_count}, SPW: {spw_up:.2f})")
    model_up_sample = train_binary_xgb_model(X_train_sample, y_train_up_sample, sample_xgb_params, scale_pos_weight=spw_up)
    probs_up, preds_up = evaluate_model(model_up_sample, X_test_sample, y_test_up_sample, "Dummy Model UP")
    
    importances_up = get_feature_importances(model_up_sample, X_train_sample.columns)
    if importances_up is not None:
        print("\nDummy Model UP - Feature Importances (Top 5):")
        print(importances_up.head())

    # Test Model DOWN
    y_train_down_sample = (y_train_orig_sample == -1).astype(int)
    y_test_down_sample = (y_test_orig_sample == -1).astype(int)
    down_neg_count = (y_train_down_sample == 0).sum()
    down_pos_count = (y_train_down_sample == 1).sum()
    spw_down = down_neg_count / down_pos_count if down_pos_count > 0 else 1

    print(f"\nTraining dummy Model DOWN (Positives: {down_pos_count}, Negatives: {down_neg_count}, SPW: {spw_down:.2f})")
    model_down_sample = train_binary_xgb_model(X_train_sample, y_train_down_sample, sample_xgb_params, scale_pos_weight=spw_down)
    probs_down, preds_down = evaluate_model(model_down_sample, X_test_sample, y_test_down_sample, "Dummy Model DOWN")
