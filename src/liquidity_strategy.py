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
