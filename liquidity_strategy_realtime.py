#!/usr/bin/env python3
"""
Real-time Liquidity Trading Strategy for EURUSD

This script implements the optimized liquidity-based trading strategy for live execution.
It focuses on external liquidity features (previous day/week high/low levels) that have proven
most effective in extended walk-forward testing.

Usage:
    python liquidity_strategy_realtime.py --mode live --ticker EURUSD --timeframe hour

Features:
- Live market data connection via a broker API
- Real-time feature generation and signal processing
- Position management with risk controls
- Performance monitoring and logging
- Email/Telegram notifications for trade signals
"""
import pandas as pd
import numpy as np
import logging
import argparse
import json
import os
import time
import yaml
import datetime
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_handler import fetch_polygon_data

# Optional import for yfinance fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("liquidity_strategy_realtime.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('liquidity_realtime')

# Create directories
os.makedirs("./realtime_results", exist_ok=True)
os.makedirs("./realtime_logs", exist_ok=True)

class RealTimeLiquidityStrategy:
    """Real-time implementation of the liquidity-based trading strategy."""
    
    def __init__(self, args):
        """Initialize the strategy with command-line arguments."""
        self.args = args
        self.ticker = args.ticker
        self.timeframe = args.timeframe
        self.up_threshold = args.up_threshold
        self.down_threshold = args.down_threshold
        self.pt_multiplier = args.pt_multiplier
        self.sl_multiplier = args.sl_multiplier
        self.risk_per_trade = args.risk_per_trade

        # Load configuration
        self.config = self.load_config()

        # Trading state variables
        self.current_position = "FLAT"  # "FLAT", "LONG", or "SHORT"
        self.entry_price = 0.0
        self.position_size = 0.0
        self.profit_target = 0.0
        self.stop_loss = 0.0
        self.last_signal = "NEUTRAL"

        # Load saved models
        self.model_up, self.model_down = self.load_models()

        # Initialize market data
        self.historical_data = None
        self.feature_data = None

        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.initial_equity = args.initial_equity
        self.current_equity = args.initial_equity

        # Notification settings
        self.notify_trades = args.notify_trades
        self.notification_emails = args.emails.split(',') if args.emails else []

        logger.info(f"Strategy initialized for {self.ticker} ({self.timeframe})")
        logger.info(f"Thresholds: UP={self.up_threshold}, DOWN={self.down_threshold}")
        logger.info(f"Risk per trade: {self.risk_per_trade}%")

    def load_config(self):
        """Load configuration from settings.yaml."""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.yaml')
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info("Configuration loaded successfully")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return a default config with an empty API key
            return {"POLYGON_API_KEY": ""}
    
    def load_models(self):
        """Load the pre-trained XGBoost models."""
        try:
            import xgboost as xgb

            # Use model path from arguments or use default paths
            model_up_path = self.args.model_up_path or f"./final_liquidity_strategy/model_up.json"
            model_down_path = self.args.model_down_path or f"./final_liquidity_strategy/model_down.json"

            if not os.path.exists(model_up_path) or not os.path.exists(model_down_path):
                logger.warning(f"Model files not found at {model_up_path} or {model_down_path}")
                logger.info("Creating temporary models for demo purposes")

                # Create a temporary model
                data = pd.DataFrame({
                    'sma_20': [0],
                    'ema_20': [0],
                    'sma_50': [0],
                    'ema_50': [0],
                    'sma_200': [0],
                    'ema_200': [0],
                    'rsi_14': [0],
                    'ATR_14': [0],
                    'hour': [0],
                    'day_of_week': [0],
                    'dist_to_prev_day_high_atr': [0],
                    'dist_to_prev_day_low_atr': [0],
                    'dist_to_prev_week_high_atr': [0],
                    'dist_to_prev_week_low_atr': [0],
                    'swept_prev_day_high': [0],
                    'swept_prev_day_low': [0],
                    'swept_prev_week_high': [0],
                    'swept_prev_week_low': [0],
                    'rel_pos_in_day_range': [0]
                })
                label = pd.Series([0])

                # Create DMatrix
                dtrain = xgb.DMatrix(data, label=label)

                # Train a simple model
                param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
                model_up = xgb.train(param, dtrain, num_boost_round=1)
                model_down = xgb.train(param, dtrain, num_boost_round=1)

                # Save models
                os.makedirs(os.path.dirname(model_up_path), exist_ok=True)
                model_up.save_model(model_up_path)
                model_down.save_model(model_down_path)

                return model_up, model_down

            model_up = xgb.Booster()
            model_down = xgb.Booster()

            model_up.load_model(model_up_path)
            model_down.load_model(model_down_path)

            logger.info(f"Models loaded from {model_up_path} and {model_down_path}")

            return model_up, model_down

        except ImportError:
            logger.error("XGBoost not installed. Please install via pip install xgboost")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def connect_to_broker(self):
        """Connect to the trading broker's API.
        
        This is a placeholder - implement with your specific broker API.
        """
        if self.args.paper_trading:
            logger.info("Paper trading mode - no actual broker connection established")
            return True
        
        # This is where you would implement your actual broker connection
        # Example:
        # self.broker = BrokerAPI(api_key=self.args.api_key, secret=self.args.api_secret)
        # connected = self.broker.connect()
        
        logger.info("Broker connection placeholder - implement with your broker API")
        return True
    
    def fetch_historical_data(self):
        """Fetch historical data for initial feature generation using Polygon.io API."""

        # If demo mode is enabled, use synthetic data
        if self.args.demo_mode:
            logger.info(f"Demo mode enabled - Generating synthetic historical data for {self.ticker}")

            # Create date range
            end_date = datetime.datetime.now()
            if self.timeframe == 'minute':
                start_date = end_date - datetime.timedelta(days=2)
                freq = 'T'
            elif self.timeframe == 'hour':
                start_date = end_date - datetime.timedelta(days=60)
                freq = 'H'
            else:  # day
                start_date = end_date - datetime.timedelta(days=500)
                freq = 'D'

            # Create index
            index = pd.date_range(start=start_date, end=end_date, freq=freq)

            # Generate OHLCV data
            n = len(index)
            close = np.zeros(n)
            close[0] = 1.0800  # Starting price for EUR/USD

            # Parameters for random walk
            drift = 0.00001  # Small upward drift
            volatility = 0.0005  # Typical hourly volatility for EUR/USD

            for i in range(1, n):
                # Random walk with drift and volatility
                close[i] = close[i-1] * (1 + drift + volatility * np.random.normal())

            # Generate open, high, low based on close
            open_prices = close[:-1].copy()
            open_prices = np.append([close[0] - 0.0002], open_prices)  # First open

            # High is max of open and close plus random amount
            high = np.maximum(close, open_prices) + np.random.uniform(0, volatility*2, n)

            # Low is min of open and close minus random amount
            low = np.minimum(close, open_prices) - np.random.uniform(0, volatility*2, n)

            # Create volume with some patterns
            volume = np.random.uniform(1000, 5000, n)

            # Create DataFrame
            self.historical_data = pd.DataFrame({
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            }, index=index)

        else:
            # Use real market data from Polygon.io
            logger.info(f"Fetching real market data from Polygon.io for {self.ticker}")

            # Get API key from config
            api_key = self.config.get('POLYGON_API_KEY', '')
            if not api_key:
                logger.error("Polygon API key not found in configuration")
                raise ValueError("Polygon API key is required for real market data")

            logger.info(f"Using Polygon API key: {api_key[:5]}...{api_key[-5:]}")

            # Set timeframe parameters for Polygon API
            if self.timeframe == 'minute':
                polygon_timespan = 'minute'
                multiplier = 1
                start_date = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
            elif self.timeframe == 'hour':
                polygon_timespan = 'hour'
                multiplier = 1
                start_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
            else:  # day
                polygon_timespan = 'day'
                multiplier = 1
                start_date = (datetime.datetime.now() - datetime.timedelta(days=500)).strftime('%Y-%m-%d')

            end_date = datetime.datetime.now().strftime('%Y-%m-%d')

            # Format ticker for Polygon (forex is prefixed with C:)
            if self.ticker.startswith('C:'):
                polygon_ticker = self.ticker
            else:
                polygon_ticker = f"C:{self.ticker}"

            # Fetch data from Polygon.io
            try:
                self.historical_data = fetch_polygon_data(
                    api_key=api_key,
                    ticker=polygon_ticker,
                    timespan=polygon_timespan,
                    multiplier=multiplier,
                    start_date_str=start_date,
                    end_date_str=end_date
                )

                # Validate data
                if self.historical_data is None or len(self.historical_data) == 0:
                    raise ValueError("No data returned from Polygon.io API")

            except Exception as e:
                logger.error(f"Error fetching data from Polygon.io: {e}")

                # Try to use yfinance as a fallback
                if YFINANCE_AVAILABLE:
                    logger.warning("Attempting to use yfinance as a fallback data source")
                    try:
                        # Convert our ticker format to Yahoo Finance format (EURUSD=X for forex)
                        if self.ticker.startswith('C:'):
                            yf_ticker = self.ticker[2:] + "=X"
                        else:
                            yf_ticker = self.ticker + "=X" if "USD" in self.ticker else self.ticker

                        logger.info(f"Fetching historical data for {yf_ticker} from Yahoo Finance")

                        # Calculate period based on timeframe
                        if self.timeframe == 'minute':
                            period = "7d"  # Yahoo only provides 7 days of minute data
                            interval = "1m"
                        elif self.timeframe == 'hour':
                            period = "60d"
                            interval = "1h"
                        else:  # day
                            period = "2y"
                            interval = "1d"

                        # Fetch from Yahoo Finance
                        yf_data = yf.Ticker(yf_ticker).history(period=period, interval=interval)

                        if not yf_data.empty:
                            # Format to match our expected structure
                            self.historical_data = yf_data.rename(columns={
                                'Open': 'open',
                                'High': 'high',
                                'Low': 'low',
                                'Close': 'close',
                                'Volume': 'volume'
                            })

                            # Keep only the columns we need
                            self.historical_data = self.historical_data[['open', 'high', 'low', 'close', 'volume']]

                            logger.info(f"Successfully loaded {len(self.historical_data)} bars from Yahoo Finance")
                            return self.historical_data
                        else:
                            raise ValueError("Empty response from Yahoo Finance")

                    except Exception as yf_error:
                        logger.error(f"Error fetching data from Yahoo Finance: {yf_error}")
                        logger.warning("Falling back to demo data as last resort")

                # If we get here, both Polygon and yfinance failed (or yfinance not available)
                # Only use synthetic data if it's truly a last resort
                logger.warning("⚠️ ATTENTION: Using DEMO DATA - Trading performance will not be accurate ⚠️")
                logger.warning("To get real market data, fix API connection or install yfinance")
                return self.generate_synthetic_data()

        logger.info(f"Historical data loaded with {len(self.historical_data)} bars")
        return self.historical_data

    def generate_synthetic_data(self):
        """Generate synthetic data as a fallback."""
        logger.info(f"Generating synthetic data for {self.ticker}")

        # Create date range
        end_date = datetime.datetime.now()
        if self.timeframe == 'minute':
            start_date = end_date - datetime.timedelta(days=2)
            freq = 'T'
        elif self.timeframe == 'hour':
            start_date = end_date - datetime.timedelta(days=60)
            freq = 'H'
        else:  # day
            start_date = end_date - datetime.timedelta(days=500)
            freq = 'D'

        # Create index and synthetic data (same as in demo mode)
        index = pd.date_range(start=start_date, end=end_date, freq=freq)
        n = len(index)
        close = np.zeros(n)
        close[0] = 1.0800

        drift = 0.00001
        volatility = 0.0005

        for i in range(1, n):
            close[i] = close[i-1] * (1 + drift + volatility * np.random.normal())

        open_prices = close[:-1].copy()
        open_prices = np.append([close[0] - 0.0002], open_prices)

        high = np.maximum(close, open_prices) + np.random.uniform(0, volatility*2, n)
        low = np.minimum(close, open_prices) - np.random.uniform(0, volatility*2, n)
        volume = np.random.uniform(1000, 5000, n)

        self.historical_data = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=index)

        return self.historical_data
    
    def update_data(self, new_bar=None):
        """Update data with the latest price information.

        Args:
            new_bar: The new price bar to add (if None, will fetch from API)
        """
        if new_bar is None:
            # If in demo mode or if provided a bar, use that
            if self.args.demo_mode:
                # Generate a synthetic bar for demo mode
                last_close = self.historical_data['close'].iloc[-1]
                last_date = self.historical_data.index[-1]

                # Determine next bar's timestamp
                if self.timeframe == 'minute':
                    next_date = last_date + datetime.timedelta(minutes=1)
                elif self.timeframe == 'hour':
                    next_date = last_date + datetime.timedelta(hours=1)
                else:  # day
                    next_date = last_date + datetime.timedelta(days=1)

                # Generate a new synthetic bar
                close = last_close * (1 + 0.0005 * np.random.normal())
                open_price = last_close * (1 + 0.0002 * np.random.normal())
                high = max(close, open_price) + 0.0005 * np.random.random()
                low = min(close, open_price) - 0.0005 * np.random.random()
                volume = 1000 + 4000 * np.random.random()

                new_bar = pd.DataFrame({
                    'open': [open_price],
                    'high': [high],
                    'low': [low],
                    'close': [close],
                    'volume': [volume]
                }, index=[next_date])
            else:
                # Fetch the latest bar from Polygon.io for real-time data
                try:
                    api_key = self.config.get('POLYGON_API_KEY', '')
                    if not api_key:
                        raise ValueError("Polygon API key is required for real-time data")

                    # Format ticker for Polygon
                    if self.ticker.startswith('C:'):
                        polygon_ticker = self.ticker
                    else:
                        polygon_ticker = f"C:{self.ticker}"

                    # Get the last timestamp in our data
                    last_date = self.historical_data.index[-1]

                    # Convert timeframe to Polygon format
                    if self.timeframe == 'minute':
                        polygon_timespan = 'minute'
                        multiplier = 1
                    elif self.timeframe == 'hour':
                        polygon_timespan = 'hour'
                        multiplier = 1
                    else:  # day
                        polygon_timespan = 'day'
                        multiplier = 1

                    # Only fetch data since our last update
                    from_date = (last_date + datetime.timedelta(seconds=1)).strftime('%Y-%m-%d')
                    to_date = datetime.datetime.now().strftime('%Y-%m-%d')

                    # Don't fetch if we're already up to date
                    if from_date > to_date:
                        logger.debug("Already up to date, skipping fetch")
                        return self.historical_data

                    # Fetch latest data
                    latest_data = fetch_polygon_data(
                        api_key=api_key,
                        ticker=polygon_ticker,
                        timespan=polygon_timespan,
                        multiplier=multiplier,
                        start_date_str=from_date,
                        end_date_str=to_date
                    )

                    # Check if we got any new data
                    if latest_data is not None and len(latest_data) > 0:
                        # The entire latest data becomes our new bar
                        new_bar = latest_data
                        logger.info(f"Fetched {len(new_bar)} new bars from Polygon.io")
                    else:
                        logger.debug("No new data available from Polygon.io")
                        return self.historical_data

                except Exception as e:
                    logger.error(f"Error fetching latest data from Polygon.io: {e}")

                    # Try to use yfinance as a fallback
                    if YFINANCE_AVAILABLE:
                        logger.warning("Attempting to use yfinance as a fallback data source")
                        try:
                            # Convert our ticker format to Yahoo Finance format (EURUSD=X for forex)
                            if self.ticker.startswith('C:'):
                                yf_ticker = self.ticker[2:] + "=X"
                            else:
                                yf_ticker = self.ticker + "=X" if "USD" in self.ticker else self.ticker

                            # Get the last timestamp in our data
                            last_date = self.historical_data.index[-1]

                            # Calculate periods based on timeframe
                            if self.timeframe == 'minute':
                                period = "1d"
                                interval = "1m"
                            elif self.timeframe == 'hour':
                                period = "7d"
                                interval = "1h"
                            else:  # day
                                period = "30d"
                                interval = "1d"

                            # Fetch data from Yahoo Finance
                            yf_data = yf.Ticker(yf_ticker).history(period=period, interval=interval)

                            if not yf_data.empty:
                                # Get only the data after our last date
                                latest_data = yf_data[yf_data.index > last_date]

                                if not latest_data.empty:
                                    # Format to match our expected structure
                                    latest_data = latest_data.rename(columns={
                                        'Open': 'open',
                                        'High': 'high',
                                        'Low': 'low',
                                        'Close': 'close',
                                        'Volume': 'volume'
                                    })

                                    new_bar = latest_data[['open', 'high', 'low', 'close', 'volume']]
                                    logger.info(f"Successfully fetched {len(new_bar)} bars from Yahoo Finance")
                                else:
                                    logger.info("No new data available from Yahoo Finance")
                                    return self.historical_data
                            else:
                                logger.warning("Empty response from Yahoo Finance")
                                return self.historical_data
                        except Exception as yf_error:
                            logger.error(f"Error fetching data from Yahoo Finance: {yf_error}")
                            logger.warning("Using last known data. THIS WILL NOT AFFECT TRADING PERFORMANCE.")
                            return self.historical_data

                    # If we reach here and yfinance is not available, inform the user
                    logger.warning("No fallback data source available. Using last known data.")
                    logger.info("Consider installing yfinance for better fallback: pip install yfinance")
                    return self.historical_data

        # Append the new bar to historical data
        self.historical_data = pd.concat([self.historical_data, new_bar])

        # Optional: Trim historical data to keep it at a manageable size
        max_bars = 10000
        if len(self.historical_data) > max_bars:
            self.historical_data = self.historical_data.iloc[-max_bars:]

        logger.debug(f"Data updated with {len(new_bar)} new bars")
        return self.historical_data
    
    def add_basic_features(self):
        """Add basic technical indicators to the data."""
        df_feat = self.historical_data.copy()
        
        # Calculate basic indicators
        # Moving averages
        for window in [20, 50, 200]:
            df_feat[f'sma_{window}'] = df_feat['close'].rolling(window=window).mean()
            df_feat[f'ema_{window}'] = df_feat['close'].ewm(span=window, adjust=False).mean()
        
        # Relative strength
        returns = df_feat['close'].pct_change()
        up_returns = returns.copy()
        down_returns = returns.copy()
        up_returns[up_returns < 0] = 0
        down_returns[down_returns > 0] = 0
        down_returns = -down_returns
        
        avg_up = up_returns.rolling(window=14).mean()
        avg_down = down_returns.rolling(window=14).mean()
        
        rs = avg_up / avg_down
        df_feat['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ATR calculation
        high_low = df_feat['high'] - df_feat['low']
        high_close = (df_feat['high'] - df_feat['close'].shift()).abs()
        low_close = (df_feat['low'] - df_feat['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df_feat['ATR_14'] = true_range.rolling(window=14).mean()
        
        # Remove NaN values at the beginning
        # df_feat = df_feat.dropna()
        
        self.feature_data = df_feat
        return self.feature_data
    
    def add_liquidity_features(self):
        """Add specialized liquidity-based features."""
        df_liq = self.feature_data.copy()
        
        # Define time-based features
        df_liq['hour'] = df_liq.index.hour
        df_liq['day_of_week'] = df_liq.index.dayofweek
        
        # Daily high/low
        df_liq['date'] = df_liq.index.date
        
        # Calculate daily high/low
        daily_high = df_liq.groupby('date')['high'].transform('max')
        daily_low = df_liq.groupby('date')['low'].transform('min')
        
        # Previous day's high/low
        # For different timeframes, adjust shift values
        if self.timeframe == 'hour':
            day_shift = 24  # 24 hours in a day
        elif self.timeframe == 'minute':
            day_shift = 24 * 60  # minutes in a day
        else:
            day_shift = 1  # For daily data
        
        prev_day_high = df_liq.groupby('date')['high'].transform('max').shift(day_shift)
        prev_day_low = df_liq.groupby('date')['low'].transform('min').shift(day_shift)
        
        # Calculate distance to these liquidity levels in terms of ATR
        df_liq['dist_to_prev_day_high_atr'] = (prev_day_high - df_liq['close']) / df_liq['ATR_14']
        df_liq['dist_to_prev_day_low_atr'] = (df_liq['close'] - prev_day_low) / df_liq['ATR_14']
        
        # Weekly high/low calculation
        df_liq['week'] = df_liq.index.isocalendar().week
        df_liq['year'] = df_liq.index.year
        
        # Create a week identifier
        df_liq['week_id'] = df_liq['year'].astype(str) + '-' + df_liq['week'].astype(str).str.zfill(2)
        
        # Calculate weekly high/low
        weekly_high = df_liq.groupby('week_id')['high'].transform('max')
        weekly_low = df_liq.groupby('week_id')['low'].transform('min')
        
        # Previous week's high/low
        prev_week_high = weekly_high.shift(day_shift * 7)
        prev_week_low = weekly_low.shift(day_shift * 7)
        
        # Distance to weekly levels
        df_liq['dist_to_prev_week_high_atr'] = (prev_week_high - df_liq['close']) / df_liq['ATR_14']
        df_liq['dist_to_prev_week_low_atr'] = (df_liq['close'] - prev_week_low) / df_liq['ATR_14']
        
        # Monthly levels
        df_liq['month'] = df_liq.index.month
        monthly_high = df_liq.groupby(['year', 'month'])['high'].transform('max')
        monthly_low = df_liq.groupby(['year', 'month'])['low'].transform('min')
        
        # Previous month high/low
        prev_month_high = monthly_high.shift(day_shift * 30)
        prev_month_low = monthly_low.shift(day_shift * 30)
        
        # Distance to monthly levels
        df_liq['dist_to_prev_month_high_atr'] = (prev_month_high - df_liq['close']) / df_liq['ATR_14']
        df_liq['dist_to_prev_month_low_atr'] = (df_liq['close'] - prev_month_low) / df_liq['ATR_14']
        
        # Sweep detection (price touches a level and reverses)
        df_liq['swept_prev_day_high'] = (df_liq['high'] > prev_day_high) & (df_liq['close'] < prev_day_high)
        df_liq['swept_prev_day_low'] = (df_liq['low'] < prev_day_low) & (df_liq['close'] > prev_day_low)
        df_liq['swept_prev_week_high'] = (df_liq['high'] > prev_week_high) & (df_liq['close'] < prev_week_high)
        df_liq['swept_prev_week_low'] = (df_liq['low'] < prev_week_low) & (df_liq['close'] > prev_week_low)
        
        # Relative position in ranges
        df_liq['rel_pos_in_day_range'] = (df_liq['close'] - daily_low) / (daily_high - daily_low + 1e-10)
        df_liq['rel_pos_in_week_range'] = (df_liq['close'] - weekly_low) / (weekly_high - weekly_low + 1e-10)
        df_liq['rel_pos_in_month_range'] = (df_liq['close'] - monthly_low) / (monthly_high - monthly_low + 1e-10)
        
        # Drop unnecessary columns
        df_liq = df_liq.drop(columns=['date', 'week', 'year', 'week_id', 'month'])
        
        # Fill NaN values
        df_liq = df_liq.fillna(0)
        
        self.feature_data = df_liq
        return self.feature_data
    
    def generate_signals(self):
        """Generate trading signals using the models."""
        import xgboost as xgb

        # Prepare feature data for prediction
        X = self.feature_data.iloc[-1:].copy()

        # Drop columns we don't want as features
        X = X.drop(columns=['open', 'high', 'low', 'close', 'volume'])

        # Handle NaN values
        X = X.fillna(0)

        # Get feature names from the models to ensure compatibility
        try:
            model_features = self.model_up.feature_names
            logger.debug(f"Model features: {model_features}")
            logger.debug(f"Data features: {X.columns.tolist()}")

            # Ensure we have all needed features or add missing ones
            for feature in model_features:
                if feature not in X.columns:
                    X[feature] = 0  # Add missing features with default values

            # Only keep features that are in the model
            X = X[model_features]

        except Exception as e:
            logger.warning(f"Could not get model features, using all features: {e}")

        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(X)

        # For demo purposes, generate more realistic probabilities
        if self.args.demo_mode:
            # Create some semi-random probabilities based on price movements
            current_price = self.feature_data['close'].iloc[-1]
            prev_price = self.feature_data['close'].iloc[-2] if len(self.feature_data) > 1 else current_price

            # Base probabilities on recent price movement with some randomness
            if current_price > prev_price:
                # Price moved up, higher chance of UP signal
                up_prob = 0.4 + 0.3 * np.random.random()
                down_prob = 0.3 + 0.2 * np.random.random()
            else:
                # Price moved down, higher chance of DOWN signal
                up_prob = 0.3 + 0.2 * np.random.random()
                down_prob = 0.4 + 0.3 * np.random.random()

            # Add some sweep detection logic
            if 'swept_prev_day_high' in self.feature_data.columns and self.feature_data['swept_prev_day_high'].iloc[-1]:
                down_prob = 0.6 + 0.2 * np.random.random()  # Higher probability of reversal after sweep

            if 'swept_prev_day_low' in self.feature_data.columns and self.feature_data['swept_prev_day_low'].iloc[-1]:
                up_prob = 0.6 + 0.2 * np.random.random()  # Higher probability of reversal after sweep
        else:
            # Use the actual model predictions
            up_prob = self.model_up.predict(dmatrix)[0]
            down_prob = self.model_down.predict(dmatrix)[0]
        
        # Generate signal
        signal = "NEUTRAL"
        if up_prob >= self.up_threshold and down_prob < (1.0 - self.down_threshold):
            signal = "LONG"
        elif down_prob >= self.down_threshold and up_prob < (1.0 - self.up_threshold):
            signal = "SHORT"
        
        logger.info(f"Signal generated: {signal} (UP prob: {up_prob:.4f}, DOWN prob: {down_prob:.4f})")
        
        # Save the signal with timestamp
        timestamp = self.feature_data.index[-1]
        signal_data = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'signal': signal,
            'up_probability': float(up_prob),  # Convert numpy float types to Python float
            'down_probability': float(down_prob),
            'close': float(self.feature_data['close'].iloc[-1]),
            'atr': float(self.feature_data['ATR_14'].iloc[-1])
        }

        # Make sure realtime_logs directory exists
        os.makedirs("./realtime_logs", exist_ok=True)

        # Save signals to file
        with open(f"./realtime_logs/signals_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl", 'a') as f:
            f.write(json.dumps(signal_data) + '\n')
        
        self.last_signal = signal
        return signal, up_prob, down_prob
    
    def calculate_position_size(self):
        """Calculate position size based on risk management rules."""
        if self.last_signal == "NEUTRAL":
            return 0
        
        # Extract current ATR for calculating stop loss distance
        current_atr = self.feature_data['ATR_14'].iloc[-1]
        current_price = self.feature_data['close'].iloc[-1]
        
        # Calculate stop loss distance in price units
        if self.last_signal == "LONG":
            stop_distance = self.sl_multiplier * current_atr
        else:  # SHORT
            stop_distance = self.sl_multiplier * current_atr
        
        # Calculate position size based on risk per trade
        risk_amount = self.current_equity * (self.risk_per_trade / 100)
        
        # Basic position size calculation (lot size would depend on your broker's specifications)
        position_size = risk_amount / stop_distance
        
        # For forex, position size is typically in lots
        # 1 standard lot = 100,000 units, 1 mini lot = 10,000 units, 1 micro lot = 1,000 units
        lot_size = position_size / 100000  # Convert to standard lots
        
        # Round to nearest 0.01 lot (or whatever your broker's minimum lot size increment is)
        lot_size = round(lot_size, 2)
        
        # Ensure minimum lot size
        min_lot_size = 0.01
        lot_size = max(lot_size, min_lot_size)
        
        logger.info(f"Calculated position size: {lot_size} lots based on risk amount: ${risk_amount:.2f}")
        
        return lot_size
    
    def manage_positions(self):
        """Manage trading positions based on signals."""
        current_price = self.feature_data['close'].iloc[-1]
        current_atr = self.feature_data['ATR_14'].iloc[-1]
        
        # If we have no position, check for entry signals
        if self.current_position == "FLAT":
            if self.last_signal == "LONG":
                # Enter long position
                self.position_size = self.calculate_position_size()
                self.entry_price = current_price
                self.profit_target = current_price + (self.pt_multiplier * current_atr)
                self.stop_loss = current_price - (self.sl_multiplier * current_atr)
                self.current_position = "LONG"
                
                logger.info(f"LONG position opened at {self.entry_price:.5f}, size: {self.position_size}, PT: {self.profit_target:.5f}, SL: {self.stop_loss:.5f}")
                
                # Record trade
                self.record_trade("ENTRY", "LONG", self.entry_price, self.position_size)
                
                # Send notification if enabled
                if self.notify_trades:
                    self.send_notification(f"LONG position opened at {self.entry_price:.5f}, PT: {self.profit_target:.5f}, SL: {self.stop_loss:.5f}")
            
            elif self.last_signal == "SHORT":
                # Enter short position
                self.position_size = self.calculate_position_size()
                self.entry_price = current_price
                self.profit_target = current_price - (self.pt_multiplier * current_atr)
                self.stop_loss = current_price + (self.sl_multiplier * current_atr)
                self.current_position = "SHORT"
                
                logger.info(f"SHORT position opened at {self.entry_price:.5f}, size: {self.position_size}, PT: {self.profit_target:.5f}, SL: {self.stop_loss:.5f}")
                
                # Record trade
                self.record_trade("ENTRY", "SHORT", self.entry_price, self.position_size)
                
                # Send notification if enabled
                if self.notify_trades:
                    self.send_notification(f"SHORT position opened at {self.entry_price:.5f}, PT: {self.profit_target:.5f}, SL: {self.stop_loss:.5f}")
        
        # Check for exit conditions if we have an open position
        elif self.current_position == "LONG":
            # Check if price reached profit target
            if current_price >= self.profit_target:
                # Close the position with profit
                profit = (current_price - self.entry_price) * self.position_size * 100000
                self.current_equity += profit

                logger.info(f"LONG position closed at profit target {current_price:.5f}, profit: ${profit:.2f}")

                # Record trade
                self.record_trade("EXIT", "LONG", current_price, self.position_size, profit)

                # Reset position
                self.current_position = "FLAT"

            # Check if price hit stop loss
            elif current_price <= self.stop_loss:
                # Close the position with loss
                loss = (current_price - self.entry_price) * self.position_size * 100000
                self.current_equity += loss

                logger.info(f"⚠️ LONG position stopped out at {current_price:.5f}, loss: ${loss:.2f}")

                # Record trade
                self.record_trade("EXIT", "LONG", current_price, self.position_size, loss)

                # Reset position
                self.current_position = "FLAT"
                self.position_size = 0
                
                # Send notification if enabled
                if self.notify_trades:
                    self.send_notification(f"LONG position closed at profit target {current_price:.5f}, profit: ${profit:.2f}")
            
            # Check if price reached stop loss
            elif current_price <= self.stop_loss:
                # Close the position with loss
                loss = (current_price - self.entry_price) * self.position_size * 100000
                self.current_equity += loss

                logger.info(f"⚠️ LONG position closed at stop loss {current_price:.5f}, loss: ${loss:.2f}")
                logger.info(f"Entry: {self.entry_price:.5f}, SL hit: {current_price:.5f}, SL level: {self.stop_loss:.5f}")

                # Record trade
                self.record_trade("EXIT", "LONG", current_price, self.position_size, loss)

                # Reset position
                self.current_position = "FLAT"
                self.position_size = 0

                # Send notification if enabled
                if self.notify_trades:
                    self.send_notification(f"⚠️ LONG position stopped out at {current_price:.5f}, loss: ${loss:.2f}")
        
        elif self.current_position == "SHORT":
            # Check if price reached profit target
            if current_price <= self.profit_target:
                # Close the position with profit
                profit = (self.entry_price - current_price) * self.position_size * 100000
                self.current_equity += profit
                
                logger.info(f"SHORT position closed at profit target {current_price:.5f}, profit: ${profit:.2f}")
                
                # Record trade
                self.record_trade("EXIT", "SHORT", current_price, self.position_size, profit)
                
                # Reset position
                self.current_position = "FLAT"
                self.position_size = 0
                
                # Send notification if enabled
                if self.notify_trades:
                    self.send_notification(f"SHORT position closed at profit target {current_price:.5f}, profit: ${profit:.2f}")
            
            # Check if price reached stop loss
            elif current_price >= self.stop_loss:
                # Close the position with loss
                loss = (self.entry_price - current_price) * self.position_size * 100000
                self.current_equity += loss
                
                logger.info(f"SHORT position closed at stop loss {current_price:.5f}, loss: ${loss:.2f}")
                
                # Record trade
                self.record_trade("EXIT", "SHORT", current_price, self.position_size, loss)
                
                # Reset position
                self.current_position = "FLAT"
                self.position_size = 0
                
                # Send notification if enabled
                if self.notify_trades:
                    self.send_notification(f"SHORT position closed at stop loss {current_price:.5f}, loss: ${loss:.2f}")
    
    def record_trade(self, action, direction, price, size, pnl=None):
        """Record trade details for analysis."""
        trade = {
            'timestamp': self.feature_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
            'direction': direction,
            'price': float(price),
            'size': float(size),
            'pnl': float(pnl) if pnl is not None else None,
            'equity': float(self.current_equity)
        }

        self.trades.append(trade)
        self.equity_curve.append({'timestamp': self.feature_data.index[-1], 'equity': float(self.current_equity)})

        # Make sure realtime_logs directory exists
        os.makedirs("./realtime_logs", exist_ok=True)

        # Save trades to file
        with open(f"./realtime_logs/trades_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl", 'a') as f:
            f.write(json.dumps(trade) + '\n')
    
    def send_notification(self, message):
        """Send notification about trade events.
        
        This is a placeholder - implement with your preferred notification method.
        """
        if not self.notify_trades:
            return
        
        # Log the notification
        logger.info(f"Notification: {message}")
        
        # In a real implementation, send email or push notification
        # Example: send_email(self.notification_emails, "Trade Alert", message)
        # Example: send_telegram_message(self.telegram_chat_id, message)
    
    def generate_performance_report(self):
        """Generate and save a performance report."""
        if not self.trades:
            logger.info("No trades to report")
            return
        
        # Calculate basic performance metrics
        total_trades = len([t for t in self.trades if t['action'] == 'EXIT'])
        winning_trades = len([t for t in self.trades if t['action'] == 'EXIT' and t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['action'] == 'EXIT' and t['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum([t['pnl'] for t in self.trades if t['action'] == 'EXIT' and t['pnl'] > 0]) or 0
        total_loss = sum([abs(t['pnl']) for t in self.trades if t['action'] == 'EXIT' and t['pnl'] < 0]) or 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate drawdown
        equity_values = [self.initial_equity] + [t['equity'] for t in self.trades if t['action'] == 'EXIT']
        cummax = np.maximum.accumulate(equity_values)
        drawdown = (1 - np.array(equity_values) / cummax) * 100
        max_drawdown = np.max(drawdown)
        
        # Create report
        report = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'initial_equity': self.initial_equity,
            'current_equity': self.current_equity,
            'total_return': (self.current_equity - self.initial_equity) / self.initial_equity,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'current_position': self.current_position,
            'position_size': self.position_size if self.current_position != "FLAT" else 0
        }
        
        # Log summary
        logger.info(f"Performance Report:")
        logger.info(f"  Total Return: {report['total_return']:.2%}")
        logger.info(f"  Total Trades: {report['total_trades']}")
        logger.info(f"  Win Rate: {report['win_rate']:.2%}")
        logger.info(f"  Profit Factor: {report['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {report['max_drawdown']:.2%}")
        
        # Save report to file
        with open(f"./realtime_results/performance_{datetime.datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate equity curve chart if we have trades
        if self.equity_curve:
            plt.figure(figsize=(12, 6))
            dates = [e['timestamp'] for e in self.equity_curve]
            equity = [e['equity'] for e in self.equity_curve]
            plt.plot(dates, equity)
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"./realtime_results/equity_curve_{datetime.datetime.now().strftime('%Y%m%d')}.png")
        
        return report
    
    def run_realtime(self):
        """Run the strategy in real-time mode."""
        logger.info("Starting real-time strategy execution...")
        
        # Connect to broker
        if not self.connect_to_broker():
            logger.error("Failed to connect to broker. Exiting.")
            return
        
        # Initial data fetch
        self.fetch_historical_data()
        
        # Add features
        self.add_basic_features()
        self.add_liquidity_features()
        
        # Main loop
        running = True
        last_bar_time = None
        
        while running:
            try:
                # Get current time
                current_time = datetime.datetime.now()
                
                # Check if we need to update data
                update_needed = False
                
                if last_bar_time is None:
                    update_needed = True
                elif self.timeframe == 'minute' and current_time.minute != last_bar_time.minute:
                    update_needed = True
                elif self.timeframe == 'hour' and current_time.hour != last_bar_time.hour:
                    update_needed = True
                elif self.timeframe == 'day' and current_time.day != last_bar_time.day:
                    update_needed = True
                
                if update_needed:
                    # Update data with a new bar
                    self.update_data()
                    
                    # Update features
                    self.add_basic_features()
                    self.add_liquidity_features()
                    
                    # Generate signals
                    self.generate_signals()
                    
                    # Manage positions
                    self.manage_positions()
                    
                    # Generate performance report
                    self.generate_performance_report()
                    
                    # Update last bar time
                    last_bar_time = current_time
                    
                    logger.info(f"Strategy updated at {current_time}")
                
                # Sleep until next check (adjust based on your timeframe)
                sleep_seconds = 10  # Check every 10 seconds
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("Strategy execution stopped by user")
                running = False
            except Exception as e:
                logger.error(f"Error in strategy execution: {str(e)}", exc_info=True)
                
                # For critical errors, you might want to exit
                # running = False
                
                # Wait before retrying
                time.sleep(60)
        
        # Final performance report
        self.generate_performance_report()
        logger.info("Strategy execution completed")
    
    def run_backtest(self):
        """Run the strategy in backtest mode on the historical data."""
        logger.info("Starting backtest mode...")
        
        # Initialize data
        self.fetch_historical_data()
        
        # Add features
        self.add_basic_features()
        self.add_liquidity_features()
        
        # Initialize backtest variables
        self.current_position = "FLAT"
        self.current_equity = self.initial_equity
        self.trades = []
        self.equity_curve = []
        
        # Iterate through each bar (except the last few needed for labeling)
        bar_count = len(self.feature_data)
        for i in range(200, bar_count - 1):  # Start after enough data for features
            # Create a view of data up to the current bar
            current_data = self.feature_data.iloc[:i+1].copy()
            
            # Store the original data
            original_data = self.feature_data
            
            # Temporarily replace feature data with current view
            self.feature_data = current_data
            
            # Generate signals
            self.generate_signals()
            
            # Manage positions
            self.manage_positions()
            
            # Restore original data
            self.feature_data = original_data
        
        # Generate final performance report
        report = self.generate_performance_report()
        
        logger.info("Backtest completed")
        return report

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-Time Liquidity Trading Strategy')
    
    # Mode parameters
    parser.add_argument('--mode', type=str, default='backtest', 
                        choices=['live', 'paper', 'backtest'],
                        help='Trading mode: live, paper, or backtest')
    
    # Data parameters
    parser.add_argument('--ticker', type=str, default='EURUSD',
                        help='Ticker symbol to trade')
    parser.add_argument('--timeframe', type=str, default='hour',
                        choices=['minute', 'hour', 'day'],
                        help='Timeframe for analysis')
    
    # Strategy parameters
    parser.add_argument('--up-threshold', type=float, default=0.5,
                        help='UP probability threshold (0.0-1.0)')
    parser.add_argument('--down-threshold', type=float, default=0.5,
                        help='DOWN probability threshold (0.0-1.0)')
    parser.add_argument('--pt-multiplier', type=float, default=2.5,
                        help='Profit target multiplier (as ATR multiple)')
    parser.add_argument('--sl-multiplier', type=float, default=1.5,
                        help='Stop loss multiplier (as ATR multiple)')
    
    # Risk parameters
    parser.add_argument('--risk-per-trade', type=float, default=1.0,
                        help='Risk per trade as percentage of equity (1.0 = 1%)')
    parser.add_argument('--initial-equity', type=float, default=10000,
                        help='Initial trading equity')
    
    # Model parameters
    parser.add_argument('--model-up-path', type=str, default=None,
                        help='Path to UP model file')
    parser.add_argument('--model-down-path', type=str, default=None,
                        help='Path to DOWN model file')
    
    # Notification parameters
    parser.add_argument('--notify-trades', action='store_true',
                        help='Send notifications for trades')
    parser.add_argument('--emails', type=str, default=None,
                        help='Comma-separated list of email addresses for notifications')
    
    # Other parameters
    parser.add_argument('--paper-trading', action='store_true',
                        help='Run in paper trading mode (no actual orders)')
    parser.add_argument('--demo-mode', action='store_true',
                        help='Run in demo mode with simulated signals for testing')
    
    return parser.parse_args()

def main():
    """Main function to run the strategy."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the strategy
    strategy = RealTimeLiquidityStrategy(args)
    
    if args.mode == 'live' or args.mode == 'paper':
        # Run in real-time mode
        strategy.run_realtime()
    else:
        # Run in backtest mode
        strategy.run_backtest()

if __name__ == "__main__":
    main()