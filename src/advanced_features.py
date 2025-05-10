import pandas as pd
import numpy as np
import logging
from collections import deque
from datetime import timedelta

# Try to import talib, use placeholder if not available
try:
    import talib
except ImportError:
    logging.warning("TA-Lib not found. Some advanced features will not be available.")
    # Create minimal talib mock functions for essential functions
    class TalibMock:
        def SMA(self, data, timeperiod):
            return pd.Series(np.nan, index=data.index)

        def EMA(self, data, timeperiod):
            return pd.Series(np.nan, index=data.index)

        def BBANDS(self, data, timeperiod, nbdevup, nbdevdn, matype):
            return (pd.Series(np.nan, index=data.index),
                    pd.Series(np.nan, index=data.index),
                    pd.Series(np.nan, index=data.index))

    talib = TalibMock()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('advanced_features')

def add_session_features(df, config):
    """
    Add trading session-based features including session high/lows and sweeps.
    
    Args:
        df: DataFrame with OHLC data (index must be datetime with timezone info)
        config: Configuration dictionary with session parameters
    
    Returns:
        DataFrame with added session features
    """
    df_session = df.copy()
    
    # Ensure index is timezone-aware
    if df_session.index.tz is None:
        df_session.index = df_session.index.tz_localize('UTC')
    
    # Add date column for grouping
    df_session['date'] = df_session.index.date
    
    # Define session hour ranges (UTC)
    asia_hours = (0, 7)    # Asian session: 00:00-07:00 UTC
    london_hours = (7, 16)  # London session: 07:00-16:00 UTC
    ny_hours = (12, 21)    # New York session: 12:00-21:00 UTC
    
    # Mark sessions
    df_session['is_asia'] = (df_session.index.hour >= asia_hours[0]) & (df_session.index.hour < asia_hours[1])
    df_session['is_london'] = (df_session.index.hour >= london_hours[0]) & (df_session.index.hour < london_hours[1])
    df_session['is_ny'] = (df_session.index.hour >= ny_hours[0]) & (df_session.index.hour < ny_hours[1])
    
    # Calculate session high/lows for the current day
    for session in ['asia', 'london', 'ny']:
        # Get high/low for each session per day
        session_data = df_session[df_session[f'is_{session}']]
        if not session_data.empty:
            session_highs = session_data.groupby('date')['high'].max()
            session_lows = session_data.groupby('date')['low'].min()
            
            # Create DataFrames with session data
            high_df = pd.DataFrame({f'{session}_high': session_highs})
            low_df = pd.DataFrame({f'{session}_low': session_lows})
            
            # Merge with original DataFrame
            df_session = pd.merge(
                df_session, 
                high_df, 
                left_on='date', 
                right_index=True, 
                how='left'
            )
            df_session = pd.merge(
                df_session, 
                low_df, 
                left_on='date', 
                right_index=True, 
                how='left'
            )
            
            # Forward fill session values for the same day
            df_session[f'{session}_high'] = df_session.groupby('date')[f'{session}_high'].ffill()
            df_session[f'{session}_low'] = df_session.groupby('date')[f'{session}_low'].ffill()
            
            # Create previous day session high/lows
            df_session[f'prev_{session}_high'] = df_session[f'{session}_high'].shift(1)
            df_session[f'prev_{session}_low'] = df_session[f'{session}_low'].shift(1)
            
            # Detect sweeps of previous session high/lows
            df_session[f'swept_prev_{session}_high'] = (
                (df_session['high'] > df_session[f'prev_{session}_high']) & 
                (df_session['close'] < df_session[f'prev_{session}_high'])
            )
            df_session[f'swept_prev_{session}_low'] = (
                (df_session['low'] < df_session[f'prev_{session}_low']) & 
                (df_session['close'] > df_session[f'prev_{session}_low'])
            )
    
    # Calculate whether price is currently trading above/below session opens
    for session in ['asia', 'london', 'ny']:
        session_data = df_session[df_session[f'is_{session}']]
        if not session_data.empty:
            # Get first candle of each session per day
            session_opens = session_data.groupby('date')['open'].first()
            opens_df = pd.DataFrame({f'{session}_open': session_opens})
            
            # Merge with main DataFrame
            df_session = pd.merge(
                df_session, 
                opens_df, 
                left_on='date', 
                right_index=True, 
                how='left'
            )
            
            # Forward fill session opens for the day
            df_session[f'{session}_open'] = df_session.groupby('date')[f'{session}_open'].ffill()
            
            # Add above/below session open features
            df_session[f'above_{session}_open'] = df_session['close'] > df_session[f'{session}_open']
    
    # Drop temporary columns
    df_session = df_session.drop(columns=['date', 'is_asia', 'is_london', 'is_ny'])
    
    return df_session

def add_external_liquidity_features(df, config):
    """
    Add features related to external liquidity (previous day/week high/lows).
    
    Args:
        df: DataFrame with OHLC data
        config: Configuration dictionary
    
    Returns:
        DataFrame with added external liquidity features
    """
    df_liq = df.copy()
    
    # Ensure index is datetime
    if not isinstance(df_liq.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex. Converting to datetime.")
        df_liq.index = pd.to_datetime(df_liq.index)
    
    # Add date columns for grouping
    df_liq['date'] = df_liq.index.date
    df_liq['week'] = df_liq.index.isocalendar().week
    df_liq['year'] = df_liq.index.year
    
    # Daily high/low
    daily_high = df_liq.groupby('date')['high'].transform('max')
    daily_low = df_liq.groupby('date')['low'].transform('min')
    
    # Get previous day's high/low
    # For each date, get the high/low, then shift to get previous day's values
    prev_day_high = df_liq.groupby('date')['high'].max().shift(1)
    prev_day_low = df_liq.groupby('date')['low'].min().shift(1)
    
    # Map these values back to all candles for each day
    prev_day_high_map = df_liq['date'].map(prev_day_high)
    prev_day_low_map = df_liq['date'].map(prev_day_low)
    
    # Add to DataFrame
    df_liq['prev_day_high'] = prev_day_high_map
    df_liq['prev_day_low'] = prev_day_low_map
    
    # Weekly high/low (more complex due to week boundaries)
    weekly_high = df_liq.groupby(['year', 'week'])['high'].transform('max')
    weekly_low = df_liq.groupby(['year', 'week'])['low'].transform('min')
    
    # Create a week identifier
    df_liq['week_id'] = df_liq['year'].astype(str) + '-' + df_liq['week'].astype(str).str.zfill(2)
    
    # Get unique week IDs in chronological order
    week_ids = df_liq['week_id'].unique()
    
    # Map to create a numeric index for weeks
    week_idx_map = {week: i for i, week in enumerate(week_ids)}
    df_liq['week_idx'] = df_liq['week_id'].map(week_idx_map)
    
    # Get previous week's high/low
    prev_week_high = {}
    prev_week_low = {}
    
    for week_id in week_ids[1:]:  # Skip first week as it has no previous
        curr_idx = week_idx_map[week_id]
        prev_idx = curr_idx - 1
        prev_week_id = week_ids[prev_idx]
        
        # Get high/low of previous week
        prev_high = df_liq[df_liq['week_id'] == prev_week_id]['high'].max()
        prev_low = df_liq[df_liq['week_id'] == prev_week_id]['low'].min()
        
        prev_week_high[week_id] = prev_high
        prev_week_low[week_id] = prev_low
    
    # Map these values back to all candles
    df_liq['prev_week_high'] = df_liq['week_id'].map(prev_week_high)
    df_liq['prev_week_low'] = df_liq['week_id'].map(prev_week_low)
    
    # Calculate distance to these liquidity levels in terms of ATR
    for level in ['prev_day_high', 'prev_day_low', 'prev_week_high', 'prev_week_low']:
        if level.endswith('high'):
            df_liq[f'dist_to_{level}_atr'] = (df_liq[level] - df_liq['close']) / df_liq['ATR_14']
        else:
            df_liq[f'dist_to_{level}_atr'] = (df_liq['close'] - df_liq[level]) / df_liq['ATR_14']
    
    # Add relative position in daily range
    df_liq['rel_pos_in_day_range'] = (df_liq['close'] - daily_low) / (daily_high - daily_low)
    
    # Detect sweeps of previous day/week high/lows
    df_liq['swept_prev_day_high'] = (df_liq['high'] > df_liq['prev_day_high']) & (df_liq['close'] < df_liq['prev_day_high'])
    df_liq['swept_prev_day_low'] = (df_liq['low'] < df_liq['prev_day_low']) & (df_liq['close'] > df_liq['prev_day_low'])
    df_liq['swept_prev_week_high'] = (df_liq['high'] > df_liq['prev_week_high']) & (df_liq['close'] < df_liq['prev_week_high'])
    df_liq['swept_prev_week_low'] = (df_liq['low'] < df_liq['prev_week_low']) & (df_liq['close'] > df_liq['prev_week_low'])
    
    # Drop temporary columns
    df_liq = df_liq.drop(columns=['date', 'week', 'year', 'week_id', 'week_idx'])
    
    return df_liq

class FVGTracker:
    """
    Class to track Fair Value Gaps (FVGs) and their mitigation status.
    
    A Fair Value Gap is a price region left unfilled after a significant move.
    - Bullish FVG: Current candle's low is higher than the previous-previous candle's high
    - Bearish FVG: Current candle's high is lower than the previous-previous candle's low
    """
    
    def __init__(self, max_fvgs=50, lookback_periods=200, atr_multiplier_significance=0.5):
        """
        Initialize FVGTracker.
        
        Args:
            max_fvgs: Maximum number of FVGs to track per direction
            lookback_periods: How far back to look when processing historical data
            atr_multiplier_significance: Multiplier of ATR to consider an FVG significant
        """
        self.bullish_fvgs = deque(maxlen=max_fvgs)  # [(timestamp, top, bottom, mitigated)]
        self.bearish_fvgs = deque(maxlen=max_fvgs)  # [(timestamp, top, bottom, mitigated)]
        self.lookback_periods = lookback_periods
        self.atr_mult = atr_multiplier_significance
    
    def process_candle(self, idx, row, df):
        """
        Process a single candle to identify and track FVGs.
        
        Args:
            idx: Index of the current candle
            row: Current candle data
            df: Full DataFrame
        
        Returns:
            dict: Features related to FVGs for this candle
        """
        candle_features = {}
        
        # Can't calculate FVGs until we have at least 3 candles
        if idx < 2:
            return {
                'bullish_fvg_formed': 0,
                'bearish_fvg_formed': 0,
                'dist_to_nearest_bullish_fvg': np.nan,
                'dist_to_nearest_bearish_fvg': np.nan,
                'nearest_bullish_fvg_size': np.nan,
                'nearest_bearish_fvg_size': np.nan,
                'nearest_bullish_fvg_age': np.nan,
                'nearest_bearish_fvg_age': np.nan,
                'has_unmitigated_bullish_fvg': 0,
                'has_unmitigated_bearish_fvg': 0
            }
        
        # Get previous candle values
        current_low = row['low']
        current_high = row['high']
        current_close = row['close']
        prev_low = df.iloc[idx-1]['low']
        prev_high = df.iloc[idx-1]['high']
        prev2_low = df.iloc[idx-2]['low']
        prev2_high = df.iloc[idx-2]['high']
        
        # Get ATR for scaling
        atr = row.get('ATR_14', 1.0)  # Default to 1.0 if ATR not available
        
        # Check for new bullish FVG
        bullish_fvg_formed = current_low > prev2_high
        if bullish_fvg_formed:
            fvg_size = current_low - prev2_high
            # Only consider significant FVGs (larger than ATR * multiplier)
            if fvg_size > atr * self.atr_mult:
                self.bullish_fvgs.append([idx, current_low, prev2_high, False])
                candle_features['bullish_fvg_formed'] = 1
            else:
                candle_features['bullish_fvg_formed'] = 0
        else:
            candle_features['bullish_fvg_formed'] = 0
        
        # Check for new bearish FVG
        bearish_fvg_formed = current_high < prev2_low
        if bearish_fvg_formed:
            fvg_size = prev2_low - current_high
            # Only consider significant FVGs
            if fvg_size > atr * self.atr_mult:
                self.bearish_fvgs.append([idx, prev2_low, current_high, False])
                candle_features['bearish_fvg_formed'] = 1
            else:
                candle_features['bearish_fvg_formed'] = 0
        else:
            candle_features['bearish_fvg_formed'] = 0
        
        # Update mitigation status of existing FVGs
        self._update_mitigation_status(current_low, current_high)
        
        # Calculate distance to nearest unmitigated FVG
        nearest_bull_dist, nearest_bull_size, nearest_bull_age, has_bull = self._get_nearest_bullish_fvg(idx, current_close)
        nearest_bear_dist, nearest_bear_size, nearest_bear_age, has_bear = self._get_nearest_bearish_fvg(idx, current_close)
        
        candle_features['dist_to_nearest_bullish_fvg'] = nearest_bull_dist
        candle_features['dist_to_nearest_bearish_fvg'] = nearest_bear_dist
        candle_features['nearest_bullish_fvg_size'] = nearest_bull_size
        candle_features['nearest_bearish_fvg_size'] = nearest_bear_size
        candle_features['nearest_bullish_fvg_age'] = nearest_bull_age
        candle_features['nearest_bearish_fvg_age'] = nearest_bear_age
        candle_features['has_unmitigated_bullish_fvg'] = 1 if has_bull else 0
        candle_features['has_unmitigated_bearish_fvg'] = 1 if has_bear else 0
        
        return candle_features
    
    def _update_mitigation_status(self, current_low, current_high):
        """
        Update the mitigation status of tracked FVGs.
        
        Args:
            current_low: Low of current candle
            current_high: High of current candle
        """
        # Update bullish FVGs
        for i, (idx, top, bottom, mitigated) in enumerate(self.bullish_fvgs):
            if not mitigated and current_low <= bottom:
                self.bullish_fvgs[i][3] = True  # Mark as mitigated
        
        # Update bearish FVGs
        for i, (idx, top, bottom, mitigated) in enumerate(self.bearish_fvgs):
            if not mitigated and current_high >= top:
                self.bearish_fvgs[i][3] = True  # Mark as mitigated
    
    def _get_nearest_bullish_fvg(self, current_idx, current_close):
        """
        Find the nearest unmitigated bullish FVG.
        
        Args:
            current_idx: Index of current candle
            current_close: Close price of current candle
            
        Returns:
            tuple: (distance, size, age, has_unmitigated)
        """
        min_dist = float('inf')
        nearest_size = np.nan
        nearest_age = np.nan
        has_unmitigated = False
        
        for idx, top, bottom, mitigated in self.bullish_fvgs:
            if not mitigated:
                has_unmitigated = True
                # Current price is below the FVG
                if current_close < bottom:
                    dist = bottom - current_close
                    age = current_idx - idx
                    size = top - bottom
                    if dist < min_dist:
                        min_dist = dist
                        nearest_size = size
                        nearest_age = age
        
        return (min_dist if has_unmitigated else np.nan, nearest_size, nearest_age, has_unmitigated)
    
    def _get_nearest_bearish_fvg(self, current_idx, current_close):
        """
        Find the nearest unmitigated bearish FVG.
        
        Args:
            current_idx: Index of current candle
            current_close: Close price of current candle
            
        Returns:
            tuple: (distance, size, age, has_unmitigated)
        """
        min_dist = float('inf')
        nearest_size = np.nan
        nearest_age = np.nan
        has_unmitigated = False
        
        for idx, top, bottom, mitigated in self.bearish_fvgs:
            if not mitigated:
                has_unmitigated = True
                # Current price is above the FVG
                if current_close > top:
                    dist = current_close - top
                    age = current_idx - idx
                    size = top - bottom
                    if dist < min_dist:
                        min_dist = dist
                        nearest_size = size
                        nearest_age = age
        
        return (min_dist if has_unmitigated else np.nan, nearest_size, nearest_age, has_unmitigated)

def add_advanced_fvg_features(df, config):
    """
    Add advanced Fair Value Gap features with tracking of mitigation status.
    
    Args:
        df: DataFrame with OHLC data
        config: Configuration dictionary
    
    Returns:
        DataFrame with added FVG features
    """
    df_fvg = df.copy()
    
    # Initialize FVG tracker
    atr_multiplier = config.get('FVG_ATR_MULTIPLIER', 0.5)
    max_fvgs = config.get('MAX_TRACKED_FVGS', 50)
    
    tracker = FVGTracker(max_fvgs=max_fvgs, atr_multiplier_significance=atr_multiplier)
    
    # Process each candle to identify FVGs and track mitigation
    fvg_features = []
    for i, (idx, row) in enumerate(df_fvg.iterrows()):
        candle_features = tracker.process_candle(i, row, df_fvg)
        fvg_features.append(candle_features)
    
    # Convert list of dictionaries to DataFrame and join with original
    fvg_df = pd.DataFrame(fvg_features, index=df_fvg.index)
    df_fvg = pd.concat([df_fvg, fvg_df], axis=1)
    
    return df_fvg

def add_premium_discount_features(df, config):
    """
    Add features related to premium/discount levels from moving averages.
    Useful for identifying when price is at extremes relative to recent price action.
    
    Args:
        df: DataFrame with OHLC data
        config: Configuration dictionary
    
    Returns:
        DataFrame with added premium/discount features
    """
    df_pd = df.copy()
    
    # Moving average periods
    ma_periods = config.get('PREMIUM_DISCOUNT_MA_PERIODS', [20, 50, 200])
    
    # Calculate moving averages
    for period in ma_periods:
        df_pd[f'SMA_{period}'] = talib.SMA(df_pd['close'], timeperiod=period)
        df_pd[f'EMA_{period}'] = talib.EMA(df_pd['close'], timeperiod=period)
        
        # Distance from MA in ATR units (normalized)
        df_pd[f'dist_to_SMA{period}_atr'] = (df_pd['close'] - df_pd[f'SMA_{period}']) / df_pd['ATR_14']
        df_pd[f'dist_to_EMA{period}_atr'] = (df_pd['close'] - df_pd[f'EMA_{period}']) / df_pd['ATR_14']
        
        # Instead of categorical labels, use numerical values for model compatibility
        df_pd[f'premium_level_SMA{period}'] = pd.cut(
            df_pd[f'dist_to_SMA{period}_atr'],
            bins=[-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf],
            labels=[1, 2, 3, 4, 5, 6, 7]  # Numerical labels instead of strings
        ).astype(float)
    
    # Calculate standard deviations from a moving average (volatility bands)
    for period in [20, 50]:
        # Calculate standard deviation of close prices
        df_pd[f'std_dev_{period}'] = df_pd['close'].rolling(window=period).std()
        
        # Create upper and lower bands at 1 and 2 standard deviations
        for n_std in [1, 2]:
            df_pd[f'upper_{n_std}std_{period}'] = df_pd[f'SMA_{period}'] + n_std * df_pd[f'std_dev_{period}']
            df_pd[f'lower_{n_std}std_{period}'] = df_pd[f'SMA_{period}'] - n_std * df_pd[f'std_dev_{period}']
            
            # Boolean features for price near bands
            df_pd[f'near_upper_{n_std}std_{period}'] = (
                df_pd['close'] > df_pd[f'upper_{n_std}std_{period}'] - 0.2 * df_pd['ATR_14']
            ) & (
                df_pd['close'] < df_pd[f'upper_{n_std}std_{period}'] + 0.2 * df_pd['ATR_14']
            )
            
            df_pd[f'near_lower_{n_std}std_{period}'] = (
                df_pd['close'] > df_pd[f'lower_{n_std}std_{period}'] - 0.2 * df_pd['ATR_14']
            ) & (
                df_pd['close'] < df_pd[f'lower_{n_std}std_{period}'] + 0.2 * df_pd['ATR_14']
            )
    
    # Use Bollinger Bands for additional context
    for period in [20, 50]:
        upper, middle, lower = talib.BBANDS(
            df_pd['close'], 
            timeperiod=period, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        df_pd[f'bb_upper_{period}'] = upper
        df_pd[f'bb_middle_{period}'] = middle
        df_pd[f'bb_lower_{period}'] = lower
        df_pd[f'bb_width_{period}'] = (upper - lower) / middle
        
        # Position within BB (0 = at lower band, 1 = at upper band)
        df_pd[f'bb_position_{period}'] = (df_pd['close'] - lower) / (upper - lower)
    
    return df_pd

def identify_breaker_blocks(df, config):
    """
    Identify potential breaker blocks.
    A breaker block is a former support/resistance area that has been broken
    and could act as resistance/support in the future.
    
    Args:
        df: DataFrame with OHLC data including swing points and BoS indicators
        config: Configuration dictionary
    
    Returns:
        DataFrame with added breaker block features
    """
    df_breakers = df.copy()
    
    # Check if required columns exist
    required_cols = ['bullish_bos', 'bearish_bos', 'last_swing_high', 'last_swing_low']
    if not all(col in df_breakers.columns for col in required_cols):
        logger.warning("Required columns for breaker block identification missing. Skipping.")
        return df_breakers
    
    # Initialize breaker features
    df_breakers['potential_bullish_breaker'] = 0
    df_breakers['potential_bearish_breaker'] = 0
    
    # Look for bearish BoS (breaking below support) - creates potential bullish breaker
    bearish_bos_indices = df_breakers[df_breakers['bearish_bos'] == 1].index
    
    for idx in bearish_bos_indices:
        if idx in df_breakers.index:
            # Get position in DataFrame
            pos = df_breakers.index.get_loc(idx)
            if pos > 0:
                # Look back up to 10 candles for the last swing low that was broken
                for i in range(1, min(11, pos + 1)):
                    prev_idx = df_breakers.index[pos - i]
                    # Mark the candle before the swing low as potential bullish breaker
                    if pos - i - 1 >= 0:
                        breaker_idx = df_breakers.index[pos - i - 1]
                        df_breakers.at[breaker_idx, 'potential_bullish_breaker'] = 1
                        break
    
    # Look for bullish BoS (breaking above resistance) - creates potential bearish breaker
    bullish_bos_indices = df_breakers[df_breakers['bullish_bos'] == 1].index
    
    for idx in bullish_bos_indices:
        if idx in df_breakers.index:
            # Get position in DataFrame
            pos = df_breakers.index.get_loc(idx)
            if pos > 0:
                # Look back up to 10 candles for the last swing high that was broken
                for i in range(1, min(11, pos + 1)):
                    prev_idx = df_breakers.index[pos - i]
                    # Mark the candle before the swing high as potential bearish breaker
                    if pos - i - 1 >= 0:
                        breaker_idx = df_breakers.index[pos - i - 1]
                        df_breakers.at[breaker_idx, 'potential_bearish_breaker'] = 1
                        break
    
    return df_breakers

def add_advanced_ict_features(df, config):
    """
    Main function to add all advanced ICT features to the DataFrame.
    
    Args:
        df: DataFrame with OHLC data and basic features
        config: Configuration dictionary
    
    Returns:
        DataFrame with added advanced ICT features
    """
    # Check if ATR is available - needed for many features
    if 'ATR_14' not in df.columns:
        logger.warning("ATR_14 not found in DataFrame. Some features may not be calculated correctly.")
    
    # Apply each feature set
    logger.info("Adding session features...")
    df = add_session_features(df, config)
    
    logger.info("Adding external liquidity features...")
    df = add_external_liquidity_features(df, config)
    
    logger.info("Adding advanced FVG features...")
    df = add_advanced_fvg_features(df, config)
    
    logger.info("Adding premium/discount features...")
    df = add_premium_discount_features(df, config)
    
    logger.info("Identifying potential breaker blocks...")
    df = identify_breaker_blocks(df, config)
    
    # Fill NaN values in boolean/integer columns with 0
    bool_cols = [col for col in df.columns if col.startswith('swept_') or 
                 col.startswith('has_') or col.startswith('near_') or
                 col.startswith('potential_')]
    df[bool_cols] = df[bool_cols].fillna(0)
    
    # Fill NaN values in distance columns with a large value
    dist_cols = [col for col in df.columns if col.startswith('dist_to_')]
    df[dist_cols] = df[dist_cols].fillna(99.0)  # Large distance value
    
    logger.info(f"Added {len(df.columns) - len(df.columns.intersection(df.columns))} advanced ICT features")
    
    return df