import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from datetime import datetime, timedelta

def get_swing_points(data_df, order=5):
    """
    Finds swing highs and lows using argrelextrema.
    'order' is the number of points on each side to define a local extremum.
    Returns two series: swing_highs and swing_lows with prices at swing indices, NaN otherwise.
    """
    high_indices = argrelextrema(data_df['high'].values, np.greater_equal, order=order)[0]
    swing_highs = pd.Series(np.nan, index=data_df.index)
    # Ensure indices are valid before assignment
    valid_high_indices = [idx for idx in high_indices if idx < len(data_df)]
    if valid_high_indices:
        swing_highs.iloc[valid_high_indices] = data_df['high'].iloc[valid_high_indices]

    low_indices = argrelextrema(data_df['low'].values, np.less_equal, order=order)[0]
    swing_lows = pd.Series(np.nan, index=data_df.index)
    valid_low_indices = [idx for idx in low_indices if idx < len(data_df)]
    if valid_low_indices:
        swing_lows.iloc[valid_low_indices] = data_df['low'].iloc[valid_low_indices]
    
    return swing_highs, swing_lows

def is_high_impact_news_active(timestamp, news_df, affected_currencies, window_minutes_before=30, window_minutes_after=30):
    """
    Checks if high impact news for relevant currencies is active around the given timestamp.
    `news_df` should be a DataFrame with 'timestamp' (UTC), 'impact' ('High', 'Medium', 'Low'), 'currency'.
    Returns True if news is active, False otherwise.
    """
    if news_df is None or news_df.empty:
        return False
    
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.to_datetime(timestamp)
        
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize('UTC')
    else:
        timestamp = timestamp.tz_convert('UTC')

    start_window = timestamp - timedelta(minutes=window_minutes_before)
    end_window = timestamp + timedelta(minutes=window_minutes_after)
    
    # Ensure news_df['timestamp'] is datetime and UTC
    if not pd.api.types.is_datetime64_any_dtype(news_df['timestamp']):
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], utc=True)
    else:
        if news_df['timestamp'].dt.tz is None:
            news_df['timestamp'] = news_df['timestamp'].dt.tz_localize('UTC')
        else:
            news_df['timestamp'] = news_df['timestamp'].dt.tz_convert('UTC')


    # Filter news for 'High' impact, relevant currencies, and within the window
    active_news = news_df[
        (news_df['impact'].isin(['High', 'Medium'])) & # Consider Medium impact too
        (news_df['currency'].isin(affected_currencies)) &
        (news_df['timestamp'] >= start_window) &
        (news_df['timestamp'] <= end_window)
    ]
    return not active_news.empty

if __name__ == '__main__':
    # Example usage for is_high_impact_news_active
    dummy_news_data = [
        {'timestamp': "2023-01-01 12:00:00", 'currency': 'USD', 'impact': 'High', 'event_name': 'NFP'},
        {'timestamp': "2023-01-01 14:00:00", 'currency': 'EUR', 'impact': 'Medium', 'event_name': 'CPI'},
        {'timestamp': "2023-01-02 08:00:00", 'currency': 'GBP', 'impact': 'Low', 'event_name': 'PMI'},
    ]
    news_calendar_df = pd.DataFrame(dummy_news_data)
    news_calendar_df['timestamp'] = pd.to_datetime(news_calendar_df['timestamp'], utc=True)

    test_time_1 = pd.to_datetime("2023-01-01 11:45:00", utc=True) # Near NFP
    test_time_2 = pd.to_datetime("2023-01-01 13:00:00", utc=True) # After NFP, before CPI window
    test_time_3 = pd.to_datetime("2023-01-01 14:15:00", utc=True) # Near CPI
    test_time_4 = pd.to_datetime("2023-01-02 08:10:00", utc=True) # Near Low impact news

    print(f"News active at {test_time_1}? {is_high_impact_news_active(test_time_1, news_calendar_df, ['USD', 'EUR'])}")
    print(f"News active at {test_time_2}? {is_high_impact_news_active(test_time_2, news_calendar_df, ['USD', 'EUR'])}")
    print(f"News active at {test_time_3}? {is_high_impact_news_active(test_time_3, news_calendar_df, ['USD', 'EUR'])}")
    print(f"News active at {test_time_4}? {is_high_impact_news_active(test_time_4, news_calendar_df, ['USD', 'EUR'])}")

