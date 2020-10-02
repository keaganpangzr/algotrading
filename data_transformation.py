import pandas as pd
import numpy as np
from api_test import get_historical_data
pd.options.display.width = 20

df_data = get_historical_data("SFM")
df_data = df_data.astype(float)

#add up/down column, 1 is week up, 0 is week down
df_data['up_down'] = np.where(df_data['5. adjusted close'] - df_data['1. open'] > 0, 1, 0)


#add target variable
def shift_transform(df_data, cols:list, lookback_period: int):
    """Transform time series to lookback matrix ie.show OHLCV of last x periods in same row"""
    df_transformed = df_data['up_down'].to_frame()

    for i in range(lookback_period):
        for col in cols.keys():
            df_shift = df_data[col].shift(-(i+1)).rename(f"t-{i+1}_{cols[col]}")
            df_transformed = df_transformed.join(df_shift)

    #drop last rows with NaN values due to lookback_period
    df_transformed.drop(df_transformed.tail(lookback_period).index,inplace=True)

    #check NaNs
    if df_transformed.isnull().values.any():
        print("NaN found in dataframe")
    else:
        return df_transformed

#columns in historical data to transform 
alphavantage_cols = {"1. open" : "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adjusted_close",
        "6. volume": "volume"}

lookback_period = 9

if __name__ == '__main__':
    print(df_data)
    print(shift_transform(df_data, alphavantage_cols, lookback_period))





