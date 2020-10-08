import requests
import json
import pprintpp
import pandas as pd
import h5py
import time

pd.options.display.width = 0

#API key: JKOM50JRCWEF9ST1
def get_historical_data(symbol):
    parameters = {"apikey":"JKOM50JRCWEF9ST1", 
            "function":"TIME_SERIES_WEEKLY_ADJUSTED", 
            "symbol":symbol, 
            "outputsize":"compact"}

    request = requests.get('https://www.alphavantage.co/query', parameters)
    request_dict = request.json()   

    df = pd.DataFrame.from_dict(request_dict["Weekly Adjusted Time Series"])
    df = df.transpose().astype(float)
    df = df.sort_index(ascending=False)

    return df

def save_to_hdf(df, symbol: str, file_path_name):
    """Save df to hdf5"""

    #w overwrites with pointer at beginning of file, use "a" for appending new info instead
    with pd.HDFStore(file_path_name, "a") as h:
        df.to_hdf(h, symbol)

if __name__ == "__main__":

    #HDF5 file path and name
    file_path_name = 'hdf/stocks_test.h5'

    #stock data to download
    stocks_list = ["TSLA", "MSFT", "IBM", "SFM", "JJAIJ" "ACI", "AA", "ARKK"]

    for stock in stocks_list:
        try:
            df = get_historical_data(stock)
            save_to_hdf(df, stock, file_path_name)
            print(f'Saved {stock} to {file_path_name}')
        
        except:
            print(f'Error saving {stock}')
        
        #max 5 requests per minute
        time.sleep(15)

            


