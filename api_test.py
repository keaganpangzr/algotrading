import requests
import json
import pprintpp
import pandas as pd
pd.options.display.width = 0

#API key: JKOM50JRCWEF9ST1
def get_historical_data(symbol):
    parameters = {"apikey":"JKOM50JRCWEF9ST1", 
            "function":"TIME_SERIES_WEEKLY_ADJUSTED", 
            "symbol":symbol, 
            "outputsize":"compact"}

    request = requests.get('https://www.alphavantage.co/query', parameters)
    request_dict = request.json()

    df = pd.DataFrame.from_dict(request_dict["Weekly Adjusted Time Series"]).transpose()
    df = df.sort_index(ascending=False)

    return df

if __name__ == "__main__":
    print(get_historical_data("SFM"))

