from candlestick_classifier import plot_candlestick
from candlestick_classifier import candlestick_classifier
from api_test import get_historical_data
import plotly.graph_objects as go

df = get_historical_data("SFM").astype(float)
print(df)

dates = [i for i in range(100)]
print(dates)


o = list(df["1. open"][0:len(dates)])
h = list(df["2. high"][0:len(dates)])
l = list(df["3. low"][0:len(dates)])
c = list(df["4. close"][0:len(dates)])

def plot_candlestick(o,h,l,c:list,dates):
    fig = go.Figure(data=[go.Candlestick(x=dates, open=o, high=h, low=l, close=c)])
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

plot_candlestick(o,h,l,c, dates)

for date in dates:
    print(date, candlestick_classifier(o[date], h[date], l[date], c[date]))