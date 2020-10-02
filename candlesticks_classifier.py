import plotly.graph_objects as go

def plot_candlestick():
    fig = go.Figure(data=[go.Candlestick(x=[123],
                open=[2], high=[5], low=[1], close=[4])])
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()


def candlestick_classifier(o, h, l, c):
    total_height = h - l
    body_height = abs(o - c)
    polarity = 'up' if c >= o else 'down'

    if body_height > 0.95 * total_height:
        return 'marubozu_' + polarity

print(candlestick_classifier(5,10,4.99,9.95))



    




