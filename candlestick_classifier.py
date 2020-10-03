import plotly.graph_objects as go


"""
1. make plot function
2. make classifier function
3. test on real data

"""

o = 102.6
h = 110
l = 100
c = 102.7

def plot_candlestick():
    fig = go.Figure(data=[go.Candlestick(x=[1,2], open=[100, o], high=[105, h], low=[99, l], close=[104, c])])
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()


def candlestick_classifier(o, h, l, c):
    total_height = h - l
    body_height = abs(o - c)
    polarity = 'up' if c >= o else 'down'

    #distance of oc avg from low as percentage of total_height
    oc_height_ratio = ((o + c)/2 - l) / total_height


    
    #print("body_height/total_height", body_height / total_height)
    #print("oc_height_ratio", oc_height_ratio)
    

    if body_height > 0.90 * total_height:
        return 'marubozu_' + polarity


    elif body_height < 0.05 * total_height:
        if oc_height_ratio < 0.06:
            return "gravestone_doji"
        elif oc_height_ratio > 0.94:
            return "dragonfly_doji"
        elif 0.25 < oc_height_ratio < 0.75:
            return "neutral_doji"

    else:
        return None

if __name__ == "__main__":

    plot_candlestick()
    print(candlestick_classifier(o,h,l,c))



    




