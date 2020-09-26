import pandas
import numpy as np
from api_test import get_historical_data

df = get_historical_data("SFM")
df = df.astype(float)

#add up/down column, 1 is week up, 0 is week down
df['up_down'] = np.where(df['5. adjusted close'] - df['1. open'] > 0, 1, 0)


print(df)
