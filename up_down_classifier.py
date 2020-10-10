import pandas as pd

#show list of stocks in hdf file
with pd.HDFStore("hdf/random_30_NIpos.h5", mode="r") as h:
    symbols = h.keys()

print('Symbols in HDF file: ', symbols)

'''
#get named stock in file using symbol as key
for symbol in symbols:
    df = pd.read_hdf("hdf/random_30_NIpos.h5", symbol)
    print(symbol, df)
'''

'''
1. Get historical data
2. Feature engineering, transformation
3. Split and label data
4. Train and pred models
'''



