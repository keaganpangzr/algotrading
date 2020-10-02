import pandas as pd
import numpy as np
from sklearn import tree
from api_test import get_historical_data
from data_transformation import shift_transform
from data_transformation import alphavantage_cols
import matplotlib.pyplot as plt
import graphviz 

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Keagan/anaconda3/envs/trading/Library/bin/graphviz'

pd.options.display.width = 20

df_data = get_historical_data("IBM").astype(float)
df_data['up_down'] = np.where(df_data['5. adjusted close'] - df_data['1. open'] > 0, 1, 0)

df_transformed = shift_transform(df_data, alphavantage_cols, 3)
print(df_transformed)

y = df_transformed['up_down']
features = df_transformed.columns[1:]

x = df_transformed[features]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(x,y)

#class names in asc numerical order
graph_source = tree.export_graphviz(classifier, out_file=None, feature_names=x.columns, class_names=["down","up"])

#export as pdf
graphviz.Source(graph_source).render("tree")












