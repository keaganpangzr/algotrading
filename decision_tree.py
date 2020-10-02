import pandas as pd
import numpy as np
from sklearn import tree
from api_test import get_historical_data
from data_transformation import shift_transform
from data_transformation import alphavantage_cols
import graphviz 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Keagan/anaconda3/envs/trading/Library/bin/graphviz'
pd.options.display.width = 20

#get historical data using alphavantage API
df_data = get_historical_data("SFM").astype(float)

#label encoded target variable
df_data['up_down'] = np.where(df_data['5. adjusted close'] - df_data['1. open'] > 0, 1, 0)

#shift transform time series
df_transformed = shift_transform(df_data, alphavantage_cols, 3)
print(df_transformed)


#features column names
features = df_transformed.columns[1:]

#train model
y = df_transformed['up_down']
X = df_transformed[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1)
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, y_train)

#test accuracy
y_pred = clf.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

#export graph as pdf, note class names are in asc numerical order by default
graph_source = tree.export_graphviz(clf, filled=True, out_file=None, feature_names=features, class_names=["down","up"])
graphviz.Source(graph_source).render("tree")


