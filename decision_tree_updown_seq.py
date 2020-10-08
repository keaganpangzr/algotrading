import pandas as pd
import numpy as np
from sklearn import tree
from api_test import get_historical_data
from data_transformation import shift_transform
import graphviz 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Keagan/anaconda3/envs/trading/Library/bin/graphviz'
pd.options.display.width = 60
#pd.set_option('display.max_rows', None)

symbols = ["ACI", "ACIW", "ACM", "ADPT", "ADSW"]


def dtree_updown_seq(symbol, max_depth, lookback_period, export: bool):

    df_data = get_historical_data(symbol)
    df_data = df_data.astype(float)

    #encode - add up/down column, 1 is week up, 0 is week down
    df_data['up_down'] = np.where((df_data['4. close'] - df_data['1. open']) > 0, 1, 0)
    df_transformed = shift_transform(df_data, {"up_down":"up_down"}, lookback_period)
    df_transformed = df_transformed.astype(int)

    #print(df_data)
    #print(df_transformed)

    #features column names
    features = df_transformed.columns[1:]

    #train model
    y = df_transformed['up_down']
    X = df_transformed[features]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1)
    clf = tree.DecisionTreeClassifier(max_depth= max_depth)
    clf = clf.fit(X_train, y_train)

    #test accuracy
    y_pred = clf.predict(X_test)
    print(symbol, 'Accuracy:', metrics.accuracy_score(y_test, y_pred))

    if export:
        #export graph as pdf, note class names are in asc numerical order by default
        graph_source = tree.export_graphviz(clf, filled=True, out_file=None, feature_names=features, class_names=["down","up"])
        graphviz.Source(graph_source).render("dtree_updown_seq_" + symbol)


for symbol in symbols:
    dtree_updown_seq(symbol, 3, 5, True)


    


