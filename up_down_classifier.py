import pandas as pd
import numpy as np
from sklearn import tree
from data_transformation import shift_transform
import graphviz 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Keagan/anaconda3/envs/trading/Library/bin/graphviz'
pd.options.display.width = 80
#pd.set_option('display.max_rows', None)


def data_preprocessing(df_data):
    '''Transform and encode data for up/down classification'''
    df_data = df_data.astype(float)
    df_data['up_down'] = np.where((df_data['4. close'] - df_data['1. open']) > 0, 1, 0)
    return df_data

def feature_engineering(df_data, lookback_period):
    df_transformed = shift_transform(df_data, {"up_down":"up_down"}, lookback_period)
    df_transformed = df_transformed.astype(int)
    return df_transformed

def dtree_classifier(df_transformed, max_depth, lookback_period, export: bool):  
    #features column names
    features = df_transformed.columns[1:]

    #train model
    y = df_transformed['up_down']
    X = df_transformed[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1)
    clf = tree.DecisionTreeClassifier(max_depth= max_depth)
    clf = clf.fit(X_train, y_train)

    #export graph as pdf, note class names are in asc numerical order by default
    if export:
        graph_source = tree.export_graphviz(clf, filled=True, out_file=None, feature_names=features, class_names=["down","up"])
        graphviz.Source(graph_source).render("dtree_updown_seq_" + symbol)

    #accuracy
    y_pred = clf.predict(X_test)
    #print(symbol, 'Accuracy:', metrics.accuracy_score(y_test, y_pred))
    return metrics.accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    
    with pd.HDFStore("hdf/random_30_NIpos.h5", mode="r") as h:
        symbols = h.keys()

    print('Symbols in HDF file: ', symbols)

    for symbol in symbols:

        df_data = pd.read_hdf("hdf/random_30_NIpos.h5", symbol)
        df_data = data_preprocessing(df_data)
        #print(df_data)

        df_transformed = feature_engineering(df_data, 3)
        #print(df_transformed)

        print(symbol, 'Decision tree: ', dtree_classifier(df_transformed, 3,3, False))







    


    