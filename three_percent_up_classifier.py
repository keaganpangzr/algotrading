import pandas as pd
import numpy as np
from sklearn import tree
from data_transformation import shift_transform
import graphviz 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/Keagan/anaconda3/envs/trading/Library/bin/graphviz'
pd.options.display.width = 500
#pd.set_option('display.max_rows', None)


def data_preprocessing(df_data):
    '''Transform and encode data for up/down classification'''
    df_data = df_data.astype(float)
    df_data['up_down'] = np.where((df_data['4. close'] - df_data['1. open'])/df_data['1. open'] > 0.03, 1, 0)
    
    
    #AV
    df_data['HLC_sum'] = df_data[['2. high', '3. low','4. close']].sum(axis=1)
    df_data['AV'] = df_data['HLC_sum'][::-1].shift(1).rolling(3).mean()/3

    #close above/below AV
    df_data['close_above_AV'] = np.where((df_data['4. close'] - df_data['AV']) > 0, 1, 0)


    return df_data


def feature_engineering(df_data, lookback_period):
    df_transformed = df_data['up_down'].to_frame()
    

    df_transformed = shift_transform(df_data, {"close_above_AV":"close_above_AV", "up_down":"up_down"}, lookback_period)
    df_transformed = df_transformed.astype(int)

    return df_transformed

def TPR(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('TPR: ', round(tp / (tp + fp), 3), "TP + FP :", tp + fp)



    







def dtree_classifier(df_transformed, max_depth, export: bool):  
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
    
    return classification_report(y_test, y_pred)
    
    #return round(metrics.accuracy_score(y_test, y_pred), 4)

def rtree_classifier(df_transformed, n_trees):  
    #features column names
    features = df_transformed.columns[1:]

    #train model
    y = df_transformed['up_down']
    X = df_transformed[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1)
    rclf = RandomForestClassifier(n_estimators=n_trees)
    rclf = rclf.fit(X_train, y_train)

    #accuracy
    y_pred = rclf.predict(X_test)
    #print(symbol, 'Accuracy:', metrics.accuracy_score(y_test, y_pred))
    #return round(metrics.accuracy_score(y_test, y_pred), 4)

    #return classification_report(y_test, y_pred)
    return TPR(y_test, y_pred)
    


def logreg_classifier(df_transformed):
    #features column names
    features = df_transformed.columns[1:]

    #train model
    y = df_transformed['up_down']
    X = df_transformed[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1)
    logreg = LogisticRegression()
    logreg = logreg.fit(X_train,y_train)
    
    #accuracy
    y_pred = logreg.predict(X_test)
    #print(symbol, 'Accuracy:', metrics.accuracy_score(y_test, y_pred))
    #return round(metrics.accuracy_score(y_test, y_pred), 4)
    return classification_report(y_test, y_pred)

def nn_classifier(df_transformed, hidden_layer_sizes:tuple):
    #features column names
    features = df_transformed.columns[1:]

    #train model
    y = df_transformed['up_down']
    X = df_transformed[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1)
    nnclf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes= hidden_layer_sizes, random_state=1, max_iter= 1500)
    nnclf = nnclf.fit(X_train, y_train)

    y_pred = nnclf.predict(X_test)
    #return round(metrics.accuracy_score(y_test, y_pred), 4)
    return classification_report(y_test, y_pred)


if __name__ == '__main__':

    #show list of all stocks
    with pd.HDFStore("hdf/random_30_NIpos.h5", mode="r") as h:
        symbols = h.keys()
    print('Symbols in HDF file: ', symbols)
        
    #debug
    df_data = pd.read_hdf("hdf/random_30_NIpos.h5", 'HBI')
    df_data = data_preprocessing(df_data)
    print(df_data)

    df_transformed = feature_engineering(df_data, 3)
    print(df_transformed)  

    #TP FP test
    print(rtree_classifier(df_transformed, 128))
    


'''

    #print(rtree_classifier(df_transformed, 128))
    print(nn_classifier(df_transformed, (150)))
    
    for symbol in symbols:
        df_data = pd.read_hdf("hdf/random_30_NIpos.h5", symbol)
        df_data = data_preprocessing(df_data)
        df_transformed = feature_engineering(df_data, 3)
        

        print(symbol, 'up incidence', df_transformed['up_down'].sum() / len(df_transformed["up_down"]),
                'Decision tree: ', dtree_classifier(df_transformed, 3, False),
                'Random forest: ', rtree_classifier(df_transformed, 128),
                'Logreg: ', logreg_classifier(df_transformed),
                'ANN: ', nn_classifier(df_transformed, (150)))
    
    
'''










    


    