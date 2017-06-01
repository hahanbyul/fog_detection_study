import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC as SVM
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neural_network import MLPClassifier as MLP

from functions import get_performance

def get_model(name):
    """
    Return corresponding sklearn classifier
    """
    if name == 'SVM' or name == 'RF' or name == 'DT':
        model = eval(f'{name}(random_state=42)')
    elif name == 'NB':
        model = eval(f'{name}()')
    elif name == 'MLP':
        model = MLP(hidden_layer_sizes=(20, 20, 20), random_state=42)

    return model

def cv_kfold(X, y, k, model_name):
    """
    Perform k-fold CV
    X, y       => data
    k          => number of cross validation (e.g. 10)
    model_name => one of ['RF', 'MLP', 'DT', 'SVM', 'NB',]
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    train_metric_list, val_metric_list = [list() for _ in range(2)]
    
    # ------------------ k-fold CV start -------------------
    for i, (train, val) in enumerate(skf.split(X_train, y_train)):
        print("===> Fold #%d" % i)
        
        X_train_ = X_train[train, :]
        y_train_ = y_train[train]
        
        X_val_   = X_train[val, :]
        y_val_   = y_train[val]
        
        # fit models
        model = get_model(model_name)
        model.fit(X_train_, y_train_)
        
        y_pred = model.predict(X_train_)
        train_metric = get_performance(y_train_, y_pred)
        train_metric_list.append(train_metric)
        
        y_pred = model.predict(X_val_)
        val_metric = get_performance(y_val_, y_pred)
        val_metric_list.append(val_metric)
    # ------------------ k-fold CV end ---------------------
        
    y_pred = model.predict(X_test)
    test_metric = get_performance(y_test, y_pred)
    
    df_train = pd.DataFrame(train_metric_list)
    df_val   = pd.DataFrame(val_metric_list)
    
    return df_train.mean(), df_val.mean(), test_metric
