import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

from functions import oned_to_twod
from functions import get_performance

# ---- value for my case ----
input_shape = (6, 125, 1)
fs, win = 50, 2.5
dropout_rate = 0.1
filter_width = int(fs*win)
# ---- value for my case ----

def create_cnn_model(num_of_filters):
    """
    Return CNN model for FOG recognition
    """
    model = Sequential()

    # First layer: CONV - POOL
    model.add(Conv2D(input_shape, num_of_filters, (1, filter_width), padding='same', kernel_constraint=maxnorm(3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Dropout(dropout_rate))

    # Second layer: CONV - POOL
    model.add(Conv2D(num_of_filters, (6, int(filter_width/2)), padding='same', kernel_constraint=maxnorm(3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def cv_kfold(X, y, k):
    """
    Perform k-fold CV
    X, y       => data
    k          => number of cross validation (e.g. 10)
    """
    model = create_cnn_model(20)
    initial_weight = model.get_weights()
    skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    
    train_metric_list, test_metric_list = [list() for _ in range(2)]
    
    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # ------------------ k-fold CV start -------------------
    for i, (train, val) in enumerate(skf.split(X, y)):
        print("==> Fold #%d" % i)
        
        D = 6
        L = int(fs*win)

        EPOCH = 200
        BATCH = 10

        X_train = oned_to_twod(X_[train], D, L)
        y_train = np_utils.to_categorical(y_[train]) 
        
        X_val  = oned_to_twod(X_[val], D, L)
        y_val  = y[val]
        
        model.set_weights(initial_weight)
        hist = model.fit(X_train, y_train, epoch=EPOCH, batch_size=BATCH, verbose=0)
        
        y_pred = model.predict_classes(X_train, verbose=0)
        train_metric = get_performance(y_train, y_pred)
        
        y_pred = model.predict_classes(X_val, verbose=0)
        test_metric  = get_performance(y_val, y_pred)
        
        train_metric_list.append(train_metric)
        test_metric_list.append(test_metric)
    # ------------------ k-fold CV end ---------------------

    y_pred = model.predict(X_test)
    test_metric = get_performance(y_test, y_pred)
        
    df_train = pd.DataFrame(train_metric_list)
    df_val   = pd.DataFrame(test_metric_list)
    
    return df_train.mean(), df_val.mean(), test_metric
