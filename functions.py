import numpy as np
from collections import namedtuple

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

def get_performance(y_true, y_pred):
    """
    Return named tuple which has accuracy, precision, recall, f1-score, sensitivity, specificity
    """
    Metric = namedtuple('Metric', 'accuracy precision recall fscore sensitivity specificity')
    a = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    se, sp = sens_spec_support(y_true, y_pred)
    return Metric(accuracy=a, precision=p, recall=r, fscore=f, sensitivity=se, specificity=sp)

def sens_spec_support(y_true, y_pred):      
    """
    Return sensitivity and specificity
    WARNING: This function supports only binary classification!
    """
    cm = confusion_matrix(y_true, y_pred)
    spec = float(cm[0][0])/np.sum(cm[0]) if np.sum(cm[0]) != 0 else 0
    sens = float(cm[1][1])/np.sum(cm[1]) if np.sum(cm[1]) != 0 else 0
    return sens, spec

def oned_to_twod(arr, nrows, ncols):        
    """
    Transform 2d array into 4d tensor for CNN input
    arr          => array of raw data
    nrows, ncols => size of 2d image
    """
    arr_2d = np.zeros((len(arr), nrows, ncols, 1))
        
    for i in range(len(arr)):
        signal = arr[i,:].reshape((nrows,ncols))
        image = signal
        arr_2d[i, :, :, 0] = image

    return arr_2d
