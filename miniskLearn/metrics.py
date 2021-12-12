import numpy as np

def accuracyScore(y_true, y_pred):
    return np.mean(y_true == y_pred)

def r2score(y_true, y_pred):
    return 1 - (((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum())

