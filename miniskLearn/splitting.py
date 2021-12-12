import numpy as np

def TrainTestSplit(X, y, train_size):
    train_indexes = np.random.choice(np.arange(0, len(X)), size=int(len(X)*train_size), replace=False)
    test_indexes = np.arange(0, len(X))[[i for i in range(0, len(X)) if i not in train_indexes]]
    X_train, y_train = X[train_indexes], y[train_indexes]
    X_test, y_test = X[test_indexes], y[test_indexes]

    return (X_train, X_test, y_train, y_test)
