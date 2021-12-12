from miniskLearn.metrics import r2score
import numpy as np
from miniskLearn.KNN.Regressor import KNNRegressor
from miniskLearn.splitting import TrainTestSplit

X = np.load("./Data/KNN/Regression/X_data.npy")
y = np.load("./Data/KNN/Regression/Y_data.npy")

X_train, X_test, y_train, y_test = TrainTestSplit(X,y,0.8)

point = np.array([[5500]])

knnr = KNNRegressor(5)
knnr.fit(X_train, y_train)

prediction = knnr.predict(point)
print("Random Point Prediction =>",prediction)

predictions = knnr.predict(X_test)
print(r2score(predictions, y_test))

from sklearn.metrics import r2_score
print(r2_score(predictions, y_test))
