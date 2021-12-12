import numpy as np
from miniskLearn.KNN.Regressor import KNNRegressor
from miniskLearn.splitting import TrainTestSplit
from miniskLearn.metrics import r2score

#Importing Data
X = np.load("./Data/KNN/Regression/X_data.npy")
y = np.load("./Data/KNN/Regression/Y_data.npy")


#Splitting Data
X_train, X_test, y_train, y_test = TrainTestSplit(X,y,0.8)

point = np.array([[5500]])


print("*"*50)
print("Mini-sklearn Class")
print("*"*50)

############################################
# Fitting KNN Regressor instance
knnr = KNNRegressor(5)
knnr.fit(X_train, y_train)
prediction = knnr.predict(point)
print("Random Point Prediction =>",prediction)

predictions = knnr.predict(X_test)
print("Accuracy =>",r2score(y_test, predictions))



############################################
print()
print("*"*50)
print("sklearn Class")
print("*"*50)
############################################

# Comparing with sk-learn KNN Regressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

KNNR = KNeighborsRegressor(5)
KNNR.fit(X_train, y_train)
prediction = KNNR.predict(point)
print("Random Point Prediction =>",prediction)

predictions = KNNR.predict(X_test)
print("Accuracy =>",r2_score(y_test, predictions))
