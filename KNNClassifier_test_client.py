import numpy as np
from miniskLearn.KNN.Classifier import KNNClassifier
from miniskLearn.metrics import accuracyScore
from miniskLearn.splitting import TrainTestSplit


#Importing Data
X = np.load("./Data/KNN/Classification/X_data.npy")
y = np.load("./Data/KNN/Classification/Y_data.npy")


#Splitting Data
X_train, X_test, y_train, y_test = TrainTestSplit(X,y,0.8)

# Taking a random point to test prediction
point = np.array([[500,35]])


print("*"*50)
print("Mini-sklearn Class")
print("*"*50)

#Fitting KNN Classifier instance
knnc = KNNClassifier(5)
knnc.fit(X_train,y_train)
prediction = knnc.predict(point)
print("Random Point Prediction =>",prediction)

predictions = knnc.predict(X_test)
print("Accuracy =>",accuracyScore(y_test, predictions))






################################################
print()
print("*"*50)
print("sklearn Class")
print("*"*50)
################################################


# Comparing with sk-learn KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

KNNC = KNeighborsClassifier(n_neighbors=5)
KNNC.fit(X_train,y_train)
prediction = KNNC.predict(point)
print("Random Point Prediction =>",prediction)

predictions = KNNC.predict(X_test)
print("Accuracy =>",accuracy_score(y_test, predictions))

