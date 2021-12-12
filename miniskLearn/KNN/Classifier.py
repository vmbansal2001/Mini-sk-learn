import numpy as np

class KNNClassifier:
    def __init__(self, n_neighbours):
        self.n_neighbours = n_neighbours
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        return self
    
    def predict(self, points):
        self.y_pred = []
        for point in points:
            distances = np.linalg.norm(self.X - point, ord=2, axis=1)
            top_indexes = np.argsort(distances)[:self.n_neighbours]
            unique, count = np.unique(self.y[top_indexes], return_counts=True)
            self.y_pred.append(unique[np.argmax(count)])
        return np.array(self.y_pred)
