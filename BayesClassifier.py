from math import log, floor
from random import shuffle
from CreateData import create_Dd_data
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class BayesClassifier:
    '''
    Implementation of the bayessian classifier

    '''
    def __init__(self):
        self.kmeans = []
        self.ksigmas = []
        self.num_datos_clase_k = []

    def train(self, X, tags, K):
        self.kmeans = np.zeros((K,len(X)))
        self.ksigmas = np.zeros((K, len(X), len(X)))
        self.num_datos_clase_k = np.zeros(K)

        for k in range(K):
            pointsink = [X.T[j] for j in range(len(X.T)) if tags[j] == k]
            self.num_datos_clase_k[k] = len(pointsink)
            self.kmeans[k] = np.mean(pointsink, axis = 0)
            for n in range(len(pointsink)):
                self.ksigmas[k] += np.outer(pointsink[n] - self.kmeans[k],\
                        pointsink[n] - self.kmeans[k])
            self.ksigmas[k] = self.ksigmas[k] / self.num_datos_clase_k[k]

    def predict(self, X):
        predictions = []
        for n in range(len(X.T)):
            predictions.append(np.argmin( \
                [(X.T[n] - self.kmeans[k]).dot(np.linalg.inv(self.ksigmas[k])).dot((X.T[n] - self.kmeans[k]).T) + log(np.linalg.det(self.ksigmas[k])) - 2*log(self.num_datos_clase_k[k] / len(X.T)) for k in range(len(self.kmeans))]))
        return predictions

if __name__ == '__main__':

    colors = ['r', 'g', 'b', 'y', 'm', '#0ff0f0', '#112f1f', '#fff000']
    D = 2
    K = 4;
    X,tags = create_Dd_data(D, K, 1, 0.2, 1300, 1400)

    # Shuffle data
    ds = list(zip(X.T,tags))
    shuffle(ds)
    X = np.array([x for (x,_) in ds]).T
    tags = np.array([t for (_,t) in ds])

    cls = BayesClassifier()
    fl = floor(len(X.T)*0.75)
    cls.train(X.T[0:fl].T, tags[0:fl], K)

    # Test with 25%
    predictions = cls.predict(X.T[fl:len(X.T)].T)
    score = [tags[fl+i] == predictions[i] for i in range(len(X.T)-fl)]
    print('Score of test dataset: %d/%d' %(score.count(True), len(score)))
    # Test with 100%
    predictions = cls.predict(X)
    score = [tags[i] == predictions[i] for i in range(len(X.T))]
    print('Score of whole dataset: %d/%d' %(score.count(True), len(score)))
