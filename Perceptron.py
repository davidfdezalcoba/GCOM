from CreateData import create_Dd_data
from random import shuffle
from math import floor
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    '''
    Implementation of the Perceptron algorithm

    '''
    def __init__(self, epochs = 1):
        self.l_rate = 0.01
        self.wg = []
        self.epochs = epochs

    def score(self, X, t):
        X = np.vstack((np.ones(X.shape[1]), X))
        return [self.wg.dot((X.T)[i])*t[i] > 0 for i in range(len(X.T))]

    def train(self, X, t):
        self.wg = np.zeros(X.shape[0] + 1)
        X = np.vstack((np.ones(X.shape[1]), X))
        for i in range(self.epochs):
            for j in range(len(X.T)):
                score = self.wg.dot(X.T[j])*t[j] > 0
                if score == False:
                    self.wg += self.l_rate*X.T[j]*t[j]

if __name__ == '__main__':

    D = 5
    K = 2
    X,tags = create_Dd_data(D, K, 4, 0.2, 1000, 1100)

    # Change tags for -1, 1
    tags = [1 if tags[i] == 0 else -1 for i in range(len(tags))]

    # Shuffle data
    ds = list(zip(X.T,tags))
    shuffle(ds)
    X = np.array([x for (x,_) in ds]).T
    tags = np.array([t for (_,t) in ds])

    cls = Perceptron(1)
    # Train with 75% of the data
    fl = floor(len(X.T)*0.75)
    cls.train(X.T[0:fl].T, tags[0:fl])
    print(cls.wg)

    # Test with 25%
    score = cls.score(X.T[fl:len(X.T)].T, tags[fl:len(tags)])
    print('Score of test dataset: %d/%d' %(score.count(True), len(score)))
    # Test with 100%
    score = cls.score(X, tags)
    print('Score of whole dataset: %d/%d' %(score.count(True), len(score)))
