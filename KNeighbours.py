from math import log, floor
from random import shuffle
from CreateData import create_Dd_data
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class KNeighbours:
    '''
    Implementation of the K-Neighbours classifier

    '''
    def __init__(self):
        return

    def getNeighbours(self, X, x, tags, K):
        distances = [(X.T[j], tags[j], np.linalg.norm(X.T[j] - x)) for j in range(len(X.T))]
        sortedDist = sorted(distances, key=lambda x : x[2])
        sortedNeigs = [x[0] for x in sortedDist]
        sortedTags = [x[1] for x in sortedDist]
        return sortedNeigs[:K], sortedTags[:K]

    def predict(self, Xtrain, Xtest, tags, K, C):
        predictions = []

        for n in range(len(Xtest.T)):
            _, neigtags = self.getNeighbours(Xtrain, Xtest.T[n], tags, K)
            counts = [neigtags.count(c) for c in range(C)]
            predictions.append(np.argmax(counts))

        return predictions

if __name__ == '__main__':

    colors = ['r', 'g', 'b', 'y', 'm', '#0ff0f0', '#112f1f', '#fff000']
    D = 5
    C = 4;
    K = 20;
    X,tags = create_Dd_data(D, C, 3.2, 0.2, 130, 140)

    # Shuffle data
    ds = list(zip(X.T,tags))
    shuffle(ds)
    X = np.array([x for (x,_) in ds]).T
    tags = np.array([t for (_,t) in ds])

    cls = KNeighbours()
    fl = floor(len(X.T)*0.75)

    # Test with 25%
    predictions = cls.predict(X.T[0:fl].T, X.T[fl:len(X.T)].T, tags[0:fl], K, C)
    score = [tags[fl+i] == predictions[i] for i in range(len(X.T)-fl)]
    print('Score of test dataset: %d/%d' %(score.count(True), len(score)))
    # Test with 100%
    predictions = cls.predict(X, X, tags, K, C)
    score = [tags[i] == predictions[i] for i in range(len(X.T))]
    print('Score of whole dataset: %d/%d' %(score.count(True), len(score)))
