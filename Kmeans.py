from CreateData import create_Dd_data
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    '''
    Implementation of the K-means clustering algorithm

    '''
    def __init__(self):
        self.clustercentroids = []
        self.clusters = []

    def clusterData(self, X, K):
        # Random centroids from X
        Cs = np.random.randint(X.shape[1], size=K)
        C = np.array([X.T[Cs[i]] for i in range(len(Cs))])
        
        C_old = np.zeros(C.shape)

        # clusters[i] == j si el dato X_i pertenece al cluster j
        self.clusters = np.zeros(len(X.T))
        error = np.linalg.norm(C - C_old)

        while error != 0: # No han cambiado las asignaciones
            # Asignación
            for i in range(len(X.T)):
                dist = np.linalg.norm(X.T[i] - C, axis = 1)
                self.clusters[i] = np.argmin(dist)

            C_old = deepcopy(C)

            # Actualización
            for i in range(K):
                kdatos = np.array([X.T[j] for j in range(len(X.T)) if self.clusters[j] == i])
                C[i] = np.mean(kdatos,axis=0)

            error = np.linalg.norm(C - C_old)

        self.clustercentroids = C

if __name__ == '__main__':

    colors = ['r', 'g', 'b', 'y', 'm', '#0ff0f0', '#112f1f', '#fff000']
    D = 2
    numClusters = 8;
    X,_ = create_Dd_data(D, 1, 3.2, 0.2, 1300, 1400)

    cls = KMeansClustering()
    cls.clusterData(X, numClusters)

    if D == 2:
        for i in range(numClusters):
            pointsink = np.array([X.T[j] for j in range(len(X.T)) if cls.clusters[j] == i])
            plt.plot(pointsink[:,0], pointsink[:,1], 'o', markersize=4, c=colors[i])
        plt.plot(cls.clustercentroids[:,0], cls.clustercentroids[:,1], '*', c='#000000')
        plt.show()
    else:
        for i in range(numClusters):
            print('Cluster %d:\n' %i)
            print(np.array([X.T[j] for j in range(len(X.T)) if cls.clusters[j] == i]))

