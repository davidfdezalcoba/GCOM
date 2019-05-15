from CreateData import create_Dd_data
import numpy as np
import matplotlib.pyplot as plt

class Fisher_Reduction:
    '''
    Fisher LDA for reducing data dimension from D to D'

    Atributes: 
        W: matrix of D' eigenvectors used to perform LDA
    '''
    def __init__(self, D_p):
        self.W = []
        self.projection = []
        self.D_p = D_p

    def computeW(self, D, K, X):
        '''
        Calculates matrix W for performing PCA reduction

        Atributes:
            D: Dimension of the initial data
            X: Data to reduce
            Epsilon: Threshold for data loss (Default 0.15)
        '''

        classMeans = np.zeros((K,D))
        num_datos_clase_k = np.zeros(K)
        totalMean = np.mean(X, axis=1)
        Sb = np.zeros((D, D))
        Sk = np.zeros((K,D,D))
        Sw = np.zeros((D,D))

        for i in range(K):
            pointsink = [X[:,j] for j in range(len(X.T)) if tags[j] == i]
            kmean = np.mean(pointsink, axis=0)
            classMeans[i] = kmean
            num_datos_clase_k[i] = len(pointsink)
            Sb += num_datos_clase_k[i] * np.outer(classMeans[i] - totalMean,\
                    classMeans[i] - totalMean)
            for p in range(len(pointsink)):
                Sk[i] += np.outer(pointsink[p] - classMeans[i],\
                        pointsink[p] - classMeans[i])
            Sw += Sk[i]

        # Form W with D_p eigenvectors associated with D_p 
        # greatest eigenvalues
        SwinvSb = np.dot(np.linalg.inv(Sw), Sb)
        eigvals, eigvecs = np.linalg.eig(SwinvSb)
        eigs = [(eigvals[i], eigvecs[:,i]) for i in range(len(eigvals))]
        sortedeigs = sorted(eigs, key = lambda x : x[0], reverse = True)
        self.W = np.array([sortedeigs[i][1] for i in range(self.D_p)]).T

    def getProjection(self, X):
        '''
        Returns the projected data
        '''
        if self.W == []:
            print('W not computed yet!')
            return
        if self.projection == []:
            self.projection = np.dot(self.W.T, X)
        return self.projection

def selectDim(D, Epsilon, singVals):
    '''Selects the dimension to reduce the data to in order
    to minimize data loss
    '''
    denom = sum(singVals**2)
    # From those values that are below epsilon,
    # get the index of the maximum
    lista = [(sum(singVals[d+1:D]**2)/denom) \
         if (sum(singVals[d+1:D]**2)/denom) < Epsilon \
         else -1 for d in range(D)]
    return np.argmax(lista)

if __name__ == '__main__':

    colors = ['r', 'g', 'b', 'y', 'm', '#0ff0f0', '#112f1f', '#fff000']
    D = 5
    D_p = 2
    K = 8
    X,tags = create_Dd_data(D, K, 0.5, 0.2)
    print(X.shape)

    cls = Fisher_Reduction(D_p)
    cls.computeW(D, K, X)

    # Get the projected data
    projection = cls.getProjection(X)

    print('Data has been reduced from %d to %d' %(D, cls.D_p))
    print('The projected data is: ')
    print(projection)

    # Plot the projected data in case of 1D and 2D
    for i in range(len(projection.T)):
        if cls.D_p == 1:
            plt.scatter(projection[:,i], np.random.normal(0,1,1)+0, color=colors[tags[i]])
        elif cls.D_p == 2:
            plt.scatter(projection[0,i], projection[1,i], color=colors[tags[i]])
    plt.show()
