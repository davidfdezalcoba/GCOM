from CreateData import create_Dd_data
import numpy as np
import matplotlib.pyplot as plt

class PCA_Reduction:
    '''
    PCA for reducing data dimension from D to D'

    Atributes: 
        W: matrix of D' eigenvectors used to perform PCA
    '''
    def __init__(self):
        self.W = []
        self.projection = []
        self.compression = []
        self.normalization = []
        self.D_p = 0

    def computeW(self, D, X, Epsilon = 0.15):
        '''
        Calculates matrix W for performing PCA reduction

        Atributes:
            D: Dimension of the initial data
            X: Data to reduce
            Epsilon: Threshold for data loss (Default 0.15)
        '''
        # Calculate Covariance matrix
        totalMean = np.mean(X.T, axis=0)
        Xm = X.T - totalMean
    
        # SVD decomposition
        u,s,v = np.linalg.svd(Xm.T, full_matrices = False)
        S = u.dot(np.diag(s**2)).dot(u.T)
    
        # Calculate best dimension to reduce to
        self.D_p = selectDim(D, Epsilon, s)
    
        # Form W with D_p eigenvectors associated with D_p 
        # greatest eigenvalues
        eigvals, eigvecs = np.linalg.eig(S)
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

    def getCompression(self, X):

        if self.W == []:
            print('W not computed yet!')
            return
        if self.compression == []:
            mean = np.mean(X.T, axis = 0)
            self.compression = np.vstack(mean) + self.W.dot(self.W.T.dot((X.T - mean).T))
        return self.compression
        

    def getNormalization(self, X):
        '''
        Returns normalized data
        '''
        totalMean = np.mean(X.T, axis=0)
        X = X - totalMean[:, np.newaxis]
        u,s,_ = np.linalg.svd(X)
        self.normalization = np.linalg.inv(np.diag(s)).dot(u.T).dot(X) 
        return self.normalization


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
    D = 4
    Epsilon = 0.15
    K = 5
    X,tags = create_Dd_data(D, K, 1.2, 0.2)

    cls = PCA_Reduction()
    cls.computeW(D, X, Epsilon)

    # Get the projected data
    projection = cls.getProjection(X)
    compression = cls.getCompression(X)
    normalization = cls.getNormalization(X)

    print('Original data is:\n')
    print(X)
    print('\nData has been reduced from %d to %d\n' %(D, cls.D_p))
    print('\nThe projected data is: \n')
    print(projection)
    
    print('\nCompressed data is:\n')
    print(compression)

    print('\nNormalized data is:\n')
    print(normalization)
    nor2 = normalization - np.mean(normalization, axis=1)[:, np.newaxis]
    print('\nMean: %f\n' %np.linalg.norm(np.mean(normalization, axis = 1)))
    print('\nCovariance matrix: \n')
    print(nor2.dot(nor2.T))

    # Plot the projected data in case of 1D and 2D
    for i in range(len(projection.T)):
        if cls.D_p == 1:
            plt.scatter(projection[:,i], np.random.normal(0,1,1)+0, color=colors[tags[i]])
        elif cls.D_p == 2:
            plt.scatter(projection[0,i], projection[1,i], color=colors[tags[i]])
    plt.show()
