import numpy as np

def create_Dd_data(D, K, sigma_class=10, sigma=0.5, min_num=100, max_num=200):
    '''Creates some random D dimensional data for testing.
    Return points X belonging to K classes given
    by tags. 
    
    Args:
        K: number of classes
        sigma_class: measures how sparse are the classes
        sigma: measures how clustered around its mean are 
               the points of each class
        min_num, max_num: minimum and maximum number of points of each class

    Returns:
        X: (N, D) array of points
        tags: list of tags [k0, k1, ..., kN]
    '''
    
    tags = []
    N_class_list = []
    mu_list = []

    mu_list = [np.random.randn(D)*sigma_class]
    
    for k in range(1, K):
        try_mu = np.random.randn(D)*sigma_class
        while min([np.linalg.norm(mu - try_mu) for mu in mu_list]) < D*sigma_class:
            try_mu = np.random.randn(D)*sigma_class
        mu_list.append(try_mu)

    for k in range(K):
        N_class = np.random.randint(min_num, max_num)
        tags += [k] * N_class
        N_class_list += [N_class]

    N = sum(N_class_list)
    X = np.zeros((D, N))
    count = 0
    for k in range(K):
        X[:, count:count + N_class_list[k]] = \
            mu_list[k][:, np.newaxis] + np.random.randn(D, N_class_list[k])*sigma
        count += N_class_list[k]

    return X, tags
