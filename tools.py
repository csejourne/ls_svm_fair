import numpy as np
import numpy.random as rng
from scipy.spatial.distance import pdist, squareform

def tmp():
    return np.zeros(4)

def one_hot(idx, length):
    """
    Create one hot vector with 1 at `idx` and of length `length`
    Args:
        idx (int or list of ints)
        length (int)
        
    returns:
        one hot vector
    """
    if type(idx) == int:
        vec = np.zeros(length)
        vec[idx] = 1
        return vec
    elif type(idx) == list:
        vec = np.zeros(length)
        vec[idx] = 1
        return vec
    else:
        raise ValueError("Weird type for idx")

def gen_dataset(mu_list, cov_list, cardinals): 
    """
    Generate a synthetic dataset on which to work. Consists of two classes each driven by a multidimensional gaussian.
    We then assign a sensitive attribute to all the population with a correlation.
    Args:
        `mu_list`: (list of length 2 of 1D array) the means of the two classes
        `cov_list`: (list of length 2 of 2D array) the covariances of the two classes
        `cardinals`: (list of 4 elements). Contains
            * number of points (Y = 1, A = 0)
            * number of points (Y = 1, A = 1)
            * number of points (Y = -1, A = 0)
            * number of points (Y = -1, A = 1)

    returns:
        `X` (2D array) of size `n` times `p`
        `y` (1D array) of size `p`
        `labels` (1D array) of size `p`
        `ind_dict`: (dict of 1D array) dict of indicator vectors (the $j_{y, a}$).
    """

    mu_pos = mu_list[0]
    mu_neg = mu_list[1]
    cov_pos = cov_list[0]
    cov_neg = cov_list[1]
    n = np.sum(cardinals)
    p = np.shape(mu_pos)[0]

    # Compute the cardinals of each class
    n_pos = (cardinals[0] + cardinals[1])
    n_neg = (cardinals[2] + cardinals[3])
    
    # Create the points from each class.
    class_pos = rng.multivariate_normal(mu_pos, cov_pos, n_pos)
    class_neg = rng.multivariate_normal(mu_neg, cov_neg, n_neg)

    ### Create data
    X = np.concatenate([class_pos, class_neg], axis=0)
    y = np.concatenate([np.ones(n_pos), -1 * np.ones(n_neg)])
    sens_labels = np.concatenate([np.zeros(cardinals[0]),
                                  np.ones(cardinals[1]), 
                                  np.zeros(cardinals[2]),
                                  np.ones(cardinals[3])])

    ### Create indicator vectors used later on.
    ind_dict = {}
    ind_dict[('pos', 0)] = np.zeros(n).reshape((-1, 1))
    ind_dict[('neg', 0)] = np.zeros(n).reshape((-1, 1))
    ind_dict[('pos', 1)] = np.zeros(n).reshape((-1, 1))
    ind_dict[('neg', 1)] = np.zeros(n).reshape((-1, 1))
    ind_dict[('pos', 0)][0: cardinals[0]] = 1
    ind_dict[('pos', 1)][cardinals[0]:cardinals[0]+cardinals[1]] = 1
    ind_dict[('neg', 0)][n_pos:n_pos+cardinals[2]] = 1
    ind_dict[('neg', 1)][n_pos+cardinals[2]:] = 1


    return X, y, sens_labels, ind_dict

def get_gaussian_kernel(X, sigma):
    K = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-K/(2*sigma))
    return K

def build_system_matrix(X, sigma, gamma, cardinals, nu_list, ind_dict):
    """
    Builds the system matrix to find the lagrangian parameters $b, \lambda_1, \lambda_{-1}, \alpha$ when using fairness
    constraints.

    Args:
        X: (n x p array)
        y: (n 1D array)
        sigma: scalar
        gamma: scalar
        cardinals: list of 4 ints as explained previously.
        nu_list: list of 2 scalars, regularization parameters.
        ind_dict: dict of indicator vectors, as explained above.

    returns:
        matrix: (n+3)x(n+3) array.
    """
    n, p = X.shape
    K = get_gaussian_kernel(X, sigma)
    nu_pos = nu_list[0]
    nu_neg = nu_list[1]

    # For H_1(X)
    H_pos = 1/cardinals[0] * K @ ind_dict[('pos', 0)] \
        - 1/cardinals[1] * K @ ind_dict[('pos', 1)]

    # For H_1(X)
    H_neg = 1/cardinals[2] * K @ ind_dict[('neg', 0)] \
        - 1/cardinals[3] * K @ ind_dict[('neg', 1)]

    # For $h_1(X)^T h_1(X)$
    h_pos_pos = (1/cardinals[0]**2) * ind_dict[('pos', 0)].T @ K @ ind_dict[('pos', 0)] \
            + (1/cardinals[1]**2) * ind_dict[('pos', 1)].T @ K @ ind_dict[('pos', 1)] \
            - 2/(cardinals[0]*cardinals[1]) * ind_dict[('pos', 0)].T @ K @ ind_dict[('pos', 1)]

    # For $h_{-1}(X)^T h_{-1}(X)$
    h_neg_neg = (1/cardinals[2]**2) * ind_dict[('neg', 0)].T @ K @ ind_dict[('neg', 0)] \
            + (1/cardinals[3]**2) * ind_dict[('neg', 1)].T @ K @ ind_dict[('neg', 1)] \
            - 2/(cardinals[2]*cardinals[3]) * ind_dict[('neg', 0)].T @ K @ ind_dict[('neg', 1)]

    # For $h_1(X)^T h_{-1}(X)$
    h_neg_pos = 1/(cardinals[0]*cardinals[2]) * ind_dict[('neg', 0)].T @ K @ ind_dict[('pos', 0)] \
            + 1/(cardinals[1]*cardinals[3]) * ind_dict[('neg', 1)].T @ K @ ind_dict[('pos', 1)] \
            - 1/(cardinals[2]*cardinals[1]) * ind_dict[('neg', 0)].T @ K @ ind_dict[('pos', 1)] \
            - 1/(cardinals[3]*cardinals[0]) * ind_dict[('neg', 1)].T @ K @ ind_dict[('pos', 0)]

    # Construct the system matrix
    matrix = np.zeros((n+3, n+3))
    matrix[3:, 0] = 1
    matrix[3:, 1] = np.squeeze(H_pos)
    matrix[3:, 2] = np.squeeze(H_neg)
    matrix[0, 3:] = 1
    matrix[1, 3:] = np.squeeze(nu_pos * H_pos.T)
    matrix[2, 3:] = np.squeeze(nu_pos * H_neg.T)
    matrix[1, 1] = 1 + nu_pos * h_pos_pos
    matrix[2, 2] = 1 + nu_neg * h_neg_neg
    matrix[1, 2] = nu_pos * h_neg_pos
    matrix[2, 1] = nu_neg * h_neg_pos
    matrix[3:, 3:] = K + n/gamma * np.eye(n)

    return matrix

def decision_fair(X, b, lambda_pos, lambda_neg, alpha, sigma, ind_dict, x_q):
    """
    Computes the decision function of fairness-enhanced LS-SVM
    
    Args:
        X: (2D array n x p) the training data set
        b: scalar
        lambda_pos: scalar lagrangian parameter
        lambda_neg: scalar lagrangian parameter
        alpha: (1D array)  lagrangian parameters
        sigma: (scalar) for kernel computation.
        x_q: (1D array) data test (query)
        ind_dict: (dict of vectors) indicator vectors for the classes.

    returns:
        scalar
    """
    n, p = X.shape
    X = X.reshape((n, 1, p))
    k_x = X - x_q.reshape((-1, p))
    k_x = np.linalg.norm(k_x, axis=2)
    k_x = np.squeeze(k_x)
    k_x = np.exp(-k_x ** 2/sigma)

    # h_1(X)^T \phi(x_q)
    n_pos_1 = np.sum(ind_dict['pos', 0])
    n_pos_2 = np.sum(ind_dict['pos', 1])
    h_pos_xq = 1/n_pos_1 * ind_dict['pos', 0].T @ k_x - 1/n_pos_2 * ind_dict['pos', 1].T @ k_x

    # h_{-1}(X)^T \phi(x_q)
    n_neg_1 = np.sum(ind_dict['neg', 0])
    n_neg_2 = np.sum(ind_dict['neg', 1])
    h_neg_xq = 1/n_neg_1 * ind_dict['neg', 0].T @ k_x - 1/n_neg_2 * ind_dict['neg', 1].T @ k_x

    tmp = alpha.T @ k_x
    pred = alpha.T @ k_x + b + lambda_pos * h_pos_xq + lambda_neg * h_neg_xq 
    pred = np.squeeze(pred)

    return pred

def decision_unfair(X, b, alpha, sigma, x_q):
    """
    Computes the decision function of classical LS-SVM
    
    Args:
        X: (2D array n x p) the training data set
        b: scalar
        alpha: (1D array)  lagrangian parameters
        sigma: (scalar) for kernel computation.
        x_q: (1D array or 2D array) data query, of shape (m , p) or (p,) where m>=1 is the number of queries.

    returns:
        scalar
    """
    n, p = X.shape
    X = X.reshape((n, 1, p))
    k_x = X - x_q.reshape((-1, p))
    k_x = np.linalg.norm(k_x, axis=2)
    k_x = np.squeeze(k_x)
    k_x = np.exp(-k_x ** 2/sigma)
    print(f"k_x shape is {k_x.shape}")

    pred = alpha.T @ k_x + b
    pred = np.squeeze(pred)

    return pred

#def comp_fairness_constraints(pred_func, X, ind_dict):
#    """
#    Computes the value of the fairness constraints, to see how well they are respected by `pred_func`
#
#    """
