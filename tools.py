import numpy as np
import numpy.random as rng
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn import metrics


def one_hot(idx, length):
    """
    Create one hot vector with 1 at `idx` and of length `length`
    Args: idx (int or list of ints)
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
        `mu_list`: (list of length 4 of 1D array) the means of the two classes
        `cov_list`: (list of length 4 of 2D array) the covariances of the 4 subclasses
            * (Y = 1, A = 0)
            * (Y = 1, A = 1)
            * (Y = -1, A = 0)
            * (Y = -1, A = 1)
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

    mu_pos_0 = mu_list[0]
    mu_pos_1 = mu_list[1]
    mu_neg_0 = mu_list[2]
    mu_neg_1 = mu_list[3]
    cov_pos_0 = cov_list[0]
    cov_pos_1 = cov_list[1]
    cov_neg_0 = cov_list[2]
    cov_neg_1 = cov_list[3]
    n = np.sum(cardinals)
    p = np.shape(mu_pos_0)[0]

    # Compute the cardinals of each class
    n_pos = (cardinals[0] + cardinals[1])
    n_neg = (cardinals[2] + cardinals[3])
    
    # Class pos
    class_pos_0 = rng.multivariate_normal(mu_pos_0, cov_pos_0, cardinals[0])
    class_pos_1 = rng.multivariate_normal(mu_pos_1, cov_pos_1, cardinals[1])
    class_pos = np.concatenate([class_pos_0, class_pos_1], axis=0)

    # Class neg
    class_neg_0 = rng.multivariate_normal(mu_neg_0, cov_neg_0, cardinals[2])
    class_neg_1 = rng.multivariate_normal(mu_neg_1, cov_neg_1, cardinals[3])
    class_neg = np.concatenate([class_neg_0, class_neg_1], axis=0)

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

def build_system_matrix(X, sigma, gamma, cardinals, nu_list, ind_dict, mode="strict"):
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
        mode: (str) can be "strict" or "relaxed". Whether we enforce the fairness constraints exactly or approximatively.

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
    if mode == "relaxed":
        matrix[3:, 0] = 1
        matrix[3:, 1] = np.squeeze(H_pos)
        matrix[3:, 2] = np.squeeze(H_neg)
        matrix[0, 3:] = 1
        matrix[1, 3:] = np.squeeze(nu_pos * H_pos.T)
        matrix[2, 3:] = np.squeeze(nu_neg * H_neg.T)
        matrix[1, 1] = 1 + nu_pos * h_pos_pos
        matrix[2, 2] = 1 + nu_neg * h_neg_neg
        matrix[1, 2] = nu_pos * h_neg_pos
        matrix[2, 1] = nu_neg * h_neg_pos
        matrix[3:, 3:] = K + n/gamma * np.eye(n)
    elif mode == "strict":
        matrix[3:, 0] = 1
        matrix[3:, 1] = np.squeeze(H_pos)
        matrix[3:, 2] = np.squeeze(H_neg)
        matrix[0, 3:] = 1
        matrix[1, 3:] = np.squeeze(H_pos.T)
        matrix[2, 3:] = np.squeeze(H_neg.T)
        matrix[1, 1] = h_pos_pos
        matrix[2, 2] = h_neg_neg
        matrix[1, 2] = h_neg_pos
        matrix[2, 1] = h_neg_pos
        matrix[3:, 3:] = K + n/gamma * np.eye(n)
    else:
        raise ValueError("Wrong value for mode")

    return matrix

def decision_fair(X, b, lambda_pos, lambda_neg, alpha, sigma, ind_dict, x_q):
    """
    Computes the decision function of fairness-enhanced LS-SVM.
    WARNING: should be used with a low number of queries, otherwise it takes a lot of memory.
    
    Args:
        X: (2D array n x p) the training data set
        b: scalar
        lambda_pos: scalar lagrangian parameter
        lambda_neg: scalar lagrangian parameter
        alpha: (1D array)  lagrangian parameters
        sigma: (scalar) for kernel computation.
        x_q: (1D array or 2D array) data query, of shape (m , p) or (p,) where m>=1 is the number of queries.
        ind_dict: (dict of vectors) indicator vectors for the classes.

    returns:
        scalar
    """
    k_x = cdist(X, x_q, metric='sqeuclidean')
    k_x = np.exp(-k_x / (2*sigma))

    # h_1(X)^T \phi(x_q)
    n_pos_1 = np.sum(ind_dict['pos', 0])
    n_pos_2 = np.sum(ind_dict['pos', 1])
    h_pos_xq = 1/n_pos_1 * ind_dict['pos', 0].T @ k_x - 1/n_pos_2 * ind_dict['pos', 1].T @ k_x

    # h_{-1}(X)^T \phi(x_q)
    n_neg_1 = np.sum(ind_dict['neg', 0])
    n_neg_2 = np.sum(ind_dict['neg', 1])
    h_neg_xq = 1/n_neg_1 * ind_dict['neg', 0].T @ k_x - 1/n_neg_2 * ind_dict['neg', 1].T @ k_x

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
    k_x = cdist(X, x_q, metric='sqeuclidean')
    k_x = np.exp(-k_x/(2*sigma))
    pred = alpha.T @ k_x + b
    pred = np.squeeze(pred)

    return pred

def comp_fairness_constraints(preds, ind_dict, with_int=False):
    """
    Computes the value of the fairness constraints, to see how well they are respected by `pred_func`.

    Args:
        preds: (array 1D) predictions to be assessed.
        ind_dict: as specified in above functions.
        with_int: (bool) whether to compute the constraints for the output (real) or when taking the sign.

    returns:
        tuple of scalars
    """ 
    if with_int:
        preds = np.sign(preds)

    # Positive constraint
    card0 = np.sum(ind_dict[('pos', 0)])
    card1 = np.sum(ind_dict[('pos', 1)])
    pos_const = 1/card0 * np.sum(np.squeeze(ind_dict[('pos', 0)]) * preds) \
            - 1/card1 * np.sum(np.squeeze(ind_dict[('pos', 1)]) * preds)
    # Negative constraint
    card0 = np.sum(ind_dict[('neg', 0)])
    card1 = np.sum(ind_dict[('neg', 1)])
    neg_const = 1/card0 * np.sum(np.squeeze(ind_dict[('neg', 0)]) * preds) \
            - 1/card1 * np.sum(np.squeeze(ind_dict[('neg', 1)]) * preds)

    return pos_const, neg_const
    
def get_metrics(preds, y, ind_dict):
    """
    The metrics that interests us.
    * Diff of False Positive Rate (FPR) between the sensitive label.
    * Diff of False Negative Rate (FPR) between the sensitive label.
    * precision
    * recall

    Args:
        preds: 1D array
        y: 1D array
        ind_dict: dict of indicator vectors as stated in aforecommented functions.

    returns:
        dictionaries with keys "gen", "0", "1". Each value of this dict is a dict as well, containing metrics for the
        specific scenario (general, for sensitive class 0, for sensitive class 1).
    """
    values = {"gen": {}, "0": {}, "1": {}} 

    true_preds = np.sign(preds)
    # Binarize labels
    preds = (1+np.sign(preds))/2
    y = (1+np.sign(y))/2
    ### So that 0 is true, 1 is false. Needed for the confusion matrix later.
    preds = 1 - preds
    y = 1 - y
    preds = preds.astype('int')
    y = y.astype('int')

    # Overall metrics.
    prec = metrics.precision_score(y, preds) 
    recall = metrics.recall_score(y, preds) 
    conf_mat = metrics.confusion_matrix(y, preds)
    #conf_mat = metrics.confusion_matrix(y, preds, normalize='all')

    values["gen"]["prec"] = np.copy(prec)
    values["gen"]["recall"] = np.copy(recall)
    values["gen"]["conf_mat"] = np.copy(conf_mat)
    values["gen"]["preds"] = true_preds

    # For the sensitive value 0.
    ind_0 = ind_dict[('pos', 0)] + ind_dict[('neg', 0)]
    ind_0 = np.nonzero(np.squeeze(ind_0))
    y_0 = y[ind_0]
    preds_0 = preds[ind_0]
    prec = metrics.precision_score(y_0, preds_0) 
    recall = metrics.recall_score(y_0, preds_0) 
    conf_mat = metrics.confusion_matrix(y_0, preds_0)

    values["0"]["prec"] = np.copy(prec)
    values["0"]["recall"] = np.copy(recall)
    values["0"]["conf_mat"] = np.copy(conf_mat)

    # For the sensitive value 1.
    ind_1 = ind_dict[('pos', 1)] + ind_dict[('neg', 1)]
    ind_1 = np.nonzero(np.squeeze(ind_1))
    y_1 = y[ind_1]
    preds_1 = preds[ind_1]
    prec = metrics.precision_score(y_1, preds_1) 
    recall = metrics.recall_score(y_1, preds_1) 
    conf_mat = metrics.confusion_matrix(y_1, preds_1)

    values["1"]["prec"] = np.copy(prec)
    values["1"]["recall"] = np.copy(recall)
    values["1"]["conf_mat"] = np.copy(conf_mat)

    return values

"""
Below are functions for formulae, for debug.
"""

def extract_W(X, mu_list, cardinals):
    n, p = X.shape
    k = len(cardinals)
    mu = [np.tile(mu_list[i], [cardinals[i], 1]) for i in range(k)]
    mu = np.concatenate(mu, axis=0)
    assert mu.shape[0] == n
    W = X - mu
    return W/np.sqrt(p)

def f(x):
    return np.exp(-x/2)

def f_p(x):
    return -1/2*np.exp(-x/2)

def f_pp(x):
    return 1/4*np.exp(-x/2)

def build_V(mu_list, cardinals, W, cov_list):
    """
    Args:
        mu_list: (list of 1d array)
        cardinals: (list of int or 1d array) 
        W: (2d array) of size p x n.
        eps: (float) scales the covariances.

    returns:
        the V matrix
    """

    V = []
    cardinals = np.array(cardinals)
    n = np.sum(cardinals)
    k = len(cardinals)
    p = W.shape[0]
    assert len(mu_list) == k
    P = np.eye(n) - 1/n * np.ones((n,n))
    W = W @ P
    
    ### Create the J matrix.
    J = np.zeros((n, k))
    idxs = np.cumsum(cardinals)
    idxs = np.insert(idxs, 0, 0)
    for i in range(k):
        tmp = list(np.arange(idxs[i], idxs[i+1]))
        J[:, i] = one_hot(tmp, n)
    V.append(J/np.sqrt(p))

    ### Build the $v_a$ and append them for V.
    for i in range(k):
        V.append(W.T @ mu_list[i].reshape((-1, 1)))

    ### Build the $\tilde{v}$.
    tilde_v = np.zeros((n, 1))
    for i in range(k):
        tmp = W[:, idxs[i]:idxs[i+1]].T @ mu_list[i].reshape((-1, 1))
        tilde_v[idxs[i]:idxs[i+1], :] = np.copy(tmp.reshape((-1, 1)))
    V.append(tilde_v)

    ### Build the $\psi_circ$ or just $\psi$
    tmp = [np.tile(np.trace(cov_list[i]), [cardinals[i], 1]) for i in range(k)]
    tmp = np.concatenate(tmp)
    psi = np.diag(W.T @ W).reshape((-1, 1))
    psi = psi - (1-1/n)*1/p * tmp
    V.append(psi)
    V.append(np.sqrt(p) * psi**2)

    ### Create psi_tilde
    cov_circ = sum([cardinals[i] * cov_list[i] for i in range(k)])/n
    t = np.array([np.trace(cov_list[i] - cov_circ)/np.sqrt(p) for i in range(k)])
    diag = sum([t[i] * J[:, i] for i in range(k)])
    diag = np.diag(np.squeeze(diag))
    psi_tilde = diag @ psi
    V.append(psi_tilde)
    
    V = np.concatenate(V, axis=-1)

    return V

def build_A_n(tau, k, p, V):
    """
    Args:
        tau: (float)
        k: (int)
        p: (int) 

    returns:
        the A_n matrix
    """
    A_n = np.zeros((2*k + 4, 2*k + 4))
    A_n[:k, :k] = 1
    A_n = - f(tau) * p /(2 * f_p(tau)) * A_n 

    #return V @ A_n @ V.T
    return A_n 

def build_A_sqrt_n(cardinals, cov_list, V):
    """
    Args:
        p: (int) 

    returns:
        the A_sqrt_n matrix
    """

    n = sum(cardinals)
    p = cov_list[0].shape[0]
    k = len(cardinals)
    ### Create `t` vector.
    cov_circ = sum([cardinals[i] * cov_list[i] for i in range(k)])/n
    t = np.array([np.trace(cov_list[i] - cov_circ)/np.sqrt(p) for i in range(k)])

    t = t.reshape((-1, 1))
    t = t.T + t
    A_sqrt_n = np.zeros((2*k + 4, 2*k + 4))
    A_sqrt_n[:k, :k] = t
    A_sqrt_n[:k, -3] = 1
    A_sqrt_n[-3, :k] = 1

    #return -np.sqrt(p)/2 * V @ A_sqrt_n @ V.T
    return -np.sqrt(p)/2 *  A_sqrt_n 

def build_A_1(cardinals, mu_list, cov_list, tau, V):
    """
    Args:
        mu_list: (list of 1d array)
        n: (1d array) 
        eps: (float) 

    returns:
        the A_1 matrix
    """

    k = len(mu_list)
    n = sum(cardinals)
    p = mu_list[0].shape[0]
    A_1 = np.zeros((2*k + 4, 2*k + 4))

    ### Create `t` vector.
    cov_circ = sum([cardinals[i] * cov_list[i] for i in range(k)])/n
    t = np.array([np.trace(cov_list[i] - cov_circ)/np.sqrt(p) for i in range(k)])

    ### Build $A_{1, 11}$
    A_1_11 = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            A_1_11[i, j] = -1/2 * np.linalg.norm(mu_list[i] - mu_list[j])**2
            A_1_11[i, j] += - f_pp(tau)/(4*f_p(tau)) * (t[i] + t[j])**2
            A_1_11[i, j] += - f_pp(tau)/(p*f_p(tau)) * np.trace(cov_list[i] @ cov_list[j])
            A_1_11[j, i] = A_1_11[i, j]
    A_1[:k, :k] = A_1_11

    ### The rest
    A_1[:k, k:2*k] = np.eye(k)
    A_1[k:2*k, :k] = np.eye(k)

    A_1[:k, -4] = -1
    A_1[-4, :k] = -1

    A_1[:k, -3] = -f_pp(tau)/(2*f_p(tau)) * np.squeeze(t)
    A_1[-3, :k] = -f_pp(tau)/(2*f_p(tau)) * np.squeeze(t)

    A_1[:k, -2] = -f_pp(tau)/(4*f_p(tau))
    A_1[-2, :k] = -f_pp(tau)/(4*f_p(tau))

    A_1[:k, -1] = -f_pp(tau)/(2*f_p(tau))
    A_1[-1, :k] = -f_pp(tau)/(2*f_p(tau))

    A_1[-3, -3] = - f_pp(tau)/(2*f_p(tau))

    #return V @ A_1 @ V.T
    return  A_1 

def build_C_1(A_n, tau, gamma):
    n = A_n.shape[0]
    P = np.eye(n) - 1/n * np.ones((n,n))
    L = gamma/(1 + gamma*f(tau)) * (np.eye(n) + gamma * P)
    C_1 = - 2*f_p(tau)/n * A_n @ L
    return C_1

def build_C_sqrt_n(A_sqrt_n, A_n, tau, gamma):
    n = A_n.shape[0]
    P = np.eye(n) - 1/n * np.ones((n,n))
    L = gamma/(1 + gamma*f(tau)) * (np.eye(n) + gamma * P)
    alpha = -2*f_p(tau)/n
    C_sqrt_n = -alpha*( A_sqrt_n @ L + alpha * A_n @ L @ A_sqrt_n @ L)

    return C_sqrt_n

def build_C_n(A_1, A_sqrt_n, A_n, tau, gamma, W):
    _, n = W.shape
    beta = f(0) - f(0) + tau * f_p(tau)
    P = np.eye(n) - 1/n * np.ones((n,n))
    L = gamma/(1 + gamma*f(tau)) * (np.eye(n) + gamma * P)
    Q = 2*f_p(tau)/n**2 * (A_1 + P @ W.T @ W @ P + \
                2*f_p(tau)/n * A_sqrt_n @ L @ A_sqrt_n)

    C_n = (-2*f_p(tau) / n) * (A_sqrt_n @ L + 2*f_p(tau)/n * A_n @ L @ A_sqrt_n @ L) @ L
    C_n = C_n - (2*f_p(tau)/n)**2 * A_sqrt_n @ L @ A_sqrt_n @ L
    C_n = C_n - 2*f_p(tau) * A_n @ L @ (Q - beta/n**2 * np.eye(n)) @ L
    return C_n

def build_D_n(C_1, A_n, tau):
    return -2*f_p(tau) * C_1 @ A_n

def build_D_sqrt_n(C_sqrt_n, C_1, A_sqrt_n, A_n, tau):
    return -2*f_p(tau) * (C_1 @ A_sqrt_n + C_sqrt_n @ A_n)

def build_D_1(C_n, C_sqrt_n, C_1, A_1, A_sqrt_n, A_n, W, tau):
    _, n = W.shape
    P = np.eye(n) - 1/n * np.ones((n, n))
    beta = f(0) - f(tau) + tau*f_p(tau)
    D_1 = -2*f_p(tau) * C_n @ A_n - 2*f_p(tau) * C_sqrt_n @ A_sqrt_n
    D_1 = D_1 + C_1 @ (-2 * f_p(tau) * (P @ W.T @ W @ P + A_1) + beta*np.eye(n))

    return D_1

""" 
From old file for debug.
""" 
def old_gen_data_gaussian(means, ns, p, eps, seed=10):
    """
    Generate classes of gaussian vectors, each class with the same covariance (`eps` * I_p)
    Args:
        means: (list of 1d array)
        ns: (1d array) number of points per class.
        p: dimension of the data points.
        eps: scale the covariance.
        seed: (int) for random reproducibility.

    returns:
        x: (2d array) the data points, of size p x n.
        mu: (2d array) the means, of size p x n.
        W: (2d array) the noise matrix, of size p x n.
    """ 

    np.random.seed(seed)
    k = len(ns)

    # Computes the means and the `mu` matrix.
    mu = [np.repeat(means[i].reshape((1, -1)), ns[i], axis=0) for i in range(k)]
    mu = np.concatenate(mu, axis=0)
    
    # Computes C. For the moment, it is simply the eps* I_n since 
    # $C_a = eps * I_n$ for all a
    C = eps*np.eye(p)
    
    # Build W (noise)
    # Note that our W is the transpose of the W of the paper.
    W = [np.random.multivariate_normal(np.zeros(p), 1/p * C, ns[i]) for i in range(len(ns))]
    W = np.concatenate(W, axis=0)

    # Final points
    x = mu + np.sqrt(p)*W

    return x.T, mu.T, W.T

def old_build_V(means, ns, W, eps):
    """
    Args:
        means: (list of 1d array)
        ns: (1d array) 
        W: (2d array) of size p x n.
        eps: (float) scales the covariances.

    returns:
        the V matrix
    """

    V = []
    n = np.sum(ns)
    k = len(ns)
    p = W.shape[0]
    assert len(means) == k
    
    ### Recenter the noise matrix to get $W^{\circ}$.
    P = np.eye(n) - 1/n * np.ones((n, n))
    W_c = W @ P

    ### Create the J matrix.
    J = np.zeros((n, k))
    idxs = np.cumsum(ns)
    idxs = np.insert(idxs, 0, 0)
    for i in range(k):
        tmp = list(np.arange(idxs[i], idxs[i+1]))
        J[:, i] = one_hot(tmp, n)
    V.append(J/np.sqrt(p))

    ### Build the $v_a$ and append them for V.
    mu_circ = np.zeros(p)
    for i in range(k):
        mu_circ += ns[i]/n * means[i]
    means_circ = [(means[i] - mu_circ).reshape((-1, 1)) for i in range(k)]

    for i in range(k):
        V.append(W_c.T @ means_circ[i])

    ### Build the $\tilde{v}$.
    tilde_v = np.zeros((n, 1))
    for i in range(k):
        tmp = W_c[:, idxs[i]:idxs[i+1]].T @ means_circ[i]
        tilde_v[idxs[i]:idxs[i+1]] = np.copy(tmp)
    V.append(tilde_v)

    ### Build the $\psi_circ$
    psi_circ = np.diag(W_c.T @ W_c).reshape((-1, 1))
    psi_circ = psi_circ - (1 - 1/n)*eps
    V.append(psi_circ)
    V.append(np.sqrt(p) * psi_circ**2)

    ### The related vector is null in our case (same covariance for all the classes)
    V.append(np.zeros((n, 1)))
    
    V = np.concatenate(V, axis=-1)

    return V

def old_build_A_n(eps, k, p, n):
    """
    Args:
        eps: (float)
        k: (int)
        n: (int) 
        p: (int) 

    returns:
        the A_n matrix
    """
    tau_circ = 2 * (n+1)/n * eps
    A_n = np.zeros((2*k + 4, 2*k + 4))
    A_n[:k, :k] = 1
    A_n = - f(tau_circ) * p /(2 * f_p(tau_circ)) * A_n 

    return A_n

def old_build_A_sqrt_n(k, n, p):
    """
    Args:
        k: (int)
        n: (int) 
        p: (int) 

    returns:
        the A_sqrt_n matrix
    """
    A_sqrt_n = np.zeros((2*k + 4, 2*k + 4))
    A_sqrt_n[:k, -3] = 1
    A_sqrt_n[-3, :k] = 1

    return -np.sqrt(p)/2 * A_sqrt_n

def old_build_A_1(means, n, eps):
    """
    Args:
        means: (list of 1d array)
        n: (1d array) 
        eps: (float) 

    returns:
        the A_1 matrix
    """

    tau = 2 * eps
    tau_circ = 2 * (n+1)/n * eps
    k = len(means)
    A_1 = np.zeros((2*k + 4, 2*k + 4))

    ### Build $A_{1, 11}$
    A_1_11 = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            A_1_11[i, j] = -1/2 * np.linalg.norm(means[i] - means[j])**2
            A_1_11[i, j] += - f_pp(tau)/f_p(tau) * eps**2
            A_1_11[j, i] = A_1_11[i, j]
    A_1[:k, :k] = A_1_11

    ### The rest
    A_1[:k, k:2*k] = np.eye(k)
    A_1[k:2*k, :k] = np.eye(k)

    A_1[:k, -4] = -1
    A_1[-4, :k] = -1

    A_1[:k, -2] = -f_pp(tau)/(4*f_p(tau))
    A_1[-2, :k] = -f_pp(tau)/(4*f_p(tau))

    A_1[:k, -1] = -f_pp(tau)/(2*f_p(tau))
    A_1[-1, :k] = -f_pp(tau)/(2*f_p(tau))

    A_1[-3, -3] = - f_pp(tau)/(2*f_p(tau))

    return A_1
