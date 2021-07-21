import numpy as np
import numpy.random as rng
import pprint
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import Config
from pathlib import Path
#from functools import partial
from scipy.spatial.distance import pdist, squareform
from tools import extract_W, build_V, build_A_n, build_A_sqrt_n, build_A_1, \
                build_C_1, build_C_sqrt_n, build_C_n, build_D_1, build_D_sqrt_n, \
                build_D_n, one_hot, gen_dataset, get_gaussian_kernel, build_system_matrix 

from tools import f, f_p, f_pp, build_objects

### NumPy print threshold size.
np.set_printoptions(threshold = 100)

### Set the seed for reproducibility
#np.random.seed(12)

### Get default config for experiments.
conf_default = Config(Path('conf_default.json'))

### Hyperparameters
mode = "strict"
gamma = 1
mean_scal = 5
cov_scal = 1
print(f"cov_scal is {cov_scal}")
#list_cardinals = [[600, 300, 300, 500]]
#p_list = [512]
list_cardinals = [[300, 150, 150, 250], [600, 300, 300, 500]]
p_list = [512, 1024]

# Monitoring purposes.
A_1_list = []
A_sqrt_n_list = []
A_n_list = []
C_1_list = []
C_sqrt_n_list = []
C_n_list = []
D_1_list = []
D_sqrt_n_list = []
D_n_list = []
K_diff_list = []
E_list = []
Om_list = []
tmp_list = []
tmp2_list = []
tmp3_list = []

for i in range(len(p_list)):
    # Class distribution
    cardinals = list_cardinals[i]
    p = p_list[i]
    print("\ncardinals: ", cardinals, "  ||  p: ", p)
    k = len(cardinals)
    n = sum(cardinals)
    sigma=p
    mu_list = [mean_scal * one_hot(0, p), mean_scal * one_hot(1, p),
               mean_scal * one_hot(2, p), mean_scal * one_hot(3, p)]
    cov_list = [cov_scal*np.eye(p), cov_scal*np.eye(p),
                cov_scal*np.eye(p), cov_scal*np.eye(p)]
    
    # Generate data.
    X, y, sens_labels, ind_dict = gen_dataset(mu_list, cov_list, cardinals)
    assert (n, p) == X.shape
    W = extract_W(X, mu_list, cardinals)
    K = get_gaussian_kernel(X, sigma)
    J, M, t, S = build_objects(mu_list, cardinals, cov_list)

    matrix_fair = build_system_matrix(X, sigma, gamma, cardinals, [1, 1], ind_dict, mode=mode)
    B_11 = matrix_fair[1:3, 1:3]
    B_12 = matrix_fair[1:3, 3:]
    B_21 = matrix_fair[3:, 1:3]
    B_22 = matrix_fair[3:, 3:]
    A_22 = matrix_fair[1:, 1:]
    
    ### Generate approximators for control purposes.
    # Build V
    V = build_V(cardinals, mu_list, cov_list, J, W.T, M, t)

    # estimator from data
    #tau = np.sum(squareform(pdist(X, 'sqeuclidean')))
    #tau = tau/(p * n *(n-1))
    # theoretical value.
    tau = np.trace(sum([cardinals[i]/n * cov_list[i] for i in range(len(cov_list))]))
    tau = 2*tau/p
    beta = f(0) - f(tau) + tau*f_p(tau)
    
    A_1 = build_A_1(mu_list, t, S, tau, V)
    A_sqrt_n = build_A_sqrt_n(t, p, V)
    A_n = build_A_n(tau, k, p, V)
    A = A_1 + A_sqrt_n + A_n

    C_n = build_C_n(A_1, A_sqrt_n, A_n, tau, gamma, W.T)
    C_sqrt_n = build_C_sqrt_n(A_sqrt_n, A_n, tau, gamma)
    C_1 = build_C_1(A_n, tau, gamma)
    C = C_n + C_sqrt_n + C_1
    
    D_1 = build_D_1(C_n, C_sqrt_n, C_1, A_1, A_sqrt_n, A_n, W.T, tau)
    D_sqrt_n = build_D_sqrt_n(C_sqrt_n, C_1, A_sqrt_n, A_n, tau)
    D_n = build_D_n(C_1, A_n, tau)
    D = D_n + D_sqrt_n + D_1

    # $B_{22}$ in the companion paper.
    Om = K + n/gamma * np.eye(n)
    P = np.eye(n) - 1/n * np.ones((n,n))
    L = gamma / (1 +gamma * f(tau)) * (np.eye(n) + f(tau)*gamma * P)
    Q = 2*f_p(tau)/n**2 * (A_1 + 1/p * W @ W.T + \
                2*f_p(tau)/n * A_sqrt_n @ L @ A_sqrt_n)
    Om = np.linalg.inv(Om)
    rest = np.linalg.inv(B_11 - B_12 @ Om @ B_21)
    E = n/gamma * (np.eye(n) - n/gamma * Om)
    E_app = f(tau)/(1 + gamma*f(tau)) * np.ones((n,n)) \
            - 1/gamma**2 * (2*f_p(tau) * L@A_sqrt_n@L \
            + L @ (n**2 * Q - beta * np.eye(n)) @ L)
    E_app1 = f(tau)/(1 + gamma*f(tau)) * np.ones((n,n)) 
    E_app2 = - 1/gamma**2 * (2*f_p(tau) * L@A_sqrt_n@L )
    E_app3 = -1/gamma**2 * L @ (n**2 * Q - beta * np.eye(n)) @ L
    tmp0 = np.sum(ind_dict[('pos', 0)])
    tmp1 = np.sum(ind_dict[('pos', 1)])
    Delta_pos = ind_dict[('pos', 0)]/tmp0 - ind_dict[('pos', 1)]/tmp1
    tmp0 = np.sum(ind_dict[('neg', 0)])
    tmp1 = np.sum(ind_dict[('neg', 1)])
    Delta_neg = ind_dict[('neg', 0)]/tmp0 - ind_dict[('neg', 1)]/tmp1
    Delta = np.concatenate([Delta_pos, Delta_neg], axis=1)
    alpha_pos = float(Delta_pos.T @ E @ Delta_pos)
    alpha_neg = float(Delta_neg.T @ E @ Delta_neg)
    alpha_pos_neg = float(- Delta_pos.T @ E @ Delta_neg)
    alpha_pos_app = float(- Delta_pos.T @ (n**2 * Q - beta*np.eye(n)) @ Delta_pos)
    alpha_neg_app = float(- Delta_neg.T @ (n**2 * Q - beta*np.eye(n)) @ Delta_neg)
    alpha_pos_neg_app = float(- Delta_pos.T @ (n**2 * Q - beta*np.eye(n)) @ Delta_neg)
    #G = n**2 * Q - beta*np.eye(n)
    #a11 = float(-Delta_neg.T @ G @ Delta_neg)
    #a12 = float(Delta_pos.T @ G @ Delta_neg)
    #a21 = float(Delta_pos.T @ G @ Delta_neg)
    #a22 = float(-Delta_neg.T @ G @ Delta_neg)
    #rest_approx = np.array([[a11, a12], [a21, a22]])
    #rest_approx = 1/(alpha_pos * alpha_neg - alpha_pos_neg**2) * rest_approx
    #K_app = - 2*f_p(tau) * (1/p * W @ W.T + A) + beta*np.eye(n)
    Delta_rot = np.concatenate([Delta_neg, -Delta_pos], axis=1)
    fact = 1/(alpha_pos * alpha_neg - alpha_pos_neg**2)
    inv_app = fact * Delta_rot.T @ E @ Delta_rot


    ### For control purposes.
    #print("For tmp")
    #E_list.append(np.linalg.norm(E, ord=2))
    tmp_list.append(alpha_neg)
    tmp2_list.append(alpha_neg_app)
    tmp3_list.append(alpha_neg - alpha_neg_app)

    #print("For K")
    #K_diff_list.append(np.linalg.norm(K - K_app, ord=2))
    #print("For A")
    #A_1_list.append(np.linalg.norm(A_1, ord=2))
    #A_sqrt_n_list.append(np.linalg.norm(A_sqrt_n, ord=2))
    #A_n_list.append(np.linalg.norm(A_n, ord=2))
    #print("For C")
    #C_1_list.append(np.linalg.norm(C_1, ord=2))
    #C_sqrt_n_list.append(np.linalg.norm(C_sqrt_n, ord=2))
    #C_n_list.append(np.linalg.norm(C_n, ord=2))
    #print("For D")
    #D_1_list.append(np.linalg.norm(D_1, ord=2))
    #D_sqrt_n_list.append(np.linalg.norm(D_sqrt_n, ord=2))
    #D_n_list.append(np.linalg.norm(D_n, ord=2))

print("")
print("Results of operator norms")
print("\t normal : ", tmp_list[1]/tmp_list[0])
print("\t approx : ", tmp2_list[1]/tmp2_list[0])
print("\t diff : ", tmp3_list[1]/tmp3_list[0])
#print("\tE order: ", E_list[1]/E_list[0])
#print("\tK: ", K_diff_list[1]/K_diff_list[0])
#print("A_1: ", A_1_list[1]/A_1_list[0])
#print("A_sqrt_n: ", A_sqrt_n_list[1]/A_sqrt_n_list[0])
#print("A_n: ", A_n_list[1]/A_n_list[0])
#print("\tC_1: ", C_1_list[1]/C_1_list[0])
#print("\tC_sqrt_n: ", C_sqrt_n_list[1]/C_sqrt_n_list[0])
#print("\tC_n: ", C_n_list[1]/C_n_list[0])
#print("Q: ", Q_list[1]/Q_list[0])
#print("D_1: ", D_1_list[1]/D_1_list[0])
#print("D_sqrt_n: ", D_sqrt_n_list[1]/D_sqrt_n_list[0])
#print("D_n: ", D_n_list[1]/D_n_list[0])
