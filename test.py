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
from tools import extract_W, build_V, build_A_n, build_A_sqrt_n, build_A_1, build_A_1_11, \
                build_C_1, build_C_sqrt_n, build_C_n, build_D_1, build_D_sqrt_n, \
                build_D_n, one_hot, gen_dataset, get_gaussian_kernel, build_system_matrix, \
                build_Delta, build_F_n, build_tilde_F_n

from tools import f, f_p, f_pp, build_objects

### NumPy print threshold size.
np.set_printoptions(threshold = 100)

### Set the seed for reproducibility
#np.random.seed(12)
#print("numpy state is: ", rng.get_state())

### Get default config for experiments.
conf_default = Config(Path('conf_default.json'))

### Hyperparameters
mode = "strict"
gamma = 1
mean_scal = 1
cov_scal = 1
print(f"cov_scal is {cov_scal}")
#list_cardinals = [[300, 150, 150, 250]]
#p_list = [512]
#list_cardinals = [[300, 150, 150, 250], [600, 300, 300, 500]]
#p_list = [512, 1024]
list_cardinals = [[600, 300, 300, 500], [1200, 600, 600, 1000]]
p_list = [1024, 2048]
#list_cardinals = [[300, 150, 150, 250], [600, 300, 300, 500], [1200, 600, 600, 1000]]
#p_list = [512, 1024, 2048]

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
var_list = []
mean_list = []
tmp_list = []
tmp2_list = []
tmp3_list = []
tmp4_list = []
tmp5_list = []
tmp6_list = []
tmp7_list = []
lambdas_list = []
lambdas_app_list = []
lambdas_app2_list = []

for i in range(len(p_list)):
    # Class distribution
    cardinals = list_cardinals[i]
    p = p_list[i]
    print("\ncardinals: ", cardinals, "  ||  p: ", p)
    k = len(cardinals)
    n = sum(cardinals)
    vec_prop = np.array(cardinals).reshape((-1, 1))
    sigma=p
    mu_list = [mean_scal * one_hot(0, p), mean_scal * one_hot(1, p),
               mean_scal * one_hot(2, p), mean_scal * one_hot(3, p)]
    #cov_list = [cov_scal*np.eye(p),  cov_scal*np.eye(p),
    #            cov_scal*np.eye(p),  cov_scal*np.eye(p)]
    cov_list = [cov_scal*np.eye(p), (1 + 2/np.sqrt(p)) * cov_scal*np.eye(p),
                cov_scal*np.eye(p), (1 + 2/np.sqrt(p)) * cov_scal*np.eye(p)]
    
    # Generate data.
    X, y, sens_labels, ind_dict = gen_dataset(mu_list, cov_list, cardinals)
    Y = np.concatenate([np.zeros(2), y])
    assert (n, p) == X.shape
    W = extract_W(X, mu_list, cardinals)
    K = get_gaussian_kernel(X, sigma)
    J, M, t, S = build_objects(mu_list, cardinals, cov_list)
    ones_k = np.ones((k, 1))
    n_signed = np.array([cardinals[0], cardinals[1], -cardinals[2], -cardinals[3]]).reshape((-1, 1))
    T = t.reshape((-1, 1)) @ ones_k.T + ones_k @ t.reshape((1, -1))

    matrix_fair = build_system_matrix(X, sigma, gamma, cardinals, [1, 1], ind_dict, mode=mode)
    rhs_fair = np.concatenate([np.zeros(3), y])
    sol_fair = sp_linalg.solve(matrix_fair, rhs_fair)
    # Build the fair decision function.
    b = sol_fair[0]
    lambda_pos = sol_fair[1]
    lambda_neg = sol_fair[2]
    alpha = sol_fair[3:]
    B_11 = matrix_fair[1:3, 1:3]
    B_12 = matrix_fair[1:3, 3:]
    B_21 = matrix_fair[3:, 1:3]
    B_22 = matrix_fair[3:, 3:]
    A_12 = matrix_fair[0, 1:].reshape((1, -1))
    A_21 = matrix_fair[1:, 0].reshape((-1, 1))
    A_22 = matrix_fair[1:, 1:]
    A_22_inv = np.linalg.inv(A_22)
    
    ### Generate approximators for control purposes.
    # Build V
    V = build_V(cardinals, mu_list, cov_list, J, W.T, M, t)
    psi = V[:, -3].reshape((-1, 1))
    tilde_psi = V[:, -1].reshape((-1, 1))
    tilde_v = V[:, -4].reshape((-1, 1))

    # estimator from data
    #tau = np.sum(squareform(pdist(X, 'sqeuclidean')))
    #tau = tau/(p * n *(n-1))
    # theoretical value.
    tau = np.trace(sum([cardinals[i]/n * cov_list[i] for i in range(len(cov_list))]))
    tau = 2*tau/p
    beta = f(0) - f(tau) + tau*f_p(tau)
    
    A_1 = build_A_1(mu_list, t, S, tau, V)
    A_1_11 = build_A_1_11(mu_list, t, S, tau)
    A_sqrt_n = build_A_sqrt_n(t, p, V)
    A_n = build_A_n(tau, k, p, V)
    A = A_1 + A_sqrt_n + A_n

    C_n = build_C_n(A_1, A_sqrt_n, A_n, tau, gamma, W.T)
    C_sqrt_n = build_C_sqrt_n(A_sqrt_n, A_n, tau, gamma)
    C_1 = build_C_1(A_n, tau, gamma)
    C = C_n + C_sqrt_n + C_1
    
    ## $B_{22}$ in the companion paper.
    Om = K + n/gamma * np.eye(n)
    Om = np.linalg.inv(Om)
    P = np.eye(n) - 1/n * np.ones((n,n))
    L = gamma / (1 +gamma * f(tau)) * (np.eye(n) + f(tau)*gamma * P)
    Q = 2*f_p(tau)/n**2 * (A_1 + 1/p * W @ W.T + \
                2*f_p(tau)/n * A_sqrt_n @ L @ A_sqrt_n)
    E = n/gamma * (np.eye(n) - n/gamma * Om)
    E_app = f(tau)/(1 + gamma*f(tau)) * np.ones((n,n)) \
            - 1/gamma**2 * (2*f_p(tau) * L@A_sqrt_n@L \
            + L @ (n**2 * Q - beta * np.eye(n)) @ L)
    Delta = build_Delta(ind_dict)
    Delta_pos = Delta[:, 0].reshape((-1, 1))
    Delta_neg = Delta[:, 1].reshape((-1, 1))
    inv_Delta = np.concatenate([Delta_neg, -Delta_pos], axis=1)
    t = t.reshape((-1, 1))
    y = y.reshape((-1, 1))
    t_diff = np.array([[t[0,0] - t[1,0]], [t[2,0]-t[3,0]]])
    tilde_F_n = build_tilde_F_n(Delta, E)
    tilde_F_n_app = build_tilde_F_n(Delta, E_app)

    ### Build the matrix 1/n_a * Tr(C_a) on the diag
    tr_diag = np.diag(np.array([np.trace(cov_list[i]) for i in range(k)]))
    tr_diag = J @ tr_diag
    tr_diag = 1/p * np.diag(np.sum(tr_diag, axis=1))
    mat_n = 2*f_p(tau) * (1/p * J @ A_1_11 @ J.T 
            + tr_diag
            + gamma*f_p(tau)/(1+gamma*f(tau)) * 1/(2*p) * J @ t @ t.T @ J.T) \
            - beta*np.eye(n)
    mat_sqrt_n = 2*f_p(tau) * (1/p * J @ M.T @ W.T + 1/p * W @ M @ J.T 
            - f_pp(tau)/(2*f_p(tau)*np.sqrt(p)) * (psi @ t.T @ J.T + J@t@psi.T)
            + 1/p*(W @ W.T - tr_diag) 
            + gamma*f_p(tau)/(2*np.sqrt(p)*(1+gamma*f_p(tau))) * (J@t@psi.T +
                psi@t.T@ J.T)
            )
    terms = [2*f_p(tau) * (1/p * J @ M.T @ W.T + 1/p * W @ M @ J.T ),
             - 2*f_p(tau) *f_pp(tau)/(2*f_p(tau)*np.sqrt(p)) * (psi @ t.T @ J.T + J@t@psi.T),
             2*f_p(tau)/p*(W @ W.T - tr_diag),
             2*f_p(tau)*gamma*f_p(tau)/(2*np.sqrt(p)*(1+gamma*f_p(tau))) * (J@t@psi.T +
                psi@t.T@ J.T)
            ]
    tilde_F_n_app2 = - inv_Delta.T @ mat_n @ inv_Delta
    F_n = build_F_n(Delta, E_app)
    
    tilde_G_n = build_tilde_F_n(Delta, mat_n)
    tilde_G_sqrt_n = build_tilde_F_n(Delta, mat_sqrt_n)

    #sol_fair_app = np.concatenate([np.array([b_app]), params_app])
    b_app = float(1/(A_12 @ A_22_inv @ A_21)) * A_12 @ A_22_inv @ Y
    b_app = float(b_app)
    params_app = A_22_inv @ (np.eye(n+2) - 1/float(A_12 @ A_22_inv @ A_21) * A_21 @ A_12 @ A_22_inv) @ Y
    sol_fair_app = np.concatenate([np.array([b_app]), params_app])

    ### Compute b_app2
    b_app2 = ones_k.T @ n_signed / n
    det_tilde_G_n = np.linalg.det(tilde_G_n)
    b_sqrt_n = ((1+gamma*f(tau))/(gamma*f_p(tau)**2) * det_tilde_G_n + 1/p * t.T
            @ J.T @ Delta @ tilde_G_n @ Delta.T @ J @ t)
    b_sqrt_n = 1/b_sqrt_n
    b_sqrt_n = b_sqrt_n * 1/np.sqrt(p) * t.T @ J.T @ Delta @ tilde_G_n @ ( 
            2*ones_k.T @ n_signed /(n**2*p) * Delta.T @ J @ A_1_11 @ vec_prop - 2/(n*p) * Delta.T @ J @ (
                A_1_11 @ ((1+gamma*f(tau))* n_signed - gamma*f(tau)*ones_k.T @ n_signed/n * vec_prop) 
                + gamma*f_p(tau)/2 * t.T @ n_signed * t)
            )
    b_app2 = b_app2 + b_sqrt_n

    ### For control purposes.
    print("For tmp")
    ones_n = np.ones((n,1))
    
    ## Debug some quantities
    t1 = gamma*f_p(tau)/(1+gamma*f(tau)) * (1/np.sqrt(p) * Delta.T @ J @ t + Delta.T @ psi 
            - 2/(n*p) * Delta.T @ J @ A_1_11 @ vec_prop
            )
    t2 = gamma*f_p(tau)/(1+gamma*f(tau))*( ones_k.T @ n_signed / n * (1/np.sqrt(p) * Delta.T @ J @ t + Delta.T @ psi) \
            - 2/(n*p) * Delta.T@J@A_1_11@((1+gamma*f(tau)) * n_signed - gamma*f(tau)*ones_k.T @ n_signed/n * vec_prop)
            - gamma*f_p(tau)/(n*p) * t.T @ n_signed * Delta.T @ J @ t
            )
    lambdas = np.array([lambda_pos, lambda_neg]).reshape((2,1))
    lambdas_app = 1/det_tilde_G_n*tilde_G_n @ Delta.T @ (C_sqrt_n + C_n) @ (y - b_app2*ones_n)
    lambdas_app2 = 1/det_tilde_G_n*tilde_G_n @ (t2 - b*t1)
    print("lambdas: ", lambdas.reshape((-1,)))
    print("lambdas_app: ", lambdas_app.reshape((-1,)))
    print("lambdas_app2: ", lambdas_app2.reshape((-1,)))
    lambdas_list.append(lambdas)
    lambdas_app_list.append(lambdas_app)
    tmp3_list.append(lambdas - lambdas_app2)

    # debug terms
    tmp_list.append((lambdas - lambdas_app).reshape((-1,)))
    tmp2_list.append((lambdas - lambdas_app2).reshape((-1,)))
    #print("t1 real: ", Delta.T @ (C_sqrt_n + C_n) @ ones_n)
    #print("t2 real: ", Delta.T @ (C_sqrt_n + C_n) @ y)
    #print("t1 app: ", t1)
    #print("t2 app: ", t2)

#print("")
print("Results of operator norms")
print("\t lambdas - lambdas_app: ", tmp_list[1]/tmp_list[0])
print("\t lambdas - lambdas_app2: ", tmp2_list[1]/tmp2_list[0])
#print("\t approx1: ", tmp2_list[1]/tmp2_list[0])
#print("\t tmp: ", tmp3_list[1]/tmp3_list[0])
#print("\t approx2: ", tmp3_list[1]/tmp3_list[0])
#print("\t diff1: ", tmp4_list[1]/tmp4_list[0])
#print("\t diff2: ", tmp5_list[1]/tmp5_list[0])
#print("\t tmp2 : ", tmp2_list[1]/tmp2_list[0])
#print("\t tmp3 : ", tmp3_list[1]/tmp3_list[0])
#print("\t tmp4 : ", tmp4_list[1]/tmp4_list[0])
#print("\t tmp5 : ", tmp5_list[1]/tmp5_list[0])
#print("\t tmp6 : ", tmp6_list[1]/tmp6_list[0])

#print("\nValues of matrices: ")
#print("tilde_F_n:\n", tilde_F_n)
#print("tilde_F_n_app:\n", tilde_F_n_app)
#print("tilde_F_n_app2:\n", tilde_F_n_app2)
