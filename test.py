import numpy as np
import numpy.random as rng
import pprint
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import Config
from pathlib import Path
from functools import partial
from tools import extract_W, build_V, build_A_n, build_A_sqrt_n, build_A_1, \
                build_C_1, build_C_sqrt_n, build_C_n, build_D_1, build_D_sqrt_n, \
                build_D_n, one_hot, gen_dataset

from tools import f, f_p, f_pp 

### NumPy print threshold size.
np.set_printoptions(threshold = 100)

### Set the seed for reproducibility
np.random.seed(12)

### Get default config for experiments.
conf_default = Config(Path('conf_default.json'))

### Hyperparameters
gamma = 1
mean_scal = 5
cov_scal = 1
print(f"cov_scal is {cov_scal}")

#list_cardinals = [[150, 350, 250], [300, 700, 500]]
#p_list = [512, 1024]
list_cardinals = [[300, 150, 150, 250], [600, 300, 300, 500]]
p_list = [512, 1024]
A_1_list = []
A_sqrt_n_list = []
A_n_list = []
C_1_list = []
C_sqrt_n_list = []
C_n_list = []
D_1_list = []
D_sqrt_n_list = []
D_n_list = []

for i in range(len(p_list)):
    cardinals = list_cardinals[i]
    p = p_list[i]
    print("cardinals are: ", cardinals)
    print("p is: ", p)
    k = len(cardinals)
    sigma=p
    n = sum(cardinals)
    
    mu_list = [mean_scal * one_hot(0, p), mean_scal * one_hot(0, p),
               mean_scal * one_hot(2, p), mean_scal * one_hot(2, p)]
    cov_list = [cov_scal*np.eye(p), cov_scal*np.eye(p),
                cov_scal*np.eye(p), cov_scal*np.eye(p)]
    
    X, y, sens_labels, ind_dict = gen_dataset(mu_list, cov_list, cardinals)
    # extract the noise. WARNING: scaled by 1/sqrt(p) as in KSC.
    W = extract_W(X, mu_list, cardinals)
    assert (n, p) == X.shape
    
    # need to transpose W for this function
    V = build_V(mu_list, cardinals, W.T, cov_list)
    
    ### Compute tau
    tau = np.trace(X @ X.T / p) / n
    
    A_1 = build_A_1(cardinals, mu_list, cov_list, tau, V)
    A_sqrt_n = build_A_sqrt_n(cardinals, cov_list, V)
    A_n = build_A_n(tau, len(cardinals), p, V)
    
#    P = np.eye(n) - 1/n * np.ones((n,n))
#    L = gamma/(1 + gamma*f(tau)) * (np.eye(n) + gamma * P)
#    
#    C_n = build_C_n(A_1, A_sqrt_n, A_n, tau, gamma, W.T)
#    C_sqrt_n = build_C_sqrt_n(A_sqrt_n, A_n, tau, gamma)
#    C_1 = build_C_1(A_n, tau, gamma)
#    
#    D_1 = build_D_1(C_n, C_sqrt_n, C_1, A_1, A_sqrt_n, A_n, W.T, tau)
#    D_sqrt_n = build_D_sqrt_n(C_sqrt_n, C_1, A_sqrt_n, A_n, tau)
#    D_n = build_D_n(C_1, A_n, tau)
#
#    print("For A")
    A_1_list.append(np.linalg.norm(A_1, ord=2))
    A_sqrt_n_list.append(np.linalg.norm(A_sqrt_n, ord=2))
    A_n_list.append(np.linalg.norm(A_n, ord=2))
#    print("For C")
#    C_1_list.append(np.linalg.norm(C_1, ord=2))
#    C_sqrt_n_list.append(np.linalg.norm(C_sqrt_n, ord=2))
#    C_n_list.append(np.linalg.norm(C_n, ord=2))
#    print("For D")
#    D_1_list.append(np.linalg.norm(D_1, ord=2))
#    D_sqrt_n_list.append(np.linalg.norm(D_sqrt_n, ord=2))
#    D_n_list.append(np.linalg.norm(D_n, ord=2))
#
print("Results of operator norms")
print("A_1: ", A_1_list[1]/A_1_list[0])
print("A_sqrt_n: ", A_sqrt_n_list[1]/A_sqrt_n_list[0])
print("A_n: ", A_n_list[1]/A_n_list[0])
#print("C_1: ", C_1_list[1]/C_1_list[0])
#print("C_sqrt_n: ", C_sqrt_n_list[1]/C_sqrt_n_list[0])
#print("C_n: ", C_n_list[1]/C_n_list[0])
#print("D_1: ", D_1_list[1]/D_1_list[0])
#print("D_sqrt_n: ", D_sqrt_n_list[1]/D_sqrt_n_list[0])
#print("D_n: ", D_n_list[1]/D_n_list[0])
