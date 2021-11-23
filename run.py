import numpy as np
import numpy.random as rng
import pprint
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import Config
from pathlib import Path
from functools import partial
from tools import gen_dataset, build_system_matrix, one_hot, \
            get_gaussian_kernel, decision_fair, decision_unfair, \
            comp_fairness_constraints, get_metrics, extract_W, \
            build_V, build_A_n, build_A_sqrt_n, build_A_1, build_objects


plt.rcParams.update({'font.size': 5})

### NumPy print threshold size.
np.set_printoptions(threshold = 100)

### Set the seed for reproducibility
np.random.seed(12)

### Get default config for experiments.
conf_default = Config(Path('conf_default.json'))

### Hyperparameters
mode = "strict" # how to enforce fairness constraints, can be {"strict", "relaxed"}
gamma = conf_default.gamma
gamma = 1
#mean_scal = conf_default.mean_scal
mean_scal = 3
#cov_scal = conf_default.cov_scal
cov_scal = 1
p = conf_default.p
cardinals = conf_default.cardinals
sigma=p
n = sum(cardinals)
nb_loops = 10
cardinals_test = [50, 50, 50, 50]
n_test = sum(cardinals_test)
k = len(cardinals)

### Data generation
#mu_list = [mean_scal * one_hot(0, p), mean_scal * one_hot(0,p), mean_scal*one_hot(1,p), mean_scal*one_hot(1,p)]
# different means
beta = 0.8
w1 = rng.multivariate_normal(np.zeros(p), np.eye(p))
w2 = rng.multivariate_normal(np.zeros(p), np.eye(p))
mu_list = [mean_scal*one_hot(0, p),
           beta*mean_scal*one_hot(0, p) + np.sqrt(1 - beta**2) * w1,
           mean_scal*one_hot(1, p),
           beta*mean_scal*one_hot(1, p) + np.sqrt(1 - beta**2) * w2
           ]
### Below `cov_list` was used for 
#cov_list = [cov_scal*np.eye(p), 2*cov_scal*np.eye(p),
#            cov_scal*np.eye(p), 2*cov_scal*np.eye(p)]
cov_list = [cov_scal*np.eye(p), (1 + 2/np.sqrt(p)) * cov_scal*np.eye(p),
            cov_scal*np.eye(p), (1 + 2/np.sqrt(p)) * cov_scal*np.eye(p)]

# Generate data.
X, y, sens_labels, ind_dict = gen_dataset(mu_list, cov_list, cardinals)
Y = np.concatenate([np.zeros(2), y])
assert (n, p) == X.shape
W = extract_W(X, mu_list, cardinals)
K = get_gaussian_kernel(X, sigma)
J, M, t, S = build_objects(mu_list, cardinals, cov_list)
V = build_V(cardinals, mu_list, cov_list, J, W.T, M, t)

### Compute tau
tau = np.trace(X @ X.T / p) / n

# Build the A matrix
A_1 = build_A_1(mu_list, t, S, tau, V)
A_sqrt_n = build_A_sqrt_n(t, p, V)
A_n = build_A_n(tau, k, p, V)
A = A_1 + A_sqrt_n + A_n

### Building the fair prediction function
# Get solution parameters.
matrix_fair = build_system_matrix(X, sigma, gamma, cardinals, [1, 1], ind_dict, mode=mode)
rhs_fair = np.concatenate([np.zeros(3), y])
sol_fair = sp_linalg.solve(matrix_fair, rhs_fair)
# Build the fair decision function.
b = sol_fair[0]
lambda_pos = sol_fair[1]
lambda_neg = sol_fair[2]
alpha = sol_fair[3:]
pred_function_fair = partial(decision_fair, X, b, lambda_pos, lambda_neg, alpha, sigma, ind_dict)

### Comparison with LS-SVM without fairness constraints.
# Get solution parameters
S = get_gaussian_kernel(X, sigma) + n/gamma * np.eye(n)
matrix_unfair = np.zeros((n+1, n+1))
matrix_unfair[0, 1:] = 1
matrix_unfair[1:, 0] = 1
matrix_unfair[1:, 1:] = S
rhs_unfair = np.concatenate([np.zeros(1), y])
sol_unfair = sp_linalg.solve(matrix_unfair, rhs_unfair)
b = sol_unfair[0]
alpha = sol_unfair[1:]
# Build the fair decision function
pred_function_unfair = partial(decision_unfair, X, b, alpha, sigma)
    
### For storing results
g_unfair = {}
g_unfair[("pos", 0)] = np.zeros((nb_loops, cardinals_test[0]))
g_unfair[("pos", 1)] = np.zeros((nb_loops, cardinals_test[1]))
g_unfair[("neg", 0)] = np.zeros((nb_loops, cardinals_test[2]))
g_unfair[("neg", 1)] = np.zeros((nb_loops, cardinals_test[3]))

g_fair = {}
g_fair[("pos", 0)] = np.zeros((nb_loops, cardinals_test[0]))
g_fair[("pos", 1)] = np.zeros((nb_loops, cardinals_test[1]))
g_fair[("neg", 0)] = np.zeros((nb_loops, cardinals_test[2]))
g_fair[("neg", 1)] = np.zeros((nb_loops, cardinals_test[3]))

for loop in range(nb_loops):
    ### Generate test dataset
    X_test, y_test, sens_labels_test, ind_dict_test = gen_dataset(mu_list, cov_list, cardinals_test)
    
    assert g_unfair.keys() == ind_dict_test.keys()
    ### Compute raw predictions in $\R$
    preds_fair = pred_function_fair(X_test)
    preds_unfair = pred_function_unfair(X_test)
    c2 = (cardinals[2] + cardinals[3])/n
    c1 = (cardinals[0] + cardinals[1])/n

    ### Extract the predictions for each subclass for plotting use later.
    for key in ind_dict.keys():
        inds = np.nonzero(np.squeeze(ind_dict_test[key])) 
        g_unfair[key][loop, :] = preds_unfair[inds]
    ### Extract the predictions for each subclass for plotting use later.
    for key in ind_dict.keys():
        inds = np.nonzero(np.squeeze(ind_dict_test[key])) 
        g_fair[key][loop, :] = preds_fair[inds]
    
    ### Remove threshold like Zhenyu
    preds_fair = preds_fair - (c1 - c2) # remove threshold
    preds_unfair = preds_unfair - (c1 - c2) # remove threshold
    
    ### Compare how well both predictions function satisfy fairness properties.
    pos_const_fair, neg_const_fair = comp_fairness_constraints(preds_fair, ind_dict_test)
    pos_const_unfair, neg_const_unfair = comp_fairness_constraints(preds_unfair, ind_dict_test)
    pos_const_fair_int, neg_const_fair_int = comp_fairness_constraints(preds_fair, ind_dict_test, with_int=True)
    pos_const_unfair_int, neg_const_unfair_int = comp_fairness_constraints(preds_unfair, ind_dict_test, with_int=True)
    print(f"FAIR: pos {pos_const_fair}, neg {neg_const_fair}")
    print(f"UNFAIR: pos {pos_const_unfair}, neg {neg_const_unfair}")
    print("")
    print(f"FAIR with int: pos {pos_const_fair_int}, neg {neg_const_fair_int}")
    print(f"UNFAIR with int: pos {pos_const_unfair_int}, neg {neg_const_unfair_int}\n")
    
    results_fair = get_metrics(preds_fair, y_test, ind_dict_test)
    results_unfair = get_metrics(preds_unfair, y_test, ind_dict_test)
    
    """
    ### Plot the results.
    """
    #### Classification results
    #fig, axs = plt.subplots(2)
    #suptitle = r"comparison"
    #fig.suptitle(suptitle)
    #
    ## Data
    #axs[0].plot(results_fair["gen"]["preds"], 'x:k', mfc='blue', mec='blue', markersize=2, linewidth=0.4)
    #axs[0].set_title("Fair classifier")
    #axs[1].plot(results_unfair["gen"]["preds"], 'x:k', mfc='blue', mec='blue', markersize=2, linewidth=0.4)
    #axs[1].set_title("Unfair classifier")
    #
    ## Markers and all
    #tmp = np.cumsum(np.array(cardinals_test))
    #n_test = np.sum(np.array(cardinals_test))
    #axs[0].plot(np.zeros(n_test), '-k', linewidth=0.5)
    #axs[1].plot(np.zeros(n_test), '-k', linewidth=0.5)
    #axs[0].plot(tmp[0], 0, '|r')
    #axs[0].plot(tmp[0], 0, '|r')
    #axs[0].plot(tmp[1], 0, '|r')
    #axs[0].plot(tmp[2], 0, '|r')
    #axs[1].plot(tmp[0], 0, '|r')
    #axs[1].plot(tmp[0], 0, '|r')
    #axs[1].plot(tmp[1], 0, '|r')
    #axs[1].plot(tmp[2], 0, '|r')
    #fig.savefig(f"results/strict/classif_{loop}.pdf")
    
### Predictions distributions.
fig, axs = plt.subplots(2)
axs[0].hist(g_fair[('pos', 0)].flatten(), 50, facecolor='blue', alpha=0.4)
axs[0].hist(g_fair[('pos', 1)].flatten(), 50, facecolor='green', alpha=0.4)
axs[0].hist(g_fair[('neg', 0)].flatten(), 50, facecolor='red', alpha=0.4)
axs[0].hist(g_fair[('neg', 1)].flatten(), 50, facecolor='yellow', alpha=0.4)
axs[0].axvline(x=c1-c2, color='red')
axs[0].set_title("fair LS-SVM")

axs[1].hist(g_unfair[('pos', 0)].flatten(), 50, facecolor='blue', alpha=0.4)
axs[1].hist(g_unfair[('pos', 1)].flatten(), 50, facecolor='green', alpha=0.4)
axs[1].hist(g_unfair[('neg', 0)].flatten(), 50, facecolor='red', alpha=0.4)
axs[1].hist(g_unfair[('neg', 1)].flatten(), 50, facecolor='yellow', alpha=0.4)
axs[1].axvline(x=c1-c2, color='red')
axs[1].set_title("unfair LS-SVM")

# Create legend
handles = [Rectangle((0, 0), 1,1,color=c) for c in ['blue', 'green', 'red', 'yellow']]
labels=["Y = 1, A = 0", "Y = 1, A = 1", "Y = -1, A = 0", "Y = -1, A = 1"]
fig.legend(handles, labels)
fig.savefig("results/distribs/distrib.pdf")
plt.show()
