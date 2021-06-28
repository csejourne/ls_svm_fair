import numpy as np
import scipy.linalg as sp_linalg
from functools import partial
from tools import gen_dataset, build_system_matrix, one_hot, \
            get_gaussian_kernel, decision_fair, decision_unfair, \
            comp_fairness_constraints


### Set the seed for reproducibility
np.random.seed(12)

### Generate the data
#mu_list = [np.array([1, 0, 0]), np.array([0, 1, 0])]
#cov_list = [np.eye(3), np.eye(3)]
#cardinals = [5, 4, 3, 2]
mean_scal = 5
cov_scal = 5e-1
p = 512
cardinals = [200, 400, 500, 300]
cardinals = [600, 200, 100, 500]
n = sum(cardinals)
#mu_list = [mean_scal * one_hot(0, p), mean_scal*one_hot(1, p)]
mu_pos = one_hot(0, p)
mu_neg = one_hot(1, p)
mu_list = [6 * mu_pos, 6*mu_neg]
cov_list = [cov_scal*np.eye(p), cov_scal*np.eye(p)]

### Hyperparameters
nu_neg = 1
nu_pos = 1
gamma = 1
sigma=p

X, y, sens_labels, ind_dict = gen_dataset(mu_list, cov_list, cardinals)
assert (n, p) == X.shape

### Compute tau
tau = np.trace(X @ X.T / p) / n

### Building the fair prediction function
# Get solution parameters.
matrix_fair = build_system_matrix(X, sigma, gamma, cardinals, [1, 1], ind_dict)
S = matrix_fair[3:, 3:]
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
S2 = get_gaussian_kernel(X, sigma) + n/gamma * np.eye(n)
matrix_unfair = np.zeros((n+1, n+1))
matrix_unfair[0, 1:] = 1
matrix_unfair[1:, 0] = 1
matrix_unfair[1:, 1:] = S2
rhs_unfair = np.concatenate([np.zeros(1), y])
sol_unfair = sp_linalg.solve(matrix_unfair, rhs_unfair)
# Build the fair decision function
pred_function_unfair = partial(decision_unfair, X, b, alpha, sigma)

### Compute predictions
preds_fair = pred_function_fair(X)
preds_unfair = pred_function_unfair(X)

### Compare how well both predictions function satisfy fairness properties.
pos_const_fair, neg_const_fair = comp_fairness_constraints(pred_function_fair, X, ind_dict)
pos_const_unfair, neg_const_unfair = comp_fairness_constraints(pred_function_unfair, X, ind_dict)
print(f"fair: {pos_const_fair}, {neg_const_fair}")
print(f"unfair: {pos_const_unfair}, {neg_const_unfair}")
