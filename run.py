import numpy as np
import pickle as pk
import numpy.random as rng
import pprint
import scipy.linalg as sp_linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import Config
from pathlib import Path
from functools import partial
from scipy.spatial.distance import cdist, pdist, squareform
from tools import extract_W, build_V, build_A_n, build_A_sqrt_n, build_A_1, build_A_1_11, \
                build_C_1, build_C_sqrt_n, build_C_n, build_D_1, build_D_sqrt_n, \
                build_D_n, one_hot, gen_dataset, get_gaussian_kernel, build_system_matrix, \
                build_Delta, build_F_n, build_tilde_F_n, decision_fair, decision_unfair, \
                comp_fair_expec_constraints, comp_fair_prob_constraints, \
                get_metrics, missclass_errors_theo, missclass_errors_exp, \
                missclass_errors_zh, tot_errors


from tools import f, f_p, f_pp, build_objects
from sys import exit

""" TODO
- check formulae for theoretical curves
"""

### NumPy print threshold size.
np.set_printoptions(threshold = 100)

### Some flags
save_arr = False
save_arr_test = False
get_arr = False
get_bk = False
do_test = True
plot_test = True

"""
We iterate over a high number of experiences to better evaluate the orders.
"""
nb_iter = 1
nb_tests = 1
nb_loops_test = 50
cardinals_test = [500, 500, 500, 500] # the base, we apply a coefficient later in the code
n_test = sum(cardinals_test)

### Lists for debuggin purposes
h_lambdas_list = []
h_b_list = []
h_alpha_list = []
h_alpha_app_list = []
h_alpha_diff_list = []
h_alpha_stds = []
h_lambdas_list = []
h_lambdas_app_list = []
h_lambdas_app2_list = []
h_lambdas_diff_list = []
h_lambdas_diff2_list = []
h_b_list = []
h_b_app_list = []
h_b_diff_list = []
h_num_list = []
h_denom_list = []
h_num_app_list = []
h_denom_app_list = []
h_num_diff_list = []
h_denom_diff_list = []
h_means_list = []
h_means_exp_list = []
h_means_zh_list = []
h_diff_means_list = []
h_means_exp_app_list = []
h_diff_means_app_list = []
h_varis_list = []
h_varis_exp_list = []
h_varis_zh_list = []
h_diff_varis_list = []
h_varis_exp_app_list = []
h_diff_varis_app_list = []
h_diff_means_exp_list = []
h_diff_std_list = []

for id_iter in range(nb_iter):
    print(f"iter {id_iter + 1}/{nb_iter}")
    ### Set the seed for reproducibility
    #np.random.seed(12)
    #print("numpy state is: ", rng.get_state())
    
    ### Get default config for experiments.
    conf_default = Config(Path('conf_default.json'))
    
    ### Hyperparameters
    mode = "strict"
    gamma = 1
    mean_scal = 3
    cov_scal = 1
    print("Experiment begin")
    #print(f"cov_scal is {cov_scal}")
    cardinals_list = [[42, 22, 66, 116]]
    #cardinals_list = [[61, 61, 61, 61]]
    #cardinals_list = [[60, 60, 60, 60], [120, 120, 120, 120]]
    #p_list = [256, 512]
    #cardinals_list = [[30, 30, 30, 30], [60, 60, 60, 60]]
    #p_list = [128, 256]
    #cardinals_list = [[60, 60, 60, 60]]
    p_list = [256]
    
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
    alpha_list = []
    alpha_app_list = []
    alpha_app2_list = []
    alpha_diff_list = []
    lambdas_list = []
    lambdas_app_list = []
    lambdas_app2_list = []
    lambdas_diff_list = []
    lambdas_diff2_list = []
    b_list = np.array([])
    b_app_list = np.array([])
    b_diff_list = np.array([])
    num_list = []
    denom_list = []
    num_app_list = []
    denom_app_list = []
    num_app2_list = []
    denom_app2_list = []
    num_diff_list = []
    denom_diff_list = []
    diff_means_exp_list = []
    
    
    for i in range(len(p_list)):
        #cardinals_test = np.array(cardinals_test) * np.array(cardinals_list[i])/np.sum(cardinals_list[i])
        #cardinals_test = list(np.rint(cardinals_test).astype(int))
        #print("cardinals_test is: ", cardinals_test)

        ### for storing test purposes
        # Means
        means_list = []
        means_zh_list = []
        means_exp_list = []
        diff_means_list = []
        diff_means_zh_list = []
        means_exp_app_list = []
        diff_means_app_list = []
        # Variances
        varis_list = []
        varis_zh_list = []
        varis_exp_list = []
        diff_varis_list = []
        diff_varis_zh_list = []
        varis_exp_app_list = []
        diff_varis_app_list = []
        # Theoretical errors
        errors_th = []
        # Experimental errors
        errors_exp = []
        errors_unfair = []
        # Zhenyu errors
        errors_zh = []

        # Class distribution
        cardinals = cardinals_list[i]
        p = p_list[i]
        print("\n\tcardinals: ", cardinals, "  ||  p: ", p)
        k = len(cardinals)
        n = sum(cardinals)
        vec_prop = np.array(cardinals).reshape((-1, 1))
        sigma=p

        ### Fairness setup
        beta_pos = 0.9
        beta_neg = beta_pos
        noise_pos = rng.multivariate_normal(np.zeros(p), np.eye(p))
        noise_neg = rng.multivariate_normal(np.zeros(p), np.eye(p))
        mu_list = [mean_scal * one_hot(0, p),
                   beta_pos**2 * mean_scal * one_hot(0, p) + np.sqrt(1 - beta_pos**2) * noise_pos,
                   mean_scal * one_hot(1, p),
                   beta_neg**2 * mean_scal * one_hot(1, p) + np.sqrt(1 - beta_neg**2) * noise_neg
                   ]
        #mu_list = [mean_scal * one_hot(0, p), mean_scal * one_hot(0, p),
        #           mean_scal * one_hot(1, p), mean_scal * one_hot(1, p)]
        #mu_list = [mean_scal * one_hot(0, p), mean_scal * one_hot(1, p),
        #           mean_scal * one_hot(2, p), mean_scal * one_hot(3, p)]

        #cov_list = [cov_scal*np.eye(p),  cov_scal*np.eye(p),
        #            (1+2/np.sqrt(p)) * cov_scal*np.eye(p),  (1+2/np.sqrt(p)) * cov_scal*np.eye(p)]
        #cov_list = [cov_scal*np.eye(p), (1 + 2/np.sqrt(p)) * cov_scal*np.eye(p),
        #            cov_scal*np.eye(p), (1 + 2/np.sqrt(p)) * cov_scal*np.eye(p)]
        # covariances from zhenyu's paper
        col = np.array([0.4**l for l in range(p)])
        C_2 = (1+2/np.sqrt(p))*sp_linalg.toeplitz(col, col.T)
        #cov_list = [np.eye(p), np.eye(p), C_2, C_2]
        cov_list = [np.eye(p), C_2, np.eye(p), C_2]
        #cov_list = [(1 + 2/np.sqrt(p))*np.eye(p), C_2, (1 + 2/np.sqrt(p))*np.eye(p), C_2]
        
        ### Generate data.
        X, y, sens_labels, ind_dict = gen_dataset(mu_list, cov_list, cardinals)
        assert (n, p) == X.shape
        W = extract_W(X, mu_list, cardinals)
        K = get_gaussian_kernel(X, sigma)
        J, M, t, S = build_objects(mu_list, cardinals, cov_list)
        mu_diff_dist = squareform(pdist(M.T, metric='sqeuclidean')) 
        ones_k = np.ones((k, 1))
        n_signed = np.array([cardinals[0], cardinals[1], -cardinals[2], -cardinals[3]]).reshape((-1, 1))
        T = t.reshape((-1, 1)) @ ones_k.T + ones_k @ t.reshape((1, -1))

        # Build the unfair objects
        matrix_unfair = np.zeros((n+1, n+1))
        matrix_unfair[0, 1:] = 1
        matrix_unfair[1:, 0] = 1
        matrix_unfair[1:, 1:] = K + n/gamma * np.eye(n)
        rhs_unfair = np.concatenate([np.zeros(1), y])
        sol_unfair = sp_linalg.solve(matrix_unfair, rhs_unfair)
        b_unfair = sol_unfair[0]
        alpha_unfair = sol_unfair[1:]

        ### Fair solution
        matrix_fair = build_system_matrix(X, sigma, gamma, cardinals, [1, 1], ind_dict, mode=mode)
        rhs_fair = np.concatenate([np.zeros(3), y])
        sol_fair = sp_linalg.solve(matrix_fair, rhs_fair)
        # Build the fair objects
        b = sol_fair[0]
        lambda_pos = sol_fair[1]
        lambda_neg = sol_fair[2]
        alpha = sol_fair[3:]
        alpha = alpha.reshape((-1, 1))
        B_11 = matrix_fair[1:3, 1:3]
        B_12 = matrix_fair[1:3, 3:]
        B_21 = matrix_fair[3:, 1:3]
        B_22 = matrix_fair[3:, 3:]
        A_12 = matrix_fair[0, 1:].reshape((1, -1))
        A_21 = matrix_fair[1:, 0].reshape((-1, 1))
        A_22 = matrix_fair[1:, 1:]
        
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
        ones_n = np.ones((n,1))
        
        A_1 = build_A_1(mu_list, t, S, tau, V)
        A_1_11 = build_A_1_11(mu_list, t, S, tau)
        A_sqrt_n = build_A_sqrt_n(t, p, V)
        A_n = build_A_n(tau, k, p, V)
        A = A_1 + A_sqrt_n + A_n
    
        C_n = build_C_n(A_1, A_sqrt_n, A_n, tau, gamma, W.T)
        C_sqrt_n = build_C_sqrt_n(A_sqrt_n, A_n, tau, gamma)
        C = C_n + C_sqrt_n
        
        ## $B_{22}$ in the companion paper.
        P = np.eye(n) - 1/n * np.ones((n,n))
        L = gamma / (1 +gamma * f(tau)) * (np.eye(n) + f(tau)*gamma * P)
        Q = 2*f_p(tau)/n**2 * (A_1 + 1/p * W @ W.T + \
                    2*f_p(tau)/n * A_sqrt_n @ L @ A_sqrt_n)
        #E = n/gamma * (np.eye(n) - n/gamma * Om)
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
        F_n = build_F_n(Delta, E_app)
        tilde_F_n_app = build_tilde_F_n(Delta, E_app)
        tilde_G_n = build_tilde_F_n(Delta, mat_n)
        tilde_G_sqrt_n = build_tilde_F_n(Delta, mat_sqrt_n)
        det_tilde_G_n = np.linalg.det(tilde_G_n)
        det_tilde_F_n_app = np.linalg.det(tilde_F_n_app)
        factor = ones_k.T @ n_signed / n

        ###### Computes approximations
        ### Compute approximation of `b`
        b_sqrt_n = (1 + gamma*f(tau))/(gamma*f_p(tau)**2) * det_tilde_G_n + 1/p * t.T@J.T@Delta@tilde_G_n@Delta.T @ J @ t
        b_sqrt_n = 1/b_sqrt_n
        b_sqrt_n = b_sqrt_n * (
                1/np.sqrt(p) * t.T @ J.T @ Delta @ tilde_G_n @ (
                2*(1+gamma*f(tau))/(n*p) * Delta.T @ J @ A_1_11 @ (factor*vec_prop - n_signed) + gamma*f_p(tau)/(n*p) * t.T @ n_signed * Delta.T @J@t)
                - (1+gamma*f(tau))*det_tilde_G_n * t.T @ n_signed / (n*f_p(tau)*np.sqrt(p))
                )
        b_app = ones_k.T @ n_signed / n + b_sqrt_n

        ### Compute approximation of $\lambda_1, \lambda_{-1}$
        # Build the R_n vector
        R_n = 1/det_tilde_G_n * tilde_G_n @ (gamma*f_p(tau)/(1+gamma*f(tau)) *
                (
                2/(n*p) * (1+gamma*f(tau))* Delta.T @ J @ A_1_11 @ (factor*vec_prop - n_signed) 
                - gamma*f_p(tau)/(n*p) * t.T @ n_signed * Delta.T @ J @ t 
                - b_sqrt_n/np.sqrt(p) * Delta.T @ J @ t)
                )
        
        ### Compute approximation of `alpha`
        alpha_n_3_2 = -(gamma*b_sqrt_n*1/(1+gamma*f(tau))*1/n 
                + 1/n * gamma*f_p(tau)/(1+gamma*f(tau)) * 1/np.sqrt(p) * t.T @ J.T @ Delta @ R_n 
                + gamma**2*f_p(tau)/(1+gamma*f(tau)) * t.T @ n_signed / (np.sqrt(p)*n**2)
                )

        alpha_app = gamma/n * P @ y + alpha_n_3_2 *ones_n

        ### Compute the expectations `E_a`
        D_cal = (gamma*f_p(tau)/n * y.T @ P + f_p(tau)*R_n.T @ Delta.T) @ psi \
                + (gamma*f_pp(tau)/(2*n)* y.T @ P + f_pp(tau)/2*R_n.T @ Delta.T) @ psi**2 \
                + (gamma*f_pp(tau)/(2*n*p)* (n_signed - factor*vec_prop).T 
                        + f_pp(tau)/(2*p) * R_n.T @ Delta.T @ J) @ t**2

        ### own formulae
        expecs = factor * np.ones((k, 1)) \
                + b_sqrt_n \
                + D_cal
        varis = np.zeros((k,1))
        y_list = np.array([1, 1, -1, -1])
        sens_list = np.array([0, 1, 0, 1])

        ### For zhenyu's formulae
        c2 = (cardinals[2] + cardinals[3])/n
        c1 = (cardinals[0] + cardinals[1])/n
        expecs_zh = (c2-c1)*np.ones((k,1))
        varis_zh = np.zeros((k,1))
        D_cal_zh = -2*f_p(tau)/p * np.linalg.norm(mu_list[0]-mu_list[2])**2 \
                + f_pp(tau)/p**2 * np.trace(cov_list[0]-cov_list[2])**2 \
                + 2*f_pp(tau)/p**2 * np.trace((cov_list[0]-cov_list[2]) @ (cov_list[0] - cov_list[2]))
        expecs_zh[0] += -2*c2**2*c1 * D_cal_zh
        expecs_zh[1] += -2*c2**2*c1 * D_cal_zh
        expecs_zh[2] += 2*c1**2*c2 * D_cal_zh
        expecs_zh[3] += 2*c1**2*c2 * D_cal_zh

        # TODO: NOTE: tau depends on the covariances. If cardinals are equal, its ok, but be careful
        print("normal covariances")
        #print("inverted covariances")
        #cov_list_old = [cov_list[0], cov_list[1], cov_list[2], cov_list[3]]
        #cov_list = [cov_list_old[1], cov_list_old[0], cov_list_old[3], cov_list_old[2]]
        for a in range(k):
            """
            Formulae for fair LS-SVM
            """
            ### Compute the expectations `E_a`
            tmp = (gamma*f_pp(tau)*t.T @ n_signed/(2*n*np.sqrt(p)) 
                    + n*alpha_n_3_2 * f_p(tau) 
                    + f_pp(tau)/np.sqrt(p) * R_n.T @ Delta.T @ J @ t) * np.trace(cov_list[a])/p
            D_cal_x = (gamma*f_p(tau)/(n*p)*(n_signed - factor*vec_prop).T 
                            + f_p(tau)/p * R_n.T @ Delta.T @ J)@mu_diff_dist[:, a] \
                + (2*gamma*f_pp(tau)/(n*p) * (n_signed - factor*vec_prop).T 
                            + 2*f_pp(tau)/p*R_n.T @ Delta.T @ J)@ S[:, a] \
                + (gamma*f_pp(tau)/(2*n*p)* t.T @ n_signed 
                            + n*alpha_n_3_2 *f_p(tau)/np.sqrt(p)
                            + f_pp(tau)/p * R_n.T @ Delta.T @ J @ t)*t[a]
            expecs[a] += np.squeeze(tmp + D_cal_x)

            ### Compute the variances `Var_a`
            mu_diff = np.stack([mu_list[b] - mu_list[a] for b in range(k)]).T
            tmp = 2/p**2 * (gamma*f_pp(tau)/(2*n*np.sqrt(p))*t.T @ n_signed 
                                + n*alpha_n_3_2*f_p(tau))**2 * np.trace(cov_list[a] @ cov_list[a])
            varis[a] += np.squeeze(tmp)

            tmp = (2*gamma*f_p(tau)/(n*p) * (n_signed - factor*vec_prop).T + 2*f_p(tau)/p*R_n.T@Delta.T@J) @ mu_diff.T @ cov_list[a] @ mu_diff @ (2*gamma*f_p(tau)/(n*p) * (n_signed - factor*vec_prop) + 2*f_p(tau)/p*J.T @ Delta @ R_n)
            varis[a] += np.squeeze(tmp)

            tmp = [(gamma**2/n**2*vec_prop[b]* (1 + factor**2 - 2*y_list[b]*factor) 
                    + R_n[int(np.floor(b/2))]**2/vec_prop[b]
                    + 2*gamma/n * (y_list[b] - factor)*(-1)**sens_list[b]*R_n[int(np.floor(b/2))]
                    ) *S[a,b] for b in range(k)]
            varis[a] += (2*f_p(tau))**2/p * np.sum(tmp)

            """
            For Zhenyu binary LS_SVM
            """
            nu_a1 = (f_pp(tau)/p**2)**2 * np.trace(cov_list[0] - cov_list[2])**2 * np.trace(cov_list[a]@cov_list[a])
            nu_a2 = 2*f_p(tau)**2/p**2 * (mu_list[2] - mu_list[0]).T @ cov_list[a] @ (mu_list[2] - mu_list[0])
            nu_a3 = 2*f_p(tau)**2/(n*p**2) * (np.trace(cov_list[0]@cov_list[a])/c1 + np.trace(cov_list[2]@cov_list[a]))
            varis_zh[a] = 8*gamma**2 * c1**2 * c2**2 *(nu_a1 + nu_a2 + nu_a3)

        """ Store approximations
        """
        cumsum_cardinals = np.append(0, np.cumsum(cardinals))
        b_list = np.append(b_list, b)
        b_app_list = np.append(b_app_list, b_app)
        b_diff_list = np.append(b_diff_list, b - b_app)
        alpha_list.append(np.copy(alpha))
        alpha_app_list.append(np.copy(alpha_app))
        alpha_diff_list.append(np.copy(alpha - alpha_app))
        lambdas_list.append(np.array([lambda_pos, lambda_neg]))
        lambdas_app_list.append(np.copy(np.squeeze(R_n)))
        lambdas_app2_list.append(np.squeeze(- F_n @ Delta.T @ (C_sqrt_n + C_n) @ (y-b*ones_n)))
        lambdas_diff_list.append(np.array([lambda_pos, lambda_neg]) - np.squeeze(R_n))
        lambdas_diff2_list.append(np.array([lambda_pos, lambda_neg]) - np.squeeze(F_n @ Delta.T @ (C_sqrt_n + C_n) @ (y-b*ones_n)))

        """
        Classify new data after fitting the fair LS-SVM
        """
        if do_test:
            for id_test in range(nb_tests):
                print(f"Doing test {(id_test+1)/nb_tests}")
                pred_function_fair = partial(decision_fair, X, b, lambda_pos, lambda_neg, alpha, sigma, ind_dict)
                pred_function_fair_app = partial(decision_fair, X, b_app, R_n[0,0], R_n[1,0], alpha_app, sigma, ind_dict)
                pred_function_unfair = partial(decision_unfair, X, b_unfair, alpha_unfair, sigma)

                ### For storing results
                g_fair = {}
                g_fair[("pos", 0)] = np.zeros((nb_loops_test, cardinals_test[0]))
                g_fair[("pos", 1)] = np.zeros((nb_loops_test, cardinals_test[1]))
                g_fair[("neg", 0)] = np.zeros((nb_loops_test, cardinals_test[2]))
                g_fair[("neg", 1)] = np.zeros((nb_loops_test, cardinals_test[3]))

                g_fair_app = {}
                g_fair_app[("pos", 0)] = np.zeros((nb_loops_test, cardinals_test[0]))
                g_fair_app[("pos", 1)] = np.zeros((nb_loops_test, cardinals_test[1]))
                g_fair_app[("neg", 0)] = np.zeros((nb_loops_test, cardinals_test[2]))
                g_fair_app[("neg", 1)] = np.zeros((nb_loops_test, cardinals_test[3]))

                g_unfair = {}
                g_unfair[("pos", 0)] = np.zeros((nb_loops_test, cardinals_test[0]))
                g_unfair[("pos", 1)] = np.zeros((nb_loops_test, cardinals_test[1]))
                g_unfair[("neg", 0)] = np.zeros((nb_loops_test, cardinals_test[2]))
                g_unfair[("neg", 1)] = np.zeros((nb_loops_test, cardinals_test[3]))
                
                for loop in range(nb_loops_test):
                    print(f"\t test {(loop+1)/nb_loops_test}")

                    ### Generate test dataset
                    X_test, y_test, sens_labels_test, ind_dict_test = gen_dataset(mu_list, cov_list, cardinals_test)

                    assert g_fair.keys() == ind_dict_test.keys()
                    ### Compute raw predictions in $\R$
                    #NOTE: preds_fair_app may not be useful, as we do approximations only for the theoretical curves?
                    preds_fair = pred_function_fair(X_test)
                    preds_fair_app = pred_function_fair_app(X_test)
                    preds_unfair = pred_function_unfair(X_test)
                    c2 = (cardinals[2] + cardinals[3])/n
                    c1 = (cardinals[0] + cardinals[1])/n
                
                    ### Extract the predictions for each subclass for plotting use later.
                    for key in ind_dict_test.keys():
                        inds = np.nonzero(np.squeeze(ind_dict_test[key])) 
                        g_fair[key][loop] = preds_fair[inds]
                        g_fair_app[key][loop] = preds_fair_app[inds]
                        g_unfair[key][loop] = preds_unfair[inds]
                    
                    ### Remove threshold like Zhenyu
                    # Test different formula for the threshold. TODO: check which one should be theoretically (remove b_sqrt_n ?)
                    #threshold = float(c1 - c2 - b_sqrt_n)
                    threshold = float(c1 - c2)
                    preds_fair = preds_fair - (threshold) # remove threshold
                    preds_fair_app = preds_fair_app - (threshold- b_sqrt_n) # remove threshold
                    preds_unfair = preds_unfair - (threshold) # remove threshold
                    
                    ### Compare how well both predictions function satisfy fairness properties.
                    #TODO: change the following function so we can use all the loops.
                    results_fair = get_metrics(preds_fair, y_test, ind_dict_test)
                    results_unfair = get_metrics(preds_unfair, y_test, ind_dict_test)
                    fair_expecs_const = comp_fair_expec_constraints(preds_fair, ind_dict_test)
                    fair_expecs_const_int = comp_fair_expec_constraints(preds_fair, ind_dict_test, with_int=True)

                ### Study the results
                # Means
                means_exp = np.array([np.mean(g_fair[('pos', 0)].flatten()), 
                            np.mean(g_fair[('pos', 1)].flatten()),
                            np.mean(g_fair[('neg', 0)].flatten()),
                            np.mean(g_fair[('neg', 1)].flatten())])
                means_exp_app = np.array([np.mean(g_fair_app[('pos', 0)].flatten()), 
                            np.mean(g_fair_app[('pos', 1)].flatten()),
                            np.mean(g_fair_app[('neg', 0)].flatten()),
                            np.mean(g_fair_app[('neg', 1)].flatten())])
                diff_means_list.append(np.copy(np.squeeze(expecs) - means_exp))
                diff_means_zh_list.append(np.copy(np.squeeze(expecs_zh) + means_exp))
                diff_means_app_list.append(np.copy(np.squeeze(expecs) - means_exp_app))
                means_exp_list.append(np.copy(means_exp))
                means_exp_app_list.append(np.copy(means_exp_app))
                means_list.append(np.copy(np.squeeze(expecs)))
                means_zh_list.append(np.copy(np.squeeze(expecs_zh)))
                const_bias = np.mean(diff_means_list)

                # Variances
                varis_exp = np.array([np.var(g_fair[('pos', 0)].flatten()), 
                            np.var(g_fair[('pos', 1)].flatten()),
                            np.var(g_fair[('neg', 0)].flatten()),
                            np.var(g_fair[('neg', 1)].flatten())])
                varis_exp_app = np.array([np.var(g_fair_app[('pos', 0)].flatten()), 
                            np.var(g_fair_app[('pos', 1)].flatten()),
                            np.var(g_fair_app[('neg', 0)].flatten()),
                            np.var(g_fair_app[('neg', 1)].flatten())])
                diff_varis_list.append(np.copy(np.squeeze(varis) - varis_exp))
                diff_varis_zh_list.append(np.copy(np.squeeze(varis_zh) - varis_exp))
                diff_varis_app_list.append(np.copy(np.squeeze(varis) - varis_exp_app))
                varis_exp_list.append(np.copy(varis_exp))
                varis_exp_app_list.append(np.copy(varis_exp_app))
                varis_list.append(np.copy(np.squeeze(varis)))
                varis_zh_list.append(np.copy(np.squeeze(varis_zh)))

                # Errors
                errors_th.append(missclass_errors_theo(expecs - const_bias, varis, threshold)) #test with not constant but class depend bias.
                errors_exp.append(missclass_errors_exp(g_fair, threshold))
                errors_unfair.append(missclass_errors_exp(g_unfair, threshold))
                errors_zh.append(missclass_errors_zh(-expecs_zh,varis_zh, threshold))

            ### Storing results.
            # means
            h_diff_means_list.append(np.array(diff_means_list))
            h_means_exp_list.append(np.array(means_exp_list))
            h_diff_means_app_list.append(np.array(diff_means_app_list))
            h_means_exp_app_list.append(np.array(means_exp_app_list))
            h_means_list.append(np.array(means_list))
            h_diff_means_exp_list.append(np.squeeze(np.array(
                    [means_exp[1] - means_exp[0], means_exp[3] - means_exp[2]]
                    )))
            # variances
            h_varis_list.append(np.array(varis_list))
            h_diff_varis_list.append(np.array(diff_varis_list))
            h_varis_exp_list.append(np.array(varis_exp_list))
            h_diff_varis_app_list.append(np.array(diff_varis_app_list))
            h_varis_exp_app_list.append(np.array(varis_exp_app_list))
            h_means_list.append(np.array(means_list))
            #std
            h_diff_std_list.append(np.squeeze(np.array(
                    [np.sqrt(varis[0]) - np.sqrt(varis_exp[0]),
                     np.sqrt(varis[1]) - np.sqrt(varis_exp[1]),
                     np.sqrt(varis[2]) - np.sqrt(varis_exp[2]),
                     np.sqrt(varis[3]) - np.sqrt(varis_exp[3])]
                    )))

            """ Plot the results.
            """
            hists = {}
            if plot_test:
                ### Predictions distributions.
                fig, axs = plt.subplots(3)
                hists[('pos', 0)] = axs[0].hist(g_fair[('pos', 0)].flatten(), 50, facecolor='blue',
                                alpha=0.4, density=True, stacked=True, edgecolor='black', linewidth=1.2)
                hists[('pos', 1)] = axs[0].hist(g_fair[('pos', 1)].flatten(), 50, facecolor='green',
                                alpha=0.4, density=True, stacked=True, edgecolor='black', linewidth=1.2)
                hists[('neg', 0)] = axs[0].hist(g_fair[('neg', 0)].flatten(), 50, facecolor='red',
                                alpha=0.4, density=True, stacked=True, edgecolor='black', linewidth=1.2)
                hists[('neg', 1)] = axs[0].hist(g_fair[('neg', 1)].flatten(), 50, facecolor='yellow',
                                alpha=0.4, density=True, stacked=True, edgecolor='black', linewidth=1.2)
                #axs[0].axvline(x=threshold, color='red')
                axs[0].axvline(x=c1 - c2, color='red')
                axs[0].set_title("fair LS-SVM")

                axs[1].hist(g_fair_app[('pos', 0)].flatten(), 50, facecolor='blue', alpha=0.4,
                                density=True, stacked=True, edgecolor='black', linewidth=1.2)
                axs[1].hist(g_fair_app[('pos', 1)].flatten(), 50, facecolor='green', alpha=0.4,
                                density=True, stacked=True, edgecolor='black', linewidth=1.2)
                axs[1].hist(g_fair_app[('neg', 0)].flatten(), 50, facecolor='red', alpha=0.4,
                                density=True, stacked=True, edgecolor='black', linewidth=1.2)
                axs[1].hist(g_fair_app[('neg', 1)].flatten(), 50, facecolor='yellow', alpha=0.4,
                                density=True, stacked=True, edgecolor='black', linewidth=1.2)
                axs[1].axvline(x=threshold, color='red')
                axs[1].set_title("fair_app LS-SVM")
                
                axs[2].hist(g_unfair[('pos', 0)].flatten(), 50, facecolor='blue', alpha=0.4,
                                density=True, stacked=True, edgecolor='black', linewidth=1.2)
                axs[2].hist(g_unfair[('pos', 1)].flatten(), 50, facecolor='green', alpha=0.4,
                                density=True, stacked=True, edgecolor='black', linewidth=1.2)
                axs[2].hist(g_unfair[('neg', 0)].flatten(), 50, facecolor='red', alpha=0.4,
                                density=True, stacked=True, edgecolor='black', linewidth=1.2)
                axs[2].hist(g_unfair[('neg', 1)].flatten(), 50, facecolor='yellow', alpha=0.4,
                                density=True, stacked=True, edgecolor='black', linewidth=1.2)
                axs[2].axvline(x=threshold, color='red')
                axs[2].set_title("unfair LS-SVM")
                
                ### Associated theoretical gaussian
                colors = ['b', 'g', 'r', 'y']
                space = np.linspace(threshold - 8*np.sqrt(varis[0]), threshold + 8*np.sqrt(varis[0]), 300)
                for a in range(k):
                    axs[0].plot(space, stats.norm.pdf(space, expecs[a], np.sqrt(varis[a])), color=colors[a])
                    axs[0].plot(space, stats.norm.pdf(space, means_exp[a], np.sqrt(varis[a])), color=colors[a], linestyle='-.')
                    axs[1].plot(space, stats.norm.pdf(space, expecs[a], np.sqrt(varis[a])), color=colors[a])
                    axs[1].plot(space, stats.norm.pdf(space, means_exp[a], np.sqrt(varis[a])), color=colors[a], linestyle='-.')
                    axs[2].plot(space, stats.norm.pdf(space, - expecs_zh[a], np.sqrt(varis_zh[a])), color=colors[a])
                
                # Create legend
                handles = [Rectangle((0, 0), 1,1,color=c) for c in ['blue', 'green', 'red', 'yellow']]
                labels=["Y = 1, A = 0", "Y = 1, A = 1", "Y = -1, A = 0", "Y = -1, A = 1"]
                fig.legend(handles, labels)
                fig.savefig("results/distribs/zhenyu_setup_fair_formulae.pdf")
                plt.show()

    ### Store history of iterations.
    cumsum_cardinals = [np.append(0, np.cumsum(cardinals_list[i])) for i in range(len(p_list))]
    h_b_list.append(b_list)
    h_b_app_list.append(b_app_list)
    h_b_diff_list.append(b_diff_list)
    h_lambdas_list.append(lambdas_list)
    h_lambdas_app_list.append(lambdas_app_list)
    h_lambdas_app2_list.append(lambdas_app2_list)
    h_lambdas_diff_list.append(lambdas_diff_list)
    h_lambdas_diff2_list.append(lambdas_diff2_list)

    ### Build the standard deviation of the coefficient of `alpha`
    alpha_stds = []
    for a in range(len(cumsum_cardinals)):
        idxs = cumsum_cardinals[a]
        alpha_stds.append(np.array([np.std(alpha_diff_list[a][idxs[i-1]:idxs[i]]) for i in range(1, len(idxs))]))
    h_alpha_stds.append(alpha_stds)

""" For saving array or restoring saved arrays
"""
if save_arr_test:
    print("Saving test arrays")
    with open("results/h_diff_means_list.pk", "wb") as fi:
        pk.dump(h_diff_means_list, fi)
    with open("results/h_means_exp_list.pk", "wb") as fi:
        pk.dump(h_means_exp_list, fi)
    with open("results/h_means_list.pk", "wb") as fi:
        pk.dump(h_means_list, fi)

if save_arr:
    with open("results/h_lambdas_list.pk", "wb") as fi:
        pk.dump(h_lambdas_list, fi)
    with open("results/h_lambdas_diff_list.pk", "wb") as fi:
        pk.dump(h_lambdas_diff_list, fi)
    with open("results/h_lambdas_diff2_list.pk", "wb") as fi:
        pk.dump(h_lambdas_diff2_list, fi)
    with open("results/h_lambdas_app_list.pk", "wb") as fi:
        pk.dump(h_lambdas_app_list, fi)
    with open("results/h_lambdas_app2_list.pk", "wb") as fi:
        pk.dump(h_lambdas_app2_list, fi)

    with open("results/diff_lambdas_pos.pk", "wb") as fi:
        pk.dump(diff_lambdas_pos, fi)
    with open("results/diff_lambdas_neg.pk", "wb") as fi:
        pk.dump(diff_lambdas_neg, fi)
    with open("results/vals_lambdas_pos.pk", "wb") as fi:
        pk.dump(vals_lambdas_pos, fi)
    with open("results/vals_lambdas_neg.pk", "wb") as fi:
        pk.dump(vals_lambdas_neg, fi)

if get_arr:
    with open("results/h_lambdas_list.pk", "wb") as fi:
        h_lambdas_list = pk.load(fi)
    with open("results/h_lambdas_diff_list.pk", "wb") as fi:
        h_lambdas_diff_list = pk.load(fi)
    with open("results/h_lambdas_diff2_list.pk", "wb") as fi:
        h_lambdas_diff2_list = pk.load(fi)
    with open("results/h_lambdas_app_list.pk", "wb") as fi:
        h_lambdas_app_list = pk.load(fi)
    with open("results/h_lambdas_app2_list.pk", "wb") as fi:
        h_lambdas_app2_list = pk.load(fi)

    with open("results/diff_lambdas_pos.pk", "wb") as fi:
        diff_lambdas_pos = pk.load(fi)
    with open("results/diff_lambdas_neg.pk", "wb") as fi:
        diff_lambdas_neg = pk.load(fi)
    with open("results/vals_lambdas_pos.pk", "wb") as fi:
        vals_lambdas_pos = pk.load(fi)
    with open("results/vals_lambdas_neg.pk", "wb") as fi:
        vals_lambdas_neg = pk.load(fi)
