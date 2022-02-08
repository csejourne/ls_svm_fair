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
from scipy.spatial.distance import pdist, squareform
from tools import extract_W, build_V, build_A_n, build_A_sqrt_n, build_A_1, build_A_1_11, \
                build_C_1, build_C_sqrt_n, build_C_n, build_D_1, build_D_sqrt_n, \
                build_D_n, one_hot, gen_dataset, get_gaussian_kernel, build_system_matrix, \
                build_Delta, build_F_n, build_tilde_F_n, decision_fair, comp_fairness_constraints, get_metrics

from tools import f, f_p, f_pp, build_objects

### NumPy print threshold size.
np.set_printoptions(threshold = 100)

### Some flags
save_arr = False
get_bk = True
do_test = True

"""
We iterate over a high number of experiences to better evaluate the orders.
"""
nb_iter = 1
h_lambdas_list = []
h_b_list = []
h_alpha_list = []
h_alpha_app_list = []
h_alpha_app2_list = []
h_alpha_diff_list = []
h_alpha_diff2_list = []
h_lambdas_app_list = []
h_lambdas_app2_list = []
h_b_list = []
h_b_app_list = []
h_b_diff_list = []
h_num_list = []
h_denom_list = []
h_num_app_list = []
h_denom_app_list = []
h_num_diff_list = []
h_denom_diff_list = []

for id_iter in range(nb_iter):
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
    print(f"cov_scal is {cov_scal}")
    list_cardinals = [[300, 150, 150, 250]]
    p_list = [512]
    #list_cardinals = [[150, 75, 75, 125], [300, 150, 150, 250], [600, 300, 300, 500]]
    #p_list = [256, 512, 1024]
    #list_cardinals = [[600, 300, 300, 500], [1200, 600, 600, 1000]]
    #p_list = [1024, 2048]
    #list_cardinals = [[1200, 600, 600, 1000], [2400, 1200, 1200, 2000]]
    #p_list = [2048, 4096]
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
    alpha_list = []
    alpha_app_list = []
    alpha_app2_list = []
    alpha_diff_list = []
    alpha_diff2_list = []
    lambdas_list = []
    lambdas_app_list = []
    lambdas_app2_list = []
    b_list = []
    b_app_list = []
    b_diff_list = []
    num_list = []
    denom_list = []
    num_app_list = []
    denom_app_list = []
    num_app2_list = []
    denom_app2_list = []
    num_diff_list = []
    denom_diff_list = []
    
    
    for i in range(len(p_list)):
        # Class distribution
        cardinals = list_cardinals[i]
        p = p_list[i]
        print("\n\tcardinals: ", cardinals, "  ||  p: ", p)
        k = len(cardinals)
        n = sum(cardinals)
        vec_prop = np.array(cardinals).reshape((-1, 1))
        sigma=p
        mu_list = [mean_scal * one_hot(0, p), mean_scal * one_hot(1, p),
                   mean_scal * one_hot(2, p), mean_scal * one_hot(3, p)]
        #mu_list = [mean_scal * one_hot(0, p), mean_scal * one_hot(0, p),
        #           mean_scal * one_hot(1, p), mean_scal * one_hot(1, p)]
        #cov_list = [cov_scal*np.eye(p),  cov_scal*np.eye(p),
        #            (1+2/np.sqrt(p)) * cov_scal*np.eye(p),  (1+2/np.sqrt(p)) * cov_scal*np.eye(p)]
        cov_list = [cov_scal*np.eye(p), (1 + 2/np.sqrt(p)) * cov_scal*np.eye(p),
                    cov_scal*np.eye(p), (1 + 2/np.sqrt(p)) * cov_scal*np.eye(p)]

        ### For testing
        nb_loops = 10
        cardinals_test = [50, 50, 50, 50]
        n_test = sum(cardinals_test)
        
        # Generate data.
        X, y, sens_labels, ind_dict = gen_dataset(mu_list, cov_list, cardinals)
        Y = np.concatenate([np.zeros(2), y])
        assert (n, p) == X.shape
        W = extract_W(X, mu_list, cardinals)
        K = get_gaussian_kernel(X, sigma)
        J, M, t, S = build_objects(mu_list, cardinals, cov_list)
        mu_diff_dist = squareform(pdist(M.T, metric='sqeuclidean')) 
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
        alpha = alpha.reshape((-1, 1))
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
        #C_1 = build_C_1(A_n, tau, gamma)
        C = C_n + C_sqrt_n #+ C_1
        
        ## $B_{22}$ in the companion paper.
        #Om = K + n/gamma * np.eye(n)
        #Om = np.linalg.inv(Om)
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
        tilde_F_n_app2 = - inv_Delta.T @ mat_n @ inv_Delta
        F_n = build_F_n(Delta, E_app)
        tilde_F_n_app = build_tilde_F_n(Delta, E_app)
        
        tilde_G_n = build_tilde_F_n(Delta, mat_n)
        tilde_G_sqrt_n = build_tilde_F_n(Delta, mat_sqrt_n)
        det_tilde_G_n = np.linalg.det(tilde_G_n)
        det_tilde_F_n_app = np.linalg.det(tilde_F_n_app)
        # Build the R_n vector
        factor = ones_k.T @ n_signed / n
        R_n = 1/det_tilde_G_n * (2*gamma*f_p(tau))/(n*p) * tilde_G_n @ Delta.T @ J @ A_1_11 @ (n_signed - factor* vec_prop)
    
        ### Compute lambdas approximates
        ones_n = np.ones((n,1))

        ### Compute approximation of `b`
        b_sqrt_n = gamma/(1+gamma*f(tau)) - 1/det_tilde_F_n_app*(gamma*f_p(tau)/(1+gamma*f(tau)))**2 * 1/p * t.T @ J.T @ Delta @ tilde_F_n_app @ Delta.T @ J @ t
        b_sqrt_n = 1/b_sqrt_n
        b_sqrt_n = b_sqrt_n * (
                - 1 * (gamma*f_p(tau)/(1+gamma*f(tau)))**2 * 1/det_tilde_G_n* 1/np.sqrt(p) * t.T @ J.T @ Delta @ tilde_G_n @ (
                2*ones_k.T @ n_signed /(n**2*p) * Delta.T @ J @ A_1_11 @ vec_prop - 2/(n*p) * Delta.T @ J @ (
                    A_1_11 @ ((1+gamma*f(tau))* n_signed - gamma*f(tau)*ones_k.T @ n_signed/n * vec_prop) 
                    + gamma*f_p(tau)/2 * t.T @ n_signed * t)
                )
                + 2*f_p(tau)/n**2 * ones_n.T @ L @ A_sqrt_n @ L @ y
            )
        b_app = ones_k.T @ n_signed / n + b_sqrt_n
        
        ### Compute the expectations `E_a`
        alpha_n_3_2 = - ( gamma/n * b_sqrt_n * 1/(1+gamma*f(tau))
                + 1/n * (gamma*f_p(tau)/(1+gamma*f(tau)))**2 * 1/det_tilde_G_n * 1/np.sqrt(p) *
                t.T @ J.T @ Delta @ tilde_G_n @ (
                    factor*2/(n*p) * Delta.T @ J @ A_1_11 @ vec_prop - 2/(n*p) * Delta.T @ J @ (
                        A_1_11 @ ((1+gamma*f(tau))*n_signed - gamma*f(tau)*factor*vec_prop)
                        + gamma*f_p(tau)/2 * t.T @ n_signed * t
                        )
                    - b_sqrt_n/np.sqrt(p) * Delta.T @ J @ t
                    )
                + gamma**2*f_p(tau)/(1+gamma*f(tau)) * 1/n**2 * t.T @ n_signed/np.sqrt(p) 
                )

        D_cal = (gamma*f_p(tau)/n * y.T @ P + f_p(tau)*R_n.T @ Delta.T) @ psi \
                + (gamma*f_pp(tau)/(2*n)* y.T @ P + f_pp(tau)/2*R_n.T @ Delta.T) @ psi**2 \
                + (gamma*f_pp(tau)/(2*n*p)* (n_signed - factor*vec_prop).T 
                        + f_pp(tau)/(2*p) * R_n.T @ Delta.T @ J) @ t**2

        expecs = factor * np.ones((k, 1))
        varis = np.zeros((k,1))
        y_list = np.array([1, 1, -1, -1])
        sens_list = np.array([0, 1, 0, 1])

        for a in range(k):
            ### Compute the expectations `E_a`
            mu_diff = np.stack([mu_list[b] - mu_list[a] for b in range(k)]).T
            tmp = (gamma*f_pp(tau)*t.T @ n_signed/(2*n*np.sqrt(p)) + n*alpha_n_3_2 * f_p(tau))* \
                        np.trace(cov_list[a])/p
            D_cal_x = (gamma*f_p(tau)/(n*p)*(n_signed - factor*vec_prop).T + f_p(tau)/p * R_n.T @ Delta.T @ J)@mu_diff_dist[:, a] \
                + 2*f_p(tau)/p*R_n.T @ Delta.T @ np.diag(np.repeat(mu_diff.T, vec_prop.flatten(), axis=0) @ W.T) \
                + (2*gamma*f_pp(tau)/(n*p) * (n_signed - factor*vec_prop).T + 2*f_pp(tau)/p*R_n.T @ Delta.T @ J)@ S[:, a] \
                + (gamma*f_pp(tau)/(2*n*p)* t.T @ n_signed + n*alpha_n_3_2 *f_p(tau)/np.sqrt(p))*t[a]
            expecs[a] += np.squeeze(tmp + D_cal + D_cal_x)

            ### Compute the variances `Var_a`
            tmp = 2/p**2 * (gamma*f_pp(tau)/(2*n*np.sqrt(p))*t.T @ n_signed + n*alpha_n_3_2*f_p(tau))**2 * np.trace(cov_list[a] @ cov_list[a])
            varis[a] += np.squeeze(tmp)
            tmp = (2*gamma*f_p(tau)/(n*p) * (n_signed - factor*vec_prop).T + 2*f_p(tau)/p*R_n.T@Delta.T@J) @ mu_diff.T @ cov_list[a] @ mu_diff @ (2*gamma*f_p(tau)/(n*p) * (n_signed - factor*vec_prop) + 2*f_p(tau)/p*J.T @ Delta @ R_n)
            varis[a] += np.squeeze(tmp)
            R_n_coef = R_n[int(np.floor(b/2))]
            tmp = [(gamma**2/n**2*vec_prop[b]* (1 + factor**2 - 2*y_list[b]*factor) 
                    + R_n_coef**2/vec_prop[b]
                    + 2*gamma/n * (y_list[b] - factor)*(-1)**sens_list[b]*R_n_coef
                    ) *S[a,b] for b in range(k)]
            varis[a] += (2*f_p(tau))**2/p * np.sum(tmp)


        """
        Classify new data after fitting the fair LS-SVM
        """

        if do_test:
            pred_function_fair = partial(decision_fair, X, b, lambda_pos, lambda_neg, alpha, sigma, ind_dict)
            ### For storing results
            g_fair = {}
            g_fair[("pos", 0)] = np.zeros((nb_loops, cardinals_test[0]))
            g_fair[("pos", 1)] = np.zeros((nb_loops, cardinals_test[1]))
            g_fair[("neg", 0)] = np.zeros((nb_loops, cardinals_test[2]))
            g_fair[("neg", 1)] = np.zeros((nb_loops, cardinals_test[3]))
            
            for loop in range(nb_loops):
                ### Generate test dataset
                X_test, y_test, sens_labels_test, ind_dict_test = gen_dataset(mu_list, cov_list, cardinals_test)
                
                assert g_fair.keys() == ind_dict_test.keys()
                ### Compute raw predictions in $\R$
                preds_fair = pred_function_fair(X_test)
                c2 = (cardinals[2] + cardinals[3])/n
                c1 = (cardinals[0] + cardinals[1])/n
            
                ### Extract the predictions for each subclass for plotting use later.
                for key in ind_dict.keys():
                    inds = np.nonzero(np.squeeze(ind_dict_test[key])) 
                    g_fair[key][loop, :] = preds_fair[inds]
                
                ### Remove threshold like Zhenyu
                preds_fair = preds_fair - (c1 - c2) # remove threshold
                
                ### Compare how well both predictions function satisfy fairness properties.
                pos_const_fair, neg_const_fair = comp_fairness_constraints(preds_fair, ind_dict_test)
                pos_const_fair_int, neg_const_fair_int = comp_fairness_constraints(preds_fair, ind_dict_test, with_int=True)
                print(f"FAIR: pos {pos_const_fair}, neg {neg_const_fair}")
                print("")
                print(f"FAIR with int: pos {pos_const_fair_int}, neg {neg_const_fair_int}")
                
                results_fair = get_metrics(preds_fair, y_test, ind_dict_test)

            """
            ### Plot the results.
            """

            ### Predictions distributions.
            fig, axs = plt.subplots()
            axs.hist(g_fair[('pos', 0)].flatten(), 50, facecolor='blue', alpha=0.4)
            axs.hist(g_fair[('pos', 1)].flatten(), 50, facecolor='green', alpha=0.4)
            axs.hist(g_fair[('neg', 0)].flatten(), 50, facecolor='red', alpha=0.4)
            axs.hist(g_fair[('neg', 1)].flatten(), 50, facecolor='yellow', alpha=0.4)
            axs.axvline(x=c1-c2, color='red')
            axs.set_title("fair LS-SVM")
            
            
            # Create legend
            handles = [Rectangle((0, 0), 1,1,color=c) for c in ['blue', 'green', 'red', 'yellow']]
            labels=["Y = 1, A = 0", "Y = 1, A = 1", "Y = -1, A = 0", "Y = -1, A = 1"]
            fig.legend(handles, labels)
            fig.savefig("results/distribs/test_py.pdf")
            plt.show()

"""
Debug approximations
"""
#        num = A_12 @ A_22_inv @ Y
#        denom = A_12 @ A_22_inv @ A_21
#
#        ### Debug formula for A_{12} A_{22}^{-1} Y
#        num_app = gamma/(1+gamma*f(tau)) * factor - 1/det_tilde_F_n_app*(gamma*f_p(tau)/(1+gamma*f(tau)))**2 * (
#                factor* 1/p * t.T @ J.T @ Delta @ (tilde_G_n +tilde_G_sqrt_n) @ Delta.T @ J @ t
#                + 1/np.sqrt(p)*t.T @ J.T @ Delta @ tilde_G_n @ (factor*Delta.T @ psi - 2/(n*p) * Delta.T @ J @ (
#                    A_1_11@( (1+gamma*f(tau))*n_signed - gamma*f(tau)*factor*vec_prop) + gamma*f_p(tau)/2 * t.T @ n_signed*t)
#                    )
#                + factor*1/np.sqrt(p)*(psi.T @ Delta -2/(n*p) * vec_prop.T @ A_1_11 @ J.T @ Delta) @ tilde_G_n @ Delta.T @ J@t
#                ) \
#                - f_p(tau)/n**2 * gamma**2/(1+gamma*f(tau)) * 1/np.sqrt(p) * vec_prop.T @ T @ n_signed
#        denom_app  = gamma/(1+gamma*f(tau)) - 1/det_tilde_F_n_app * (gamma*f_p(tau)/(1+gamma*f(tau)))**2 * (
#                1/p * t.T @ J.T @ Delta @ (tilde_G_n + tilde_G_sqrt_n) @ Delta.T @ J @ t
#                + 1/np.sqrt(p) * t.T @ J.T @ Delta @ tilde_G_n @ (Delta.T @ psi - 2/(n*p) * Delta.T @ J@A_1_11@vec_prop)
#                + 1/np.sqrt(p) * (psi.T @Delta - 2/(n*p)*vec_prop.T @ A_1_11 @ J.T @ Delta ) @ tilde_G_n @ Delta.T@J@t
#                )
#
#        ### Compute approximation of `b`
#        b_sqrt_n = gamma/(1+gamma*f(tau)) - 1/det_tilde_F_n_app*(gamma*f_p(tau)/(1+gamma*f(tau)))**2 * 1/p * t.T @ J.T @ Delta @ tilde_F_n_app @ Delta.T @ J @ t
#        b_sqrt_n = 1/b_sqrt_n
#        b_sqrt_n = b_sqrt_n * (
#                - 1 * (gamma*f_p(tau)/(1+gamma*f(tau)))**2 * 1/det_tilde_G_n* 1/np.sqrt(p) * t.T @ J.T @ Delta @ tilde_G_n @ (
#                2*ones_k.T @ n_signed /(n**2*p) * Delta.T @ J @ A_1_11 @ vec_prop - 2/(n*p) * Delta.T @ J @ (
#                    A_1_11 @ ((1+gamma*f(tau))* n_signed - gamma*f(tau)*ones_k.T @ n_signed/n * vec_prop) 
#                    + gamma*f_p(tau)/2 * t.T @ n_signed * t)
#                )
#                + 2*f_p(tau)/n**2 * ones_n.T @ L @ A_sqrt_n @ L @ y
#            )
#        b_app = ones_k.T @ n_signed / n + b_sqrt_n
#
#        ### For alpha
#        alpha_app = ((C_sqrt_n + C_n).T @ Delta @ F_n @ Delta.T @ (C_sqrt_n + C_n) + L/n + 2*f_p(tau)/n**2 * L @ A_sqrt_n @ L) @ (y - b*ones_n)
#        alpha_app2 = gamma/n*(y - factor*ones_n - b_sqrt_n * 1/(1+gamma*f(tau))*ones_n) \
#                - 1/n*(gamma*f_p(tau)/(1+gamma*f(tau)))**2 * 1/det_tilde_G_n * 1/np.sqrt(p)*t.T @ J.T @ Delta @ tilde_G_n @ (
#                        factor*2/(n*p) * Delta.T @ J @ A_1_11 @ vec_prop - 2/(n*p) * Delta.T @ J @ (
#                            A_1_11 @ ((1+gamma*f(tau))*n_signed - gamma*f(tau)*factor * vec_prop) + gamma*f_p(tau)/2 * t.T @ n_signed*t
#                            )
#                        - b_sqrt_n * 1/np.sqrt(p) * Delta.T @ J @ t
#                        ) * ones_n \
#                - gamma*f_p(tau)/n**2 * t.T @ n_signed/np.sqrt(p) * L @ ones_n
#                #+ 2*f_p(tau)/n**2 * L @ A_sqrt_n @ L @ (y - b*ones_n)
#        ### For `lambdas`
#        t1 = gamma*f_p(tau)/(1+gamma*f(tau)) * (1/np.sqrt(p) * Delta.T @ J @ t + Delta.T @ psi 
#                - 2/(n*p) * Delta.T @ J @ A_1_11 @ vec_prop
#                )
#        t2 = gamma*f_p(tau)/(1+gamma*f(tau))*( ones_k.T @ n_signed / n * (1/np.sqrt(p) * Delta.T @ J @ t + Delta.T @ psi) \
#                - 2/(n*p) * Delta.T@J@A_1_11@((1+gamma*f(tau)) * n_signed - gamma*f(tau)*ones_k.T @ n_signed/n * vec_prop)
#                - gamma*f_p(tau)/(n*p) * t.T @ n_signed * Delta.T @ J @ t
#                )
#        lambdas = np.array([lambda_pos, lambda_neg]).reshape((2,1))
#        lambdas_app = 1/det_tilde_G_n*tilde_G_n @ Delta.T @ (C_sqrt_n + C_n) @ (y - b_app*ones_n)
#        lambdas_app2 = 1/det_tilde_G_n*tilde_G_n @ (t2 - b*t1)
#        lambdas = lambdas.reshape((-1,))
#        lambdas_app = lambdas_app.reshape((-1,))
#        lambdas_app2 = lambdas_app2.reshape((-1,))
#
#        ### Store information
#        # For `b`
#        b_list.append(b)
#        b_app_list.append(b_app)
#        b_diff_list.append(b - b_app)
#        num_list.append(num)
#        denom_list.append(denom)
#        num_app_list.append(num_app)
#        denom_app_list.append(denom_app)
#        # For `alpha`
#        alpha_list.append(alpha)
#        alpha_app_list.append(alpha_app)
#        alpha_app2_list.append(alpha_app2)
#        alpha_diff_list.append(alpha - alpha_app)
#        alpha_diff2_list.append(alpha - alpha_app2)
#        # For `lambdas`
#        lambdas_list.append(lambdas)
#        lambdas_app_list.append(lambdas_app)
#        lambdas_app2_list.append(lambdas_app)
#    
#        ### other stuff
#        tmp_list.append(1/n**2 * L @ A_sqrt_n @ L @ ones_n)
#        tmp2_list.append(1/n**2 * L @ A_sqrt_n @ L @ y)
#    #print("")
#    #print("Results of operator norms")
#
#    ### Store information from the iterations.
#    h_alpha_list.append(alpha_list)
#    h_alpha_app_list.append(alpha_app_list)
#    h_alpha_app2_list.append(alpha_app2_list)
#    h_alpha_diff_list.append(alpha_diff_list)
#    h_alpha_diff2_list.append(alpha_diff2_list)
#    h_lambdas_list.append(lambdas_list)
#    h_lambdas_app_list.append(lambdas_app_list)
#    h_lambdas_app2_list.append(lambdas_app2_list)
#    h_b_list.append(b_list)
#    h_b_app_list.append(b_app_list)
#    h_b_diff_list.append(b_diff_list)
#    h_num_list.append(num_list)
#    h_denom_list.append(denom_list)
#    h_num_app_list.append(num_app_list)
#    h_denom_app_list.append(denom_app_list)
#
#    ### Save
#    if save_arr:
#        np.save(open('results/lambdas.npy', 'wb'), np.array(h_lambdas_list))
#        np.save(open('results/lambdas_app.npy', 'wb'), np.array(h_lambdas_app_list))
#        np.save(open('results/lambdas_app2.npy', 'wb'), np.array(h_lambdas_app2_list))
#        np.save(open('results/b_diff.npy', 'wb'), np.array(h_b_diff_list))
#        np.save(open('results/num_list.npy', 'wb'), np.array(h_num_list))
#        np.save(open('results/denom_list.npy', 'wb'), np.array(h_denom_list))
#        np.save(open('results/denom_app_list.npy', 'wb'), np.array(h_denom_app_list))
#        np.save(open('results/num_app_list.npy', 'wb'), np.array(h_num_app_list))
#        pk.dump(np.array(h_alpha_list), open('results/alpha.pk', 'wb'))
#        pk.dump(np.array(h_alpha_app_list), open('results/alpha_app.pk', 'wb'))
#        pk.dump(np.array(h_alpha_app2_list), open('results/alpha_app2.pk', 'wb'))
#
#h_denom_list = np.squeeze(np.array(h_denom_list))
#h_denom_app_list = np.squeeze(np.array(h_denom_app_list))
#h_num_list = np.squeeze(np.array(h_num_list))
#h_num_app_list = np.squeeze(np.array(h_num_app_list))
#h_b_diff_list = np.squeeze(np.array(h_b_diff_list))
#h_lambdas_list = np.squeeze(np.array(h_lambdas_list))
#h_lambdas_app_list = np.squeeze(np.array(h_lambdas_app_list))
#h_lambdas_app2_list = np.squeeze(np.array(h_lambdas_app2_list))
#
#if get_bk:
#    h_lambdas_list = np.load(open('results/lambdas.npy', 'rb'))
#    h_lambdas_app_list = np.load(open('results/lambdas_app.npy', 'rb'))
#    h_lambdas_app2_list = np.load(open('results/lambdas_app2.npy', 'rb'))
#    h_b_diff_list = np.load(open('results/b_diff.npy', 'rb'))
#    h_num_list = np.load(open('results/num_list.npy', 'rb'))
#    h_denom_list = np.load(open('results/denom_list.npy', 'rb'))
#    h_denom_app_list = np.load(open('results/denom_app_list.npy', 'rb'))
#    h_num_app_list = np.load(open('results/num_app_list.npy', 'rb'))
#    h_alpha_list = pk.load(open('results/alpha.pk', 'rb'))
#    h_alpha_app_list = pk.load(open('results/alpha_app.pk', 'rb'))
#    h_alpha_app2_list = pk.load(open('results/alpha_app2.pk', 'rb'))
