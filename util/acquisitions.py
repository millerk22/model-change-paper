# author: Kevin Miller

import numpy as np
from scipy.stats import norm
from .al_util import *
import time

MODELS = ['gr', 'probit-log', 'probit-norm', 'ce', 'log', 'probitnorm', 'mgr']

def sgn(x):
    if x >= 0:
        return 1.
    else:
        return -1.

def mc_full(Cand, m, C, modelname, gamma=0.1):
    if modelname not in MODELS:
        raise ValueError("%s is not a valid model name, must be in %s" % (modelname, MODELS))
    if len(m.shape) > 1: # Multiclass case
        if modelname == 'gr':
            #return np.array([ np.sqrt(np.inner(m[k,:], m[k,:])[0,0] + 1. - 2.*np.max(m[k,:])) * np.linalg.norm(C[:,k])/(gamma**2. + C[k,k]) for k in Cand])
            m_minus_yk = m[Cand,:]
            m_minus_yk.ravel()[[m.shape[1]*i for i in range(m_minus_yk.shape[0])] + np.argmax(m_minus_yk, axis=1)] -= 1.
            m_minus_yk_norms = np.linalg.norm(m_minus_yk, axis=1)
            Ck_norms = np.linalg.norm(C[:, Cand], axis=0)
            return Ck_norms*m_minus_yk_norms/(gamma**2. + np.diag(C)[Cand])
            #return np.array([np.sqrt(np.inner(m[k,:], m[k,:]) + 1. - 2.*np.max(m[k,:])) * np.linalg.norm(C[:,k])/(gamma**2. + C[k,k]) for k in Cand])
        else:
            raise NotImplementedError("Have not implemented full storage model change calculation for multiclass besides Gaussian Regression ('gr')")
    else:
        if modelname == 'probit-log' or modelname == 'log':
            return np.array([np.absolute(jac_calc2(m[k], sgn(m[k]), gamma))/(1. + C[k,k]*hess_calc2(m[k], sgn(m[k]), gamma)) \
                           * np.linalg.norm(C[:,k]) for k in Cand])
        elif modelname == 'probit-norm' or modelname == 'probitnorm':
            return np.array([np.absolute(jac_calc(m[k], sgn(m[k]), gamma))/(1. + C[k,k]*hess_calc(m[k], sgn(m[k]), gamma)) \
                           * np.linalg.norm(C[:,k]) for k in Cand])
        else:
            return np.array([np.absolute(m[k] - sgn(m[k]))/(gamma**2. + C[k,k]) * np.linalg.norm(C[:,k]) for k in Cand])

def mc_reduced(C_a, alpha, v_Cand, modelname, uks=None, gamma=0.1, verbose=False, greedy=True):
    if modelname not in MODELS:
        raise ValueError("%s is not a valid model name, must be in %s" % (modelname, MODELS))

    if modelname == 'ce':
        num_cand, M = v_Cand.shape
        nc = alpha.shape[0]//M

        if uks is None:
            uks = v_Cand @ (alpha.reshape(nc, M).T)

        piks = np.exp(uks/gamma)
        piks /= np.sum(piks, axis=1)[:,np.newaxis]

        C_aV_candT = np.empty((M*nc, num_cand*nc))
        for c in range(nc):
            C_aV_candT[:,c*num_cand:(c+1)*num_cand] = C_a[:,c*M:(c+1)*M] @ v_Cand.T

        mc_vals = []
        for k in range(num_cand):

            inds_k_in_cand = [k + c*num_cand for c in range(nc)]
            CVT_k = C_aV_candT[:, inds_k_in_cand]

            Bk = np.diag(piks[k,:]) - np.outer(piks[k,:], piks[k,:])

            if np.linalg.norm(Bk, ord='fro') > 1e-7:
                # print("big")
                # print(piks[k,:])
                # print(Bk)
                uk, sk, vkt = np.linalg.svd(Bk)
                Tk = uk * np.sqrt(sk)[np.newaxis, :]


                Gk = np.empty((nc, nc))
                for c in range(nc):
                    Gk[c,:] = v_Cand[k,:][np.newaxis,:] @ C_aV_candT[c*M:(c+1)*M, inds_k_in_cand]

                TkGk = Tk @ Gk
                Mk = np.eye(nc) - Tk.T @ np.linalg.inv(gamma*np.eye(nc) + TkGk @ Tk.T) @ TkGk

                if not greedy:
                    Mkpi_k_yk_mat = CVT_k @ (Mk @ (np.tile(piks[k,:][:,np.newaxis], (1,nc))  - np.eye(nc)))
                    #mc_vals_for_k = [np.linalg.norm(Mkpi_k_yk_mat[:,c]) for c in range(nc)]
                    mc_vals_for_k = np.linalg.norm(Mkpi_k_yk_mat, axis=0)
                    argmin_mcvals = np.argmin(mc_vals_for_k)
                    argmax_piks = np.argmax(piks[k,:])

                    if (argmin_mcvals != argmax_piks) and verbose:
                        print("%d (index in Cand) did Not choose choice that we thought" % k)
                    mc_vals.append(np.min(mc_vals_for_k))
                else:
                    y_c = np.zeros(nc)
                    y_c[np.argmax(piks[k,:])] = 1.
                    mc_vals.append(np.linalg.norm(CVT_k @ (Mk @ (piks[k,:] - y_c))))
            else:
                # print("really small")
                # print(piks[k,:])
                # print(Bk)
                #print(np.linalg.norm(Bk, ord='fro'))
                #VK = np.kron(np.eye(nc), v_Cand[k,:][:, np.newaxis])
                # print(np.linalg.norm(VK @ Bk @ VK.T))
                # print()
                if not greedy:
                    Mkpi_k_yk_mat = CVT_k @ (np.tile(piks[k,:][:,np.newaxis], (1,nc))  - np.eye(nc))
                    #mc_vals_for_k = [np.linalg.norm(Mkpi_k_yk_mat[:,c]) for c in range(nc)]
                    mc_vals_for_k = np.linalg.norm(Mkpi_k_yk_mat, axis=0)
                    argmin_mcvals = np.argmin(mc_vals_for_k)
                    argmax_piks = np.argmax(piks[k,:])

                    if (argmin_mcvals != argmax_piks) and verbose:
                        print("%d (index in Cand) did Not choose choice that we thought" % k)
                    mc_vals.append(np.min(mc_vals_for_k))
                else:
                    y_c = np.zeros(nc)
                    y_c[np.argmax(piks[k,:])] = 1.
                    mc_vals.append(np.linalg.norm(CVT_k @ (piks[k,:] - y_c)))


        return np.array(mc_vals)



    else:
        if uks is None: # if have not already calculated MAP estimator on full (as we are doing in our Reduced model), then do this calculation
            uks = v_Cand @ alpha
        C_a_vk = C_a @ (v_Cand.T)
        if modelname == 'probit-log' or modelname == 'log':
            return np.array([np.absolute(jac_calc2(uks[i], sgn(uks[i]),gamma))/(1. + \
                np.inner(v_Cand[i,:], C_a_vk[:,i])*hess_calc2(uks[i], sgn(uks[i]),gamma))* np.linalg.norm(C_a_vk[:,i]) \
                            for i in range(v_Cand.shape[0])])
        elif modelname == 'probit-norm' or modelname == 'probitnorm':
            return np.array([np.absolute(jac_calc(uks[i], sgn(uks[i]),gamma))/(1. + \
                np.inner(v_Cand[i,:], C_a_vk[:,i])*hess_calc(uks[i], sgn(uks[i]),gamma))* np.linalg.norm(C_a_vk[:,i]) \
                            for i in range(v_Cand.shape[0])])
        else:
            # Multiclass GR
            if len(alpha.shape) > 1:
                #raise NotImplementedError("Multiclass GR not implemented yet in the reduced case.")
                uks_minus_yk = uks.copy()
                uks_minus_yk.ravel()[[uks.shape[1]*i for i in range(uks.shape[0])]+ np.argmax(uks, axis=1)] -= 1.
                uks_minus_yk_norms = np.linalg.norm(uks_minus_yk, axis=1)
                C_a_vk_norms = np.linalg.norm(C_a_vk, axis=0)
                return np.array([C_a_vk_norms[i]*uks_minus_yk_norms[i]/(gamma**2. + \
                    np.inner(v_Cand[i,:], C_a_vk[:,i])) for i in range(v_Cand.shape[0])])

            # Binary GR
            else:
                return np.array([np.absolute(uks[i] - sgn(uks[i]))/(gamma**2. + np.inner(v_Cand[i,:], C_a_vk[:,i]))
                               * np.linalg.norm(C_a_vk[:,i]) for i in range(v_Cand.shape[0])])


def mc_avg_reduced(model, Cand, beta=0.0):
    assert not model.full_storage
    if not model.modelname in ['mgr', 'gr']:
        raise ValueError("{} currently only implemented for 'MGR' and 'GR' models..")

    v_Cand = model.v[Cand, :]
    uks = v_Cand @ model.alpha
    C_a_vk = model.C_a @ (v_Cand.T)
    # Multiclass GR
    if len(model.alpha.shape) > 1:
        uks_minus_yk = uks.copy()
        uks_minus_yk.ravel()[[uks.shape[1]*i for i in range(uks.shape[0])]+ np.argmax(uks, axis=1)] -= 1.
        uks_minus_yk_norms = np.linalg.norm(uks_minus_yk, axis=1)
        C_a_vk_norms = np.linalg.norm(C_a_vk, axis=0)
        return np.array([C_a_vk_norms[i]**(1. + beta) * (uks_minus_yk_norms[i]**(1. - beta))/(model.gamma**2. + \
            np.inner(v_Cand[i,:], C_a_vk[:,i])) for i in range(v_Cand.shape[0])])

    # Binary GR
    else:
        return np.array([np.absolute(uks[i] - sgn(uks[i]))**(1.- beta)/(model.gamma**2. + np.inner(v_Cand[i,:], C_a_vk[:,i]))
                       * np.linalg.norm(C_a_vk[:,i])**(1.+beta) for i in range(v_Cand.shape[0])])

def mc_app_full_red(model_, Cand):
    assert not model_.full_storage
    lam_bar_inv = 1./(model_.d[-1] + model_.tau**2.)
    N, M = model_.v.shape
    v_Cand = model_.v[model_.unlabeled, :].T
    C_a_vk = model_.C_a @ v_Cand
    C_a_vk_norms = np.linalg.norm(C_a_vk, axis=0)
    v_Cand_norms = np.linalg.norm(v_Cand, axis=0)**2. # calculated ahead of time
    A = model_.v @ (C_a_vk - lam_bar_inv * v_Cand)  # can calculate 2nd half ahead of time
    A[model_.unlabeled, range(len(model_.unlabeled))] += lam_bar_inv
    A_norms = np.linalg.norm(A, axis=0)
    if len(model_.alpha.shape) > 1:
        return np.array([ C_a_vk_norms[i]/(model_.gamma**2. + np.inner(v_Cand[:,i], C_a_vk[:,i]) + lam_bar_inv*(1. - v_Cand_norms[i]))
                     *A_norms[i] for i,k in enumerate(model_.unlabeled)])
    else:
        return np.array([ np.absolute(model_.m[k] - np.sign(model_.m[k]))/(model_.gamma**2. + np.inner(v_Cand[:,i], C_a_vk[:,i]) + lam_bar_inv*(1. - v_Cand_norms[i]))
                     *A_norms[i] for i,k in enumerate(model_.unlabeled)])

def mcavg_app_full_red(model_, Cand, beta=0.0):
    assert not model_.full_storage
    lam_bar_inv = 1./(model_.d[-1] + model_.tau**2.)
    N, M = model_.v.shape
    v_Cand = model_.v[model_.unlabeled, :].T
    C_a_vk = model_.C_a @ v_Cand
    C_a_vk_norms = np.linalg.norm(C_a_vk, axis=0)
    v_Cand_norms = np.linalg.norm(v_Cand, axis=0)**2. # calculated ahead of time
    A = model_.v @ (C_a_vk - lam_bar_inv * v_Cand)  # can calculate 2nd half ahead of time
    A[model_.unlabeled, range(len(model_.unlabeled))] += lam_bar_inv
    A_norms = np.linalg.norm(A, axis=0)
    if len(model_.alpha.shape) > 1:
        return np.array([ C_a_vk_norms[i]**(1.-beta)/(model_.gamma**2. + np.inner(v_Cand[:,i], C_a_vk[:,i]) + lam_bar_inv*(1. - v_Cand_norms[i]))
                     *A_norms[i]**(1.+beta) for i,k in enumerate(model_.unlabeled)])
    else:
        return np.array([ np.absolute(model_.m[k] - np.sign(model_.m[k]))**(1.-beta)/(model_.gamma**2. + np.inner(v_Cand[:,i], C_a_vk[:,i]) + lam_bar_inv*(1. - v_Cand_norms[i]))
                     *A_norms[i]**(1.+beta) for i,k in enumerate(model_.unlabeled)])
