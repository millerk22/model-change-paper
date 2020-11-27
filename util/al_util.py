import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy.optimize import root
import scipy.linalg as sla
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
import time

class Classifier(object):
    def __init__(self, name, gamma, tau, v=None, w=None, Ct=None):
        self.gamma = gamma
        self.tau = tau
        self.v = v
        self.w = w
        if Ct is not None:
            self.Ct = Ct
        else:
            self.d = (self.tau ** (2.)) * ((self.w + self.tau**2.) ** (-1.))
            self.Ct = (self.v * self.d) @ self.v.T
        self.name = name
        return

    def get_m(self, Z, y):
        if self.name == "probit":
            if len(y) <= len(self.w):
                return probit_map_dr(Z, y, self.gamma, self.Ct)
            else:
                return probit_map_st(Z, y, self.gamma, 1./self.d, self.v)
        elif self.name == "probit2":
            if len(y) <= len(self.w):
                return probit_map_dr2(Z, y, self.gamma, self.Ct)
            else:
                return probit_map_st2(Z, y, self.gamma, 1./self.d, self.v)
        elif self.name == "gr":
            #return gr_map(Z, y, self.gamma, self.Ct)
            return gr_map(Z, y, self.gamma, 1./self.d, self.v)
        else:
            pass

    def get_C(self, Z, y, m):
        if self.name in ["probit", "probit-st"]:
            if len(y) > len(self.w):
                return Hess_inv_st(m, Z, y, 1./self.d, self.v, self.gamma)
            else:
                return Hess_inv(m, Z, y, self.gamma, self.Ct)
        elif self.name in ["probit2", "probit2-st"]:
            if len(y) > len(self.w):
                return Hess_inv_st2(m, Z, y, 1./self.d, self.v, self.gamma)
            else:
                return Hess2_inv(m, Z, y, self.gamma, self.Ct)
        elif self.name in ["gr"]:
            #return gr_C(Z, self.gamma, self.Ct)
            return gr_C(Z, self.gamma, 1./self.d, self.v)
        else:
            pass


class Classifier_HF(object):
    def __init__(self, tau, L):
        self.tau = tau
        if sps.issparse(L):
            self.sparse = True
            self.L = L + self.tau**2. * sps.eye(L.shape[0])
        else:
            self.sparse = False
            self.L = L + self.tau**2. * np.eye(L.shape[0])
        return

    def get_m(self, Z, y):
        Zbar = list(filter(lambda x: x not in Z, range(self.L.shape[0])))
        self.m = -self.get_C(Z, Zbar=Zbar) @ self.L[np.ix_(Zbar, Z)] @ y
        return self.m

    def get_C(self, Z, Zbar=None):
        if Zbar is None:
            Zbar = list(filter(lambda x: x not in Z, range(self.L.shape[0])))
        if self.sparse:
            self.C = scipy.sparse.linalg.inv(self.L[np.ix_(Zbar, Zbar)]).toarray() # inverse will be dense anyway
            return self.C
        else:
            self.C = sla.inv(self.L[np.ix_(Zbar, Zbar)])
            return self.C


###############################################################################
############### Gaussian Regression Helper Functions ##########################


def get_init_post(C_inv, labeled, gamma2):
    """
    calculate the risk of each unlabeled point
    C_inv: prior inverse (i.e. graph Laplacian L)
    """
    N = C_inv.shape[0]
    #unlabeled = list(filter(lambda x: x not in labeled, range(N)))
    B_diag = [1 if i in labeled else 0 for i in range(N)]
    B = sp.sparse.diags(B_diag, format='csr')
    return sp.linalg.inv(C_inv + B/gamma2)


def calc_next_m(m, C, y, lab, k, y_k, gamma2):
    ck = C[k,:]
    ckk = ck[k]
    ip = np.dot(ck[lab], y[lab])
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck
    return m_k


def get_probs(m, sigmoid=False):
    if sigmoid:
        return 1./(1. + np.exp(-3.*m))
    m_probs = m.copy()
    # simple fix to get probabilities that respect the 0 threshold
    m_probs[np.where(m_probs >0)] /= 2.*np.max(m_probs)
    m_probs[np.where(m_probs <0)] /= -2.*np.min(m_probs)
    m_probs += 0.5
    return m_probs


def EEM_full(k, m, C, y, lab, unlab, m_probs, gamma2):
    N = C.shape[0]
    m_at_k = m_probs[k]
    m_k_p1 = calc_next_m(m, C, y, lab, k, 1., gamma2)
    m_k_p1 = get_probs(m_k_p1)
    risk = m_at_k*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in range(N)])
    m_k_m1 = calc_next_m(m, C, y, lab, k, -1., gamma2)
    m_k_m1 = get_probs(m_k_m1)
    risk += (1.-m_at_k)*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in range(N)])
    return risk


def EEM_opt_record(m, C, y, labeled, unlabeled, gamma2):
    m_probs = get_probs(m)
    N = C.shape[0]
    risks = [EEM_full(j, m, C, y, labeled, unlabeled, m_probs, gamma2) for j in range(N)]
    k = np.argmin(risks)
    return k, risks


def Sigma_opt(C, unlabeled, gamma2):
    sums = np.sum(C[np.ix_(unlabeled,unlabeled)], axis=1)
    sums = np.asarray(sums).flatten()**2.
    s_opt = sums/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(s_opt)]
    return k_max


def V_opt_record(C, unlabeled, gamma2):
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    v_opt = ips/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max, v_opt


def V_opt_record2(u, C, unlabeled, gamma2, lam):
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    v_opt = ips/(gamma2 + np.diag(C)[unlabeled])
    print(np.max(v_opt), np.min(v_opt))
    v_opt += lam*(np.max(v_opt) + np.min(v_opt))*0.5*(1./np.absolute(u[unlabeled])) # add term that bias toward decision boundary
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max, v_opt


def Sigma_opt_record(C, unlabeled, gamma2):
    sums = np.sum(C[np.ix_(unlabeled,unlabeled)], axis=1)
    sums = np.asarray(sums).flatten()**2.
    s_opt = sums/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(s_opt)]
    return k_max, s_opt




################################################################################
################## Plotting Helper Functions ###################################


def plot_iter(m, X, labels, labeled, k_next=-1, title=None, subplot=False):
    '''
    Assuming labels are +1, -1
    '''
    m1 = np.where(m >= 0)[0]
    m2 = np.where(m < 0)[0]

    #sup1 = list(set(labeled).intersection(set(np.where(labels == 1)[0])))
    #sup2 = list(set(labeled).intersection(set(np.where(labels == -1)[0])))

    corr1 = list(set(m1).intersection(set(np.where(labels == 1)[0])))
    incorr1 = list(set(m2).intersection(set(np.where(labels == 1)[0])))
    corr2 = list(set(m2).intersection(set(np.where(labels == -1)[0])))
    incorr2 = list(set(m1).intersection(set(np.where(labels == -1)[0])))

    print("\tnum incorrect = %d" % (len(incorr1) + len(incorr2)))

    if k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title(r'Dataset with Label for %s added' % str(k_next))
    elif k_next == -1:
        if not title:
            plt.title(r'Dataset with Initial Labeling')
        else:
            plt.title(title)

    plt.scatter(X[corr1,0], X[corr1,1], marker='x', c='b', alpha=0.2)
    plt.scatter(X[incorr1,0], X[incorr1,1], marker='x', c='r', alpha=0.2)
    plt.scatter(X[corr2,0], X[corr2,1], marker='o', c='r',alpha=0.15)
    plt.scatter(X[incorr2,0], X[incorr2,1], marker='o', c='b',alpha=0.15)
    # plt.scatter(X[corr1,0], X[corr1,1], marker='x', c='b', alpha=0.8)
    # plt.scatter(X[incorr1,0], X[incorr1,1], marker='x', c='r', alpha=0.8)
    # plt.scatter(X[corr2,0], X[corr2,1], marker='o', c='r',alpha=0.8)
    # plt.scatter(X[incorr2,0], X[incorr2,1], marker='o', c='b',alpha=0.8)

    sup1 = list(set(labeled).intersection(set(np.where(labels == 1)[0])))
    sup2 = list(set(labeled).intersection(set(np.where(labels == -1)[0])))
    plt.scatter(X[sup1,0], X[sup1,1], marker='x', c='k', alpha=1.0)
    plt.scatter(X[sup2,0], X[sup2,1], marker='o', c='k', alpha=1.0)
    plt.axis('equal')
    if subplot:
        return
    plt.show()
    return


def get_acc(u, labels, unlabeled = None):
    u_ = np.sign(u)
    u_[u_ == 0] = 1
    if unlabeled is None:
        corr = sum(1.*(u_ == labels))
        return corr, corr/u.shape[0]
    else:
        corr = sum(1.*(u_[unlabeled] == labels[unlabeled]))
        return corr, corr/len(unlabeled)
def get_acc_multi(u, labels, unlabeled=None):
    """
    Assuming that u and labels are NOT in one-hot encoding. i.e. ith entry of
    u and labels is the integer class in {0,1,2,...num_classes}.
    """
    if unlabeled is None:
        corr = sum(1.*(u == labels))
        return corr, corr/u.shape[0]
    else:
        corr = sum(1.*(u[unlabeled] == labels[unlabeled]))
        return corr, corr/len(unlabeled)






################################################################################
########## MAP estimators of Probit model (with normal distribution's pdf )
################################################################################


def pdf_deriv(t, gamma):
    return -t*norm.pdf(t, scale = gamma)/(gamma**2)

def jac_calc(uj_, yj_, gamma):
    return -yj_*(norm.pdf(uj_*yj_, scale=gamma)/norm.cdf(uj_*yj_, scale=gamma))


def hess_calc(uj_, yj_, gamma):
    return (norm.pdf(uj_*yj_,scale=gamma)**2 - pdf_deriv(uj_*yj_, gamma)*norm.cdf(uj_*yj_,scale=gamma))/(norm.cdf(uj_*yj_,scale=gamma)**2)

def Hess(u, y, labeled, Lt, gamma, debug=False):
    """
    Assuming matrices are sparse, since L_tau should be relatively sparse,
        and we are perturbing with diagonal matrix.
    """
    H_d = np.zeros(u.shape[0])
    for j, yj in zip(labeled, y):
        H_d[j] = hess_calc(u[j], yj, gamma)
    if debug:
        print(H_d[np.nonzero(H_d)])
    # if np.any(H_d == np.inf):
    #     print('smally')
    return Lt + sp.sparse.diags(H_d, format='csr')


def J(u, y, labeled, Lt, gamma, debug=False):
    vec = np.zeros(u.shape[0])
    for j, yj in zip(labeled,y):
        vec[j] = -yj*norm.pdf(u[j]*yj, gamma)/norm.cdf(u[j]*yj, gamma)
    if debug:
        print(vec[np.nonzero(vec)])
    return Lt @ u + vec


def probit_map_dr(Z_, yvals, gamma, Ct):
    """
    Probit MAP estimator, using dimensionality reduction via Representer Theorem.
    *** This uses cdf of normal distribution ***
    """
    Ctp = Ct[np.ix_(Z_,Z_)]
    Jj = len(yvals)

    def f(x):
        vec = [jac_calc(x[j], yj, gamma) for j,yj in enumerate(yvals)]
        if np.any(vec == np.inf):
            print('smally in f')
        return x + Ctp @ vec

    def fprime(x):
        H = Ctp * np.array([hess_calc(x[j],yj, gamma)
                            for j, yj in enumerate(yvals)])
        if np.any(H == np.inf):
            print('smally')
        H[np.diag_indices(Jj)] += 1.0
        return H

    x0 = np.random.rand(Jj)
    x0[np.array(yvals) < 0] *= -1
    res = root(f, x0, jac=fprime)
    #print(np.allclose(0., f(res.x)))
    tmp = sla.inv(Ctp) @ res.x
    return Ct[:, Z_] @ tmp



################################################################################
################### Probit with logit likelihood
################################################################################

###################### Psi = cdf of logistic #######################
def log_pdf(t, g):
    return np.exp(-t/g)/(g*(1. + np.exp(-t/g)**2.))


def log_cdf(t, g):
    return 1.0/(1.0 + np.exp(-t/g))


def log_pdf_deriv(t, g):
    return -np.exp(-t/g)/((g*(1. + np.exp(-t/g)))**2.)

def jac_calc2(uj_, yj_, gamma):
    return -yj_*(np.exp(-uj_*yj_/gamma))/(gamma*(1.0 + np.exp(-uj_*yj_/gamma)))

def hess_calc2(uj_, yj_, gamma):
    return np.exp(-uj_*yj_/gamma)/(gamma**2. * (1. + np.exp(-uj_*yj_/gamma))**2.)


def Hess2(u, y, labeled, Lt, gamma, debug=False):
    """
    Assuming matrices are sparse, since L_tau should be relatively sparse,
        and we are perturbing with diagonal matrix.
    """
    H_d = np.zeros(u.shape[0])
    for j, yj in zip(labeled, y):
        H_d[j] = hess_calc2(u[j], yj, gamma)
    if debug:
        print(H_d[np.nonzero(H_d)])
    if np.any(H_d == np.inf):
        print('smally')
    return Lt + sp.sparse.diags(H_d, format='csr')


def J2(u, y, labeled, Lt, gamma, debug=False):
    vec = np.zeros(u.shape[0])
    for j, yj in zip(labeled,y):
        vec[j] = jac_calc2(u[j], yj, gamma)
    if debug:
        print(vec[np.nonzero(vec)])
    return Lt @ u + vec


def probit_map_dr2(Z_, yvals, gamma, Ct):
    """
    Probit MAP estimator, using dimensionality reduction via Representer Theorem.
    *** This uses logistic cdf ***
    """
    Ctp = Ct[np.ix_(Z_,Z_)]
    Jj = len(yvals)

    def f(x):
        vec = [jac_calc2(x[j], yj, gamma)
               for j,yj in enumerate(yvals)]
        return x + Ctp @ vec

    def fprime(x):
        H = Ctp * np.array([hess_calc2(x[j], yj, gamma)
                            for j, yj in enumerate(yvals)])
        if np.any(H == np.inf):
            print('smally')
        H[np.diag_indices(Jj)] += 1.0
        return H

    x0 = np.random.rand(Jj)
    x0[np.array(yvals) < 0] *= -1
    res = root(f, x0, jac=fprime)
    tmp = sla.inv(Ctp) @ res.x
    return Ct[:, Z_] @ tmp



################################################################################
# Spectral truncation
################################################################################
"""

def pdf_deriv(t, gamma):
    return -t*norm.pdf(t, scale = gamma)/(gamma**2)

def jac_calc(uj_, yj_, gamma):
    return -yj_*(norm.pdf(uj_*yj_, scale=gamma)/norm.cdf(uj_*yj_, scale=gamma))


def hess_calc(uj_, yj_, gamma):
    return (norm.pdf(uj_*yj_,scale=gamma)**2 - pdf_deriv(uj_*yj_, gamma)*norm.cdf(uj_*yj_,scale=gamma))/(norm.cdf(uj_*yj_,scale=gamma)**2)

"""
def probit_map_st(Z, y, gamma, w, v):
    N = v.shape[0]
    n = v.shape[1]
    def f(x):
        vec = np.zeros(N)
        tmp = v @ x
        for i, yi in zip(Z, y):
            vec[i] = - jac_calc(tmp[i], yi, gamma)
            #vec[i] = yi*norm.pdf(tmp[i]*yi, scale=gamma)/norm.cdf(tmp[i]*yi, scale=gamma)
        return w * x  - v.T @ vec
    def fprime(x):
        tmp = v @ x
        vec = np.zeros(N)
        for i, yi in zip(Z, y):
            vec[i] = -hess_calc(tmp[i], yi, gamma)
            #vec[i] = (pdf_deriv(tmp[i]*yi,gamma)*norm.cdf(tmp[i]*yi,scale=gamma)
            #             - norm.pdf(tmp[i]*yi,scale=gamma)**2)/ (norm.cdf(tmp[i]*yi,scale=gamma)**2)
        H = (-v.T * vec) @ v
        H[np.diag_indices(n)] += w
        return H
    x0 = np.random.rand(len(w))
    res = root(f, x0, jac=fprime)
    #print(f"Root Finding is successful: {res.success}")
    return v @ res.x

# OLD Code
# def Hess_inv_st(u, Z, y, w, v, gamma, debug=False):
#     """
#     Assuming matrices are sparse, since L_tau should be relatively sparse,
#         and we are perturbing with diagonal matrix.
#     """
#     H_d = np.zeros(u.shape[0])
#     for j, yj in zip(Z, y):
#         H_d[j] = hess_calc(u[j], yj, gamma)
#     post = sp.sparse.diags(w, format='csr') \
#            + v.T @ sp.sparse.diags(H_d, format='csr') @ v
#     return v @ sp.linalg.inv(post) @ v.T

def Hess_inv_st(u, Z, y, w, v, gamma, debug=False):
    H_d = np.zeros(len(Z))
    vZ = v[Z,:]
    for i, (j, yj) in enumerate(zip(Z, y)):
        H_d[i] = hess_calc2(u[j], yj, gamma)
    post = sp.sparse.diags(w, format='csr') \
           + vZ.T @ sp.sparse.diags(H_d, format='csr') @ vZ
    return v @ sp.linalg.inv(post) @ v.T



def probit_map_st2(Z, y,  gamma, w, v):
    N = v.shape[0]
    n = v.shape[1]
    def f(x):
        vec = np.zeros(N)
        tmp = v @ x
        for i, yi in zip(Z, y):
            vec[i] = - jac_calc2(tmp[i], yi, gamma)
            #vec[i] = yi*norm.pdf(tmp[i]*yi, scale=gamma)/norm.cdf(tmp[i]*yi, scale=gamma)
        return w * x  - v.T @ vec
    def fprime(x):
        tmp = v @ x
        vec = np.zeros(N)
        for i, yi in zip(Z, y):
            vec[i] = -hess_calc2(tmp[i], yi, gamma)
            #vec[i] = (pdf_deriv(tmp[i]*yi,gamma)*norm.cdf(tmp[i]*yi,scale=gamma)
            #             - norm.pdf(tmp[i]*yi,scale=gamma)**2)/ (norm.cdf(tmp[i]*yi,scale=gamma)**2)
        H = (-v.T * vec) @ v
        H[np.diag_indices(n)] += w
        return H
    x0 = np.random.rand(len(w))
    res = root(f, x0, jac=fprime)
    #print(f"Root Finding is successful: {res.success}")
    return v @ res.x

# OLD code
# def Hess_inv_st2(u, Z, y, w, v, gamma, debug=False):
#     """
#     Assuming matrices are sparse, since L_tau should be relatively sparse,
#         and we are perturbing with diagonal matrix.
#     """
#     H_d = np.zeros(u.shape[0])
#     for j, yj in zip(Z, y):
#         H_d[j] = hess_calc2(u[j], yj, gamma)
#     post = sp.sparse.diags(w, format='csr') \
#            + v.T @ sp.sparse.diags(H_d, format='csr') @ v
#     return v @ sp.linalg.inv(post) @ v.T

def Hess_inv_st2(u, Z, y, w, v, gamma, debug=False):
    H_d = np.zeros(len(Z))
    vZ = v[Z,:]
    for i, (j, yj) in enumerate(zip(Z, y)):
        H_d[i] = hess_calc2(u[j], yj, gamma)
    post = sp.sparse.diags(w, format='csr') \
           + vZ.T @ sp.sparse.diags(H_d, format='csr') @ vZ
    return v @ sp.linalg.inv(post) @ v.T


def Hess_inv(u, Z, y, gamma, Ct):
    Ctp = Ct[np.ix_(Z, Z)]
    H_d = np.zeros(len(y))
    for i, (j, yj) in enumerate(zip(Z, y)):
        H_d[i] = 1./hess_calc(u[j], yj, gamma)
    temp = sp.linalg.inv(sp.sparse.diags(H_d, format='csr') + Ctp)
    return Ct - Ct[:, Z] @ temp @ Ct[Z, :]

def Hess2_inv(u, Z, y, gamma, Ct):
    Ctp = Ct[np.ix_(Z, Z)]
    H_d = np.zeros(len(y))
    for i, (j, yj) in enumerate(zip(Z, y)):
        H_d[i] = 1./hess_calc2(u[j], yj, gamma)
    temp = sp.linalg.inv(sp.sparse.diags(H_d, format='csr') + Ctp)
    return Ct - Ct[:, Z] @ temp @ Ct[Z, :]

# OLD inefficient
# def gr_C(Z, gamma, Ct):
#     Ctp = Ct[np.ix_(Z, Z)]
#     H_d = np.ones(len(Z)) * (gamma * gamma)
#     temp = sp.linalg.inv(sp.sparse.diags(H_d, format='csr') + Ctp)
#     return Ct - Ct[:, Z] @ temp @ Ct[Z, :]

def gr_C(Z, gamma, d, v):
    H_d = len(Z) *[1./gamma**2.]
    vZ = v[Z,:]
    post = sp.sparse.diags(d, format='csr') \
           + vZ.T @ sp.sparse.diags(H_d, format='csr') @ vZ
    return v @ sp.linalg.inv(post) @ v.T

def gr_map(Z, y, gamma, d, v):
    C = gr_C(Z, gamma, d, v)
    return (C[:,Z] @ y)/(gamma * gamma)
