# Python implementation of the Laplace Kernel RKHS data-based norm of Karzand and Nowak
# @author : Kevin Miller

'''
Currently only implemented for binary (as is the case in Karzand and Nowak paper)

So current implementation doesn't scale well, because we are calculating the Gram matrix on the go.
    * Try doing Nystrom on Gram matrix, use this to compare?
    * Could use the full matrix, and show how long it takes then? This will probably do better than our method?

THOUGHT- Should we just phrase my method as a generalization and speed improvement of Nowak and Karzand?
    * Then all the tests their better performance can be put into perspective.
'''

import numpy as np
from scipy.spatial.distance import cdist
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pickle



class RKHSClassifierOld(object):
    def __init__(self, X, sigma):
        self.X = X
        self.N = self.X.shape[0]
        self.sigma = sigma
        self.f = None

    def calculate_model(self, labeled, y):
        self.labeled = labeled
        self.unlabeled = list(filter(lambda x: x not in self.labeled, range(self.N)))
        self.y = y

        self.K_inv = np.linalg.inv(np.exp(-cdist(self.X[self.labeled, :], self.X[self.labeled,:]) / self.sigma))
        self.f = np.empty(self.N)
        self.f[self.labeled] = self.y
        self.K_ul = np.exp(-cdist(self.X[self.unlabeled, :], self.X[self.labeled,:]) / self.sigma)
        self.f[self.unlabeled] = self.K_ul @ self.K_inv @ np.array(self.y) # Kul @ Kll^{-1} y

    def update_model_old(self, Q, yQ):
        len_lab_old = len(self.labeled)
        if self.f is None:
            print("No previous model calculated, so we will do initial calculation")
            self.calculate_model(Q, yQ)
            return
        self.labeled += Q
        self.y += yQ

        aQ = np.exp(-cdist(self.X[self.labeled,:], self.X[Q, :])/ self.sigma)
        aQ1 = aQ[:len_lab_old,:]
        aQ2 = aQ[-len(Q):,:]
        Z = np.linalg.inv(aQ2 - aQ1.T @ self.K_inv @ aQ1)
        A12 = self.K_inv @ aQ1 @ Z
        A11 = A12 @ aQ1.T
        A11.ravel()[::len_lab_old+1] += 1.
        A11 = A11 @ self.K_inv
        self.K_inv = np.hstack((A11, -A12))
        self.K_inv = np.vstack((self.K_inv, np.hstack((-A12.T, Z))))
        unl_Qi, unl_Q = zip(*list(filter(lambda x: x[1] not in Q, enumerate(self.unlabeled))))
        K_Qu_Q = np.exp(-cdist(self.X[unl_Q,:], self.X[Q,:])/self.sigma)
        self.K_ul = np.hstack((self.K_ul[unl_Qi, :], K_Qu_Q))
        self.unlabeled = list(filter(lambda x: x not in self.labeled, range(self.N)))
        self.f[Q] = np.array(yQ)
        self.f[self.unlabeled] = self.K_ul @ self.K_inv @ np.array(self.y) # Kul @ Kll^{-1} y
        return

    def update_model(self, Q, yQ):
        unl_Q = [self.unlabeled.index(k) for k in Q]
        unl_notQ = list(filter(lambda x: x not in unl_Q, range(len(self.unlabeled))))
        notQ = list(filter(lambda x: x not in Q, self.unlabeled))

        # update f
        SQ_inv = np.linalg.inv(np.exp(-cdist(self.X[Q,:], self.X[Q,:])/self.sigma))
        K_notQ_Q = np.exp(-cdist(self.X[notQ,:], self.X[Q,:])/self.sigma)
        sc = K_notQ_Q - self.K_ul[unl_notQ,:] @ self.K_inv @ self.K_ul[unl_Q,:].T
        self.f[notQ] += sc @ SQ_inv @ (np.array(yQ) - self.f[Q])
        self.f[Q] = yQ

        # update self.K_inv, self.K_ul for future calculations
        # calculate self.K_inv new by block matrix inversion formula
        Mat = self.K_ul[unl_Q,:].T @ SQ_inv @ self.K_ul[unl_Q,:] @ self.K_inv
        Mat.ravel()[::len(self.labeled)+1] += 1.0
        new_Kinv_lower_left = -SQ_inv @ self.K_ul[unl_Q,:] @ self.K_inv
        bottom_new_Kinv = np.hstack((new_Kinv_lower_left, SQ_inv))
        self.K_inv = self.K_inv @ Mat
        self.K_inv = np.hstack((self.K_inv, new_Kinv_lower_left.T))
        self.K_inv = np.vstack((self.K_inv, bottom_new_Kinv))

        # self.K_ul
        self.K_ul = self.K_ul[unl_notQ,:]
        self.K_ul = np.hstack((self.K_ul, K_notQ_Q))

        self.labeled.extend(Q)
        self.unlabeled = notQ

        return
    # def look_ahead_db_norm(self, k):
    #     ak = self.K_ul[self.unlabeled.index(k), :]
    #     Z = 1./(1. - np.inner(ak, self.K_inv @ ak))
    #     A12 = self.K_inv @ ak * Z
    #     akK_inv = self.K_inv @ ak
    #     A11 = self.K_inv + Z * (np.outer(akK_inv, akK_inv))
    #     K_inv_new = np.hstack((A11, -A12[:,np.newaxis]))
    #     K_inv_new = np.vstack((K_inv_new, np.hstack((-A12[np.newaxis,:], np.array([[Z]])))))
    #     f_k = self.f.copy()
    #     f_k[k] = np.sign(self.f[k])# f_k [k] = yk = lowest interpolating label
    #     unl_i, unl_k = zip(*list(filter(lambda x : x[1] != k, enumerate(self.unlabeled))))
    #     K_ku_k = np.exp(-cdist(self.X[unl_k,:], self.X[k,:][np.newaxis,:])/self.sigma)
    #     f_k[list(unl_k)] = np.hstack((self.K_ul[unl_i, :], K_ku_k)) @ K_inv_new \
    #                             @ np.array(self.y + [f_k[k]])
    #     return np.linalg.norm(f_k - self.f)#**2./len(self.unlabeled)
    #
    # def look_ahead_db_norm2(self, k):
    #     modelk = RKHSClassifier(self.X, self.sigma)
    #     modelk.calculate_model(self.labeled[:] + [k], self.y[:] + [np.sign(self.f[k])])
    #
    #     return np.linalg.norm(modelk.f - self.f)

    def look_ahead_db_norms(self, Cand):
        unl_Cand = [self.unlabeled.index(k) for k in Cand]
        KCandl = self.K_ul[unl_Cand,:]
        sc = np.exp(-cdist(self.X[self.unlabeled, :], self.X[Cand,:])/self.sigma) \
                - self.K_ul @ self.K_inv @ KCandl.T
        return (1. - np.absolute(self.f[Cand]))*np.linalg.norm(sc, axis=0)/sc[unl_Cand, range(len(Cand))]

        vals = []
        for ii, i in enumerate(unl_Cand):
            vals.append((1. - np.absolute(self.f[self.unlabeled[i]]))/(1. - B[i,ii]) * np.linalg.norm(B[:,ii]))
        return np.array(vals)

    # def look_ahead_db_norms2(self, Cand):
    #     unl_Cand = list(filter(lambda x: self.unlabeled[x] in Cand, range(len(self.unlabeled))))
    #     sc_submatrix = np.exp(-cdist(self.X[self.unlabeled, :], self.X[Cand,:])/self.sigma) \
    #                         - self.K_ul @ self.K_inv @ self.K_ul[unl_Cand,:].T
    #     return np.abs(1. - self.f[Cand]) *  np.linalg.norm(sc_submatrix, axis=0) / np.abs(sc_submatrix[unl_Cand, range(len(Cand))])

    # def look_ahead_db_norms2(self, Cand):
    #     unl_Cand = [self.unlabeled.index(k) for k in Cand]
    #     sc = np.exp(-cdist(self.X[self.unlabeled, :], self.X[self.unlabeled,:])/self.sigma) \
    #                         - self.K_ul @ self.K_inv @ self.K_ul.T
    #     return (1. - np.absolute(self.f[Cand])) *  np.linalg.norm(sc[:,unl_Cand], axis=0) / np.abs(sc[unl_Cand, unl_Cand])





class RKHSClassifier(object):
    def __init__(self, X, sigma):
        self.N = X.shape[0]
        self.sigma = sigma
        self.f = None
        self.K = np.exp(-cdist(X, X)/self.sigma) # calculate full, dense kernel matrix upfront
        self.modelname = 'rkhs'

    def calculate_model(self, labeled, y):
        self.labeled = labeled
        self.unlabeled = list(filter(lambda x: x not in self.labeled, range(self.N)))
        self.y = y

        self.K_inv = np.linalg.inv(self.K[np.ix_(self.labeled, self.labeled)])
        self.f = np.empty(self.N)
        self.f[self.labeled] = self.y
        self.K_ul = self.K[np.ix_(self.unlabeled, self.labeled)]
        self.f[self.unlabeled] = self.K_ul @ self.K_inv @ np.array(self.y) # Kul @ Kll^{-1} y

    def update_model(self, Q, yQ):
        unl_Q = [self.unlabeled.index(k) for k in Q]
        unl_notQ = list(filter(lambda x: x not in unl_Q, range(len(self.unlabeled))))
        notQ = list(filter(lambda x: x not in Q, self.unlabeled))

        # update f
        SQ_inv = np.linalg.inv(self.K[np.ix_(Q, Q)])
        K_notQ_Q = self.K[np.ix_(notQ, Q)]
        sc = K_notQ_Q - self.K_ul[unl_notQ,:] @ self.K_inv @ self.K_ul[unl_Q,:].T
        self.f[notQ] += sc @ SQ_inv @ (np.array(yQ) - self.f[Q])
        self.f[Q] = yQ

        # update self.K_inv, self.K_ul for future calculations
        # calculate self.K_inv new by block matrix inversion formula
        Mat = self.K_ul[unl_Q,:].T @ SQ_inv @ self.K_ul[unl_Q,:] @ self.K_inv
        Mat.ravel()[::len(self.labeled)+1] += 1.0
        new_Kinv_lower_left = -SQ_inv @ self.K_ul[unl_Q,:] @ self.K_inv
        bottom_new_Kinv = np.hstack((new_Kinv_lower_left, SQ_inv))
        self.K_inv = self.K_inv @ Mat
        self.K_inv = np.hstack((self.K_inv, new_Kinv_lower_left.T))
        self.K_inv = np.vstack((self.K_inv, bottom_new_Kinv))

        # self.K_ul
        self.K_ul = self.K_ul[unl_notQ,:]
        self.K_ul = np.hstack((self.K_ul, K_notQ_Q))

        self.labeled.extend(Q)
        self.unlabeled = notQ

        return


    def look_ahead_db_norms(self, Cand):
        unl_Cand = [self.unlabeled.index(k) for k in Cand]
        KCandl = self.K_ul[unl_Cand,:]
        sc = self.K[np.ix_(self.unlabeled, Cand)] - self.K_ul @ self.K_inv @ KCandl.T
        return (1. - np.absolute(self.f[Cand]))*np.linalg.norm(sc, axis=0)/sc[unl_Cand, range(len(Cand))]



if __name__ == "__main__":
    import sys
    sys.path.append('..')
    import os
    from al_util import get_acc
    parser = ArgumentParser(description="Read in previous RKHS run and check classifier")
    parser.add_argument("--loc", default='../checker2/db-rkhs-2000-0.1-0.1/rand-top-5-100-1.txt', type=str)
    parser.add_argument("--Xloc", default='../checker2/X_labels.npz', type=str)
    parser.add_argument("--sigma", default='0.1', type=str)
    args = parser.parse_args()
    print(float(args.sigma))
    labeled = []
    with open(args.loc, 'r') as f:
        for i, line in enumerate(f.readlines()):
            # read in init_labeled, and initial accuracy
            line = line.split(',')
            labeled.extend([int(x) for x in line[:-2]])
            if i == 0:
                num_init = len(labeled)

    lab_set = set(labeled)
    print(len(lab_set), len(labeled))
    data = np.load(args.Xloc, allow_pickle=True)
    X, labels = data['X'], data['labels']

    model = RKHSClassifier(X, sigma=float(args.sigma))
    model_new = RKHSClassifierNew(X, sigma=float(args.sigma))


    model.calculate_model(labeled[:20], list(labels[labeled[:20]]))
    model_new.calculate_model(labeled[:20], list(labels[labeled[:20]]))

    assert np.allclose(model.f, model_new.f)

    Cand = list(np.random.choice(model.unlabeled, 50))
    orig_vals = model.look_ahead_db_norms(Cand[:])
    new_vals = model_new.look_ahead_db_norms(Cand[:])

    assert np.allclose(orig_vals, new_vals)

    model.update_model(Cand[:5], list(labels[Cand[:5]]))
    model_new.update_model(Cand[:5], list(labels[Cand[:5]]))

    assert np.allclose(model.f, model_new.f)
    print("passed all tests!")
    # for i in np.arange(0,100,10):
    #     K = num_init+i*5
    #     model = RKHSClassifier(X, sigma=float(args.sigma))
    #     model.calculate_model(labeled[:K], list(labels[labeled[:K]]))
    #     model2 = RKHSClassifier(X, sigma=float(args.sigma))
    #     model2.calculate_model(labeled[:K], list(labels[labeled[:K]]))
    #
    #
    #
    #     assert np.allclose(model.f, model2.f)
    #
    #     Q = list(np.random.choice(model.unlabeled, 5))
    #
    #     model3 = RKHSClassifier(X, sigma=float(args.sigma))
    #     model3.calculate_model(labeled[:K] + Q, list(labels[labeled[:K]]) + list(labels[Q]))
    #
    #     #print(list(labels[Q]))
    #     model.update_model_old(Q, list(labels[Q]))
    #     model2.update_model(Q, list(labels[Q]))
    #
    #     if np.allclose(model.f, model2.f):
    #         print("both models are close in the update")
    #     else:
    #         print("true to old - " + str(np.allclose(model.f, model3.f)))
    #         print("true to new - " + str(np.allclose(model2.f, model3.f)))
    #         print(np.linalg.norm(model.f - model2.f)/np.linalg.norm(model.f))
    #         # print(labels[model.labeled], model.f[model.labeled])
    #         # print(labels[model2.labeled], model.f[model2.labeled])
    #         plt.scatter(range(model.N), model3.f, label='true', marker='^')
    #         plt.scatter(range(model.N), model.f, label='old', marker='x')
    #         plt.scatter(range(model.N), model2.f, label='new', marker='.')
    #
    #         plt.legend()
    #         plt.savefig('./comp-%d.png' % i)
    #         plt.show(0)
    #         plt.close()
    #         print()
    #     #assert np.allclose(model.f, model2.f)
    #
    #     # print(os.path.exists('rkhs-model-0.npz'))
    #     # data = np.load('rkhs-model-%d.npz' % i)
    #     # print(type(data))
    #     # print(list(data.keys()))
    #     # saved_f, saved_lab = data['f'], data['lab']
    #     # print(model.labeled)
    #     # print(saved_lab)
    #     # print(np.allclose(saved_f, model.f))
    #     # print(K, len(model.labeled), get_acc(model.f, labels, unlabeled=model.unlabeled)[1], get_acc(saved_f, labels, unlabeled=model.unlabeled)[1])
    #     # fig, (ax1, ax2) = plt.subplots(1,2)
    #     # ax1.scatter(X[:, 0], X[:, 1], c=labels)
    #     # ax1.set_title("Ground Truth")
    #     # ax2.scatter(X[:,0], X[:,1], c=np.sign(model.f))
    #     # ax2.scatter(X[labeled[:K],0], X[labeled[:K],1], c='k', marker='^')
    #     # ax2.set_title("Calculated -- acc = {:.4f}".format(get_acc(model.f, labels, unlabeled=model.unlabeled)[1]))
    #     # plt.savefig('./check2-rkhs-%d.png' % K)
    #     # plt.show(0)
