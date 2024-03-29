import numpy as np
import time
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_moons, make_blobs

from .activelearner import *
from .gbssl import *

import mlflow





BMODELNAMES = ['gr', 'log', 'probitnorm']
MMODELNAMES = ['mgr', 'ce']
OTHERMODELNAMES = ['rkhs', 'hf']
ACQS = ['mc', 'uncertainty', 'rand', 'vopt', 'sopt', 'mbr', 'mcgreedy', 'mcavg', 'mcavgf', 'mcf']

def create_checkerboard2(N):
    X = np.random.rand(N,2)
    labels = []
    for x in X:
        i, j = 0,0
        if 0.25 <= x[0] and x[0] < 0.5:
            i = 1
        elif 0.5 <= x[0] and x[0] < 0.75:
            i = 2
        elif 0.75 <= x[0]:
            i = 3

        if 0.25 <= x[1] and x[1] < 0.5:
            j = 1
        elif 0.5 <= x[1] and x[1] < 0.75:
            j = 2
        elif 0.75 <= x[1]:
            j = 3

        labels.append((i+j) % 2)
    return X, np.array(labels)


def create_binary_clusters():
    np.random.seed(4)
    Xm, labelsm = make_moons(200, shuffle=False, noise=0.12)
    X1, labels1 = make_blobs([50,60, 40, 30, 40], 2, shuffle=False, centers=[[1.6,-1.3],[1.3,1.7], [0.5, 2.4], [0.2,-1.], [-1.7,2.2]], cluster_std=[.26, .23, .23, .26, .23])
    labels1 = labels1 % 2
    X2 = np.random.randn(100,2) @ np.array([[.4, 0.],[0.,.3]]) + np.array([-1.5,-.8])
    X3 = np.random.randn(70,2) @ np.array([[.4, 0.],[0.,.3]]) + np.array([2.5,2.8])
    x11, x12 = np.array([-2., 0.8])[np.newaxis, :], np.array([-.2,2.])[np.newaxis, :]
    l1 = (x11 + np.linspace(0,1, 80)[:, np.newaxis] @ (x12 - x11))  + np.random.randn(80, 2)*0.18
    x21, x22 = np.array([2.5, -1.5])[np.newaxis, :], np.array([2.5, 2.])[np.newaxis, :]
    l2 = (x21 + np.linspace(0,1, 90)[:, np.newaxis] @ (x22 - x21))  + np.random.randn(90, 2)*0.2


    X = np.concatenate((Xm, X1, X2, X3, l1, l2))
    labels = np.concatenate((labelsm, labels1, np.zeros(100), np.ones(70), np.ones(80), np.zeros(90)))

    return X, labels


def create_checkerboard3(N):
    X = np.random.rand(N,2)
    labels = []
    for x in X:
        i, j = 0,0
        if 0.33333 <= x[0] and x[0] < 0.66666:
            i = 1
        elif 0.66666 <= x[0]:
            i = 2

        if 0.33333 <= x[1] and x[1] < 0.66666:
            j = 1
        elif 0.66666 <= x[1]:
            j = 2

        labels.append(3*j + i)
    labels = np.array(labels)
    labels[labels == 4] = 0
    labels[labels == 8] = 0
    labels[labels == 5] = 1
    labels[labels == 6] = 1
    labels[labels == 3] = 2
    labels[labels == 7] = 2
    return X, labels



def run_binary(w, v, tau, gamma, oracle, init_labeled, num_al_iters, B_per_al_iter, modelname='gr', acq='mc',
                          cand='rand', select_method='top', full=False,
                          verbose=False):
    '''
    Inputs:
      w = eigenvalue numpy array
      v = eigenvectors numpy array (columns)
      oracle = "labels" ground truth numpy array, in {0, 1, ..., n_c} or {-1, 1}
      init_labeled = list of indices that are initially labeled, per ordering in oracle and rows of v
      num_al_iters = total number of active learning iterations to perform
      B_per_al_iter = batch size B that will be done on each iteration
      acq = string that refers to the acquisition function to be tried in this experiment

    Outputs:
      labeled : list of indices of labeled points chosen throughout whole active learning process
      acc : list of length (num_al_iters + 1) corresponding to the accuracies of the current classifer at each AL iteration
    '''



    if modelname not in BMODELNAMES:
        raise ValueError("modelname %s not in list of possible modelnames : \n%s" % (
            modelname, str(BMODELNAMES)))
    if acq not in ACQS:
        raise ValueError(
            "acq = %s is not a valid acquisition function currently implemented:\n\t%s" % (acq, str(ACQS)))


    N, M = v.shape
    if M < N:
        truncated = True
    else:
        truncated = False

    if -1 not in np.unique(oracle):
        oracle[oracle == 0] = -1

    if truncated and not full:
        print("Binary %s Reduced Model -- i.e. not storing full C covariance matrix" % modelname)
        model = BinaryGraphBasedSSLModelReduced(
            modelname, gamma, tau, w=w, v=v)
    elif truncated and full:
        print("Binary %s FULL Model, but Truncated eigenvalues" % modelname)
        model = BinaryGraphBasedSSLModel(modelname, gamma, tau, w=w, v=v)
    else:
        print("Binary %s FULL Model, with ALL eigenvalues" % modelname)
        model = BinaryGraphBasedSSLModel(modelname, gamma, tau, w=w, v=v)

    # train the initial model, record accuracy
    model.calculate_model(labeled=init_labeled[:], y=list(oracle[init_labeled]))
    acc = get_acc(model.m, oracle, unlabeled=model.unlabeled)[1]

    mlflow.log_metric('init_acc', acc)


    # instantiate ActiveLearner object
    print("ActiveLearner Settings:\n\tacq = \t%s\n\tcand = \t%s" % (acq, cand))
    print("\tselect_method = %s, B = %d" % (select_method, B_per_al_iter))
    AL = ActiveLearner(acquisition=acq, candidate=cand)


    iter_acc = []
    iter_time = []
    al_choices = []
    for al_iter in range(num_al_iters):
        if verbose or (al_iter % 10 == 0):
            print("AL Iteration %d, acc=%1.6f" % (al_iter + 1, acc))
        # select query points via active learning
        tic = time.perf_counter()
        Q = AL.select_query_points(
            model, B_per_al_iter, method=select_method, verbose=verbose)
        toc = time.perf_counter()

        # query oracle
        yQ = list(oracle[Q])


        # update model, and calculate updated model's accuracy
        model.update_model(Q, yQ)
        acc = get_acc(model.m, oracle, unlabeled=model.unlabeled)[1]
        iter_acc.append(acc)
        iter_time.append(toc - tic)
        al_choices.append(Q)

    np.savez('tmp/iter_stats.npz', al_choices=np.array(al_choices), iter_acc=np.array(iter_acc), iter_time=np.array(iter_time))
    mlflow.log_artifact('tmp/iter_stats.npz')

    return


def run_rkhs_hf(oracle, init_labeled, num_al_iters, B_per_al_iter, modelname='rkhs', h=0.1, delta=0.1, X=None, L=None,
                          cand='rand', select_method='top', acq='db', verbose=False):
    '''
    Inputs:
      X = dataset
      oracle = "labels" ground truth numpy array, in {0, 1, ..., n_c} or {-1, 1}
      init_labeled = list of indices that are initially labeled, per ordering in oracle and rows of v
      num_al_iters = total number of active learning iterations to perform
      B_per_al_iter = batch size B that will be done on each iteration

    Outputs:
      labeled : list of indices of labeled points chosen throughout whole active learning process
      acc : list of length (num_al_iters + 1) corresponding to the accuracies of the current classifer at each AL iteration
    '''




    if modelname == 'rkhs':
        assert X is not None
        model = RKHSClassifier(X, sigma=h) # bandwidth from Karzand paper
    else:
        assert L is not None
        model = HFGraphBasedSSLModel(delta, L)

    # train the initial model, record accuracy
    if len(np.unique(oracle)) > 2:
        # calculate one-hot labels for oracle
        enc = OneHotEncoder()
        enc.fit(oracle.reshape((-1, 1)))
        oracle_onehot = enc.transform(oracle.reshape((-1, 1))).todense()
        y_init = oracle_onehot[init_labeled]
    else:
        # binary case
        if -1 not in np.unique(oracle):
            oracle[oracle == 0] = -1
        y_init = list(oracle[init_labeled])

    model.calculate_model(labeled=init_labeled[:], y=y_init)
    if model.nc > 2:
        acc = get_acc_multi(np.argmax(model.f, axis=1),
                             oracle, unlabeled=model.unlabeled)[1]
    else:
        acc = get_acc(model.f, oracle, unlabeled=model.unlabeled)[1]

    mlflow.log_metric('init_acc', acc)

    # instantiate ActiveLearner object
    print("ActiveLearner Settings:\n\t{} {}".format(modelname.upper(), acq.upper()))
    print("\tselect_method = %s, B = %d" % (select_method, B_per_al_iter))
    AL = ActiveLearner(acquisition=acq, candidate=cand)


    iter_acc = []
    iter_time = []
    al_choices = []
    for al_iter in range(num_al_iters):
        if verbose or (al_iter % 10 == 0):
            print("AL Iteration %d, acc=%1.6f" % (al_iter + 1, acc))
        # select query points via active learning
        tic = time.perf_counter()
        Q = AL.select_query_points(
            model, B_per_al_iter, method=select_method, verbose=verbose)
        toc = time.perf_counter()

        # query oracle
        if model.nc > 2:
            yQ = oracle_onehot[Q]
        else:
            yQ = list(oracle[Q])


        # update model, and calculate updated model's accuracy
        model.update_model(Q, yQ)
        if model.nc > 2:
            acc = get_acc_multi(np.argmax(model.f, axis=1),
                                 oracle, unlabeled=model.unlabeled)[1]
        else:
            acc = get_acc(model.f, oracle, unlabeled=model.unlabeled)[1]
        iter_acc.append(acc)
        iter_time.append(toc - tic)
        al_choices.append(Q)

    np.savez('tmp/iter_stats.npz', al_choices=np.array(al_choices), iter_acc=np.array(iter_acc), iter_time=np.array(iter_time))
    mlflow.log_artifact('tmp/iter_stats.npz')

    return


def run_multi(w, v, tau, gamma, oracle, init_labeled, num_al_iters, B_per_al_iter,
                         modelname='mgr', acq='mc', cand='rand', select_method='top', full=False,
                         verbose=False):
    '''
    Inputs:
      w = eigenvalue numpy array
      v = eigenvectors numpy array (columns)
      oracle = "labels" ground truth numpy array, in {0, 1, ..., n_c} or {-1, 1}
      init_labeled = list of indices that are initially labeled, per ordering in oracle and rows of v
      num_al_iters = total number of active learning iterations to perform
      B_per_al_iter = batch size B that will be done on each iteration
      acq = string that refers to the acquisition function to be tried in this experiment

    Outputs:
      labeled : list of indices of labeled points chosen throughout whole active learning process
      acc : list of length (num_al_iters + 1) corresponding to the accuracies of the current classifer at each AL iteration
    '''

    if modelname not in MMODELNAMES:
        raise ValueError("modelname %s not in list of possible modelnames : \n%s" % (
            modelname, str(MMODELNAMES)))
    if acq not in ACQS:
        raise ValueError(
            "acq = %s is not a valid acquisition function currently implemented:\n\t%s" % (acq, str(ACQS)))

    N, M = v.shape
    if M < N:
        truncated = True
    else:
        truncated = False

    if modelname == 'mgr':  # GR is implemented in the Binary model since it requires same storage structure
        if truncated and not full:
            print(
                "Multi %s Reduced Model -- i.e. not storing full C covariance matrix" % modelname)
            model = BinaryGraphBasedSSLModelReduced(
                modelname, gamma, tau, w=w, v=v)
        elif truncated and full:
            print("Multi %s FULL Model, but Truncated eigenvalues" % modelname)
            model = BinaryGraphBasedSSLModel(modelname, gamma, tau, w=w, v=v)
        else:
            print("Multi %s FULL Model, with ALL eigenvalues" % modelname)
            model = BinaryGraphBasedSSLModel(modelname, gamma, tau, w=w, v=v)
    else:
        print("Multi %s Reduced Model -- i.e. not storing full C covariance matrix" % modelname)
        model = CrossEntropyGraphBasedSSLModelReduced(gamma, tau, w=w, v=v)


    # calculate one-hot labels for oracle
    enc = OneHotEncoder()
    enc.fit(oracle.reshape((-1, 1)))
    oracle_onehot = enc.transform(oracle.reshape((-1, 1))).todense()

    # train the initial model, record accuracy
    model.calculate_model(
        labeled=init_labeled[:], y=oracle_onehot[init_labeled])
    acc = get_acc_multi(np.argmax(model.m, axis=1),
                         oracle, unlabeled=model.unlabeled)[1]
    mlflow.log_metric('init_acc', acc)


    # instantiate ActiveLearner object
    print("ActiveLearner Settings:\n\tacq = \t%s\n\tcand = \t%s" % (acq, cand))
    print("\tselect_method = %s, B = %d" % (select_method, B_per_al_iter))
    AL = ActiveLearner(acquisition=acq, candidate=cand)

    iter_acc = []
    iter_time = []
    al_choices = []
    beta = 0.
    for al_iter in range(num_al_iters):
        if verbose or (al_iter % 1 == 0):
            print("AL Iteration %d, acc=%1.6f" % (al_iter + 1, acc))
            if acq in ['mcavg', 'mcavgf']:
                #beta = 1./(1. + al_iter // 10)
                beta = (1. - (al_iter/float(num_al_iters)))
                if beta < 0:
                    beta = 0.0
                # if al_iter < 8:
                #     beta = 1.0
                # else:
                #     beta = 0.0
                print("\tbeta = {:.3f}".format(beta))
        # select query points via active learning
        tic = time.perf_counter()
        Q = AL.select_query_points(
            model, B_per_al_iter, method=select_method, verbose=verbose, mcavg_beta=beta)
        toc = time.perf_counter()

        # query oracle
        yQ = oracle_onehot[Q]

        # update model, and calculate updated model's accuracy
        model.update_model(Q, yQ)
        acc = get_acc_multi(np.argmax(model.m, axis=1),
                         oracle, unlabeled=model.unlabeled)[1]
        iter_acc.append(acc)
        iter_time.append(toc - tic)
        al_choices.append(Q)


    np.savez('tmp/iter_stats.npz', al_choices=np.array(al_choices), iter_acc=np.array(iter_acc), iter_time=np.array(iter_time))
    mlflow.log_artifact('tmp/iter_stats.npz')

    return





























#
#
# def run_test(oracle, init_labeled, num_al_iters, B_per_al_iter, modelname='gr', acq='mc',
#                           cand='rand', select_method='top', w=None, v=None, tau=0.1, gamma=0.1,
#                           X=None, L=None, h=0.1, delta=0.1,full=False, verbose=False):
#
#     # if modelname not in BMODELNAMES:
#     #     raise ValueError("modelname %s not in list of possible modelnames : \n%s" % (
#     #         modelname, str(BMODELNAMES)))
#     # if acq not in ACQS:
#     #     raise ValueError(
#     #         "acq = %s is not a valid acquisition function currently implemented:\n\t%s" % (acq, str(ACQS)))
#
#     if v is not None:
#         N, M = v.shape
#         if M < N:
#             truncated = True
#         else:
#             truncated = False
#
#     if modelname in BMODELNAMES:
#         assert v is not None
#         assert w is not None
#         if -1 not in np.unique(oracle):
#             oracle[oracle == 0] = -1
#         if truncated and not full:
#             print("Binary %s Reduced Model -- i.e. not storing full C covariance matrix" % modelname)
#             model = BinaryGraphBasedSSLModelReduced(
#                 modelname, gamma, tau, w=w, v=v)
#         elif truncated and full:
#             print("Binary %s FULL Model, but Truncated eigenvalues" % modelname)
#             model = BinaryGraphBasedSSLModel(modelname, gamma, tau, w=w, v=v)
#         else:
#             print("Binary %s FULL Model, with ALL eigenvalues" % modelname)
#             model = BinaryGraphBasedSSLModel(modelname, gamma, tau, w=w, v=v)
#
#         ylab = list(oracle[init_labeled])
#
#     elif modelname in MMODELNAMES:
#         assert v is not None
#         assert w is not None
#         if modelname == 'mgr':  # GR is implemented in the Binary model since it requires same storage structure
#             if truncated and not full:
#                 print(
#                     "Multi %s Reduced Model -- i.e. not storing full C covariance matrix" % modelname)
#                 model = BinaryGraphBasedSSLModelReduced(
#                     modelname, gamma, tau, w=w, v=v)
#             elif truncated and full:
#                 print("Multi %s FULL Model, but Truncated eigenvalues" % modelname)
#                 model = BinaryGraphBasedSSLModel(modelname, gamma, tau, w=w, v=v)
#             else:
#                 print("Multi %s FULL Model, with ALL eigenvalues" % modelname)
#                 model = BinaryGraphBasedSSLModel(modelname, gamma, tau, w=w, v=v)
#         else:
#             print("Multi %s Reduced Model -- i.e. not storing full C covariance matrix" % modelname)
#             model = CrossEntropyGraphBasedSSLModelReduced(gamma, tau, w=w, v=v)
#
#         enc = OneHotEncoder()
#         enc.fit(oracle.reshape((-1, 1)))
#         oracle_onehot = enc.transform(oracle.reshape((-1, 1))).todense()
#         ylab = oracle_onehot[init_labeled]
#
#     elif modelname in OTHERMODELNAMES:
#         if modelname == 'rkhs':
#             assert X is not None
#             assert acq == 'db'
#             model = RKHSClassifier(X, sigma=h) # bandwidth from Karzand paper
#         else:
#             assert L is not None
#             assert acq in ['vopt', 'sopt']
#             model = HFGraphBasedSSLModel(delta, L)
#
#         ylab = list(oracle[init_labeled])
#     else:
#         raise ValueError("{} is not a valid model name")
#
#
#
#     # train the initial model, record accuracy
#     model.calculate_model(labeled=init_labeled[:], y=ylab[:])
#     acc = get_acc(model.m, oracle, unlabeled=model.unlabeled)[1]
#     mlflow.log_metric('init_acc', acc)
#
#
#     # instantiate ActiveLearner object
#     print("ActiveLearner Settings:\n\tacq = \t%s\n\tcand = \t%s" % (acq, cand))
#     print("\tselect_method = %s, B = %d" % (select_method, B_per_al_iter))
#     AL = ActiveLearner(acquisition=acq, candidate=cand)
#
#
#     iter_acc = []
#     iter_time = []
#     al_choices = []
#     for al_iter in range(num_al_iters):
#         if verbose or (al_iter % 10 == 0):
#             print("AL Iteration %d, acc=%1.6f" % (al_iter + 1, acc))
#         # select query points via active learning
#         tic = time.perf_counter()
#         Q = AL.select_query_points(
#             model, B_per_al_iter, method=select_method, verbose=verbose)
#         toc = time.perf_counter()
#
#         # query oracle
#         yQ = list(oracle[Q])
#
#
#         # update model, and calculate updated model's accuracy
#         model.update_model(Q, yQ)
#         acc = get_acc(model.m, oracle, unlabeled=model.unlabeled)[1]
#         iter_acc.append(acc)
#         iter_time.append(toc - tic)
#         al_choices.append(Q)
#
#     np.savez('tmp/iter_stats.npz', al_choices=np.array(al_choices), iter_acc=np.array(iter_acc), iter_time=np.array(iter_time))
#     mlflow.log_artifact('tmp/iter_stats.npz')
#
#     return
#
#
#
#
#
# def get_data_from_runs(acq, modelname, M, tau, gamma, cand, select_method, B, num_al_iters, runs=[1], root_filename='./'):
#     parent_filename = root_filename + "%s-%s-%d-%s-%s/" % (acq, modelname, M, str(tau), str(gamma))
#     if not os.path.exists(parent_filename):
#         raise ValueError("data at %s does not exist..." % parent_filename)
#     RUNS = {}
#     for run in runs:
#         experiment_name = "%s-%s-%d-%d-%d.txt" % (cand, select_method, B, num_al_iters, run)
#         if not os.path.exists(parent_filename + experiment_name):
#             print('Run #%d that you requested does not exist at %s, skipping' % (run, parent_filename + experiment_name))
#         else:
#             with open(parent_filename + experiment_name, 'r') as f:
#                 for i, line in enumerate(f.readlines()):
#                     # read in init_labeled, and initial accuracy
#                     if i == 0:
#                         line = line.split(',')
#                         RUNS[run] = {'init_labeled': [int(x) for x in line[:-2]], 'acc':[float(line[-1])], 'times':[], 'choices':[]}
#                     else:
#                         line = line.split(',')
#                         RUNS[run]['acc'].append(float(line[-1]))
#                         RUNS[run]['choices'].extend(int(x) for x in line[:-2])
#                         RUNS[run]['times'].append(float(line[-2]))
#
#     return RUNS
#
# def get_avg_acc_from_runs_dict(RUNS, runs=[1]):
#     count = len(runs)
#     accs = []
#     for run in runs:
#         if run not in RUNS:
#             print("Run #%d not in RUNS dictionary given, skipping..." % run)
#         else:
#             accs.append(RUNS[run]['acc'])
#     if len(accs) == 0:
#         print("No valid runs found, returning None")
#         return
#     accs = np.array(accs)
#     return np.average(accs, axis=0), np.std(accs, axis=0)
