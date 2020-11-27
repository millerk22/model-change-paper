import numpy as np
import time
import copy
import sys
from sklearn.preprocessing import OneHotEncoder
sys.path.append('..')
from util.activelearner import *
from util.gbssl import *


BMODELNAMES = ['gr', 'log', 'probitnorm']
MMODELNAMES = ['gr', 'ce']
ACQS = ['mc', 'uncertainty', 'rand', 'vopt', 'sopt', 'mbr', 'mcgreedy']


def run_experiment_binary(w, v, tau, gamma, oracle, init_labeled, num_al_iters, B_per_al_iter, acq='mc',
                          cand='rand', select_method='top', modelname='gr', full=False, root_filename='./',
                          run=-1, verbose=False):
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

    if run == -1:
        raise ValueError("Need to include which run this is. Make sure you know which initially labeled set corresponds to this run")

    if modelname not in BMODELNAMES:
        raise ValueError("modelname %s not in list of possible modelnames : \n%s" % (
            modelname, str(BMODELNAMES)))
    if acq not in ACQS:
        raise ValueError(
            "acq = %s is not a valid acquisition function currently implemented:\n\t%s" % (acq, str(ACQS)))

    parent_filename = root_filename + "%s-%s-%d-%s-%s/" % (acq, modelname, v.shape[1], str(tau), str(gamma))
    if not os.path.exists(parent_filename):
        os.makedirs(parent_filename)

    experiment_name = parent_filename + "%s-%s-%d-%d-%d.txt" % (cand, select_method, B_per_al_iter, num_al_iters, run)

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


    # write initial line of file
    f = open(experiment_name, 'w')
    f.write(','.join(str(n) for n in init_labeled) + ',--,%1.6f\n' % acc )

    # instantiate ActiveLearner object
    print("ActiveLearner Settings:\n\tacq = \t%s\n\tcand = \t%s" % (acq, cand))
    print("\tselect_method = %s, B = %d" % (select_method, B_per_al_iter))
    AL = ActiveLearner(acquisition=acq, candidate=cand)

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
        f.write(','.join(str(n) for n in Q) + ',%f,%1.6f\n' %((toc - tic), acc))


    f.close()
    return





def run_experiment_multi(w, v, tau, gamma, oracle, init_labeled, num_al_iters, B_per_al_iter, acq='mc',
                         cand='rand', select_method='top', modelname='gr', full=False, root_filename='./',
                          run=-1, verbose=False):
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
    if run == -1:
        raise ValueError("Need to include which run this is. Make sure you know which initially labeled set corresponds to this run")


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

    if modelname == 'gr':  # GR is implemented in the Binary model since it requires same storage structure
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
        model = SoftmaxGraphBasedSSLModelReduced(gamma, tau, w=w, v=v)


    parent_filename = root_filename + "%s-%s-%d-%s-%s/" % (acq, modelname, v.shape[1], str(tau), str(gamma))
    if not os.path.exists(parent_filename):
        os.makedirs(parent_filename)

    experiment_name = parent_filename + "%s-%s-%d-%d-%d.txt" % (cand, select_method, B_per_al_iter, num_al_iters, run)


    # calculate one-hot labels for oracle
    enc = OneHotEncoder()
    enc.fit(oracle.reshape((-1, 1)))
    oracle_onehot = enc.transform(oracle.reshape((-1, 1))).todense()

    # train the initial model, record accuracy
    model.calculate_model(
        labeled=init_labeled[:], y=oracle_onehot[init_labeled])
    acc = get_acc_multi(np.argmax(model.m, axis=1),
                         oracle, unlabeled=model.unlabeled)[1]

    # write initial line of file
    f = open(experiment_name, 'w')
    f.write(','.join(str(n) for n in init_labeled) + ',--,%1.6f\n' % acc )

    # instantiate ActiveLearner object
    print("ActiveLearner Settings:\n\tacq = \t%s\n\tcand = \t%s" % (acq, cand))
    print("\tselect_method = %s, B = %d" % (select_method, B_per_al_iter))
    AL = ActiveLearner(acquisition=acq, candidate=cand)

    for al_iter in range(num_al_iters):
        if verbose or (al_iter % 10 == 0):
            print("AL Iteration %d, acc=%1.6f" % (al_iter + 1, acc))
        # select query points via active learning
        tic = time.perf_counter()
        Q = AL.select_query_points(
            model, B_per_al_iter, method=select_method, verbose=verbose)
        toc = time.perf_counter()

        # query oracle
        yQ = oracle_onehot[Q]

        # update model, and calculate updated model's accuracy
        model.update_model(Q, yQ)
        acc = get_acc_multi(np.argmax(model.m, axis=1),
                         oracle, unlabeled=model.unlabeled)[1]
        f.write(','.join(str(n) for n in Q) + ',%f,%1.6f\n' %((toc - tic), acc))

    f.close()
    return







def get_data_from_runs(acq, modelname, M, tau, gamma, cand, select_method, B, num_al_iters, runs=[1], root_filename='./'):
    parent_filename = root_filename + "%s-%s-%d-%s-%s/" % (acq, modelname, M, str(tau), str(gamma))
    if not os.path.exists(parent_filename):
        raise ValueError("data at %s does not exist..." % parent_filename)
    RUNS = {}
    for run in runs:
        experiment_name = "%s-%s-%d-%d-%d.txt" % (cand, select_method, B, num_al_iters, run)
        if not os.path.exists(parent_filename + experiment_name):
            print('Run #%d that you requested does not exist at %s, skipping' % (run, parent_filename + experiment_name))
        else:
            with open(parent_filename + experiment_name, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    # read in init_labeled, and initial accuracy
                    if i == 0:
                        line = line.split(',')
                        RUNS[run] = {'init_labeled': [int(x) for x in line[:-2]], 'acc':[float(line[-1])], 'times':[], 'choices':[]}
                    else:
                        line = line.split(',')
                        RUNS[run]['acc'].append(float(line[-1]))
                        RUNS[run]['choices'].extend(int(x) for x in line[:-2])
                        RUNS[run]['times'].append(float(line[-2]))

    return RUNS

def get_avg_acc_from_runs_dict(RUNS, runs=[1]):
    count = len(runs)
    accs = []
    for run in runs:
        if run not in RUNS:
            print("Run #%d not in RUNS dictionary given, skipping..." % run)
        else:
            accs.append(RUNS[run]['acc'])
    if len(accs) == 0:
        print("No valid runs found, returning None")
        return
    accs = np.array(accs)
    return np.average(accs, axis=0), np.std(accs, axis=0)
