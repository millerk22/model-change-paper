# mlflow utilization
import numpy as np
import scipy.sparse as sps
import argparse
import time
from sklearn.model_selection import train_test_split
import copy
import os
import math
import sys
import matplotlib.pyplot as plt
from util.graph_manager import GraphManager
from util.runs_util import *

import mlflow
from util.mlflow_util import *


# These are acq-model pairs available for a multiclass problem
ACQ_MODELS = ['vopt--mgr', 'sopt--mgr', 'sopt--hf', 'mc--mgr', 'mcgreedy--ce', 'mc--ce', \
        'rand--ce', 'rand--mgr', 'vopt--hf', 'uncertainty--mgr', 'uncertainty--ce']

EXPERIMENT_NAME = 'dataset'

GRAPH_PARAMS = {
    'knn' :10,
    'sigma' : 3.,
    'normalized' : True,
    'zp_k' : 5
}





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run Active Learning experiment on Multiclass dataset located at --data_root.')
    parser.add_argument('--data_root', default='./data/dataset/', type=str, help='Location of data X with labels.')
    parser.add_argument('--n_eigs', default=50, dest='M', type=int, help='Number of eigenvalues for spectral truncation')
    parser.add_argument('--tau-gr', default=0.01, dest='tau_gr', type=float, help='value of diagonal perturbation and scaling of MGR (not HF)')
    parser.add_argument('--gamma-gr', default=0.1, dest='gamma_gr', type=float, help='value of noise parameter of MGR (not HF)')
    parser.add_argument('--tau-ce', default=0.01, dest='tau_ce', type=float, help='value of diagonal perturbation and scaling of CE model')
    parser.add_argument('--gamma-ce', default=0.1, dest='gamma_ce', type=float, help='value of noise parameter of CE model')
    parser.add_argument('--delta', default=0.01, type=float, help='value of diagonal perturbation of unnormalized graph Laplacian for HF model.')
    parser.add_argument('--B', default=5, type=int, help='batch size for AL iterations')
    parser.add_argument('--al_iters', default=11, type=int, help='number of active learning iterations to perform.')
    parser.add_argument('--candidate-method', default='rand', type=str, dest='cand', help='candidate set selection method name ["rand", "full"]')
    parser.add_argument('--candidate-percent', default=0.1, type=float, dest='cand_perc', help='if --candidate-method == "rand", then this is the percentage of unlabeled data to consider')
    parser.add_argument('--select_method', default='top', type=str, help='how to select which points to query from the acquisition values. in ["top", "prop"]')
    parser.add_argument('--lab-start', default=3, type=int, dest='lab_start', help='size of initially labeled set.')
    parser.add_argument('--metric', default='euclidean', type=str, help='metric name ("euclidean" or "cosine") for graph construction')
    parser.add_argument('--name', default=EXPERIMENT_NAME, dest='experiment_name', help='Name for this dataset/experiment run ')
    parser.add_argument('--runs', default=5, type=int, help='Number of trials to run')
    args = parser.parse_args()

    GRAPH_PARAMS['n_eigs'] = args.M
    GRAPH_PARAMS['metric'] = args.metric



    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')

    # Load in or Create the Dataset

    if not os.path.exists(args.data_root + 'X_labels.npz'):
        raise ValueError("Cannot find previously saved data at {}".format(args.data_root + 'X_labels.npz'))

    print("Loading data at {}".format(args.data_root + 'X_labels.npz'))
    data = np.load(args.data_root + 'X_labels.npz', allow_pickle=True)
    X, labels = data['X'], data['labels'].flatten()
    N = X.shape[0]


    # Load in or calculate eigenvectors, using mlflow IN GraphManager

    gm = GraphManager()
    evals, evecs = gm.from_features(X, knn=GRAPH_PARAMS['knn'], sigma=GRAPH_PARAMS['sigma'],
                        normalized=GRAPH_PARAMS['normalized'], n_eigs=GRAPH_PARAMS['n_eigs'],
                        zp_k=GRAPH_PARAMS['zp_k'], metric=GRAPH_PARAMS['metric']) # runs mlflow logging in this function call

    L = None
    # If we are doing a run with the HF model, we need the unnormalized graph Laplacian
    if 'hf' in ''.join(ACQ_MODELS):
        print("Since HF model include in ACQ_MODELS, calculating (or finding) the full graph laplacian L...")
        prev_run = get_prev_run('GraphManager.from_features',
                                GRAPH_PARAMS,
                                tags={"X":str(X)},
                                git_commit=None)

        url_data = urllib.parse.urlparse(os.path.join(prev_run.info.artifact_uri,
                        'W.npz'))
        path = urllib.parse.unquote(url_data.path)
        W = sps.load_npz(path)
        L = sps.csr_matrix(gm.compute_laplacian(W, normalized=False)) + args.delta**2. * sps.eye(N)


    # Run the experiments
    print("--------------- Parameters for the Run of AL Experiments -----------------------")
    print("\tacq_models = %s" % str(ACQ_MODELS))
    print("\tal_iters = %d, B = %d, M = %d" % (args.al_iters, args.B, args.M))
    print("\tcand=%s, select_method=%s" % (args.cand, args.select_method))
    print("\tnum_init_labeled = %d" % (args.lab_start))
    print("\ttau = %1.6f, gamma = %1.6f, tau_ce = %1.6f, gamma_ce = %1.6f" % (args.tau_gr, args.gamma_gr, args.tau_ce, args.gamma_ce))
    print("\tdelta = {:.6f}".format(args.delta))
    print("\tnumber of runs = {}".format(args.runs))
    print("\n\n")
    ans = input("Do you want to proceed with this test?? [y/n] ")
    while ans not in ['y','n']:
        ans = input("Sorry, please input either 'y' or 'n'")
    if ans == 'n':
        print("Not running test, exiting...")
    else:

        client = mlflow.tracking.MlflowClient()

        mlflow.set_experiment(args.experiment_name)
        experiment = client.get_experiment_by_name(args.experiment_name)

        for i, seed in enumerate(j**2 + 3 for j in range(args.runs)):
            print("=======================================")
            print("============= Run {}/{} ===============".format(i+1, args.runs))
            print("=======================================")
            np.random.seed(seed)
            init_labeled, unlabeled = train_test_split(np.arange(N), train_size=args.lab_start, stratify=labels)
            init_labeled, unlabeled = list(init_labeled), list(unlabeled)

            params_shared = {
                'init_labeled': init_labeled,
                'run': i,
                'al_iters' : args.al_iters,
                'B' : args.B,
                'cand' : args.cand,
                'select' : args.select_method
            }
            query = 'attributes.status = "FINISHED"'
            for key, val in params_shared.items():
                query += ' and params.{} = "{}"'.format(key, val)


            already_completed = [run.data.tags['mlflow.runName'] for run in client.search_runs([experiment.experiment_id], filter_string=query)]


            if len(already_completed) > 0:
                print("Run {} already completed:".format(i))
                for thing in sorted(already_completed, key= lambda x : x[0]):
                    print("\t", thing)
                print()


            np.save('tmp/init_labeled', init_labeled)

            for acq, model in (am.split('--') for am in ACQ_MODELS):
                if model == 'hf':
                    run_name = "{}-{}-{:.2f}-{}".format(acq, model, args.delta, i)
                elif model == 'ce':
                    run_name = "{}-{}-{:.2f}-{:.2f}-{}-{}".format(acq, model, args.tau_ce, args.gamma_ce, args.M, i)
                else:
                    run_name = "{}-{}-{:.2f}-{:.2f}-{}-{}".format(acq, model, args.tau_gr, args.gamma_gr, args.M, i)

                if run_name not in already_completed:
                    labeled = copy.deepcopy(init_labeled)
                    with mlflow.start_run(run_name=run_name) as run:
                        # run AL test
                        mlflow.log_params(params_shared)
                        mlflow.log_artifact('tmp/init_labeled.npy')

                        if model == 'ce':
                            mlflow.log_params({
                                'tau' : args.tau_ce,
                                'gamma' : args.gamma_ce,
                                'M' : args.M
                            })
                            run_multi(evals, evecs, args.tau_ce, args.gamma_ce, labels, labeled, args.al_iters, args.B,
                                            modelname=model, acq=acq, cand=args.cand, select_method=args.select_method, verbose=False)
                        elif model == 'mgr':
                            mlflow.log_params({
                                'tau' : args.tau_gr,
                                'gamma' : args.gamma_gr,
                                'M' : args.M
                            })
                            run_multi(evals, evecs, args.tau_gr, args.gamma_gr, labels, labeled, args.al_iters, args.B,
                                            modelname=model, acq=acq, cand=args.cand, select_method=args.select_method, verbose=False)
                        elif model == 'hf':
                            mlflow.log_param('delta', args.delta)
                            run_rkhs_hf(labels, labeled, args.al_iters, args.B, delta=args.delta, L=L,
                                            modelname=model, acq=acq, cand=args.cand, select_method=args.select_method, verbose=False)
                        else:
                            raise ValueError("{} is not a valid multiclass model".format(model))


        # Clean up tmp file
        print("Cleaning up files in ./tmp/")
        if os.path.exists('tmp/init_labeled.npy'):
            os.remove('tmp/init_labeled.npy')
        if os.path.exists('tmp/iter_stats.npz'):
            os.remove('tmp/iter_stats.npz')
        if os.path.exists('tmp/W.npz'):
            os.remove('tmp/W.npz')
        if os.path.exists('tmp/eig.npz'):
            os.remove('tmp/eig.npz')
