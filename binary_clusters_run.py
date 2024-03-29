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


ACQ_MODELS = ['vopt--gr', 'sopt--gr', 'db--rkhs', 'mc--gr', 'mc--log', 'mc--probitnorm', 'sopt--hf', 'vopt--hf',
                'uncertainty--gr', 'uncertainty--log', 'uncertainty--probitnorm', 'rand--gr', 'rand--log', 'rand--probitnorm']

GRAPH_PARAMS = {
    'knn' :10,
    'sigma' : 3.,
    'normalized' : True,
    'zp_k' : 5
}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run Active Learning experiment on Binary Clusters dataset')
    parser.add_argument('--data-root', default='./data/binary_clusters/', dest='data_root', type=str, help='Location of data X with labels.')
    parser.add_argument('--num-eigs', default=50, dest='M', type=int, help='Number of eigenvalues for spectral truncation')
    parser.add_argument('--tau', default=0.005, type=float, help='value of diagonal perturbation and scaling of GBSSL models (minus HF)')
    parser.add_argument('--gamma', default=0.1, type=float, help='value of noise parameter to be shared across all GBSSL models (minus HF)')
    parser.add_argument('--delta', default=0.01, type=float, help='value of diagonal perturbation of unnormalized graph Laplacian for HF model.')
    parser.add_argument('--h', default=0.1, type=float, help='kernel width for RKHS model.')
    parser.add_argument('--B', default=5, type=int, help='batch size for AL iterations')
    parser.add_argument('--al-iters', default=100, type=int, dest='al_iters', help='number of active learning iterations to perform.')
    parser.add_argument('--candidate-method', default='rand', type=str, dest='cand', help='candidate set selection method name ["rand", "full"]')
    parser.add_argument('--candidate-percent', default=0.1, type=float, dest='cand_perc', help='if --candidate-method == "rand", then this is the percentage of unlabeled data to consider')
    parser.add_argument('--select-method', default='top', type=str, dest='select_method', help='how to select which points to query from the acquisition values. in ["top", "prop"]')
    parser.add_argument('--runs', default=5, type=int, help='Number of trials to run')
    parser.add_argument('--lab-start', default=2, dest='lab_start', type=int, help='Number of initially labeled points.')
    parser.add_argument('--metric', default='euclidean', type=str, help='metric name ("euclidean" or "cosine") for graph construction')
    parser.add_argument('--name', default='binary-clusters', dest='experiment_name', help='Name for this dataset/experiment run ')
    args = parser.parse_args()


    GRAPH_PARAMS['n_eigs'] = args.M
    GRAPH_PARAMS['metric'] = args.metric

    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')

    # Load in or Create the Dataset
    if not os.path.exists(args.data_root + 'X_labels.npz'):
        print("Cannot find previously saved data at {}".format(args.data_root + 'X_labels.npz'))
        print("so creating the dataset and labels")
        X, labels = create_binary_clusters()
        N = X.shape[0]
        os.makedirs(args.data_root)
        np.savez(args.data_root + 'X_labels.npz', X=X, labels=labels)
    else:
        data = np.load(args.data_root + 'X_labels.npz')
        X, labels = data['X'], data['labels']
        N = X.shape[0]

    labels[labels == 0] = -1


    # Load in or calculate eigenvectors, using mlflow IN Graph_manager
    gm = GraphManager()
    evals, evecs = gm.from_features(X, knn=GRAPH_PARAMS['knn'], sigma=GRAPH_PARAMS['sigma'],
                        normalized=GRAPH_PARAMS['normalized'], n_eigs=GRAPH_PARAMS['n_eigs'],
                        zp_k=GRAPH_PARAMS['zp_k'], metric=GRAPH_PARAMS['metric']) # runs mlflow logging in this function call

    print(evals[:6])
    # If we are doing a run with the HF model, we need the unnormalized graph Laplacian
    L = None
    if 'hf' in ''.join(ACQ_MODELS):
        prev_run = get_prev_run('GraphManager.from_features',
                                GRAPH_PARAMS,
                                tags={"X":str(X), "N":str(X.shape[0])},
                                git_commit=None)

        url_data = urllib.parse.urlparse(os.path.join(prev_run.info.artifact_uri,
                        'W.npz'))
        path = urllib.parse.unquote(url_data.path)
        W = sps.load_npz(path)
        L = sps.csr_matrix(gm.compute_laplacian(W, normalized=False)) + args.delta**2. * sps.eye(N)


    # Run the experiments
    print("--------------- Parameters for the Run of Experiments -----------------------")
    print("\tacq_models = %s" % str(ACQ_MODELS))
    print("\tal_iters = %d, B = %d, M = %d" % (args.al_iters, args.B, args.M))
    print("\tcand=%s, select_method=%s" % (args.cand, args.select_method))
    print("\tnum_init_labeled = %d" % (args.lab_start))
    print("\ttau = %1.6f, gamma = %1.6f, delta = %1.6f, h = %1.6f" % (args.tau, args.gamma, args.delta, args.h))
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
            init_labeled, unlabeled = train_test_split(np.arange(N), train_size=2, stratify=labels)#list(np.random.choice(range(N), 10, replace=False))
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
                elif model == 'rkhs':
                    run_name = "{}-{}-{:.2}-{}".format(acq, model, args.h, i)
                else:
                    run_name = "{}-{}-{:.3f}-{:.3f}-{}-{}".format(acq, model, args.tau, args.gamma, args.M, i)

                if run_name not in already_completed:
                    labeled = copy.deepcopy(init_labeled)
                    with mlflow.start_run(run_name=run_name) as run:
                        # run AL test
                        mlflow.log_params(params_shared)
                        mlflow.log_artifact('tmp/init_labeled.npy')

                        if model not in ['hf', 'rkhs']:
                            mlflow.log_params({
                                'tau' : args.tau,
                                'gamma' : args.gamma,
                                'M' : args.M
                            })
                            run_binary(evals, evecs, args.tau, args.gamma, labels, labeled, args.al_iters, args.B,
                                            modelname=model, acq=acq, cand=args.cand, select_method=args.select_method, verbose=False)
                        else:
                            if model == 'hf':
                                mlflow.log_param('delta', args.delta)
                            else:
                                mlflow.log_param('h', args.h)
                            run_rkhs_hf(labels, labeled, args.al_iters, args.B, h=args.h, delta=args.delta, X=X, L=L,
                                            modelname=model, acq=acq, cand=args.cand, select_method=args.select_method, verbose=False)


        # Clean up tmp file
        print("Cleaning up files in ./tmp/")
        if os.path.exists('tmp/init_labeled.npy'):
            os.remove('tmp/init_labeled.npy')
        if os.path.exists('tmp/iter_stats.npz'):
            os.remove('tmp/iter_stats.npz')
