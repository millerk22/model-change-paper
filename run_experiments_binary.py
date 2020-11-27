# Script for running AL experiment -- Checkerboard 2 dataset
# author: Kevin Miller
import numpy as np
import argparse
import time
import copy
import os
import sys
sys.path.append('..')
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from util.Graph_manager import Graph_manager
from util.activelearner import *
from util.gbssl import *
from util.experiment_util import *


DATA_FILENAME = './data/data.npz' # CHANGE THIS LINE TO HAVE FILEPATH TO .npz file with
                                # labels, eigenvalues, and eigenvectors


## Define hyperparameters of the runs
al_iters, B = 100, 5
cand_, select_ = 'rand', 'top'
M = 50
init_labeled_perc = 0.002
tau, gamma = 0.1, 0.1
acq_models = ['vopt--gr', 'sopt--gr', 'mc--gr', 'mc--log', 'rand--gr', 'uncertainty--gr', \
               'uncertainty--probitnorm', 'mc--probitnorm', 'uncertainty--log' ]



################ CHANGE THIS load_data function to align with keys in .npz file given in DATA_FILENAME
def load_data():
    '''
    Load in the labels, eigenvalues (w), and eigenvectors (v) for the dataset.
        * DATA_FILENAME = filepath to .npz file where data is stored
        * M = number of eigenvalues to consider
    '''
    data = np.load(DATA_FILENAME, allow_pickle=True)
    return data['labels'], data['w'][:M], data['v'][:,:M]


def many_experiments(root_filename, w, v, labels, num_runs=5, new=False):
    if not os.path.exists(root_filename):
        ans = input("The root file %s you gave does not currently exist, do you want to create it? [y/n]" % root_filename)
        while ans not in ['y','n']:
            ans = input("Please input either 'y' or 'n'")
        if ans == 'y':
            os.makedirs(root_filename)
        else:
            print("Not finishing test, exiting...")
            return
    D = int(1./init_labeled_perc) # for getting balanced initial class labels
    for run_ in range(1, num_runs+1):
        print("--------- Run #%d of %d-----------" %(run_, num_runs))
        acq_models_run = acq_models[:]
        labeled_orig = []
        if not new:
            # find existing files in root_filename that correspond to run_, get already done acq-models
            for file in os.listdir(root_filename):
                filesplit = file.split('-')
                if len(filesplit) > 1:
                    _acq, _model = filesplit[:2]
                    filename = root_filename + file + "/%s-%s-%d-%d-%d.txt" % (cand_, select_, B, al_iters, run_)
                    if len(labeled_orig) == 0:
                        if os.path.exists(filename):
                            print("Found previous initially labeled set, using those for this run of experiments...")
                            with open(filename, 'r') as f:
                                labeled_orig = [int(x) for x in f.readline().split(',')[:-2]]
                    if '%s--%s'%(_acq, _model) in acq_models_run:
                        #print("found %s--%s in acq_models_run" % (_acq, _model))
                        if os.path.exists(filename):
                            print("Experiment %s already exists, skipping..." % filename)
                            acq_models_run.pop(acq_models_run.index('%s--%s' % (_acq, _model)))




        if len(labeled_orig) == 0:
            print("didnt find previously recored initially labeled set, defining new")
            # define initially labeled set for the run
            for c in np.unique(labels):
                class_c = np.where(labels == c)[0]
                labeled_orig += list(np.random.choice(class_c, len(class_c)//D))

        if run_ == 1:
            print("Starting with %d labeled points" % len(labeled_orig))
        print(acq_models_run)
        for acq_mod in acq_models_run:
            acq, modelname = acq_mod.split('--')
            run_experiment_binary(wM, vM, tau, gamma, labels, labeled_orig[:], al_iters, B,
                    acq=acq, cand=cand_, select_method=select_, modelname=modelname, run=run_,
                     root_filename=root_filename)
            print()
        print()


    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run Active Learning experiment on Binary dataset')
    parser.add_argument('--save_root', default='./dumb/', type=str, help='Location to store the results of the experiment')
    parser.add_argument('--runs', default=5, type=int, help='Number of trials to run')
    parser.add_argument('--new', default=False, type=bool, help='Rerun all tests if True, otherwise only run \
                            tests that have not been done, loading in previously chosen initial points.')
    args = parser.parse_args()

    print("Loading data...")
    labels, wM, vM = load_data()

    # Run the experiments
    print("--------------- Parameters for the Run of Experiments -----------------------")
    print("\tacq_models = %s" % str(acq_models))
    print("\tal_iters = %d, B = %d, M = %d" % (al_iters, B, M))
    print("\tcand=%s, select_method=%s" % (cand_, select_))
    print("\tinit_labeled_perc = %1.6f, num_init_labeled = %d" % (init_labeled_perc, int(vM.shape[0]*init_labeled_perc)))
    print("\ttau = %1.6f, gamma = %1.6f" % (tau, gamma))
    print("\tRUN NEW TESTS OF PREVIOUSLY DONE TESTS = %s" % str(args.new))
    print("\n\n")
    ans = input("Do you want to proceed with this test?? [y/n] ")
    while ans not in ['y','n']:
        ans = input("Sorry, please input either 'y' or 'n'")
    if ans == 'n':
        print("Not running test, exiting...")
    else:
        many_experiments(args.save_root, wM, vM, labels, num_runs=args.runs, new=args.new)
