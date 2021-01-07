import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
import os
import sys
sys.path.append('..')

import mlflow
from util.mlflow_util import load_uri

# Plotting configurations

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default='checker2', help="Location of Checkerboard[2,3] of which you want to plot the AL choices")
    parser.add_argument("--run", default=0, type=int, help="Specify which run's results you want to plot; default is 0")
    parser.add_argument('--data_root', default='../data/checker2/', type=str, help='Location of data X with labels.')
    args = parser.parse_args()

    if '3' in args.experiment and '2' in args.data_root:
        print("You have given data_root = {} with experiment = {}, which could be different datasets...".format(args.data_root, args.experiment))
        ans = input()
    else:
        ans = 'y'

    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment)




    if ans in ['y', 'yes']:
        mlflow.set_tracking_uri('../mlruns')
        client = mlflow.tracking.MlflowClient()


        checker_data = np.load(os.path.join(args.data_root, 'X_labels.npz'))
        X, labels = checker_data['X'], checker_data['labels']

        assert X.shape[1] == 2  # ensure that our data is plottable
        gt_colors = np.array(len(labels)*['r'])
        if len(np.unique(labels)) == 3:
            gt_colors[labels == 0] = 'r'
            gt_colors[labels == 1] = 'b'
            gt_colors[labels == 2] = 'g'
        else:
            gt_colors[labels == 1] = 'r'
            if -1 not in labels:
                gt_colors[labels == 0] = 'b'
            else:
                gt_colors[labels == -1] = 'b'

        ### Ground Truth Plot
        plt.figure(figsize=(4,4))
        plt.scatter(X[:,0], X[:,1], c=gt_colors)
        plt.yticks([0, 0.5, 1.0])
        plt.tight_layout()
        plotname = './{}/gt.pdf'.format(args.experiment)
        print("Saving {}...".format(plotname))
        plt.savefig(plotname)
        plt.show()


        mlflow.set_experiment(args.experiment)
        experiment = client.get_experiment_by_name(args.experiment)
        query = 'attributes.status = "FINISHED" and params.run = "{}"'.format(args.run)
        runs = client.search_runs(experiment.experiment_id, filter_string=query)

        for r in runs:
            acq, modelname = r.data.tags['mlflow.runName'].split('-')[:2]
            init_labeled = load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy'))
            iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
            choices = iter_stats['al_choices'].flatten()
            assert len(choices.flatten()) == len(list(set(list(choices.flatten())))) # ensure we didn't have a problem of choosing labeled points
            
            plt.figure(figsize=(4,4))
            plt.scatter(X[:,0], X[:,1], c=gt_colors, alpha=0.8)
            plt.scatter(X[init_labeled,0], X[init_labeled,1], marker='*', c='gold', s=90, edgecolors='k', linewidths=0.4)
            plt.scatter(X[choices,0], X[choices,1], marker='*', c='gold', s=90, edgecolors='k', linewidths=0.4)
            plt.tight_layout()
            plt.xticks([], [])
            plt.yticks([], [])
            plotname = './{}/{}-{}.pdf'.format(args.experiment, acq, modelname)
            print("Saving {}...".format(plotname))
            plt.savefig(plotname)
            plt.show()
    else:
        print("Exiting...")
