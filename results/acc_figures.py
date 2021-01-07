import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
import sys
sys.path.append('..')

from util.mlflow_util import load_uri
import mlflow

from IPython import embed

# MATLPLOTLIV settings
plt.style.use('default')
plt.style.use('seaborn-colorblind')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=26)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


acq_model2label_marker = {
    'vopt-mgr' : ('VOPT-MGR', '+'),
    'sopt-mgr' : ('SOPT-MGR', 'x'),
    'sopt-hf' : ('SOPT-HF', 's'),
    'vopt-hf' : ('VOPT-HF', 's'),
    'vopt-gr' : ('VOPT-GR', '+'),
    'sopt-gr' : ('SOPT-GR', 'x'),
    'mc-mgr' : ('MC-MGR', '*'),
    'mcgreedy-ce' : ('MC-CE', '&'),
    'mc-ce' : ('MC-CE', '&'),
    'mc-probitnorm' : ('MC-P', 'h'),
    'mc-gr' : ('MC-GR', '*'),
    'mc-log' : ('MC-LOG', 'o'),
    'rand-ce' : ('RAND-CE', '>'),
    'rand-mgr' : ('RAND-MGR', '<'),
    'rand-gr' : ('RAND-GR', '<'),
    'rand-log' : ('RAND-LOG', 'v'),
    'rand-probitnorm' : ('RAND-P', '^'),
    'db-rkhs' : ('DB-RKHS', 'p'),
    'uncertainty-mgr' : ('UNC-MGR', 'P'),
    'uncertainty-mg' : ('UNC-GR', 'P'),
    'uncertainty-ce' : ('UNC-CE', '8'),
    'uncertainty-log' : ('UNC-LOG', '8'),
    'uncertainty-probitnorm' : ('UNC-P', 'D'),
}





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="checker2", help="Experiment name")
    parser.add_argument("--skip", default=3)
    args = parser.parse_args()

    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment)

    # Check MLFLOW runs for the experiment, get all the runs and plot
    mlflow.set_tracking_uri('../mlruns')
    client = mlflow.tracking.MlflowClient()
    mlflow.set_experiment(args.experiment)
    experiment = client.get_experiment_by_name(args.experiment)
    query = 'attributes.status = "FINISHED"'
    runs = client.search_runs(experiment.experiment_id, filter_string=query)
    exp_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in runs]))

    plt.figure(figsize=(10,6))
    first = True
    for exp_name in exp_names:
        exp_runs = [r for r in runs if exp_name in r.data.tags['mlflow.runName']]
        acq_model = '-'.join(exp_runs[0].data.tags['mlflow.runName'].split('-')[:2])
        lbl, mrkr = acq_model2label_marker[acq_model]
        
        ACC = np.zeros(int(exp_runs[0].data.params['al_iters']) + 1)
        for r in exp_runs:
            acc = np.array([r.data.metrics['init_acc']])
            iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))


            ACC += np.concatenate((acc, iter_stats['iter_acc']))
        ACC /= float(len(exp_runs))

        if first:
            num_init_labeled = len(load_uri(os.path.join(exp_runs[0].info.artifact_uri, 'init_labeled.npy')))
            B, al_iters = int(exp_runs[0].data.params['B']), int(exp_runs[0].data.params['al_iters'])
            dom = [num_init_labeled + B*i for i in range(al_iters+1)]
            first = False


        plt.scatter(dom[::args.skip], ACC[::args.skip], marker=mrkr, label=lbl)
        plt.plot(dom[::args.skip], ACC[::args.skip], linewidth=0.5)

    plt.legend()
    plt.xlabel("Number of labeled points, $|\\mathcal{L}|$")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    # to see the plot before saving it
    embed()
    plt.savefig('{}/acc.pdf'.format(args.experiment))
    plt.show()
