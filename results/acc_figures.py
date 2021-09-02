import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
import sys
sys.path.append('..')

from util.mlflow_util import load_uri
import mlflow

from cycler import cycler

print(plt.style.available)

# MATLPLOTLIB settings
plt.style.use('default')
plt.style.use('seaborn-deep')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y', 'cyan', 'brown', 'k', 'gray', 'orange','purple', 'pink'])
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


acq_model2label_marker_color = {
    'vopt-mgr' : ('VOPT-MGR', '+', 'k'),
    'sopt-mgr' : ('SOPT-MGR', 'x', 'brown'),
    'sopt-hf' : ('SOPT-HF', 'v', 'pink'),
    'vopt-hf' : ('VOPT-HF', '^', 'm'),
    'vopt-gr' : ('VOPT-GR', '+', 'k'),
    'sopt-gr' : ('SOPT-GR', 'x', 'brown'),
    'mc-mgr' : ('MC-MGR', '*', 'g'),
    'mcgreedy-ce' : ('MCG-CE', 'v', 'cyan'),
    'mc-ce' : ('MC-CE', 'v', 'b'),
    'mc-probitnorm' : ('MC-P', 's', 'y'),
    'mc-gr' : ('MC-GR', '*', 'g'),
    'mc-log' : ('MC-LOG', 'o', 'b'),
    'rand-mgr' : ('RAND-MGR', '<', 'purple'),
    'rand-log' : ('RAND-LOG', '<', 'purple'),
    'rand-ce' : ('RAND-CE', '<', 'purple'),
    'rand-gr' : ('RAND-GR', '<', 'purple'),
    'db-rkhs' : ('DB-RKHS', '>', 'r'),
    'uncertainty-mgr' : ('UNC-MGR', 'P', 'orange'),
    'uncertainty-gr' : ('UNC-GR', 'P', 'orange'),
    'uncertainty-ce' : ('UNC-CE', 'X', 'gray'),
    'uncertainty-log' : ('UNC-LOG', 'X', 'gray'),
    'uncertainty-probitnorm' : ('UNC-P', 'd', 'gray'),
}

mlflow.set_tracking_uri('../mlruns')
client = mlflow.tracking.MlflowClient()

save_root = './for-paper-other/'

# ## Binary Clusters

# +
exp_name = 'sequential-binary-clusters'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 2
first = True
not_plot = ['rand-gr', 'rand-probitnorm', 'uncertainty-gr', 'uncertainty-probitnorm', 'vopt-gr', 'sopt-gr']

plt.figure(figsize=(8,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[:50:skip], ACC[:50:skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[:50:skip], ACC[:50:skip], linewidth=0.9, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
# plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# -

checkdata = np.load('../data/binary_clusters2/X_labels.npz', allow_pickle=True)
X = checkdata['X']
labels = checkdata['labels']
clrs = np.array(X.shape[0]*['r'])
clrs[labels == 0] = 'b'
for r in all_runs:
    print(r.data.tags['mlflow.runName'])
    if r.data.tags['mlflow.runName'][-1] != '0':
        continue
    iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
    choices = iter_stats['al_choices'].flatten()
    init_labeled = load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy'))
    choices = np.concatenate((init_labeled, choices))[:350]
    if np.max(choices) > 1999:
        print('found checker run with more than 2000 nodes')
        continue
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clrs)
    plt.scatter(X[choices[:50],0], X[choices[:50],1], marker='*', s=90, c='gold', linewidths=0.6, edgecolors='k')
    #plt.title(r.data.tags['mlflow.runName'])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('{}/{}.pdf'.format(exp_save_root, r.data.tags['mlflow.runName']))
    plt.show()

# +
# Redo of experiment -- changed tau = 0.001 and gamma = 0.5
exp_name = 'sequential-binary-clusters3'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 2
first = True
not_plot = ['rand-gr', 'rand-probitnorm', 'uncertainty-gr', 'uncertainty-probitnorm', 'vopt-gr', 'sopt-gr']

plt.figure(figsize=(8,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[:50:skip], ACC[:50:skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[:50:skip], ACC[:50:skip], linewidth=0.9, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# -

checkdata = np.load('../data/binary_clusters2/X_labels.npz', allow_pickle=True)
X = checkdata['X']
labels = checkdata['labels']
clrs = np.array(X.shape[0]*['r'])
clrs[labels == 0] = 'b'
for r in all_runs:
    print(r.data.tags['mlflow.runName'])
    if r.data.tags['mlflow.runName'][-1] != '0':
        continue
    iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
    choices = iter_stats['al_choices'].flatten()
    init_labeled = load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy'))
    choices = np.concatenate((init_labeled, choices))[:350]
    if np.max(choices) > 1999:
        print('found checker run with more than 2000 nodes')
        continue
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clrs)
    plt.scatter(X[choices[:50],0], X[choices[:50],1], marker='*', s=90, c='gold', linewidths=0.6, edgecolors='k')
    #plt.title(r.data.tags['mlflow.runName'])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('{}/{}.pdf'.format(exp_save_root, r.data.tags['mlflow.runName']))
    plt.show()

# # Binary Clusters 3 - Sequential

# +
# Redo of experiment -- changed tau = 0.001 and gamma = 0.5
exp_name = 'binary-clusters3-sequential'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 2
first = True
not_plot = ['rand-gr', 'rand-probitnorm', 'uncertainty-gr', 'uncertainty-probitnorm', 'vopt-gr', 'sopt-gr']

plt.figure(figsize=(8,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[:100:skip], ACC[:100:skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[:100:skip], ACC[:100:skip], linewidth=0.9, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# -

checkdata = np.load('../data/binary_clusters3/X_labels.npz', allow_pickle=True)
X = checkdata['X']
labels = checkdata['labels']
clrs = np.array(X.shape[0]*['r'])
clrs[labels == 0] = 'b'
for r in all_runs:
    print(r.data.tags['mlflow.runName'])
    if r.data.tags['mlflow.runName'][-1] != '0':
        continue
    iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
    choices = iter_stats['al_choices'].flatten()
    init_labeled = load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy'))
    choices = np.concatenate((init_labeled, choices))[:350]
    if np.max(choices) > 1999:
        print('found checker run with more than 2000 nodes')
        continue
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clrs)
    plt.scatter(X[choices[:50],0], X[choices[:50],1], marker='*', s=90, c='gold', linewidths=0.6, edgecolors='k')
    #plt.title(r.data.tags['mlflow.runName'])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('{}/{}.pdf'.format(exp_save_root, r.data.tags['mlflow.runName']))
    plt.show()

plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=clrs)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('{}/{}.pdf'.format(exp_save_root, 'bc-gt'))
plt.show()

# # Binary Clusters3 - Batch

# +
# Redo of experiment -- changed tau = 0.001 and gamma = 0.5
exp_name = 'binary-clusters3-batch'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 1
first = True
not_plot = ['rand-gr', 'rand-probitnorm', 'uncertainty-gr', 'uncertainty-probitnorm', 'vopt-gr', 'sopt-gr']

plt.figure(figsize=(8,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[:20:skip], ACC[:20:skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[:20:skip], ACC[:20:skip], linewidth=0.9, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# -

plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()

checkdata2 = np.load('../data/binary_clusters_check/X_labels.npz', allow_pickle=True)
X2 = checkdata2['X']
labels2 = checkdata2['labels']

print(np.allclose(X2, X))



# ## Checker 2

# +
exp_name = 'checker2'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 2
first = True
not_plot = ['rand-gr', 'rand-probitnorm', 'uncertainty-gr', 'uncertainty-probitnorm', 'vopt-gr', 'sopt-gr']

plt.figure(figsize=(8,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[::skip], ACC[::skip], linewidth=0.9, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# -
checkdata = np.load('../data/checker2/X_labels.npz', allow_pickle=True)
X = checkdata['X']
labels = checkdata['labels']
clrs = np.array(X.shape[0]*['r'])
clrs[labels == 0] = 'b'
for r in all_runs:
    if r.data.tags['mlflow.runName'][-1] != '0':
        continue
    iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
    choices = iter_stats['al_choices'].flatten()
    init_labeled = load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy'))
    choices = np.concatenate((init_labeled, choices))[:350]
    if np.max(choices) > 1999:
        print('found checker run with more than 2000 nodes')
        continue
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clrs)
    plt.scatter(X[choices,0], X[choices,1], marker='*', s=90, c='gold', linewidths=0.6, edgecolors='k')
    #plt.title(r.data.tags['mlflow.runName'])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('{}/{}.pdf'.format(exp_save_root, r.data.tags['mlflow.runName']))
    plt.show()

# # Sequential Checker2

# +
exp_name = 'sequential-checker2'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 5
tot = 55
first = True
not_plot = ['rand-gr', 'rand-probitnorm', 'uncertainty-gr', 'uncertainty-probitnorm', 'vopt-gr', 'sopt-gr']

plt.figure(figsize=(7,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[::skip][:tot], ACC[::skip][:tot], marker=mrkr, label=lbl, s=40, c=clr)
    plt.plot(dom[::skip][:tot], ACC[::skip][:tot], linewidth=0.9, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# -
checkdata = np.load('../data/checker2/X_labels.npz', allow_pickle=True)
X = checkdata['X']
labels = checkdata['labels']
clrs = np.array(X.shape[0]*['r'])
clrs[labels == 0] = 'b'
for r in all_runs:
    if r.data.tags['mlflow.runName'][-1] != '0':
        continue
    iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
    choices = iter_stats['al_choices'].flatten()
    init_labeled = load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy'))
    choices = np.concatenate((init_labeled, choices))[:350]
    if np.max(choices) > 1999:
        print('found checker run with more than 2000 nodes')
        continue
    print(r.data.tags['mlflow.runName'])
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clrs)
    plt.scatter(X[choices,0], X[choices,1], marker='*', s=110, c='yellow', linewidths=0.2, edgecolors='k')
    #plt.title(r.data.tags['mlflow.runName'])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('{}/{}.pdf'.format(exp_save_root, r.data.tags['mlflow.runName']),bbox_inches = 'tight',
    pad_inches = 0)
    
    plt.show()

# +
checkdata = np.load('../data/checker2/X_labels.npz', allow_pickle=True)
X = checkdata['X']
labels = checkdata['labels']
clrs = np.array(X.shape[0]*['r'])
clrs[labels == 0] = 'b'

i1, i2 = None, None
i = 0 
while (i1 is None or i2 is None) and i < 2000:
    x, y = X[i,0], X[i,1]
    if i1 is None:
        if 0.55 <= x <= 0.7 and 0.3 <= y <= 0.45:
            i1 = i
            
    if i2 is None:
        if 0.3 <= x <= 0.45 and 0.3 <= y <= 0.45:
            i2 = i

    i += 1
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=clrs)
# plt.scatter(X[i1,0], X[i1,1], marker='*', s=90, c='gold', linewidths=0.4, edgecolors='k')
# plt.scatter(X[i2,0], X[i2,1], marker='*', s=90, c='gold', linewidths=0.4, edgecolors='k')
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('{}/gt.pdf'.format(exp_save_root), bbox_inches = 'tight',
    pad_inches = 0)
plt.show()
# -

labeled_handchosen = [i1, i2]
print(labeled_handchosen)

# +
exp_name = 'handchosen-checker2'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 5
tot = -1
first = True
not_plot = ['rand-gr', 'rand-probitnorm', 'uncertainty-gr', 'uncertainty-probitnorm', 'vopt-gr', 'sopt-gr']

plt.figure(figsize=(7,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
#     runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName'] and r.data.params['cand'] == 'full']
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[::skip][:tot], ACC[::skip][:tot], marker=mrkr, label=lbl, s=40, c=clr)
    plt.plot(dom[::skip][:tot], ACC[::skip][:tot], linewidth=0.9, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
#plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# -
checkdata = np.load('../data/checker2/X_labels.npz', allow_pickle=True)
X = checkdata['X']
labels = checkdata['labels']
clrs = np.array(X.shape[0]*['r'])
clrs[labels == 0] = 'b'
for r in all_runs:
    if r.data.tags['mlflow.runName'][-1] != '1' or r.data.params['cand'] != 'full':
        continue
    iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
    choices = iter_stats['al_choices'].flatten()
    init_labeled = load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy'))
    choices = np.concatenate((init_labeled, choices))
    if np.max(choices) > 1999:
        print('found checker run with more than 2000 nodes')
        continue
    print(r.data.tags['mlflow.runName'])
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clrs)
    plt.scatter(X[choices,0], X[choices,1], marker='*', s=100, c='gold', linewidths=0.4, edgecolors='k')
    #plt.title(r.data.tags['mlflow.runName'])
    plt.xticks([], [])
    plt.yticks([], [])
    #plt.savefig('{}/{}.pdf'.format(exp_save_root, r.data.tags['mlflow.runName']))
    plt.show()







# # Binary MNIST


# +
exp_name = 'sequential-binary-mnist'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 1
tot = 100
first = True
not_plot = ['rand-log', 'rand-probitnorm', 'uncertainty-log', 'uncertainty-probitnorm']

plt.figure(figsize=(10,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[::skip][:tot], ACC[::skip][:tot], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[::skip][:tot], ACC[::skip][:tot], linewidth=0.9, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# -


# # Checker 3

# +
exp_name = 'checker3'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 3
first = True
not_plot = ['rand-mgr', 'uncertainty-mgr']

plt.figure(figsize=(10,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[::skip], ACC[::skip], linewidth=0.5, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
#plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# +
# Plot figure
skip = 3
first = True

for modelname in ['mgr', 'ce']:
    plt.figure(figsize=(7,5))
    for setup_name in setup_names:
        
        if modelname not in setup_name.split('-')[1]:
            continue

        runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
        acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
        lbl, mrkr, clr = acq_model2label_marker_color[acq_model]

        al_iters = int(runs[0].data.params['al_iters'])
        ACC = np.zeros(al_iters + 1)

        print(len(runs), setup_name)
        for r in runs:
            acc = np.array([r.data.metrics['init_acc']])
            iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
            ACC += np.concatenate((acc, iter_stats['iter_acc']))

        ACC /=  float(len(runs))

        if first:
            num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
            B = int(runs[0].data.params['B'])
            dom = [num_init_labeled + B*i for i in range(al_iters+1)]
            first = False

        plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
        plt.plot(dom[::skip], ACC[::skip], linewidth=1.5, c=clr)

    plt.legend()
    plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig('{}/acc-{}.pdf'.format(exp_save_root, modelname))
    plt.show()
# -
checkdata = np.load('../data/checker3/X_labels.npz', allow_pickle=True)
X = checkdata['X']
labels = checkdata['labels']
clrs = np.array(X.shape[0]*['r'])
clrs[labels == 0] = 'b'
clrs[labels== 1] = 'g'
for r in all_runs:
    if r.data.tags['mlflow.runName'][-1] != '1':
        continue
    iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
    #print(list(iter_stats.keys()))
    choices = iter_stats['al_choices'].flatten()
    init_labeled = load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy'))
    choices = np.concatenate((init_labeled, choices))[:]
    if np.max(choices) > 2999:
        continue
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=clrs)
    plt.scatter(X[choices,0], X[choices,1], marker='*', s=90, c='gold', linewidths=0.6, edgecolors='k')
    #plt.title(r.data.tags['mlflow.runName'])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig('{}/{}.pdf'.format(exp_save_root, r.data.tags['mlflow.runName']))
    plt.show()


# # MNIST

# +
exp_name = 'mnist'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 3
first = True
not_plot = ['rand-mgr', 'uncertainty-mgr']

plt.figure(figsize=(7,5))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[::skip], ACC[::skip], linewidth=0.5, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
#plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# +
# Plot figure
skip = 3
first = True

for modelname in ['mgr', 'ce']:
    plt.figure(figsize=(7,5))
    mm = 1.0
    for setup_name in setup_names:
        
        if modelname not in setup_name.split('-')[1]:
            continue

        runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
        acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
        lbl, mrkr, clr = acq_model2label_marker_color[acq_model]

        al_iters = int(runs[0].data.params['al_iters'])
        ACC = np.zeros(al_iters + 1)

        print(len(runs), setup_name)
        for r in runs:
            acc = np.array([r.data.metrics['init_acc']])
            iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
            ACC += np.concatenate((acc, iter_stats['iter_acc']))

        ACC /=  float(len(runs))

        if first:
            num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
            B = int(runs[0].data.params['B'])
            dom = [num_init_labeled + B*i for i in range(al_iters+1)]
            first = False

        plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
        plt.plot(dom[::skip], ACC[::skip], linewidth=1.5, c=clr)
        if min(ACC[::skip]) < mm:
            mm = min(ACC[::skip])

    plt.legend()
    plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
    plt.ylabel("Accuracy")
    plt.ylim([mm,1.0])
    plt.xticks([i*100 for i in range(6)])
    plt.tight_layout()
    plt.savefig('{}/acc-{}2.pdf'.format(exp_save_root, modelname))
    plt.show()
# -
# # Salinas

# +
exp_name = 'salinas'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 3
first = True
not_plot = [] #['rand-mgr', 'uncertainty-mgr']

plt.figure(figsize=(10,6))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) in not_plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[::skip], ACC[::skip], linewidth=0.5, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
#plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# +
# Plot figure
skip = 3
first = True

for modelname in ['mgr', 'ce']:
    plt.figure(figsize=(7,5))
    mm = 1.0
    for setup_name in setup_names:
        
        if modelname not in setup_name.split('-')[1]:
            continue

        runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
        acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
        lbl, mrkr, clr = acq_model2label_marker_color[acq_model]

        al_iters = int(runs[0].data.params['al_iters'])
        ACC = np.zeros(al_iters + 1)

        print(len(runs), setup_name)
        for r in runs:
            acc = np.array([r.data.metrics['init_acc']])
            iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
            ACC += np.concatenate((acc, iter_stats['iter_acc']))

        ACC /=  float(len(runs))

        if first:
            num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
            B = int(runs[0].data.params['B'])
            dom = [num_init_labeled + B*i for i in range(al_iters+1)]
            first = False

        plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
        plt.plot(dom[::skip], ACC[::skip], linewidth=1.5, c=clr)
        if min(ACC[::skip]) < mm:
            mm = min(ACC[::skip])

    plt.legend()
    plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
    plt.ylabel("Accuracy")
    plt.ylim([mm, 0.9])
    plt.xticks([i*100 for i in range(6)])
    plt.tight_layout()
    plt.savefig('{}/acc-{}2.pdf'.format(exp_save_root, modelname))
    plt.show()
# -
# # Urban

# +
exp_name = 'urban'
exp_save_root = os.path.join(save_root, exp_name)

if not os.path.exists(exp_save_root):
    os.makedirs(exp_save_root)

experiment = client.get_experiment_by_name(exp_name)

query = 'attributes.status = "FINISHED"'
all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
print(len(all_runs))
setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
print(setup_names)

# +
# Plot figure
skip = 3
first = True
not_plot = ['mc-mgr', 'mc-ce']

plt.figure(figsize=(7,5))
for setup_name in setup_names:
    if '-'.join(setup_name.split('-')[:2]) not in plot:
        continue
        
    runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
    acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    
    al_iters = int(runs[0].data.params['al_iters'])
    ACC = np.zeros(al_iters + 1)
    
    print(len(runs), setup_name)
    for r in runs:
        acc = np.array([r.data.metrics['init_acc']])
        iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
        ACC += np.concatenate((acc, iter_stats['iter_acc']))
    
    ACC /=  float(len(runs))
    
    if first:
        num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
        B = int(runs[0].data.params['B'])
        dom = [num_init_labeled + B*i for i in range(al_iters+1)]
        first = False
    
    plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
    plt.plot(dom[::skip], ACC[::skip], linewidth=0.5, c=clr)
    
plt.legend()
plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
plt.ylabel("Accuracy")
plt.tight_layout()
#plt.savefig('{}/acc.pdf'.format(exp_save_root))
plt.show()
# +
# Plot figure
skip = 3
first = True

for modelname in ['mgr', 'ce']:
    plt.figure(figsize=(7,5))
    mm = 1.0
    for setup_name in setup_names:
        
        if modelname not in setup_name.split('-')[1]:
            continue

        runs = [r for r in all_runs if setup_name in r.data.tags['mlflow.runName']]
        acq_model = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
        lbl, mrkr, clr = acq_model2label_marker_color[acq_model]

        al_iters = int(runs[0].data.params['al_iters'])
        ACC = np.zeros(al_iters + 1)

        print(len(runs), setup_name)
        for r in runs:
            acc = np.array([r.data.metrics['init_acc']])
            iter_stats = load_uri(os.path.join(r.info.artifact_uri, 'iter_stats.npz'))
            ACC += np.concatenate((acc, iter_stats['iter_acc']))

        ACC /=  float(len(runs))

        if first:
            num_init_labeled = len(load_uri(os.path.join(r.info.artifact_uri, 'init_labeled.npy')))
            B = int(runs[0].data.params['B'])
            dom = [num_init_labeled + B*i for i in range(al_iters+1)]
            first = False

        plt.scatter(dom[::skip], ACC[::skip], marker=mrkr, label=lbl, s=50, c=clr)
        plt.plot(dom[::skip], ACC[::skip], linewidth=1.5, c=clr)
        if min(ACC[::skip]) < mm:
            mm = min(ACC[::skip])

    plt.legend()
    plt.xlabel("Number of labeled points, $|\mathcal{L}|$")
    plt.ylabel("Accuracy")
    plt.ylim([mm, 1.0])
    plt.xticks([i*100 for i in range(6)])
    plt.tight_layout()
    plt.savefig('{}/acc-{}2.pdf'.format(exp_save_root, modelname))
    plt.show()
# -
# # Timing

from collections import defaultdict

mlflow.set_tracking_uri('../mlruns-old')
client = mlflow.tracking.MlflowClient()

# +
TIMES = defaultdict(list)
sizes = [2000, 5000, 10000, 20000]

for i, exp_name in enumerate(['checker2', '5K-checker2', '10K-c2', '20K-c2']):
    experiment = client.get_experiment_by_name(exp_name)
    query = 'attributes.status = "FINISHED"'
    query += ' and params.B = "{}"'.format(B)
    all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
    setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
    print(setup_names)
    
    
    for setup_name in setup_names:
        if 'rand' in setup_name or 'uncertainty' in setup_name:
            continue
        runs = [r for r in all_runs if (setup_name in r.data.tags['mlflow.runName'] and r.data.tags['mlflow.runName'][-1] == '0')]
        print(len(runs))

        for r in runs:
            a_uri_split = str(r.info.artifact_uri).split('/')
            a_uri_split[7] += '-old'
            a_uri = '/'.join(a_uri_split)
            iter_stats = load_uri(os.path.join(a_uri, 'iter_stats.npz'))
            choices = iter_stats['al_choices'].flatten()
            init_labeled = load_uri(os.path.join(a_uri, 'init_labeled.npy'))
            choices = np.concatenate((init_labeled, choices))
            k = '-'.join(r.data.tags['mlflow.runName'].split('-')[:2])
            times = iter_stats['iter_time']
            if i == 0:
                if np.max(choices) < 2000:
                    TIMES[k].append((2, np.average(times)))
                    break
            else:
                TIMES[k].append((sizes[i]//1000, np.average(times)))
                break

                
for k in TIMES:
    print(k)
    print(TIMES[k])
    


# +
TIMESMC = defaultdict(list)
mlflow.set_tracking_uri('../mlruns')
client = mlflow.tracking.MlflowClient()
B = 5
for exp_name in ['checker3', 'salinas', 'mnist', 'urban']:

    exp_save_root = os.path.join(save_root, exp_name)
    if not os.path.exists(exp_save_root):
        os.makedirs(exp_save_root)

    experiment = client.get_experiment_by_name(exp_name)


    query = 'attributes.status = "FINISHED"'
    print(query)
    all_runs = client.search_runs(experiment.experiment_id, filter_string=query)
    setup_names = sorted(set(['-'.join(r.data.tags['mlflow.runName'].split('-')[:-1]) for r in all_runs]))
    print(setup_names)

    for setup_name in setup_names:
        if 'rand' in setup_name or 'uncertainty' in setup_name:
            continue
        runs = [r for r in all_runs if (setup_name in r.data.tags['mlflow.runName'] and r.data.tags['mlflow.runName'][-1] == '0')]
        
        iter_stats = load_uri(os.path.join(runs[0].info.artifact_uri, 'iter_stats.npz'))
        choices = iter_stats['al_choices'].flatten()
        init_labeled = load_uri(os.path.join(runs[0].info.artifact_uri, 'init_labeled.npy'))
        choices = np.concatenate((init_labeled, choices))
        k = '-'.join(runs[0].data.tags['mlflow.runName'].split('-')[:2])
        times = iter_stats['iter_time']
        TIMESMC[k].append((exp_name, np.average(times)))

        
for k in TIMESMC:
    print(k)
    print(TIMESMC[k])


# +
name2size = {
    'checker3' : 3000,
    'salinas' : 7148,
    'mnist' : 70000,
    'urban' : 94129
}



fig, ax = plt.subplots(1,1, figsize=(8,5))
lines = []
Names = []
for k in TIMES:
    if 'sopt' in k: continue
    digs, times = zip(*TIMES[k])
    acq_model = '-'.join(k.split("-")[:2]).lower()
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    p1, = ax.loglog(1000*np.array(digs), times, c=clr)
    p2 = ax.scatter(1000*np.array(digs), times, marker=mrkr, c=clr, s=50)
    lines.append((p1, p2))
    Names.append(lbl)

# plt.legend()
# plt.show()


for k in TIMESMC:
    if 'sopt' in k: continue
    names, times = zip(*TIMESMC[k])
    acq_model = '-'.join(k.split("-")[:2]).lower()
    lbl, mrkr, clr = acq_model2label_marker_color[acq_model]
    numbers = np.array([name2size[name] for name in names])
    p1, = ax.loglog(numbers, times, '--', c=clr)
    p2 = ax.scatter(numbers, times, marker=mrkr, c=clr, s=50)
    lines.append((p1, p2))
    Names.append(lbl)

print(lines)
print(Names)
ax.legend(lines, Names, bbox_to_anchor=(1.01, 1.0))
ax.set_xlabel("Size of Dataset, $N$")
ax.set_ylabel("Avg. AL Query Time")
plt.savefig(os.path.join(save_root, "timing.pdf"), bbox_inches = "tight")
plt.tight_layout()
plt.show()
# -

70000./20.


20./70000.


