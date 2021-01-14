import numpy as np
import copy
import time
import os
import os.path
from collections import defaultdict
from heapq import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from .al_util import *

################################################################################
#################### Dijkstra ##################################

# W is the weight matrix (CSR format), with weight in it. Cost will be 1/weight
# STARTS is dictionary of class : [node in class]
# NOTE this is old code -- doesn't combine the dictionaries for the different classes
def dijkstra_for_al_csr(W, STARTS, radius=None):
    DIST = {}
    for k, start in STARTS.items(): # for each class represented in STARTS
        #print("Calculating for class %s" % str(k))
        # instantiate considered, seen, and dist
        considered, done, dist = [(0, s) for s in start], set(), {}
        heapify(considered) # make considered a heapq

        # while still have elements in considered...
        while considered:

            # get the dist, node id, and shortest path to the node corresponding to the top of heap
            (curr_dist, curr) = heappop(considered)

            if curr not in done: # if this node is not in done
                done.add(curr)   # add curr to the done set

                #for c, v in W.get(curr, ()):  # get the current node's list of edges in W,
                r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
                for v, weight in zip(r.col, r.data):
                    if v in done or v in start: continue  # if the node v in done, skip it

                    prev_dist = dist.get(v, None) # get the current dist to v (if already calculated)

                    new_dist = curr_dist + 1./weight # add cost = 1/weight

                    if prev_dist is None or new_dist < prev_dist:   # if we haven't seen v or if v's cost has been updated
                        if radius is None or (radius is not None and  new_dist <= radius):
                            dist[v] = new_dist     # update distance in the dist dictionary and
                            heappush(considered, (new_dist, v))  # add v to the heap or update it on the heap
        DIST[k] = dist

    return DIST







# W is the weight matrix (CSR format), with weight in it. Cost will be 1/weight
# start is a list of labeled nodes (agnostic to class)
def dijkstra_csr(W, start, radius=None):
    # if set(class_ordering) != set(STARTS.keys()):
    #     raise ValueError("class_ordering %s and keys in STARTS %s do not line up")

    DIST = {x : 0.0 for x in start}
    # instantiate considered, seen, and dist
    considered, done = [(0, s) for s in start], set()
    heapify(considered) # make considered a heapq


    # while still have elements in considered...
    while considered:
        # get the dist, node id, and shortest path to the node corresponding to the top of heap
        (curr_dist, curr) = heappop(considered)

        if curr not in done: # if this node is not in done
            done.add(curr)   # add curr to the done set

            r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
            for v, weight in zip(r.col, r.data):
                if v in done or v in start: continue  # if the node v in done, skip it

                if v not in DIST:
                    prev_dist = np.inf
                else:
                    if DIST[v] == 0.0:# if this is an already labeled point, skip it
                        continue
                    prev_dist = DIST[v] # get the current dist to v (if already calculated, or is np.inf from being seen in other class)

                new_dist = curr_dist + 1./weight # add cost = 1/weight

                if radius is None or (radius is not None and new_dist <= radius):
                    DIST[v] = new_dist
                    if new_dist < radius:
                        heappush(considered, (new_dist, v))


    return DIST


# W is the weight matrix (CSR format), with weight in it. Cost will be 1/weight
# start is a list of labeled nodes (agnostic to class)
def dijkstra_csr_update(W, DIST, start, radius=None):
    # if set(class_ordering) != set(STARTS.keys()):
    #     raise ValueError("class_ordering %s and keys in STARTS %s do not line up")


    # instantiate considered, seen, and dist
    considered, done = [(0, s) for s in start], set()
    heapify(considered) # make considered a heapq


    # while still have elements in considered...
    while considered:
        # get the dist, node id, and shortest path to the node corresponding to the top of heap
        (curr_dist, curr) = heappop(considered)

        if curr not in done: # if this node is not in done
            done.add(curr)   # add curr to the done set

            r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
            for v, weight in zip(r.col, r.data):
                if v in done or v in start: continue  # if the node v in done, skip it

                if v not in DIST:
                    prev_dist = np.inf
                else:
                    if DIST[v] == 0.0:# if this is an already labeled point, skip it
                        continue
                    prev_dist = DIST[v] # get the current dist to v (if already calculated, or is np.inf from being seen in other class)

                new_dist = curr_dist + 1./weight # add cost = 1/weight

                if radius is None or (radius is not None and new_dist <= radius):
                    DIST[v] = new_dist
                    if new_dist < radius:
                        heappush(considered, (new_dist, v))


    return DIST















# W is the weight matrix (CSR format), with weight in it. Cost will be 1/weight
# STARTS is dictionary of class : [node in class]
def dijkstra_for_al_csr_joint(W, STARTS, radius=None, class_ordering=[1,-1]):
    if set(class_ordering) != set(STARTS.keys()):
        raise ValueError("class_ordering %s and keys in STARTS %s do not line up")

    DIST_joint = {}
    for i, k in enumerate(class_ordering):# for each class represented in STARTS
        start = STARTS[k]
        #print("Calculating for class %s" % str(k))
        # instantiate considered, seen, and dist
        considered, done = [(0, s) for s in start], set()
        heapify(considered) # make considered a heapq


        # while still have elements in considered...
        while considered:

            # get the dist, node id, and shortest path to the node corresponding to the top of heap
            (curr_dist, curr) = heappop(considered)

            if curr not in done: # if this node is not in done
                done.add(curr)   # add curr to the done set

                r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
                for v, weight in zip(r.col, r.data):
                    if v in done or v in start: continue  # if the node v in done, skip it

                    new_dist = curr_dist + 1./weight # add cost = 1/weight

                    if v not in DIST_joint:
                        # prev_dist is np.inf
                        if radius is None or (radius is not None and new_dist <= radius):
                            class_dists = len(class_ordering)*[np.inf]
                            class_dists[i] = new_dist
                            DIST_joint[v] = class_dists
                            if new_dist < radius:
                                heappush(considered, (new_dist, v))

                    else:
                        prev_dist = DIST_joint[v][i] # get the current dist to v (if already calculated, or is np.inf from being seen in other class)
                        if new_dist < prev_dist:
                            if radius is None or (radius is not None and new_dist <= radius):
                                DIST_joint[v][i] = new_dist         # update distance in the dist dictionary

                                if new_dist < radius:               # if we are "less than the radius", we want to add it to the heap
                                    heappush(considered, (new_dist, v))  # b/c if we are "at the radius", adding any weight will not enter the if statement


    return DIST_joint



# W is the weight matrix (CSR format), with weight in it. Cost will be 1/weight
# STARTS is dictionary of class : [node in class]
def dijkstra_for_al_csr_joint_old2(W, STARTS, radius=None, class_ordering=[1,-1], bandwidth=0.1, phase1=True):
    if set(class_ordering) != set(STARTS.keys()):
        print(class_ordering)
        print(list(STARTS.keys()))
        raise ValueError("class_ordering %s and keys in STARTS %s do not line up")

    DIST_joint = {}
    candidates = set([])
    for i, k in enumerate(class_ordering):# for each class represented in STARTS
        start = STARTS[k]
        #print("Calculating for class %s" % str(k))
        # instantiate considered, seen, and dist
        considered, done = [(0, s) for s in start], set()
        heapify(considered) # make considered a heapq


        # while still have elements in considered...
        while considered:

            # get the dist, node id, and shortest path to the node corresponding to the top of heap
            (curr_dist, curr) = heappop(considered)

            if curr not in done: # if this node is not in done
                done.add(curr)   # add curr to the done set

                r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
                for v, weight in zip(r.col, r.data):
                    if v in done or v in start: continue  # if the node v in done, skip it

                    new_dist = curr_dist + 1./weight # add cost = 1/weight

                    if v not in DIST_joint:
                        # prev_dist is np.inf
                        if radius is None or (radius is not None and new_dist <= radius):
                            class_dists = len(class_ordering)*[np.inf]
                            class_dists[i] = new_dist
                            DIST_joint[v] = class_dists
                            if new_dist < radius:
                                heappush(considered, (new_dist, v))

                            if phase1:
                                if radius - new_dist < 0.5:
                                    candidates.add(v)
                    else:
                        prev_dist = DIST_joint[v][i] # get the current dist to v (if already calculated, or is np.inf from being seen in other class)
                        if new_dist < prev_dist:
                            if radius is None or (radius is not None and new_dist <= radius):
                                DIST_joint[v][i] = new_dist         # update distance in the dist dictionary

                                if new_dist < radius:               # if we are "less than the radius", we want to add it to the heap
                                    heappush(considered, (new_dist, v))  # b/c if we are "at the radius", adding any weight will not enter the if statement

                                min_dists = sorted(DIST_joint[v])[:2] # get the distances to v from the 2 "closest" classes
                                if phase1:
                                    if min_dists[1] == np.inf:
                                        if radius - min_dists[0] < 0.5:
                                            candidates.add(v)
                                    else:
                                        if v in candidates:
                                            candidates.remove(v) # this is the case when one was added earlier, but don't want it in phase 1

                                else:
                                    if min_dists[1] - min_dists[0] <= bandwidth:
                                        candidates.add(v)





    return DIST_joint, list(candidates)


# W is the weight matrix (CSR format), with weight in it. Cost will be 1/weight
# STARTS is dictionary of class : [node in class]
def dijkstra_for_al_csr_joint_old(W, STARTS, radius=None, class_ordering=[1,-1]):
    if set(class_ordering) != set(STARTS.keys()):
        raise ValueError("class_ordering %s and keys in STARTS %s do not line up")

    DIST_joint = {}
    for i, k in enumerate(class_ordering):# for each class represented in STARTS
        start = STARTS[k]
        #print("Calculating for class %s" % str(k))
        # instantiate considered, seen, and dist
        considered, done = [(0, s) for s in start], set()
        heapify(considered) # make considered a heapq

        # while still have elements in considered...
        while considered:

            # get the dist, node id, and shortest path to the node corresponding to the top of heap
            (curr_dist, curr) = heappop(considered)

            if curr not in done: # if this node is not in done
                done.add(curr)   # add curr to the done set

                r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
                for v, weight in zip(r.col, r.data):
                    if v in done or v in start: continue  # if the node v in done, skip it

                    if v not in DIST_joint:
                        DIST_joint[v] = len(class_ordering)*[np.inf]

                    prev_dist = DIST_joint[v][i] # get the current dist to v (if already calculated, or is np.inf from being seen in other class)

                    new_dist = curr_dist + 1./weight # add cost = 1/weight



                    if new_dist < prev_dist:   # if we haven't seen v or if v's cost has been updated
                        if radius is None or (radius is not None and new_dist <= radius):
                            DIST_joint[v][i] = new_dist         # update distance in the dist dictionary and
                            heappush(considered, (new_dist, v))  # add v to the heap or update it on the heap

    return DIST_joint



# W is the Weight matrix, with weight in it. Cost = 1/weight
# DIST is dictionary of class : dist dict,    dist is dictionary of node : distance from labeled node in class to that node
# LABELED contains both the previously labeled as well as the added points
def dijkstra_for_al_update_csr_joint(W, DIST_joint, ADDED, LABELED, radius=None, class_ordering=[1,-1]):
    if set(class_ordering) != set(ADDED.keys()):
        raise ValueError("class_ordering %s and keys in ADDED %s do not line up")


    for i, k in enumerate(class_ordering):# for each class represented in STARTS
        start = ADDED[k]
        #print("Calculating for class %s" % str(k))
        # instantiate considered, seen, and dist
        considered, done = [(0, s) for s in start], set()
        heapify(considered) # make considered a heapq

        # while still have elements in considered...
        while considered:

            # get the dist, node id, and shortest path to the node corresponding to the top of heap
            (curr_dist, curr) = heappop(considered)

            if curr not in done: # if this node is not in done
                done.add(curr)   # add curr to the done set

                r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
                for v, weight in zip(r.col, r.data):
                    if v in done or v in LABELED: continue  # if the node v in done, skip it

                    new_dist = curr_dist + 1./weight # add cost = 1/weight

                    if v not in DIST_joint:
                        # prev_dist is np.inf
                        if radius is None or (radius is not None and new_dist <= radius):
                            class_dists = len(class_ordering)*[np.inf]
                            class_dists[i] = new_dist
                            DIST_joint[v] = class_dists
                            if new_dist < radius:
                                heappush(considered, (new_dist, v))

                    else:
                        prev_dist = DIST_joint[v][i] # get the current dist to v (if already calculated, or is np.inf from being seen in other class)
                        if new_dist < prev_dist:
                            if radius is None or (radius is not None and new_dist <= radius):
                                DIST_joint[v][i] = new_dist         # update distance in the dist dictionary

                                if new_dist < radius:               # if we are "less than the radius", we want to add it to the heap
                                    heappush(considered, (new_dist, v))  # b/c if we are "at the radius", adding any weight will not enter the if statement




    return DIST_joint



# W is the Weight matrix, with weight in it. Cost = 1/weight
# DIST is dictionary of class : dist dict,    dist is dictionary of node : distance from labeled node in class to that node
# LABELED contains both the previously labeled as well as the added points
def dijkstra_for_al_update_csr_joint_old2(W, DIST_joint, ADDED, LABELED, candidates, radius=None, class_ordering=[1,-1], bandwidth=0.1, phase1=True):
    if set(class_ordering) != set(ADDED.keys()):
        raise ValueError("class_ordering %s and keys in ADDED %s do not line up")


    for i, k in enumerate(class_ordering):# for each class represented in STARTS
        start = ADDED[k]
        #print("Calculating for class %s" % str(k))
        # instantiate considered, seen, and dist
        considered, done = [(0, s) for s in start], set()
        heapify(considered) # make considered a heapq

        # while still have elements in considered...
        while considered:

            # get the dist, node id, and shortest path to the node corresponding to the top of heap
            (curr_dist, curr) = heappop(considered)

            if curr not in done: # if this node is not in done
                done.add(curr)   # add curr to the done set

                r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
                for v, weight in zip(r.col, r.data):
                    if v in done or v in LABELED: continue  # if the node v in done, skip it

                    new_dist = curr_dist + 1./weight # add cost = 1/weight

                    if v not in DIST_joint:
                        # prev_dist is np.inf
                        if radius is None or (radius is not None and new_dist <= radius):
                            class_dists = len(class_ordering)*[np.inf]
                            class_dists[i] = new_dist
                            DIST_joint[v] = class_dists
                            if new_dist < radius:
                                heappush(considered, (new_dist, v))

                            if phase1:
                                if radius - new_dist < 0.5:
                                    candidates.add(v)
                    else:
                        prev_dist = DIST_joint[v][i] # get the current dist to v (if already calculated, or is np.inf from being seen in other class)
                        if new_dist < prev_dist:
                            if radius is None or (radius is not None and new_dist <= radius):
                                DIST_joint[v][i] = new_dist         # update distance in the dist dictionary

                                if new_dist < radius:               # if we are "less than the radius", we want to add it to the heap
                                    heappush(considered, (new_dist, v))  # b/c if we are "at the radius", adding any weight will not enter the if statement

                                min_dists = sorted(DIST_joint[v])[:2] # get the distances to v from the 2 "closest" classes
                                if phase1:
                                    if min_dists[1] == np.inf:
                                        if radius - min_dists[0] < 0.5:
                                            candidates.add(v)
                                        elif v in candidates:
                                            candidates.remove(v)
                                    else:
                                        if v in candidates:
                                            candidates.remove(v) # this is the case when one was added earlier, but don't want it in phase 1

                                else:
                                    if min_dists[1] - min_dists[0] <= bandwidth:
                                        candidates.add(v)


    return DIST_joint, list(candidates)

# W is the Weight matrix, with weight in it. Cost = 1/weight
# DIST is dictionary of class : dist dict,    dist is dictionary of node : distance from labeled node in class to that node
# LABELED contains both the previously labeled as well as the added points
def dijkstra_for_al_update_csr_joint_old(W, DIST_joint, ADDED, LABELED, radius=None, class_ordering=[1,-1]):
    if set(class_ordering) != set(ADDED.keys()):
        raise ValueError("class_ordering %s and keys in STARTS %s do not line up")

    for i, k in enumerate(class_ordering):# for each class represented in STARTS
        start = ADDED[k]
        #print("Calculating for class %s" % str(k))
        # instantiate considered, seen, and dist
        considered, done = [(0, s) for s in start], set()
        heapify(considered) # make considered a heapq

        # while still have elements in considered...
        while considered:

            # get the dist, node id, and shortest path to the node corresponding to the top of heap
            (curr_dist, curr) = heappop(considered)

            if curr not in done: # if this node is not in done
                done.add(curr)   # add curr to the done set

                r = W.getrow(curr).tocoo() # get the data of the current node's list of edges in W. NOTE : does not allow sinks
                for v, weight in zip(r.col, r.data):
                    if v in done or v in LABELED: continue  # if the node v in done, skip it

                    if v not in DIST_joint:
                        DIST_joint[v] = len(class_ordering)*[np.inf]

                    prev_dist = DIST_joint[v][i] # get the current dist to v (if already calculated, or is np.inf from being seen in other class)

                    new_dist = curr_dist + 1./weight # add cost = 1/weight

                    if new_dist < prev_dist:   # if we haven't seen v or if v's cost has been updated
                        if radius is None or (radius is not None and new_dist <= radius):
                            DIST_joint[v][i] = new_dist         # update distance in the dist dictionary and
                            heappush(considered, (new_dist, v))  # add v to the heap or update it on the heap


    return DIST_joint


def test_dijkstra_batch_al(W, labels, v=None, w=None, tau=0.1, gamma=0.1, B=5, coverage=0.95, \
                    n_start=10, num_batches=10, rad=5., bandwidth=0.5, class_ordering=[1,-1], \
                    model_classifier_name="probit2", acc_classifier_name=None, acquisition=None, \
                    seed=42, exact_update=False, verbose=False, X=None):
    if v is None or w is None:
        raise ValueError("Need to give eigenvectors in variable v and eigenvalues in w for accuracy calculations.")

    N = W.shape[0]

    model_classifier = Classifier(model_classifier_name, gamma, tau, v=v, w=w)
    if acc_classifier_name is None:
        acc_classifier = model_classifier
    else:
        acc_classifier = Classifier(acc_classifier_name, gamma, tau, v=v, w=w)
    np.random.seed(seed)

    labeled_orig = []
    for c in class_ordering:
        labeled_orig += list(np.random.choice(np.where(labels ==  c)[0],
                        size=n_start//len(class_ordering), replace=False))

    m = model_classifier.get_m(labeled_orig, labels[labeled_orig]) # Initial MAP estimator
    acc_m = acc_classifier.get_m(labeled_orig, labels[labeled_orig])
    C = model_classifier.get_C(labeled_orig, labels[labeled_orig], m)
    labeled = copy.deepcopy(labeled_orig)
    unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))


    if X is not None:
        plot_iter(acc_m, X, labels, labeled, subplot=True)
        plt.title("Initial")
        plt.show()
    acc = []
    acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])


    for i in range(num_batches):
        if verbose:
            print("Batch {}/{}".format(i+1, num_batches))
        if i == 0:
            fid = {}
            for c in class_ordering:
                fid[c] = [x for x in labeled if labels[x] == c]
            D_joint = dijkstra_for_al_csr_joint(W, fid, radius=rad, class_ordering=class_ordering)
        else:
            fid_new = {}
            for c in class_ordering:
                fid_new[c] = [k for k in k_batch if labels[k] == c]
            D_joint = dijkstra_for_al_update_csr_joint(W, D_joint, fid_new, labeled, radius=rad, class_ordering=class_ordering)    # run the Dijkstra on each class.

        print(len(D_joint))
        # Define the candidate set based on which "phase" of Dijkstra search we are in
        candidates = []
        if len(D_joint) > coverage*N:
            # let's get down on the decision boundary
            if verbose:
                print("\tPhase 2")
            for k in D_joint:
                if k not in labeled:
                    mm, MM = min(D_joint[k]), max(D_joint[k])
                    if MM-mm < bandwidth:
                        candidates.append(k)
        else:
            # still exploring the manifold
            if verbose:
                print("\tPhase 1")
            for k in D_joint:
                if k not in labeled:
                    mm, MM = min(D_joint[k]), max(D_joint[k])
                    if MM == np.inf:
                        if mm > rad - 0.5:
                            candidates.append(k)
        if acquisition is None:
            # i.e. random sampling in candidate set
            k_batch = list(np.random.choice(candidates, B, replace=False))
        else:
            k_batch = get_k_batch(candidates, labeled, B, acquisition, gamma=gamma, u0=m, X=X, labels=labels, C=C)

        labeled += k_batch
        unlabeled = list(filter(lambda x: x not in labeled, range(len(labels))))

        if exact_update == True:
            m = model_classifier.get_m(labeled, labels[labeled])
            C = model_classifier.get_C(labeled, labels[labeled], m)
        else:
            for k in k_batch: # do updates in sequence of rank one updates
                a = labels[k]
                m -= jac_calc2(m[k], a, gamma) / (1. + C[k,k] * hess_calc2(m[k], a, gamma))*C[k,:]
                C -= hess_calc2(m[k], a, gamma)/(1. + C[k,k]*hess_calc2(m[k], a, gamma))*np.outer(C[k,:], C[k,:])

        acc_m = acc_classifier.get_m(labeled, labels[labeled])
        if X is not None:
            plot_iter(acc_m, X, labels, labeled, subplot=True)
            plt.title("Iter %d" % (i+1))
            plt.show()
        acc.append(get_acc(acc_m, labels, unlabeled = unlabeled)[1])
        if verbose:
            print("\tAccuracy = %1.4f" % acc[-1])
            print()
    return acc, labeled




def get_k_batch(candidates, labeled, B, acquisition, gamma=0.1, u0=None, X=None, labels=None, C=None, Lt=None, y=None):
    if acquisition == "modelchange_batch":
        al_choices, al_choices_full = mc_batch(candidates, labeled, B, u0=u0, gamma=gamma, C=C)
    elif acquisition == "modelchange_batch_exact":
        al_choices, al_choices_full = mc_batch_exact(candidates, labeled, B, u0=u0, gamma=gamma, C=C)
    else:
        print("Did not recognize acquisition function %s, performing random sampling on candidate set" % acquisition)
        return list(np.random.choice(candidates, B, replace=False))
    return list(al_choices)


def mc_batch(candidates, labeled, B, u0=None, gamma=0.1, C=None, X=None, labels=None, Lt=None, y=None):
    if C is None or u0 is None:
        raise ValueError("Model Change requires that u0 = previous MAP estimator and C is posterior covariance matrix")
    N, Nu = C.shape[0], C.shape[0] - len(labeled)
    m = u0
    tic = time.clock()
    mc = np.array([min(np.absolute(jac_calc2(m[k], -1, gamma)/(1. + C[k,k]*hess_calc2(m[k], -1, gamma ))), \
       np.absolute(jac_calc2(m[k], 1, gamma)/(1. + C[k,k]*hess_calc2(m[k], 1, gamma )))) \
                   * np.linalg.norm(C[k,:]) for k in candidates])
    mc_p = (mc - np.min(mc))/(np.max(mc) - np.min(mc))
    avg = np.average(mc_p)
    # print("average of mc_p = %1.6f" % avg)
    # plt.hist(mc_p, bins=30)
    # plt.title('Modelchange histogram')
    # plt.show()
    mc_p[mc_p < avg] = 0.
    mc_pn = mc_p/np.sum(mc_p)
    mc_p_exp = np.exp(3.*mc_p)
    mc_p_expn = mc_p_exp/np.sum(mc_p_exp)


    # plt.subplot(1,2,1)
    # plt.scatter(range(Nu), mc_p, marker='.')
    # plt.title('Model Change Prob - linear')
    # plt.subplot(1,2,2)
    # plt.scatter(range(Nu), mc_p_exp, marker='.')
    # plt.title("Model Change Prob - exp")
    # plt.show()

    al_choices = list(np.random.choice(candidates, B, p= mc_pn, replace=False))
    al_choices_full = list(np.array(candidates)[np.where(mc_p > avg)])

    toc = time.clock()
    print("MC-Rand took %1.6f seconds" % (toc - tic) )

    # plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    # plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    # plt.title(r"Nonzero entries of $\mathbf{q}$")
    # plt.legend()
    # plt.show()
    # plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    # plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    # plt.title(r"Chosen B entries of $\mathbf{q}$")
    # plt.legend()
    # plt.show()

    return al_choices, al_choices_full


def mc_batch_exact(candidates, labeled, B, u0=None, gamma=0.1, C=None, X=None, labels=None, Lt=None, y=None):
    if C is None or u0 is None:
        raise ValueError("Model Change requires that u0 = previous MAP estimator and C is posterior covariance matrix")
    N, Nu = C.shape[0], C.shape[0] - len(labeled)
    m = u0
    tic = time.clock()
    mc = np.array([min(np.absolute(jac_calc2(m[k], -1, gamma)/(1. + C[k,k]*hess_calc2(m[k], -1, gamma ))), \
       np.absolute(jac_calc2(m[k], 1, gamma)/(1. + C[k,k]*hess_calc2(m[k], 1, gamma )))) \
                   * np.linalg.norm(C[k,:]) for k in candidates])

    avg = np.average(mc)
    al_choices = list(np.array(candidates)[(-mc).argsort()[:B]])
    al_choices_full = list(np.array(candidates)[np.where(mc > avg)])

    toc = time.clock()
    print("MC-Exact took %1.6f seconds" % (toc - tic) )

    # plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    # plt.scatter(X[al_choices_full,0], X[al_choices_full,1], marker='s', c='g', alpha=1.0, label='query choices')
    # plt.title(r"Nonzero entries of $\mathbf{q}$")
    # plt.legend()
    # plt.show()
    # plot_iter(u0, X, labels, labeled + al_choices, subplot=True)
    # plt.scatter(X[al_choices,0], X[al_choices,1], marker='s', c='g', alpha=1.0, label='query choices')
    # plt.title(r"Top B entries of $\mathbf{q}$")
    # plt.legend()
    # plt.show()

    return al_choices, al_choices_full
