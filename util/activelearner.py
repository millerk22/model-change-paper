# Active Learner object class
# author: Kevin Miller, ksmill327@gmail.com
#
# Need to implement acquisition_values function more efficiently, record values if debug is on.
import numpy as np
from .dijkstra import *
from .al_util import * 
from .acquisitions import *
from .rkhs import *


ACQUISITIONS = ['mc', 'uncertainty', 'rand', 'vopt', 'sopt', 'mbr', 'mcgreedy', 'db']
# MODELS = ['gr', 'probit-log', 'probit-norm', 'softmax', 'log', 'probitnorm']
CANDIDATES = ['rand', 'full', 'dijkstra']
SELECTION_METHODS = ['top', 'prop', '']

def sgn(x):
    if x >= 0:
        return 1.
    else:
        return -1.

def acquisition_values(acq, Cand, model):
    if acq == "mc":
        if model.full_storage:
            return mc_full(Cand, model.m, model.C, model.modelname, gamma=model.gamma)
        else:
            return mc_reduced(model.C_a, model.alpha, model.v[Cand,:], model.modelname, uks=model.m[Cand], gamma=model.gamma)
    elif acq == "mcgreedy":
        return mc_reduced(model.C_a, model.alpha, model.v[Cand,:], model.modelname, uks=model.m[Cand], gamma=model.gamma, greedy=True)
    elif acq == "uncertainty":
        if len(model.m.shape) > 1: # entropy calculation
            #print('multi unc')
            probs = np.exp(model.m[Cand])
            probs /= np.sum(probs, axis=1)[:, np.newaxis]
            return -np.sum(probs*np.log(probs), axis=1)
        else:
            return -np.absolute(model.m[Cand])  # ensuring a "max" formulation for acquisition values
    elif acq == "rand":
        return np.random.rand(len(Cand))
    elif acq == "vopt":
        if model.modelname == 'hf':
            return model.vopt_vals(Cand)
        else:
            if model.full_storage:
                ips = np.array([np.inner(model.C[k,:], model.C[k,:]) for k in Cand]).flatten()
                if model.modelname in ["gr", "mgr"]:
                    return ips/(model.gamma**2. + np.diag(model.C)[Cand])
                if model.modelname == "probit-norm":
                    return ips * np.array([hess_calc(model.m[k], sgn(model.m[k]), model.gamma)/ \
                            (hess_calc(model.m[k], sgn(model.m[k]), model.gamma)*model.C[k,k] + 1.) for k in Cand])
                if model.modelname == "probit-log":
                    return ips * np.array([hess_calc2(model.m[k], sgn(model.m[k]), model.gamma)/ \
                            (hess_calc2(model.m[k], sgn(model.m[k]), model.gamma)*model.C[k,k] + 1.) for k in Cand])
            else:
                uks = model.m[Cand]
                C_a_vk = model.C_a @ (model.v[Cand,:].T)
                ips = np.array([np.inner(C_a_vk[:,i],C_a_vk[:,i]) for i in range(len(Cand))])
                if model.modelname in ["gr", "mgr"]:
                    return ips / (model.gamma**2. + np.array([np.inner(model.v[Cand[i],:], C_a_vk[:,i]) for i in range(len(Cand))]))

                if model.modelname == 'probit-log' or model.modelname == 'log':
                    return ips * np.array([hess_calc2(model.m[k], sgn(model.m[k]), model.gamma)/ \
                            (hess_calc2(model.m[k], sgn(model.m[k]), model.gamma)*np.inner(model.v[Cand,:][i,:], C_a_vk[:,i]) + 1.) for i,k in enumerate(Cand)])

                if model.modelname == 'probit-norm' or model.modelname == 'probitnorm':
                    return ips * np.array([hess_calc(model.m[k], sgn(model.m[k]), model.gamma)/ \
                            (hess_calc(model.m[k], sgn(model.m[k]), model.gamma)*np.inner(model.v[Cand,:][i,:], C_a_vk[:,i]) + 1.) for i,k in enumerate(Cand)])

    elif acq == "sopt":
        if model.modelname == 'hf':
            return model.sopt_vals(Cand)
        else:
            if model.full_storage:
                sums = np.sum(model.C[Cand,:], axis=1).flatten()**2.
                if model.modelname in ["gr", "mgr"]:
                    return sums/(model.gamma**2. + np.diag(model.C)[Cand])
                if model.modelname == "probit-norm":
                    return sums * np.array([hess_calc(model.m[k], sgn(model.m[k]), model.gamma)/ \
                            (hess_calc(model.m[k], sgn(model.m[k]), model.gamma)*model.C[k,k] + 1.) for k in Cand])
                if model.modelname == "probit-log":
                    return sums * np.array([hess_calc2(model.m[k], sgn(model.m[k]), model.gamma)/ \
                            (hess_calc2(model.m[k], sgn(model.m[k]), model.gamma)*model.C[k,k] + 1.) for k in Cand])
            else:
                uks = model.m[Cand]
                C_a_vk = model.C_a @ (model.v[Cand,:].T)
                VTones = np.sum(model.v, axis=0).flatten()
                tops = np.array([np.inner(VTones, C_a_vk[:,i]) for i in range(len(Cand))])**2.
                if model.modelname in ['gr', 'mgr']:
                    return tops / (model.gamma**2. + np.array([np.inner(model.v[Cand[i],:], C_a_vk[:,i]) for i in range(len(Cand))]))

                if model.modelname == 'probit-log' or model.modelname == 'log':
                    return tops * np.array([hess_calc2(model.m[k], sgn(model.m[k]), model.gamma)/ \
                            (hess_calc2(model.m[k], sgn(model.m[k]), model.gamma)*np.inner(model.v[Cand,:][i,:], C_a_vk[:,i]) + 1.) for i,k in enumerate(Cand)])

                if model.modelname == 'probit-norm' or model.modelname == 'probitnorm':
                    return tops * np.array([hess_calc(model.m[k], sgn(model.m[k]), model.gamma)/ \
                            (hess_calc(model.m[k], sgn(model.m[k]), model.gamma)*np.inner(model.v[Cand,:][i,:], C_a_vk[:,i]) + 1.) for i,k in enumerate(Cand)])
    elif acq == "mbr":
        raise NotImplementedError()
    elif acq == "db":
        if model.modelname != 'rkhs':
            raise NotImplementedError("Databased norm is for RKHS model only")
        else:
            return model.look_ahead_db_norms(Cand)
    else:
        raise ValueError("Acquisition function %s not yet implemented" % str(acq))

    return





class ActiveLearner(object):
    def __init__(self, acquisition='mc', candidate='full', candidate_frac=0.1, W=None, r=None):
        if acquisition not in ACQUISITIONS:
            raise ValueError("Acquisition function name %s not valid, must be in %s" % (str(acquisition), str(ACQUISITIONS)))
        self.acquisition = acquisition
        if candidate not in CANDIDATES:
            raise ValueError("Candidate Set Selection name %s not valid, must be in %s" % (str(candidate), str(CANDIDATES)))
        self.candidate = candidate
        if (candidate_frac < 0. or candidate_frac > 1. ) and self.candidate == 'rand':
            print("WARNING: Candidate fraction must be between 0 and 1 for 'rand' candidate selection, setting to default 0.1")
            self.candidate_frac = 0.1
        else:
            self.candidate_frac = candidate_frac
        # if modelname not in MODELS:
        #     raise ValueError("Model name %s not valid, must be in %s" % (str(modelname), str(MODELS)))
        # self.modelname = modelname

        if self.candidate == 'dijkstra':
            self.W = W
            if self.W is None:
                raise ValueError("Candidate set selection %s requires W to be non-empty" % candidate)
            self.DIST = {}
            if r is None:
                self.dijkstra_r = 5.0
            else:
                self.dijkstra_r = r
        # else:
        #     # If weight matrix is passed to ActiveLearner but not doing Dijkstra, ignore it
        #     if self.W is not None:
        #         self.W = None


    def select_query_points(self, model, B=1, method='top', prop_func=None, prop_sigma=0.8, debug=False, verbose=False):
        if method not in SELECTION_METHODS:
            raise ValueError("Selection method %s not valid, must be one of %s" % (method, SELECTION_METHODS))

        if verbose:
            print("Active Learner settings:")
            print("\tacquisition function = %s" % self.acquisition)
            print("\tB = %d" % B)
            print("\tcandidate set = %s" % self.candidate)
            print("\tselection method = %s" % method)

        # Define the candidate set
        if self.candidate is "rand":
            Cand = np.random.choice(model.unlabeled, size=int(self.candidate_frac * len(model.unlabeled)), replace=False)
        elif self.candidate is "dijkstra":
            raise NotImplementedError("Have not implemented the dikstra candidate selection for this class")
        else:
            Cand = model.unlabeled
        if debug:
            self.Cand = Cand
        # Compute acquisition values -- save as object attribute for later plotting
        self.acq_vals = acquisition_values(self.acquisition, Cand, model)
        if len(self.acq_vals.shape) > 1:
            print("WARNING: acq_vals is of shape %s, should be one-dimensional. MIGHT CAUSE PROBLEM" % str(self.acq_vals.shape))

        # based on selection method, choose query points
        if B == 1:
            if method != 'top':
                print("Warning : B = 1 but election method is not 'top'. Overriding selection method and selecting top choice for query point.")
            return [Cand[np.argmax(self.acq_vals)]]
        else:
            if method == 'top':
                return [Cand[k] for k in (-self.acq_vals).argsort()[:B]]

            elif method == 'prop':
                if prop_func is None:
                    # if not given a customized proportionality sampling function, use this default.
                    # (1) normalize to be 0 to 1
                    acq_vals = (self.acq_vals - np.min(self.acq_vals))/(np.max(self.acq_vals) - np.min(self.acq_vals))
                    p = np.exp(acq_vals/prop_sigma)
                    p /= np.sum(p)
                else:
                    p = prop_func(acq_vals)
                if debug:
                    return list(np.random.choice(Cand, B, replace=False, p=p)), p, acq_vals, Cand
                return list(np.random.choice(Cand, B, replace=False, p=p))
            else:
                raise ValueError("Have not implemented this selection method, %s. Somehow got passed other parameter checks..." % method)
