import sys
sys.path.append('..')
from util.mlflow_util import load_uri, get_prev_run
import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla
import os
import mlflow
from sklearn.neighbors import NearestNeighbors

METRICS = ['euclidean', 'cosine']

class GraphManager(object):
    """
    Mostly Graph methods

    Necessary Fields:
    self.param

    Optional Fields:
    """

    def __init__(self):
        return


    def from_features(self, X, knn=15, sigma=3., normalized=True, n_eigs=None, zp_k=None, metric='euclidean', debug=False):
        """
        load from features using params
        params:
            knn
            sigma
            normalized
            n_eigs
            zp_k
        """
        assert metric in METRICS

        print("Computing (or retrieving) graph evals and evecs with parameters:")
        print("\tN = {}, knn = {}, sigma = {:.2f}".format(X.shape[0], knn, sigma))
        print("\tnormalized = {}, n_eigs = {}, zp_k = {}".format(normalized, n_eigs, zp_k))
        print("\tmetric = {}".format(metric))
        print()


        if not debug:
            params = {
                'knn' : knn,
                'sigma' : sigma,
                'normalized' : normalized,
                'n_eigs' : n_eigs,
                'zp_k' : zp_k,
                'metric' : metric
            }
            prev_run = get_prev_run('GraphManager.from_features',
                                    params,
                                    tags={"X":str(X)},
                                    git_commit=None)
            if prev_run is not None:
                print('Found previous eigs')
                eigs = load_uri(os.path.join(prev_run.info.artifact_uri,
                                'eigs.npz'))
                return eigs['w'], eigs['v']

        print('Did not find previous eigs, computing from scratch...')
        W = self.compute_similarity_graph(X=X, knn=knn, sigma=sigma,
                        zp_k=zp_k, metric=metric)

        L = self.compute_laplacian(W, normalized=normalized)
        w, v = self.compute_spectrum(L, n_eigs=n_eigs)
        L = sps.csr_matrix(L)
        self.W = W
        if debug:
            return w, v

        with mlflow.start_run(nested=True):
            np.savez('./tmp/eigs.npz', w=w, v=v)
            sps.save_npz('./tmp/W.npz', W)
            mlflow.set_tag('function', 'GraphManager.from_features')
            mlflow.set_tag('X', str(X))
            mlflow.log_params(params)
            mlflow.log_artifact('./tmp/eigs.npz')
            mlflow.log_artifact('./tmp/W.npz')
            os.remove('./tmp/eigs.npz')
            os.remove('./tmp/W.npz')
            return w, v

    def compute_similarity_graph(self, X, knn=15, sigma=3., zp_k=None, metric='euclidean', maxN=5000):
        """
        Computes similarity graph using parameters specified in self.param
        """
        N = X.shape[0]
        if knn is None:
            if N < maxN:
                knn = N
            else:
                print("Parameter knn was given None and N > maxN, so setting knn=15")
                knn = 15
        if N < maxN:
            print("Calculating NN graph with SKLEARN NearestNeighbors...")

            if knn > N / 2:
                nn = NearestNeighbors(n_neighbors=knn, algorithm='brute').fit(X)
            else:
                nn = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(X)

            # construct CSR matrix representation of the k-NN graph
            A_data, A_ind = nn.kneighbors(X, knn, return_distance=True)
        else:
            print("Calculating NN graph with NNDescent package since N = {} > {}".format(N, maxN))
            from pynndescent import NNDescent
            index = NNDescent(X, metric=metric)
            index.prepare()
            A_ind, A_data = index.query(X, k=knn)

        # modify from the kneighbors_graph function from sklearn to
        # accomodate Zelnik-Perona scaling
        n_nonzero = N * knn
        A_indptr = np.arange(0, n_nonzero + 1, knn)
        if zp_k is not None:
            k_dist = A_data[:,zp_k][:,np.newaxis]
            k_dist[k_dist < 1e-4] = 1e-4
            A_data /= np.sqrt(k_dist * k_dist[A_ind,0])

        A_data = np.ravel(A_data)
        W = sps.csr_matrix((np.exp(-(A_data ** 2)/sigma),
                            A_ind.ravel(),
                            A_indptr),
                            shape=(N, N))
        W = (W + W.T)/2
        W.setdiag(0)
        W.eliminate_zeros()

        return W


    def compute_laplacian(self, W, normalized=True):
        """
        Computes the graph Laplacian using parameters specified in self.params
        """
        if normalized:
            L = sps.csgraph.laplacian(W, normed=True)
        else:
            L = sps.csgraph.laplacian(W, normed=False)
        return L


    def compute_spectrum(self, L, n_eigs):
        """
        Computes first n_eigs smallest eigenvalues and eigenvectors
        """
        N = L.shape[0]
        if n_eigs is None:
            n_eigs = N
        if n_eigs > int(N/2):
            w, v = sla.eigh(L.toarray(), eigvals=(0,n_eigs-1))
        else:
            w, v = sps.linalg.eigsh(L, k=n_eigs, which='SM')
        return w, v
