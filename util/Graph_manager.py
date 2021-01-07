import sys
sys.path.append('..')
from util.mlflow_util import load_uri, get_prev_run
import numpy as np
import scipy.sparse as sps
import scipy.linalg as sla
import os
import mlflow
from sklearn.neighbors import NearestNeighbors


class Graph_manager(object):
    """
    Mostly Graph methods

    Necessary Fields:
    self.param

    Optional Fields:
    self.distance_mat
    self.similarity_mat
    self.laplacian
    self.eigenvalues
    self.eigenvectors
    """

    def __init__(self):
        return

    def __del__(self):
        try:
            os.remove('./eigs.npz')
        except:
            pass
        return

    def load_data_obj(self, data_obj):
        """
        Load from a Data_obj
        Sets:
            self.data_obj
        """
        self.data_obj = data_obj
        pass

    def update_param(self, new_param):
        """
        new_param : dictionary with param name : param value
        """
        pass


    # Needed for all Graph_manager

    def get_spectrum(self, ks, store=False):
        """
        Returns eigenvalues and eigenvectors specified in ks
        If they are computed before, compute them
        """
        pass


    def from_features(self, X, params, debug=False):
        """
        load from features using params
        params:
            knn
            sigma
            Ltype
            n_eigs
            zp_k
        """
        if not debug:
            prev_run = get_prev_run('Graph_manager.from_features',
                                    params,
                                    tags={"X":str(X)},
                                    git_commit=None)
            if prev_run is not None:
                print('Found previous eigs')
                eigs = load_uri(os.path.join(prev_run.info.artifact_uri,
                                'eigs.npz'))
                return eigs['w'], eigs['v']

        print('Compute eigs from scratch')
        W = self.compute_similarity_graph(
            X            = X,
            knn          = params['knn'],
            sigma        = params['sigma'],
            zp_k         = params['zp_k'])


        self.W = W
        A = self.compute_laplacian(W,
            Ltype = params['Ltype'])
        w, v = self.compute_spectrum(A, n_eigs=params['n_eigs'])
        self.L = sps.csr_matrix(A)

        if debug:
            return w, v

        with mlflow.start_run(nested=True):
            np.savez('./eigs.npz', w=w, v=v)
            sps.save_npz('./W.npz', W)
            mlflow.set_tag('function', 'Graph_manager.from_features')
            mlflow.set_tag('X', str(X))
            mlflow.log_params(params)
            mlflow.log_artifact('./eigs.npz')
            mlflow.log_artifact('./W.npz')
            return w, v

    def sqdist(self, X, Y):
        """
        Computes dense pairwise euclidean distance between X and Y
        """
        m = X.shape[1]
        n = Y.shape[1]
        Yt = Y.T
        XX = np.sum(X*X, axis=0)
        YY = np.sum(Yt*Yt, axis=1).reshape(n, 1)
        return np.tile(XX, (n, 1)) + np.tile(YY, (1, m)) - 2*Yt.dot(X)

    def compute_similarity_graph(self, X, knn, sigma=1, zp_k=None):
        """
        Computes similarity graph using parameters specified in self.param
        """
        # Probably we want to set all default parameters in one place
        N = len(X)
        if knn is None:
            knn = N

        if knn > N / 2:
            nn = NearestNeighbors(n_neighbors=knn, algorithm='brute').fit(X)
        else:
            nn = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(X)

        # modify from the kneighbors_graph function from sklearn to
        # accomodate Zelnik-Perona scaling
        n_nonzero = N * knn
        A_indptr = np.arange(0, n_nonzero + 1, knn)
        # construct CSR matrix representation of the k-NN graph
        A_data, A_ind = nn.kneighbors(X, knn, return_distance=True)


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

    def compute_similarity_graph_cosine(self, X, knn, sigma=1, zp_k=None):
        """
        Computes similarity graph using parameters specified in self.param
        """
        # Probably we want to set all default parameters in one place
        N = len(X)
        if knn is None:
            knn = N

        if knn > N / 2:
            nn = NearestNeighbors(n_neighbors=knn, algorithm='brute').fit(X/np.linalg.norm(X, axis=1)[:,np.newaxis])
        else:
            nn = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree', leaf_size=200).fit(X/np.linalg.norm(X, axis=1)[:,np.newaxis])

        # modify from the kneighbors_graph function from sklearn to
        # accomodate Zelnik-Perona scaling
        n_nonzero = N * knn
        A_indptr = np.arange(0, n_nonzero + 1, knn)
        # construct CSR matrix representation of the k-NN graph
        A_data, A_ind = nn.kneighbors(X, knn, return_distance=True)

        # ZP scaling on the normalized distances, wait to do the cosine similarity after
        if zp_k is not None:
            k_dist = A_data[:,zp_k+1][:,np.newaxis]
            k_dist[k_dist < 1e-4] = 1e-4
            A_data /= np.sqrt(k_dist * k_dist[A_ind,0])

        A_data = np.ravel(A_data[:,1:])
        W = sps.csr_matrix((1. - 0.5*A_data**2., # cosine similarity for the weights
                            A_ind[:,1:].ravel(),
                            A_indptr),
                            shape=(N, N))
        W = (W + W.T)/2

        return W

    # def compute_cost(self, X, knn, sigma=1, zp_k=None):
    #     """
    #     Computes similarity graph using parameters specified in self.param
    #     """
    #     # Probably we want to set all default parameters in one place
    #     N = len(X)
    #     if knn is None:
    #         knn = N
    #
    #     if knn > N / 2:
    #         nn = NearestNeighbors(n_neighbors=knn, algorithm='brute').fit(X)
    #     else:
    #         nn = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(X)
    #
    #     # modify from the kneighbors_graph function from sklearn to
    #     # accomodate Zelnik-Perona scaling
    #     n_nonzero = N * knn
    #     A_indptr = np.arange(0, n_nonzero + 1, knn)
    #     # construct CSR matrix representation of the k-NN graph
    #     A_data, A_ind = nn.kneighbors(X, knn+1, return_distance=True)
    #     A_data, A_ind = A_data[:,1:], A_ind[:,1:]
    #     if zp_k is not None:
    #         k_dist = A_data[:,zp_k][:,np.newaxis]
    #         k_dist[k_dist < 1e-4] = 1e-4
    #         A_data /= np.sqrt(k_dist * k_dist[A_ind,0])
    #
    #     #A_data = np.ravel(A_data)
    #     C = sps.lil_matrix((np.exp((A_data.ravel() ** 2)/sigma),
    #                         A_ind.ravel(),
    #                         A_indptr),
    #                         shape=(N, N))
    #     C = (C + C.T)/2
    #
    #     return C

    def compute_laplacian(self, W, Ltype):
        """
        Computes the graph Laplacian using parameters specified in self.params
        """
        if Ltype == 'normed':
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
