import numpy as np
from scipy.spatial import distance
import time
import pandas as pd


class fisvdd:
    def __init__(self, data, sigma, eps_cp=1e-8, eps_ol=1e-8):

        """Default.

        Inputs
        ----------
        data: 		input streaming data
        sigma: 		Gaussian similarity bandwidth
        eps_cp:		epsilon for close points
        eps_ol:     epsilon for outliers
        inv_A:		inverse of similarity matrix
        alpha:		the alpha values of support vectors
        sv:			support vectors
        obj_val:	stored objective values
        score:		score value for KKT condition
        sim_vec:    temporarily stored similarity vector

        """

        self.data = data
        self.sigma = sigma
        self.eps_cp = eps_cp
        self.eps_ol = eps_ol

        self.inv_A = np.array([1])
        self.alpha = np.array([1])
        self.sv = np.array([self.data[0]])
        self.obj_val = []
        self.score = 1
        self.index = [0]

    def _print_res(self):
        # print("\nalpha -------")
        # print(self.alpha)
        print(len(self.alpha))

        # print("\nsupport vector -------")
        # print(self.sv)

    def find_sv(self):

        """
        FISVDD main function.
        """
        # for new_data in self.data[1:]:
        #     print(new_data)
        a = self.data[1:]
        for i in range(self.data.shape[0]-1):
            new_data = self.data[1:][i]
            new_data = np.array([new_data])

            score, sim_vec = self.score_fcn(new_data)
            if score > 0:
                self.expand(new_data, sim_vec, i+1)

                if min(self.alpha) < 0:
                    backup, data_out_index = self.shrink()
                    for each in backup:
                        each = np.array([each])
                        score, sim_vec = self.score_fcn(each)
                        if score > 0:
                            self.expand(each, sim_vec, data_out_index)

                self.model_update()

            self.obj_val.append(self.score)

        return

    def up_inv(self, prev_inv, v):

        """
        Calculate the inverse of A_(k+1) based on Lemma 2.

        Inputs
        ----------
        prev_inv: inverse of A_k
        v: Similarity vector between new data point and support vectors

        Returns
        -------
        inverse of A_(k+1)
        """
        p = np.dot(prev_inv, v)
        beta = 1 - np.dot(v, p)
        A = prev_inv + np.outer(p, p) / beta
        C = - p / beta
        B = np.reshape(- p / beta, (len(C), 1))
        D = 1 / beta
        res = np.vstack((np.hstack((A, B)), np.hstack((C, D))))
        return res

    def down_inv(self, next_inv):

        """Calculate the inverse of A_k based on Lemma 3.

        Inputs
        ----------
        next_inv: inverse of A_(k+1)

        Returns
        -------
        inverse of A_k
        """

        lamb = next_inv[-1, -1]
        u = next_inv[:-1, -1]
        res = next_inv[:-1, :-1] - np.outer(u, u) / lamb
        return res

    def expand(self, new_sv, new_sim_vec, index_i):

        """Expand the support vector set according to algorithm 1.

        Inputs
        ----------
        new_sv: The new support vector

        """

        self.inv_A = self.up_inv(self.inv_A, new_sim_vec)
        self.alpha = np.sum(self.inv_A, axis=1)
        self.sv = np.vstack((self.sv, new_sv))
        self.index.append(index_i)

    def shrink(self):

        """
        Shrink the support vector set according to algorithm 2.
        """

        backup = []
        while True:
            min_ind = np.where(self.alpha == min(self.alpha))[0][0]
            data_out = self.sv[min_ind, :]
            data_out_index = self.index[min_ind]
            backup.append(data_out)
            pInd = np.where(self.alpha > min(self.alpha))
            self.sv = self.sv[pInd]
            index = np.array(self.index).reshape(len(self.index), 1)[pInd]
            self.index = list(index[:, 0])
            self.inv_A = self.perm(self.inv_A, min_ind)
            self.inv_A = self.down_inv(self.inv_A)
            self.alpha = np.sum(self.inv_A, axis=1)
            if min(self.alpha) > 0:
                break
        return backup, data_out_index

    def perm(self, A, ind):

        """
        Permutation function
        """

        n = A.shape[1]
        perm_vec = np.arange(n)
        perm_vec[:ind] = np.arange(0, ind)
        perm_vec[ind:n-1] = np.arange(ind+1, n)
        perm_vec[n-1] = ind
        temp = A[:, perm_vec]
        res = temp[perm_vec, :]
        return res

    def score_fcn(self, new_data):

        """
        Score function
        """

        dist_sq = distance.cdist(new_data, self.sv)[0]
        cur_sim_vec = np.exp(-np.square(dist_sq) / (2.0 * self.sigma * self.sigma))
        m = max(cur_sim_vec)
        if m < self.eps_ol or m > 1 - self.eps_cp:
            res = -1
        else:
            res = self.score - np.dot(self.alpha, cur_sim_vec)
        return res, cur_sim_vec

    def model_update(self):

        """
        Update score and alpha values of the model
        """

        self.score = 1 / sum(self.alpha)
        self.alpha = self.alpha / sum(self.alpha)
