"""
Created on Wed Nov 26 12:42:56 2019

@author: MrBrown
"""

import numpy as np
from numba import jit
from scipy import optimize as opt
import multiprocessing as mp
from tqdm import tqdm_notebook
import poibin as pb
from functools import partial
from scipy.stats import poisson, norm

@jit(nopython=True)
def vec2mat(x, y):
    return np.atleast_2d(x).T @ np.atleast_2d(y)

@jit(nopython=True)
def numba_sum(array):
    return array.sum()

# This and the next one are not saving so much time. I'll see if keeping them by testing the whole package
@jit(nopython=True)
def numba_1_arr(array):
    return 1 - array

@jit(nopython=True)
def numba_sqrt(array):
    return np.sqrt(array)

@jit(nopython=True)
def sum_combination(x, y):
    """
    This function computes a matrix in which each element is the exponential
    of the sum of the corresponding elements of x and y
    """
    return np.exp(np.atleast_2d(x).T + np.atleast_2d(y))

@jit(nopython=True)
def v_probs_from_fitnesses(x_i, x_j, y):
    return x_i * x_j * (y ** 2) / ((1 + x_i * y) * (1 + x_j * y))

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, threads_num=4):
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(threads_num)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(threads_num)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]

@jit(nopython=True)
def v_list_from_v_mat(v_mat):
    n_rows = len(v_mat)
    v_list = []
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if v_mat[i, j] > 0:
                v_list.append((i, j, v_mat[i, j]))
    return v_list

class PvalClass:
    def __init__(self):
        ##### Full problem parameters
        self.x = None
        self.y = None
        self.avg_mat = None
        self.avg_v_mat = None
        self.n_rows = None
        self.n_cols = None
        self.pval_list = None
        self.light_mode = None
        self.method = None
        self.threads_num = None
    
    def set_fitnesses(self, x, y):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.n_rows = len(x)
        self.n_cols = len(y)
        self.light_mode = True
        self.avg_mat = None
    
    def set_avg_mat(self, avg_mat):
        self.avg_mat = np.array(avg_mat, dtype=float)
        self.n_rows, self.n_cols = self.avg_mat.shape
        self.light_mode = False
        self.x = None
        self.y = None
    
    def pval_calculator(self, v):
        i = v[0]
        j = v[1]
        if self.method == 'poisson':
            if self.light_mode:
                avg_v = numba_sum(v_probs_from_fitnesses(self.x[i], self.x[j], self.y))
            else:
                avg_v = self.avg_v_mat[i, j]
            return (i, j, poisson.sf(k=v[2] - 1, mu=avg_v))
        elif self.method == 'normal':
            if self.light_mode:
                probs = v_probs_from_fitnesses(self.x[i], self.x[j], self.y)
            else:
                probs = self.avg_mat[i] * self.avg_mat[j]
            avg_v = numba_sum(probs)
            sigma_v = numba_sqrt(numba_sum(probs * numba_1_arr(probs)))
            return (i, j, norm.cdf((v[2] + 0.5 - avg_v) / sigma_v))
        elif self.method == 'rna':
            if self.light_mode:
                probs = v_probs_from_fitnesses(self.x[i], self.x[j], self.y)
            else:
                probs = self.avg_mat[i] * self.avg_mat[j]
            avg_v = numba_sum(probs)
            var_v_arr = probs * numba_1_arr(probs)
            sigma_v = numba_sqrt(numba_sum(var_v_arr))
            gamma_v = (sigma_v ** (-3)) * numba_sum(var_v_arr * (numba_1_arr(2 * probs)))
            eval_x = (v[2] + 0.5 - avg_v) / sigma_v
            pval_temp = norm.cdf(eval_x) + gamma_v * (1 - eval_x ** 2) * norm.pdf(eval_x) / 6
            if pval_temp < 0:
                return (i, j, 0)
            elif pval_temp > 1:
                return (i, j, 1)
            else:
                return (i, j, pval_temp)
    
    def pval_calculator_poibin(self, v_couple):
        if self.light_mode:
            probs = v_probs_from_fitnesses(self.x[v_couple[0][0]], self.x[v_couple[0][1]], self.y)
        else:
            probs = self.avg_mat[v_couple[0][0]] * self.avg_mat[v_couple[0][1]]
        pb_obj = pb.PoiBin(probs)
#         pvals_list = np.zeros(len(v_couple[2]), dtype=float)
        pvals_list = [(v[0], v[1], pb_obj.pval(int(v[2]))) for v in v_couple]
#         for v_index in v_couple[2]:
#             inverse_pos = np.where(pos_v == v_index)[0]
#             pvals_list[inverse_pos] = (i, j, pb_obj.pval(int(unique_v[v_index])))
        return pvals_list
    
#     Va assolutamente rivisto, quei np.where perdono un sacco di tempo
    def _calculate_pvals(self, v_list):
        """
        Internal method for calculating pvalues given an overlap list.
        v_list contains triplets of two nodes and their number of v-motifs (common neighbors).
        The parallelization is different from poibin solver and other types of solvers.
        """
        if self.method != 'poibin':
            if self.method == 'poisson' and not self.light_mode:
                self.avg_v_mat = self.avg_mat @ self.avg_mat.T
            func = partial(self.pval_calculator)
            self.pval_list = parmap(func, tqdm_notebook(v_list))
        else:
            if self.light_mode:
                r_x, r_invert = np.unique(self.x, return_inverse=True)
                r_n_rows = len(r_x)
            else:
                r_deg, r_invert = np.unique(self.avg_mat.sum(1), return_inverse=True)
                r_n_rows = len(r_deg)
            v_list_coupled = []
            for i in range(r_n_rows):
                pos_i = np.where(r_invert == i)[0]
                for j in range(i, r_n_rows):
                    pos_j = np.where(r_invert == j)[0]
                    red_v_list = [v for v in v_list if (v[0] in pos_i and v[1] in pos_j) or (v[0] in pos_j and v[1] in pos_i)]
                    if len(red_v_list) > 0:
                        v_list_coupled.append(red_v_list)
            if self.threads_num > 1:
                pval_list_coupled = parmap(self.pval_calculator_poibin, tqdm_notebook(v_list_coupled))
            else:
                for v_couple in v_list_coupled:
                    pval_list_coupled.append(self.pval_calculator_poibin(v_couple))
            self.pval_list = [pval for pvals_set in pval_list_coupled for pval in pvals_set]
        return
    
    def compute_pvals(self, observed_net, method='poisson', threads_num=1): # If threads_num == 1 it should not be parallelized
        self.method = method
        self.threads_num = threads_num
        if self.light_mode: # It should be a list containing (vertex1, vertex2, number_of_v_motifs)
            self._calculate_pvals(observed_net)
        else: # In this case it should be the matrix of V-motifs: biad_mat @ biad_mat.T
            v_list = v_list_from_v_mat(observed_net)
            self._calculate_pvals(v_list)