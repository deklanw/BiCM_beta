import poibin as pb
import numpy as np
from scipy.stats import poisson, norm
from scipy import sparse
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm_notebook
from BiCM_class import BiCM_class as bicm
from numba import jit
from Pval_class import PvalClass as pval_class
import multiprocessing as mp


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


def check_sol(biad_mat, avg_bicm, return_error=False, in_place=False):
    """
        This function prints the rows sums differences between two matrices, that originally are the biadjacency matrix and its bicm average matrix.
        The intended use of this is to check if an average matrix is actually a solution for a bipartite configuration model.
        
        If return_error is set to True, it returns 1 if the sum of the differences is bigger than 1.
        
        If in_place is set to True, it checks and sums also the total error entry by entry.
        The intended use of this is to check if two solutions are the same solution.
    """
    error = 0
    if (avg_bicm < 0).sum() != 0:
        print('NEGATIVE ENTRIES IN THE AVERAGE MATRIX!')
        error = 1
    rows_error_vec = np.abs(np.sum(biad_mat, axis=0) - np.sum(avg_bicm, axis=0))
    err_rows = np.max(rows_error_vec)
    print('max rows error =', err_rows)
    cols_error_vec = np.abs(np.sum(biad_mat, axis=1) - np.sum(avg_bicm, axis=1))
    err_cols = np.max(cols_error_vec)
    print('max columns error =', err_cols)
    tot_err = np.sum(rows_error_vec) + np.sum(cols_error_vec)
    print('total error =', tot_err)
    if tot_err > 1:
        error = 1
        print('WARNING total error > 1')
        if tot_err > 10:
            print('total error > 10')
    if err_rows + err_cols > 1:
        print('max error > 1')
        error = 1
        if err_rows + err_cols > 10:
            print('max error > 10')
    if in_place:
        diff_mat = np.abs(biad_mat - avg_bicm)
        print('In-place total error:', np.sum(diff_mat))
        print('In-place max error:', np.max(diff_mat))
    if return_error:
        return error
    else:
        return


def check_sol_light(x, y, rows_deg, cols_deg, return_error=False):
    """
    Light version of the check_sol function, working only on the fitnesses and the degree sequences.
    """
    error = 0
    rows_error_vec = np.abs([(x[i] * y / (1 + x[i] * y)).sum() - rows_deg[i] for i in range(len(x))])
    err_rows = np.max(rows_error_vec)
    print('max rows error =', err_rows)
    cols_error_vec = np.abs([(x * y[j] / (1 + x * y[j])).sum() - cols_deg[j] for j in range(len(y))])
    err_cols = np.max(cols_error_vec)
    print('max columns error =', err_cols)
    tot_err = np.sum(rows_error_vec) + np.sum(cols_error_vec)
    print('total error =', tot_err)
    if tot_err > 1:
        error = 1
        print('WARNING total error > 1')
        if tot_err > 10:
            print('total error > 10')
    if err_rows + err_cols > 1:
        print('max error > 1')
        error = 1
        if err_rows + err_cols > 10:
            print('max error > 10')
    if return_error:
        return error
    else:
        return


@jit(nopython=True)
def vec2mat(x, y):
    """
    Given two vectors x_i, y_j returns the matrix of products x_i * y_j
    """
    return np.atleast_2d(x).T @ np.atleast_2d(y)


def bicm_from_fitnesses(x, y):
    """
    Rebuilds the average probability matrix of the bicm from the fitnesses
    """
    avg_mat = vec2mat(x, y)
    avg_mat /= 1 + avg_mat
    return avg_mat


def initialize_avg_mat(biad_mat):
    """
    Reduces the matrix eliminating empty or full rows or columns.
    It repeats the process on the so reduced matrix until no more reductions are possible. 
    For instance, a perfectly nested matrix will be reduced until all entries are set to 0 or 1.
    """
    avg_mat = np.zeros_like(biad_mat, dtype=float)
    rows_num, cols_num = biad_mat.shape
    rows_degs = biad_mat.sum(1)
    cols_degs = biad_mat.sum(0)
    good_rows = np.arange(rows_num)
    good_cols = np.arange(cols_num)
    zero_rows = np.where(rows_degs == 0)[0]
    zero_cols = np.where(cols_degs == 0)[0]
    full_rows = np.where(rows_degs == cols_num)[0]
    full_cols = np.where(cols_degs == rows_num)[0]
    while zero_rows.size + zero_cols.size + full_rows.size + full_cols.size > 0:
        biad_mat = biad_mat[np.delete(np.arange(biad_mat.shape[0]), zero_rows), :]
        biad_mat = biad_mat[:, np.delete(np.arange(biad_mat.shape[1]), zero_cols)]
        good_rows = np.delete(good_rows, zero_rows)
        good_cols = np.delete(good_cols, zero_cols)
        full_rows = np.where(biad_mat.sum(1) == biad_mat.shape[1])[0]
        full_cols = np.where(biad_mat.sum(0) == biad_mat.shape[0])[0]
        avg_mat[good_rows[full_rows][:, None], good_cols] = 1
        avg_mat[good_rows[:, None], good_cols[full_cols]] = 1
        good_rows = np.delete(good_rows, full_rows)
        good_cols = np.delete(good_cols, full_cols)
        biad_mat = biad_mat[np.delete(np.arange(biad_mat.shape[0]), full_rows), :]
        biad_mat = biad_mat[:, np.delete(np.arange(biad_mat.shape[1]), full_cols)]
        zero_rows = np.where(biad_mat.sum(1) == 0)[0]
        zero_cols = np.where(biad_mat.sum(0) == 0)[0]
    return biad_mat, avg_mat, good_rows, good_cols


def initialize_fitnesses(rows_degs, cols_degs):
    """
    Reduces the matrix eliminating empty or full rows or columns.
    It does not repeat the process several times but it gives a warning in case it should be repeated. 
    For instance, a perfectly triangular matrix will be reduced until all entries are set to 0 or 1.
    """
    rows_num = len(rows_degs)
    cols_num = len(cols_degs)
    x = np.zeros(rows_num, dtype=float)
    y = np.zeros(cols_num, dtype=float)
    good_rows = np.arange(rows_num)
    good_cols = np.arange(cols_num)
    if np.any(np.isin(rows_degs, (0, cols_num))) or np.any(np.isin(cols_degs, (0, rows_num))):
        print('''
              WARNING: this system has at least a row or column that is empty or full. This may cause some convergence issues.
              Please use bicm_calculator with the biadjacency matrix, or clean your data from these nodes. 
              ''')
        zero_rows = np.where(rows_degs == 0)[0]
        zero_cols = np.where(cols_degs == 0)[0]
        full_rows = np.where(rows_degs == cols_num)[0]
        full_cols = np.where(cols_degs == rows_num)[0]
        x[full_rows] = np.inf
        y[full_cols] = np.inf
        bad_rows = np.concatenate((zero_rows, full_rows))
        bad_cols = np.concatenate((zero_cols, full_cols))
        good_rows = np.delete(np.arange(rows_num), bad_rows)
        good_cols = np.delete(np.arange(cols_num), bad_cols)
    
    return x, y, good_rows, good_cols


def bicm_calculator(full_biad_mat, method='iterative', initial_guess=None, tolerance=10e-8, print_counter=False):
    """
    Computes the bicm given a binary biadjacency matrix.
    Returns the average biadjacency matrix with the probabilities of connection.
    If the biadjacency matrix has empty or full rows or columns, the corresponding entries are automatically set to 0 or 1. 
    """
    
    biad_mat, avg_mat, good_rows, good_cols = initialize_avg_mat(full_biad_mat)
    
    if len(biad_mat) > 0:  # Every time the matrix is not perfectly nested
        rows_degs = biad_mat.sum(1)
        cols_degs = biad_mat.sum(0)
        bicm_obj = bicm(np.concatenate((rows_degs, cols_degs)), len(rows_degs), len(cols_degs))
        if method == 'root':
            bicm_obj.solve_root(initial_guess=initial_guess)
        elif method == 'ls':
            bicm_obj.solve_least_squares(initial_guess=initial_guess)
        elif method == 'iterative':
            bicm_obj.solve_iterative(initial_guess=initial_guess, tolerance=tolerance, print_counter=print_counter)
        elif method == 'newton':
            bicm_obj.solve_iterative(initial_guess=initial_guess, tolerance=tolerance, newton=True, print_counter=print_counter)
        r_avg_mat = bicm_from_fitnesses(bicm_obj.x, bicm_obj.y)
        avg_mat[good_rows[:, None], good_cols] = np.copy(r_avg_mat)
    
    check_sol(full_biad_mat, avg_mat)
    return avg_mat


def bicm_light(rows_degs, cols_degs, initial_guess=None, method='iterative', tolerance=10e-8, print_counter=False):
    """
    This function computes the bicm without using matrices, processing only the rows and columns degrees
    and returning only the fitnesses instead of the average matrix.
    """
    x, y, good_rows, good_cols = initialize_fitnesses(rows_degs, cols_degs)
    rows_degs = rows_degs[good_rows]
    cols_degs = cols_degs[good_cols]
    
    bicm_obj = bicm(np.concatenate((rows_degs, cols_degs)), len(rows_degs), len(cols_degs))
    if method == 'root':
        bicm_obj.solve_root(initial_guess=initial_guess)
    elif method == 'ls':
        bicm_obj.solve_least_squares(initial_guess=initial_guess, tolerance=tolerance)
    elif method == 'iterative':
        bicm_obj.solve_iterative(initial_guess=initial_guess, tolerance=tolerance, print_counter=print_counter)
    elif method == 'newton':
        bicm_obj.solve_iterative(newton=True, initial_guess=initial_guess, tolerance=tolerance, print_counter=print_counter)
    x[good_rows] = bicm_obj.x
    y[good_cols] = bicm_obj.y
    check_sol_light(x, y, rows_degs, cols_degs)
    return x, y


def projection_calculator(biad_mat, avg_mat, alpha=0.05, rows=True, sparse_mode=True, method='poisson', threads_num=4, return_pvals=False):
    """
    Calculates the projection on the rows layer (columns layers if rows is set to False).
    Returns an edge list of the indices of the vertices that share a link in the projection.
    
    alpha is the parameter of the FDR validation.
    method can be set to 'poibin', 'poisson', 'normal' and 'rna' according to the desired poisson binomial approximation to use.
    threads_num is the number of threads to launch when calculating the p-values.
    """
    if not rows:
        biad_mat = biad_mat.T
        avg_mat = avg_mat.T
    rows_num = biad_mat.shape[0]
    if sparse_mode:
        v_mat = (sparse.csr_matrix(biad_mat) * sparse.csr_matrix(biad_mat.T)).toarray()
    else:
        v_mat = biad_mat @ biad_mat.T
    np.fill_diagonal(v_mat, 0)
    pval_obj = pval_class()
    pval_obj.set_avg_mat(avg_mat)
    pval_obj.compute_pvals(v_mat, method=method, threads_num=threads_num)
    pval_list = np.array(pval_obj.pval_list, dtype=np.dtype([('source', int), ('target', int), ('pval', float)]))
    if return_pvals:
        return [(v[0], v[1], v[2]) for v in pval_list]
    eff_fdr_th = pvals_validator(pval_list['pval'], rows_num, alpha=alpha)
    return np.array([(v[0], v[1]) for v in pval_list if v[2] <= eff_fdr_th])


def indexes_edgelist(edgelist):
    """
    Creates a new edgelist with the indexes of the nodes instead of the names.
    Returns also two dictionaries that keep track of the nodes.
    """
    edgelist = np.array(edgelist, dtype=np.dtype([('source', np.int64), ('target', np.int64)]))
    unique_rows, rows_degs = np.unique(edgelist['source'], return_counts=True)
    unique_cols, cols_degs = np.unique(edgelist['target'], return_counts=True)
    rows_dict = dict(enumerate(unique_rows))
    cols_dict = dict(enumerate(unique_cols))
    inv_rows_dict = {v: k for k, v in rows_dict.items()}
    inv_cols_dict = {v: k for k, v in cols_dict.items()}
    edgelist_new = [(inv_rows_dict[edge[0]], inv_cols_dict[edge[1]]) for edge in edgelist]
    edgelist_new = np.array(edgelist_new, dtype=np.dtype([('rows', int), ('columns', int)]))
    return edgelist_new, rows_degs, cols_degs, rows_dict, cols_dict


@jit(nopython=True)
def vmotifs_from_edgelist(edgelist, rows_num, rows_deg):
    """
    From the edgelist returns an edgelist of the rows, weighted by the couples' v-motifs number.
    """
    edgelist = edgelist[np.argsort(edgelist['rows'])]
    cols_edli = edgelist['columns']
    v_list = []
    for i in range(rows_num - 1):
        start_i = rows_deg[:i].sum()
        i_neighbors = cols_edli[start_i: start_i + rows_deg[i]]
        start_j = start_i
        for j in range(i + 1, rows_num):
            start_j += rows_deg[j - 1]
            j_neighbors = cols_edli[start_j: start_j + rows_deg[j]]
            v_ij = len(set(i_neighbors) & set(j_neighbors))
            if v_ij > 0:
                v_list.append((i, j, v_ij))
    return v_list


def pvals_validator(pvals, rows_num, alpha=0.05):
    sorted_pvals = np.sort(pvals)
    multiplier = 2 * alpha / (rows_num * (rows_num - 1))
    try:
        eff_fdr_pos = np.where(sorted_pvals <= (np.arange(1, len(sorted_pvals) + 1) * multiplier))[0][-1]
    except:
        print('No V-motifs will be validated. Try increasing alpha')
        eff_fdr_pos = 0
    eff_fdr_th = eff_fdr_pos * multiplier
    return eff_fdr_th


def projection_calculator_light(edgelist, x, y, alpha=0.05, rows=True, method='poisson', threads_num=4, return_pvals=False):
    """
    Calculate the projection given only the edge list of the network, the fitnesses of the rows layer and the fitnesses of the columns layer.
    By default, the projection is calculated using a Poisson approximation. Other implemented choices are 'poibin' for the original Poisson-binomial
    distribution, 'normal' for the normal approximation and 'rna' for the refined normal approximation.
    """
    if not rows:
        edgelist = [(edge[1], edge[0]) for edge in edgelist]
    edgelist, order = np.unique(edgelist, axis=0, return_index=True)
    edgelist = edgelist[np.argsort(order)]  # np.unique does not preserve the order
    edgelist, rows_degs, cols_degs, rows_dict, cols_dict = indexes_edgelist(edgelist)
    rows_num = len(rows_degs)
    cols_num = len(cols_degs)
    v_list = vmotifs_from_edgelist(edgelist, rows_num, rows_degs)
    pval_obj = pval_class()
    pval_obj.set_fitnesses(x, y)
    pval_obj.compute_pvals(v_list, method=method, threads_num=threads_num)
    pval_list = np.array(pval_obj.pval_list, dtype=np.dtype([('source', int), ('target', int), ('pval', float)]))
    if return_pvals:
        return [(rows_dict[pval[0]], rows_dict[pval[1]], pval[2]) for pval in pval_list]
    eff_fdr_th = pvals_validator(pval_list['pval'], rows_num, alpha=alpha)
    return np.array([(rows_dict[v[0]], rows_dict[v[1]]) for v in pval_list if v[2] <= eff_fdr_th])
