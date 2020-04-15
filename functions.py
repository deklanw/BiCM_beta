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

def pval_calculator_poibin(v_mat, avg_mat, r_invert_rows_degs, degrees_couple, method='poibin', r_invert_rows_degs_in=[]): #v_mat should have a zero diagonal
    # If v_mat is passed as a couple, it means that the two matrices of the directed network have been given
    # Another possibility would be to pass only the matrix that has to be used, if it is one (undirected, many-columns network)
    # or if it is the difference of the two matrices
    directed = False
    if len(avg_mat) == 2:
#         out_mat = v_mat[0]
#         in_mat = v_mat[1]
        directed = True
        avg_mat_out = avg_mat[0]
        avg_mat_in = avg_mat[1]
    i, j = degrees_couple
    i_nodes = np.where(r_invert_rows_degs == i)[0]
    if len(r_invert_rows_degs_in) == 0:
        j_nodes = np.where(r_invert_rows_degs == j)[0]
    else:
        j_nodes = np.where(r_invert_rows_degs_in == j)[0]
    p_val = np.ones((len(i_nodes), len(j_nodes)))
    if len(set(i_nodes) | set(j_nodes)) > 1: # Check that there is not only one node with such degree
        if directed:
#             v_mat_couple = out_mat[i_nodes] @ (in_mat[j_nodes].T) sbagliato
            v_mat_couple = v_mat[i_nodes[:, np.newaxis], j_nodes]
            tot_v_couple = v_mat_couple.sum()
        else:
            v_mat_couple = v_mat[i_nodes[:, np.newaxis], j_nodes]
            tot_v_couple = v_mat_couple.sum()
        if tot_v_couple > 0: #It means that there is at least one v-motifs between the two groups
            if directed:
                probs = avg_mat_out[i_nodes[0]] * avg_mat_in[j_nodes[0]]
            else:
                probs = avg_mat[i_nodes[0]] * avg_mat[j_nodes[0]]
            if method == 'poibin':
                pb_obj = pb.PoiBin(probs)# Check indexes
#                 print('I\'m here')
                for ii in range(len(i_nodes)):
                    for jj in range(len(j_nodes)):
                        v_couple = v_mat_couple[ii, jj]
                        if v_couple > 0:
                            p_val[ii, jj] = pb_obj.pval(int(v_couple))
            elif method == 'poisson':
                avg_v_couple = np.sum(probs)
                for ii in range(len(i_nodes)):
                    for jj in range(len(j_nodes)):
                        if i_nodes[ii] == j_nodes[jj]:
                            p_val[ii, jj] = 0
                            continue
                        v_couple = v_mat_couple[ii, jj]
                        if v_couple > 0:
                            p_val[ii, jj] = poisson.sf(k=v_couple - 1, mu=avg_v_couple)
#                             p_val[ii, jj] = tb.p_val_poisson(v_couple, avg_v_couple)
            elif method == 'normal':
                avg_v_couple = np.sum(probs)
                for ii in range(len(i_nodes)):
                    for jj in range(len(j_nodes)):
                        v_couple = v_mat_couple[ii, jj]
                        if v_couple > 0:
                            sigma_couple = np.sqrt(np.sum(probs * (1 - probs)))
                            p_val[ii, jj] = norm.cdf((v_couple + 0.5 - avg_v_couple) / sigma_couple)
            elif method == 'rna':
                avg_v_couple = np.sum(probs)
                for ii in range(len(i_nodes)):
                    for jj in range(len(j_nodes)):
                        v_couple = v_mat_couple[ii, jj]
                        if v_couple > 0:
                            var_arr = probs * (1 - probs)
                            sigma_couple = np.sqrt(np.sum(var_arr))
                            gamma_couple = (sigma_couple ** (-3)) * np.sum(var_arr * (1 - 2 * probs))
                            eval_x = (v_couple + 0.5 - avg_v_couple) / sigma_couple
                            p_val_temp = norm.cdf(eval_x) + gamma_couple * (1 - eval_x ** 2) * norm.pdf(eval_x) / 6
                            if p_val_temp < 0:
                                p_val[ii, jj] = 0
                            elif p_val_temp > 1:
                                p_val[ii, jj] = 1
                            else:
                                p_val[ii, jj] = p_val_temp 
    return p_val

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
        return(error)
    else:
        return

@jit(nopython=True)
def vec2mat(x, y):
    """
    Given two vectors x_i, y_j returns the matrix of products x_i * y_j
    """
    return np.atleast_2d(x).T @ np.atleast_2d(y)

# Come fare a indicare righe e colonne automaticamente imposte a 0 o 1? (perfectly nested)
#  Ho messo un semplice warning. Il metodo che lavora sulle matrici è già ok, ma il ridotto?
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
    zero_cols = np.where(cols_degs ==  0)[0]
    full_rows = np.where(rows_degs ==  cols_num)[0]
    full_cols = np.where(cols_degs ==  rows_num)[0]
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
        zero_cols = np.where(biad_mat.sum(0) ==  0)[0]
    
    return biad_mat, avg_mat, good_rows, good_cols

def initialize_fitnesses(rows_degs, cols_degs):
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
        zero_cols = np.where(cols_degs ==  0)[0]
        full_rows = np.where(rows_degs ==  cols_num)[0]
        full_cols = np.where(cols_degs ==  rows_num)[0]
        x[full_rows] = np.inf
        y[full_cols] = np.inf
        bad_rows = np.concatenate((zero_rows, full_rows))
        bad_cols = np.concatenate((zero_cols, full_cols))
        good_rows = np.delete(np.arange(rows_num), bad_rows)
        good_cols = np.delete(np.arange(cols_num), bad_cols)
    
    return x, y, good_rows, good_cols

def bicm_calculator(biad_mat, method='newton', initial_guess=None, tolerance=10e-8, print_counter=False):
    """
    Computes the bicm given a binary biadjacency matrix.
    Returns the average biadjacency matrix with the probabilities of connection.
    If the biadjacency matrix has empty or full rows or columns, the corresponding entries are automatically set to 0 or 1. 
    """
    
    biad_mat, avg_mat, good_rows, good_cols = initialize_avg_mat(biad_mat)
    
    if len(biad_mat) > 0: # Every time the matrix is not perfectly nested
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
    
#     check_sol(biad_mat, avg_mat)
    return avg_mat

def bicm_light(rows_degs, cols_degs, initial_guess=None, method='iterative', tolerance=10e-8):
    """
    This function computes the bicm without using matrices, processing only the rows and columns degrees
    and returning only the fitnesses instead of the average matrix.
    """
    x, y, good_rows, good_cols = initialize_fitnesses(rows_degs, cols_degs)
    rows_degs = rows_degs[good_rows]
    cols_degs = cols_degs[good_cols]
    
    bicm_obj = bicm(np.concatenate((rows_degs, cols_degs)), len(rows_degs), len(cols_degs))
    if method == 'root':
        bicm_obj.solve_root(initial_guess=initial_guess, tolerance=tolerance)
    elif method == 'ls':
        bicm_obj.solve_least_squares(initial_guess=initial_guess, tolerance=tolerance)
    elif method == 'iterative':
        bicm_obj.solve_iterative(initial_guess=initial_guess, tolerance=tolerance)
    elif method == 'newton':
        bicm_obj.solve_iterative(newton=True, initial_guess=initial_guess, tolerance=tolerance)
    x[good_rows] = bicm_obj.x
    y[good_cols] = bicm_obj.y
    return x, y

def pval_calculator_old(v_mat, avg_mat, r_invert_rows_degs, degrees_couple, method='poisson', r_invert_rows_degs_in=[]):
    # v_mat should have a zero diagonal
    # If v_mat and avg_mat are passed as a couple, it means that the two matrices of the directed network have been given
    # Returns a matrix of p-values
    directed = False
    if len(avg_mat) == 2:
        directed = True
        avg_mat_out = avg_mat[0]
        avg_mat_in = avg_mat[1]
    i, j = degrees_couple
    i_nodes = np.where(r_invert_rows_degs == i)[0]
    if len(r_invert_rows_degs_in) == 0:
        j_nodes = np.where(r_invert_rows_degs == j)[0]
    else:
        j_nodes = np.where(r_invert_rows_degs_in == j)[0]
    p_val = np.ones((len(i_nodes), len(j_nodes)))
    if len(set(i_nodes) | set(j_nodes)) > 1: # Check that there is not only one node with such degree
        if directed:
            v_mat_couple = v_mat[i_nodes[:, np.newaxis], j_nodes]
            tot_v_couple = v_mat_couple.sum()
        else:
            v_mat_couple = v_mat[i_nodes[:, np.newaxis], j_nodes]
            tot_v_couple = v_mat_couple.sum()
        if tot_v_couple > 0: # It means that there is at least one v-motifs between the two groups
            if directed:
                probs = avg_mat_out[i_nodes[0]] * avg_mat_in[j_nodes[0]]
            else:
                probs = avg_mat[i_nodes[0]] * avg_mat[j_nodes[0]]
            if method == 'poibin':
                pb_obj = pb.PoiBin(probs)
                for ii in range(len(i_nodes)):
                    for jj in range(len(j_nodes)):
                        if i_nodes[ii] == j_nodes[jj]:
                            continue
                        v_couple = v_mat_couple[ii, jj]
                        if v_couple > 0:
                            p_val[ii, jj] = pb_obj.pval(int(v_couple))
            elif method == 'poisson':
                avg_v_couple = np.sum(probs)
                for ii in range(len(i_nodes)):
                    for jj in range(len(j_nodes)):
                        if i_nodes[ii] == j_nodes[jj]:
                            continue
                        v_couple = v_mat_couple[ii, jj]
                        if v_couple > 0:
                            p_val[ii, jj] = poisson.sf(k=v_couple - 1, mu=avg_v_couple)
            elif method == 'normal':
                avg_v_couple = np.sum(probs)
                for ii in range(len(i_nodes)):
                    for jj in range(len(j_nodes)):
                        if i_nodes[ii] == j_nodes[jj]:
                            continue
                        v_couple = v_mat_couple[ii, jj]
                        if v_couple > 0:
                            sigma_couple = np.sqrt(np.sum(probs * (1 - probs)))
                            p_val[ii, jj] = norm.cdf((v_couple + 0.5 - avg_v_couple) / sigma_couple)
            elif method == 'rna':
                avg_v_couple = np.sum(probs)
                for ii in range(len(i_nodes)):
                    for jj in range(len(j_nodes)):
                        if i_nodes[ii] == j_nodes[jj]:
                            p_val[ii, jj] = 0
                            continue
                        v_couple = v_mat_couple[ii, jj]
                        if v_couple > 0:
                            var_arr = probs * (1 - probs)
                            sigma_couple = np.sqrt(np.sum(var_arr))
                            gamma_couple = (sigma_couple ** (-3)) * np.sum(var_arr * (1 - 2 * probs))
                            eval_x = (v_couple + 0.5 - avg_v_couple) / sigma_couple
                            p_val_temp = norm.cdf(eval_x) + gamma_couple * (1 - eval_x ** 2) * norm.pdf(eval_x) / 6
                            if p_val_temp < 0:
                                p_val[ii, jj] = 0
                            elif p_val_temp > 1:
                                p_val[ii, jj] = 1
                            else:
                                p_val[ii, jj] = p_val_temp 
    return p_val

def p_val_reconstruction(pval_arrays_list, r_invert_rows_degs, degrees_couples, r_invert_rows_degs_in=[]):
    p_val_mat = np.ones((len(r_invert_rows_degs), len(r_invert_rows_degs)))
    array_index = 0
    for (i, j) in degrees_couples:
        i_nodes = np.where(r_invert_rows_degs == i)[0]
        if len(r_invert_rows_degs_in) == 0:
            j_nodes = np.where(r_invert_rows_degs == j)[0]
            p_val_mat[i_nodes[:, np.newaxis], j_nodes] = pval_arrays_list[array_index]
            p_val_mat[j_nodes[:, np.newaxis], i_nodes] = pval_arrays_list[array_index].T
        else:
            j_nodes = np.where(r_invert_rows_degs_in == j)[0]
            p_val_mat[i_nodes[:, np.newaxis], j_nodes] = pval_arrays_list[array_index]
        array_index += 1
    return p_val_mat

def projection_calculator(biad_mat, avg_mat, alpha=0.05, rows=True, method='poisson', threads_num=4, return_pvals=False):
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
    
    v_mat = (sparse.csr_matrix(biad_mat) * sparse.csr_matrix(biad_mat.T)).toarray()
    np.fill_diagonal(v_mat, 0)
    
    pval_obj = pval_class()
    pval_obj.set_avg_mat(avg_mat)
    pval_obj.compute_pvals(v_mat, method=method, threads_num=threads_num)
    pval_list = np.array(pval_obj.pval_list, dtype=np.dtype([('source', int), ('target', int), ('pval', float)]))
#     pval_list = pval_obj.pval_list
    if return_pvals:
        return [(v[0], v[1], v[2]) for v in pval_list]
#         return pval_list
    eff_fdr_th = pvals_validator(pval_list['pval'], rows_num, alpha=alpha)
#     eff_fdr_th = pvals_validator(pval_list, rows_num, alpha=alpha)
    return np.array([(v[0], v[1]) for v in pval_list if v[2] <= eff_fdr_th])
#     r_rows_degs, r_invert_rows_degs = np.unique(biad_mat.sum(1), 0, 1)
    
# #     start = time.time()
# #     print(time.strftime("%H:%M:%S"))
#     func = partial(pval_calculator_old, v_mat, avg_mat, r_invert_rows_degs, method=method)
    
#     couples = []
#     for i in range(len(r_rows_degs)):
#         for j in range(i, len(r_rows_degs)):
#             couples.append((i, j))
    
#     with Pool(processes=threads_num) as pool:
#                 p_vals_couples = np.array(pool.map(func, tqdm_notebook(couples)))
    
#     p_vals = p_val_reconstruction(p_vals_couples, r_invert_rows_degs, couples)
    
# #     print(time.strftime("%H:%M:%S"))
# #     finish = time.time()
    
#     p_vals_array = np.zeros(rows_num ** 2)
#     index = 0
#     for p_vals_couple in p_vals_couples:
#         p_vals_array[index: index + np.prod(p_vals_couple.shape)] = p_vals_couple.flatten()
#         index += np.prod(p_vals_couple.shape)
    
#     p_vals_array = p_vals_array[p_vals_array > 0]
    
#     ord_p_vals = np.argsort(p_vals_array)
#     sorted_p_vals = p_vals_array[ord_p_vals]
#     multiplier = 2 * alpha / (rows_num * (rows_num - 1))
#     try:
#         eff_fdr_pos = np.where(sorted_p_vals <= (np.arange(1, len(sorted_p_vals) + 1) * multiplier))[0][-1]
#     except:
#         print('No V-motifs will be validated. Try increasing alpha')
#         eff_fdr_pos = 0
#     eff_fdr_th = eff_fdr_pos * multiplier
#     np.fill_diagonal(p_vals, 1)
    
#     return np.array(list(zip(np.where(p_vals <= eff_fdr_th)[0], np.where(p_vals <= eff_fdr_th)[1])))

# def indexes_edgelist(edgelist):
#     """
#     Creates a new edgelist with the indexes of the nodes instead of the names.
#     Returns also two dictionaries that keep track of the nodes.
#     """
#     rows_dict = {}
#     cols_dict = {}
#     inv_rows_dict = {}
#     inv_cols_dict = {}
#     rows_degs = []
#     cols_degs = []
#     rows_i = 0
#     cols_i = 0
#     edgelist_new = []
#     for edge in edgelist:
#         if edge[0] not in rows_dict.values():
#             rows_dict.update({rows_i : edge[0]})
#             inv_rows_dict.update({edge[0] : rows_i})
#             rows_degs.append(0)
#             rows_degs[rows_i] += 1
#             rows_i += 1
#         else:
#             rows_degs[inv_rows_dict[edge[0]]] += 1
#         if edge[1] not in cols_dict.values():
#             cols_dict.update({cols_i : edge[1]})
#             inv_cols_dict.update({edge[1] : cols_i})
#             cols_degs.append(0)
#             cols_degs[cols_i] += 1
#             cols_i += 1
#         else:
#             cols_degs[inv_cols_dict[edge[1]]] += 1
#         edgelist_new.append((inv_rows_dict[edge[0]], inv_cols_dict[edge[1]]))
#     edgelist_new = np.array(edgelist_new, dtype=np.dtype([('rows', int), ('columns', int)]))
#     return edgelist_new, np.array(rows_degs), np.array(cols_degs), rows_dict, cols_dict

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

# def vmotifs_from_edgelist(edgelist, rows_num, cols_num):
#     """
#     From the edgelist returns an edgelist of the rows, weighted by the couples' v-motifs number.
#     """
#     v_list = []
#     for rows_i in range(rows_num - 1):
#         i_neighbors = edgelist['columns'][np.where(edgelist['rows'] == rows_i)]
#         for rows_j in range(rows_i + 1, rows_num):
#             j_neighbors = edgelist['columns'][np.where(edgelist['rows'] == rows_j)]
#             v_ij = len(set(i_neighbors) & set(j_neighbors))
#             if v_ij > 0:
#                 v_list.append((rows_i, rows_j, v_ij))
#     return v_list

@jit(nopython=True)
def vmotifs_from_edgelist(edgelist, rows_num, cols_num):
    """
    From the edgelist returns an edgelist of the rows, weighted by the couples' v-motifs number.
    """
    edgelist = edgelist[np.argsort(edgelist['rows'])]
    rows_edli = edgelist['rows']
    cols_edli = edgelist['columns']
    v_list = []
    i = 0
    scanned_rows = 0
    while scanned_rows < rows_num - 1:
        start_i = i
        rows_i = rows_edli[start_i]
        while rows_edli[i] == rows_i:
            i += 1
        i_neighbors = cols_edli[start_i: i]
        start_j = i
        rows_j = rows_edli[start_j]
        for j in range(i, len(rows_edli)):
            if rows_edli[j] != rows_j or j == len(rows_edli) - 1:
                j_neighbors = cols_edli[start_j: j]
                if j == len(rows_edli) - 1:
                    j_neighbors = cols_edli[start_j:]
                v_ij = len(set(i_neighbors) & set(j_neighbors))
                if v_ij > 0:
                    v_list.append((rows_i, rows_j, v_ij))
                start_j = j
                rows_j = rows_edli[j]
        scanned_rows += 1
    return v_list

@jit(nopython=True)
def vmotif_iterator(cols_edli, rows_deg, rows_num, i):
    start_i = rows_deg[:i].sum()
    i_neighbors = cols_edli[start_i: start_i + rows_deg[i]]
    start_j = start_i
    vi_list = []
    for j in range(i + 1, rows_num):
        start_j += rows_deg[j - 1]
        j_neighbors = cols_edli[start_j: start_j + rows_deg[j]]
        v_ij = len(set(i_neighbors) & set(j_neighbors))
        if v_ij > 0:
            vi_list.append((i, j, v_ij))
    return vi_list

@jit(nopython=True)
def vmotifs_from_edgelist2(edgelist, rows_num, cols_num, rows_deg, cols_deg):
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

def myravel(mylist):
    return [i for sublist in mylist for i in sublist]

# @jit(nopython=True)
def vmotifs_from_edgelist3(edgelist, rows_num, cols_num, rows_deg, cols_deg):
    """
    From the edgelist returns an edgelist of the rows, weighted by the couples' v-motifs number.
    """
    edgelist = edgelist[np.argsort(edgelist['rows'])]
    cols_edli = edgelist['columns']
    func = partial(vmotif_iterator, cols_edli, rows_deg, rows_num)
    v_list = parmap(func, np.arange(rows_num - 1))
    v_list = myravel(v_list)
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
    if not rows:
        edgelist = [(edge[1], edge[0]) for edge in edgelist]
#     nodename_type = type(edgelist[0][0])
    edgelist, order = np.unique(edgelist, axis=0, return_index=True)
    edgelist = edgelist[np.argsort(order)] # np.unique does not preserve the order
    edgelist, rows_degs, cols_degs, rows_dict, cols_dict = indexes_edgelist(edgelist)
    rows_num = len(rows_degs)
    cols_num = len(cols_degs)
    v_list = vmotifs_from_edgelist(edgelist, rows_num, cols_num)
    pval_obj = pval_class()
    pval_obj.set_fitnesses(x, y)
    pval_obj.compute_pvals(v_list, method=method, threads_num=threads_num)
    pval_list = np.array(pval_obj.pval_list, dtype=np.dtype([('source', int), ('target', int), ('pval', float)]))
    if return_pvals:
        return [(rows_dict[pval[0]], rows_dict[pval[1]], pval[2]) for pval in pval_list]
    eff_fdr_th = pvals_validator(pval_list['pval'], rows_num, alpha=alpha)
    return np.array([(rows_dict[v[0]], rows_dict[v[1]]) for v in pval_list if v[2] <= eff_fdr_th])
