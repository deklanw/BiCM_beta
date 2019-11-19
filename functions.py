import poibin as pb
import numpy as np
from scipy.stats import poisson, norm
from scipy import sparse
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm_notebook
from BiCM_class import BiCM_class as bicm
from numba import jit


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

def bicm_calculator(biad_mat, method='root', initial_guess=None):
    """
    Computes the bicm given a binary biadjacency matrix.
    Returns the average biadjacency matrix with the probabilities of connection.
    If the biadjacency matrix has empty or full rows or columns, the corresponding entries are automatically set to 0 or 1. 
    """
    
    avg_mat = np.zeros_like(biad_mat, dtype=float)
    
    # This next block reduces the matrix eliminating all zeros and all one columns.
    # Once it has done it, it repeats the process on the reduced matrix, if needed
    # A perfectly nested matrix will thus be reduced to the point that all entries are set automatically to 0 or 1
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
    
    if len(biad_mat) > 0: # Every time the matrix is not perfectly nested
        rows_degs = biad_mat.sum(1)
        cols_degs = biad_mat.sum(0)
        if method == 'iterative':
            
        else:
            bicm_obj = bicm(np.concatenate((rows_degs, cols_degs)), len(rows_degs), len(cols_degs))
            if method == 'root':
                bicm_obj.solve_root(initial_guess=initial_guess)
            elif method == 'ls':
                bicm_obj.solve_least_squares(initial_guess=initial_guess)
    #         xy = np.concatenate((bicm_obj.x, bicm_obj.y))
            r_avg_mat = bicm_from_fitnesses(bicm_obj.x, bicm_obj.y)
        avg_mat[good_rows[:, None], good_cols] = np.copy(r_avg_mat)
    
    check_sol(biad_mat, avg_mat)
    
    return avg_mat

def bicm_light(rows_degs, cols_degs, initial_guess='jerry', method='root'):
    """
    This function computes the bicm without using matrices, processing only the rows and columns degrees
    and returning only the fitnesses instead of the average matrix.
    """
    rows_num = len(cols_degs)
    cols_num = len(rows_degs)
    x = np.zeros(rows_num, dtype=float)
    y = np.zeros(cols_num, dtype=float)
    
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
    
    rows_degs = rows_degs[good_rows]
    cols_degs = cols_degs[good_cols]
    bicm_obj = bicm(np.concatenate((rows_degs, cols_degs)), len(rows_degs), len(cols_degs))
    if method == 'root':
        bicm_obj.solve_root(initial_guess=initial_guess)
    elif method == 'ls':
        bicm_obj.solve_least_squares(initial_guess=initial_guess)
    x[good_rows] = bicm_obj.x
    y[good_rows] = bicm_obj.y
    return x, y

def pval_calculator(v_mat, avg_mat, r_invert_rows_degs, degrees_couple, method='poisson', r_invert_rows_degs_in=[]):
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

def projection_calculator(biad_mat, avg_mat, alpha=0.05, rows=True, method='poisson', threads_num=4):
    """
    Calculates the projection on the rows layer (columns layers if rows is set to False).
    Returns an edge list of the indices of the vertices that share a link in the projection.
    
    alpha is the parameter of the FDR validation.
    method can be set to 'poibin', 'poisson', 'normal' and 'rna' according to the desire poisson binomial approximation to use.
    threads_num is the number of threads to launch when calculating the p-values.
    """
    
    if not rows:
        biad_mat = biad_mat.T
        avg_mat = avg_mat.T
    
    expected_V = np.dot(avg_mat, avg_mat.T)
    
    v_mat = (sparse.csr_matrix(biad_mat) * sparse.csr_matrix(biad_mat.T)).toarray()
    np.fill_diagonal(v_mat, 0)
    
    r_rows_degs, r_invert_rows_degs = np.unique(biad_mat.sum(1), 0, 1)
    
#     start = time.time()
#     print(time.strftime("%H:%M:%S"))
    func = partial(pval_calculator, v_mat, avg_mat, r_invert_rows_degs, method=method)
    
    couples = []
    for i in range(len(r_rows_degs)):
        for j in range(i, len(r_rows_degs)):
            couples.append((i, j))
    
    with Pool(processes=threads_num) as pool:
                p_vals_couples = np.array(pool.map(func, tqdm_notebook(couples)))
    
    p_vals = p_val_reconstruction(p_vals_couples, r_invert_rows_degs, couples)
    
#     print(time.strftime("%H:%M:%S"))
#     finish = time.time()
    
    p_vals_array = np.zeros(biad_mat.shape[0] ** 2)
    index = 0
    for p_vals_couple in p_vals_couples:
        p_vals_array[index: index + np.prod(p_vals_couple.shape)] = p_vals_couple.flatten()
        index += np.prod(p_vals_couple.shape)
    
    p_vals_array = p_vals_array[p_vals_array > 0]
    
    ord_p_vals = np.argsort(p_vals_array)
    sorted_p_vals = p_vals_array[ord_p_vals]
    
    try:
        eff_fdr_pos = np.where(sorted_p_vals <= np.arange(1, len(sorted_p_vals) + 1) / len(sorted_p_vals) * alpha)[0][-1]
    except:
        eff_fdr_pos = 0
    eff_fdr_th = eff_fdr_pos / len(sorted_p_vals) * alpha
    
    np.fill_diagonal(p_vals, 1)
    
    return np.array(list(zip(np.where(p_vals < eff_fdr_th)[0], np.where(p_vals < eff_fdr_th)[1])))