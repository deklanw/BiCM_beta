"""
Created on Wed Nov 6 10:27:44 2019

@authors: lePiddu, MrBrown
"""

import numpy as np
from numba import jit
from scipy import optimize as opt


@jit(nopython=True)
def vec2mat(x, y):
    return np.atleast_2d(x).T @ np.atleast_2d(y)


@jit(nopython=True)
def sum_combination(x, y):
    """
    This function computes a matrix in which each element is the exponential
    of the sum of the corresponding elements of x and y
    """
    return np.exp(np.atleast_2d(x).T + np.atleast_2d(y))


@jit(nopython=True)
def eqs(xx, d_rows, d_cols, multiplier_rows, multiplier_cols, nrows, out_res):
    xixa = vec2mat(xx[:nrows], xx[nrows:])
    xixa /= 1 + xixa
    out_res[:nrows] = (xixa * multiplier_cols).sum(axis=1) - d_rows
    out_res[nrows:] = (xixa.T * multiplier_rows).sum(axis=1) - d_cols


@jit(nopython=True)
def jac_root(xx, multiplier_rows, multiplier_cols, nrows, ncols, out_J_T):
    xixa_tilde = 1 / (1 + vec2mat(xx[:nrows], xx[nrows:]))
    xixa_tilde *= xixa_tilde
    lower_block_T = xixa_tilde * xx[nrows:]
    up_left_block = (lower_block_T * multiplier_cols).sum(axis=1) * \
                        np.eye(nrows)
    upper_block_T = xixa_tilde.T * xx[:nrows]
    lo_right_block = (upper_block_T * multiplier_rows).sum(axis=1) * \
                        np.eye(ncols)
    out_J_T[:nrows, :nrows] = up_left_block
    out_J_T[nrows:, nrows:] = lo_right_block
    # The blocks are inverted because 'col_deriv'=True takes the transpose of the Jacobian, which is faster
    out_J_T[nrows:, :nrows] = (upper_block_T.T * multiplier_cols).T
#     out_J_T[:nrows, nrows:] = (upper_block_T.T * multiplier_cols)
    out_J_T[:nrows, nrows:] = (lower_block_T.T * multiplier_rows).T
#     out_J_T[nrows:, :nrows] = (lower_block_T.T * multiplier_rows)


@jit(nopython=True)  # For Numba compatibility this is outside of the class
def solve_iterations(n_edges, r_dseq_rows, r_dseq_cols, rows_multiplicity, cols_multiplicity, initial_x, initial_y,
                     tolerance=1e-10, max_iter=5000, print_counter=True):
    '''
    Numerically solve the system of nonlinear equations
    we encounter when solving for the Lagrange multipliers
    n_edges: total (not reduced) number of edges.
    r_dseq_rows, r_dseq_cols: reduced rows and columns degree sequences.
    rows_multiplicity, cols_multiplicity: the multiplicities of every degree in the two sequences.
    @param tolerance: solver continues iterating until the
    difference between two consecutive solutions
    is less than tolerance.
    @param max_iter: maximum number of iterations.
    '''
    X = initial_x
    Y = initial_y
    x = r_dseq_rows * 0.
    y = r_dseq_cols * 0.
    change = 1
    counter = 0
    for counter in range(max_iter):
        for i in range(len(x)):
            x[i] = r_dseq_rows[i] / np.sum(Y * cols_multiplicity / (1. + X[i] * Y))
        for i in range(len(y)):
            y[i] = r_dseq_cols[i] / np.sum(X * rows_multiplicity / (1. + X * Y[i]))
        change = max(np.max(np.abs(X - x)), np.max(np.abs(Y - y)))
        X[:] = x
        Y[:] = y
        if change < tolerance: 
            break
    if change > tolerance:
        raise Exception("Solver did not converge. Try increasing max_iter")
    if print_counter:
        print("Iterations count:",  counter)
    return x, y


@jit(nopython=True)
def jac_newton(x, y, rows_deg, cols_deg, rows_multiplicity, cols_multiplicity):
    rows_num = len(rows_deg)
    cols_num = len(cols_deg)
    my_jac = np.empty((rows_num + cols_num, rows_num + cols_num))
    rows_der = - np.array([np.sum((y ** 2) * cols_multiplicity / ((1 + x[i] * y) ** 2)) - rows_deg[i] / (x[i] ** 2) for i in range(rows_num)])
    my_jac[:rows_num, :rows_num] = rows_der * np.eye(rows_num)
    cols_der = - np.array([np.sum((x ** 2) * rows_multiplicity / ((1 + y[i] * x) ** 2)) - cols_deg[i] / (y[i] ** 2) for i in range(cols_num)])
    my_jac[rows_num:, rows_num:] = cols_der * np.eye(cols_num)
    my_jac[:rows_num, rows_num:] = cols_multiplicity / ((1 + vec2mat(x, y)) ** 2)
    my_jac[rows_num:, :rows_num] = rows_multiplicity / ((1 + vec2mat(y, x)) ** 2)
    return my_jac


@jit(nopython=True)  # For Numba compatibility this is outside of the class
def solve_newton(n_edges, r_dseq_rows, r_dseq_cols, rows_multiplicity, cols_multiplicity, initial_x, initial_y,
                     tolerance=1e-10, max_iter=5000, print_counter=True, relax_multiplier=0.8, min_relax=0.001):
    '''
    Numerically solve the system of nonlinear equations
    we encounter when solving for the Lagrange multipliers
    n_edges: total (not reduced) number of edges.
    r_dseq_rows, r_dseq_cols: reduced rows and columns degree sequences.
    rows_multiplicity, cols_multiplicity: the multiplicities of every degree in the two sequences.
    @param tolerance: solver continues iterating until the
    difference between two consecutive solutions
    is less than tolerance.
    @param max_iter: maximum number of iterations.
    '''
    x = initial_x
    y = initial_y
    
    num_rows = len(r_dseq_rows)
    num_cols = len(r_dseq_cols)
    old_change = np.inf
    change = old_change
    relax = relax_multiplier
    f_x = - np.concatenate((np.array([r_dseq_rows[i] / x[i] - np.sum(y * cols_multiplicity / (1 + x[i] * y)) for i in range(num_rows)]), 
                            np.array([r_dseq_cols[j] / y[j] - np.sum(x * rows_multiplicity / (1 + x * y[j])) for j in range(num_cols)])))
    counter = 0
    for counter in range(max_iter):
        h = np.linalg.solve(jac_newton(x, y, r_dseq_rows, r_dseq_cols, rows_multiplicity, cols_multiplicity), f_x)
        x -= h[:num_rows]
        y -= h[num_rows:]
        f_x = - np.concatenate((np.array([r_dseq_rows[i] / x[i] - np.sum(y * cols_multiplicity / (1 + x[i] * y)) for i in range(num_rows)]), 
                                np.array([r_dseq_cols[j] / y[j] - np.sum(x * rows_multiplicity / (1 + x * y[j])) for j in range(num_cols)])))
        change = np.max(np.abs(f_x))
        if change < tolerance:
            break
        if change > old_change:
            x += h[:num_rows]
            y += h[num_rows:]
            while change > old_change and relax > min_relax:
                
                x2 = x - relax * h[:num_rows]
                y2 = y - relax * h[num_rows:]
                f_x = - np.concatenate((np.array([r_dseq_rows[i] / x[i] - np.sum(y * cols_multiplicity / (1 + x[i] * y)) for i in range(num_rows)]),
                                        np.array([r_dseq_cols[j] / y[j] - np.sum(x * rows_multiplicity / (1 + x * y[j])) for j in range(num_cols)])))
                change = np.max(np.abs(f_x))
                relax *= relax_multiplier
            relax = relax_multiplier
            if change > old_change:
                x -= h[:num_rows]
                y -= h[num_rows:]
                f_x = - np.concatenate((np.array([r_dseq_rows[i] / x[i] - np.sum(y * cols_multiplicity / (1 + x[i] * y)) for i in range(num_rows)]),
                                        np.array([r_dseq_cols[j] / y[j] - np.sum(x * rows_multiplicity / (1 + x * y[j])) for j in range(num_cols)])))
                old_change = np.max(np.abs(f_x))
            else:
                x = x2[:]
                y = y2[:]
                old_change = change
        else:
            old_change = change
    if change > tolerance:
        raise Exception("Solver did not converge. Try increasing max_iter")
    if print_counter:
        print("Iterations count:",  counter)
    return x, y


class BiCM_class:
    def __init__(self, degree_sequence, n_rows, n_cols):
        # Full problem parameters
        self.dseq_rows = np.copy(degree_sequence[:n_rows])
        self.dseq_cols = np.copy(degree_sequence[n_rows:])
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.dim = self.n_rows + self.n_cols
        self.n_edges = np.sum(self.dseq_rows)
        self.x = None
        self.y = None
        self.xy = None
        self.initial_guess = None
        # Reduced problem parameters
        self.is_reduced = False
        self.r_dseq_rows = None
        self.r_dseq_cols = None
        self.r_n_rows = None
        self.r_n_cols = None
        self.r_invert_dseq_rows = None
        self.r_invert_dseq_cols = None
        self.r_dim = None
        self.rows_multiplicity = None
        self.cols_multiplicity = None
        # Problem solutions
        self.J_T = None
        self.r_x = None
        self.r_y = None
        self.r_xy = None
        # Problem (reduced) residuals
        self.residuals = None
        self.final_result = None

    def degree_degeneration(self):
        self.r_dseq_rows, self.r_invert_dseq_rows, self.rows_multiplicity \
            = np.unique(self.dseq_rows, return_index=False, return_inverse=True, return_counts=True)
        self.r_dseq_cols, self.r_invert_dseq_cols, self.cols_multiplicity \
            = np.unique(self.dseq_cols, return_index=False, return_inverse=True, return_counts=True)
        self.r_n_rows = self.r_dseq_rows.size
        self.r_n_cols = self.r_dseq_cols.size
        self.r_dim = self.r_n_rows + self.r_n_cols
        self.is_reduced = True

    def _equations(self, x):
        eqs(x, self.r_dseq_rows, self.r_dseq_cols,
            self.rows_multiplicity, self.cols_multiplicity,
            self.r_n_rows, self.residuals)

    def _jacobian_root(self, x):
        jac_root(x, self.rows_multiplicity, self.cols_multiplicity,
            self.r_n_rows, self.r_n_cols, self.J_T)

    def _residuals_jacobian(self, x):
        self._equations(x)
        self._jacobian_root(x)
        return self.residuals, self.J_T
    
    def _set_initial_guess(self):
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code starts with a reduced number of rows and columns.
        if self.initial_guess is None:
            self.r_x = self.r_dseq_rows / (np.sqrt(self.n_edges) + 1)  # This +1 increases the stability of the solutions.
            self.r_y = self.r_dseq_cols / (np.sqrt(self.n_edges) + 1)
        elif self.initial_guess == 'random':
            self.r_x = np.random.rand(self.r_n_rows).astype(np.float64)
            self.r_y = np.random.rand(self.r_n_cols).astype(np.float64)
        elif self.initial_guess == 'uniform':
            self.r_x = np.ones(self.r_n_rows, dtype=np.float64)  # All probabilities will be 1/2 initially
            self.r_y = np.ones(self.r_n_cols, dtype=np.float64)
        elif self.initial_guess == 'degrees':
            self.r_x = self.r_dseq_rows.astype(np.float64)
            self.r_y = self.r_dseq_cols.astype(np.float64)

    def _initialize_problem(self):
        if ~self.is_reduced:
            self.degree_degeneration()
        self._set_initial_guess()
        self.J_T = np.empty((self.r_dim, self.r_dim), dtype=np.float64)
        self.residuals = np.empty(self.r_dim, dtype=np.float64)

    def _set_solved_problem(self, solution):
        self.r_xy = solution.x
        self.final_result = solution.fun
        self.r_x = self.r_xy[:self.r_n_rows]
        self.r_y = self.r_xy[self.r_n_rows:]
        self.x = self.r_x[self.r_invert_dseq_rows]
        self.y = self.r_y[self.r_invert_dseq_cols]

    def _clean_problem(self):
        self.J_T = None
        self.residuals = None
    
    def solve_root(self, initial_guess=None, method='hybr', scale=None):
        self.initial_guess = initial_guess
        self._initialize_problem()
        x0 = np.concatenate((self.r_x, self.r_y))
        opz = {'col_deriv': True, 'diag': scale}
        res = opt.root(self._residuals_jacobian, x0,
                       method=method, jac=True, options=opz)
        self._set_solved_problem(res)
        print(self.residuals)
        self._clean_problem()
    
    def solve_iterative(self, newton=False, initial_guess=None, tolerance=1e-8, max_iter=5000, print_counter=False):
        self.initial_guess = initial_guess
        self._initialize_problem()
        if newton:
            self.r_x, self.r_y = solve_newton(self.n_edges, self.r_dseq_rows.astype(float), self.r_dseq_cols.astype(float),
                                              self.rows_multiplicity, self.cols_multiplicity, self.r_x, self.r_y,
                                              tolerance=tolerance, print_counter=print_counter, max_iter=max_iter)
        else:
            self.r_x, self.r_y = solve_iterations(self.n_edges, self.r_dseq_rows.astype(float), self.r_dseq_cols.astype(float),
                                                  self.rows_multiplicity, self.cols_multiplicity, self.r_x, self.r_y,
                                                  tolerance=tolerance, print_counter=print_counter, max_iter=max_iter)
        self.x = self.r_x[self.r_invert_dseq_rows]
        self.y = self.r_y[self.r_invert_dseq_cols]

    def solve_partial(self, constrained_layer=0):
        if ~self.is_reduced:
            self.degree_degeneration()
        if constrained_layer == 0:
            self.r_xy = np.concatenate((self.r_dseq_rows / self.n_cols,
                                        np.ones(self.r_n_cols)))
            self.r_x = self.r_xy[:self.r_n_rows]
            self.r_y = self.r_xy[self.r_n_rows:]
            self.x = self.r_x[self.r_invert_dseq_rows]
            self.y = self.r_y[self.r_invert_dseq_cols]
        else:
            self.r_xy = np.concatenate((np.ones(self.r_n_rows),
                                        self.r_dseq_cols / self.n_rows))
            self.r_x = self.r_xy[:self.r_n_rows]
            self.r_y = self.r_xy[self.r_n_rows:]
            self.x = self.r_x[self.r_invert_dseq_rows]
            self.y = self.r_y[self.r_invert_dseq_cols]
