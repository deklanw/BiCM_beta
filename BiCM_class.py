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
def jac(xx, multiplier_rows, multiplier_cols, nrows, ncols, out_J_T):
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

@jit(nopython=True)
def solve_iterations(n_edges, r_dseq_rows, r_dseq_cols, rows_multiplicity, cols_multiplicity,
                     tolerance=1e-8, max_iter=5000, print_counter=False):
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
    X = r_dseq_rows / np.sqrt(n_edges)
    Y = r_dseq_cols / np.sqrt(n_edges)
    x = r_dseq_rows * 0
    y = r_dseq_cols * 0
    change = 1
    t1 = time.time()
    for counter in range(max_iter):
        for i in range(r_n):
            x[i] = self.r_dseq_rows[i] / np.sum(Y * cols_multiplicity / (1. + X[i] * Y))
        for i in range(r_m):
            y[i] = self.r_dseq_cols[i] / np.sum(X * rows_multiplicity / (1. + X * Y[i]))
        change = max(np.max(np.abs(X - x)), np.max(np.abs(Y - y)))
        X[:] = x
        Y[:] = y
        if change < tolerance: 
            break
    t2 = time.time()
    if change > tolerance:
        raise Exception("Solver did not converge. Try increasing max_iter")
    if print_counter == True:
        print("Solver converged in {} iterations.".format(counter))
    return x, y

class BiCM_class:
    def __init__(self, degree_sequence, n_rows, n_cols):
        ##### Full problem parameters
        self.dseq_rows = np.copy(degree_sequence[:n_rows])
        self.dseq_cols = np.copy(degree_sequence[n_rows:])
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.dim = self.n_rows + self.n_cols
        self.n_edges = np.sum(self.dseq_rows)
        self.x = None
        self.y = None
        self.xy = None
        ##### Reduced problem parameters
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
        #### Problem solutions
        self.J_T = None
        self.r_x = None
        self.r_y = None
        self.r_xy = None
        #### Problem (reduced) residuals
        self.residuals = None
        self.final_result = None

    def degree_degeneration(self):
        self.r_dseq_rows, self.r_invert_dseq_rows, self.rows_multiplicity \
            = np.unique(self.dseq_rows, 0, 1, 1)
        self.r_dseq_cols, self.r_invert_dseq_cols, self.cols_multiplicity \
            = np.unique(self.dseq_cols, 0, 1, 1)
        self.r_n_rows = self.r_dseq_rows.size
        self.r_n_cols = self.r_dseq_cols.size
        self.r_dim = self.r_n_rows + self.r_n_cols
        self.is_reduced = True

    def _equations(self, x):
        eqs(x, self.r_dseq_rows, self.r_dseq_cols, \
            self.rows_multiplicity, self.cols_multiplicity, \
            self.r_n_rows, self.residuals)

    def _jacobian(self, x):
        jac(x, self.rows_multiplicity, self.cols_multiplicity, \
            self.r_n_rows, self.r_n_cols, self.J_T)

    def _cost_gradient(self, x):
        self._equations(x)
        self._jacobian(x)
        return 0.5 * np.dot(self.residuals, self.residuals), \
                np.dot(self.J_T, self.residuals)

    def _residuals_jacobian(self, x):
        self._equations(x)
        self._jacobian(x)
        return self.residuals, self.J_T

    def _equations_rvalue(self, x):
        self._equations(x)
        return self.residuals

    def _jacobian_rvalue(self, x):
        self._jacobian(x)
        return self.J_T

    def _hessian_vector_product(self, x, p):
        return np.dot(self.J_T.T, np.dot(self.J_T, p))

    def _hessian(self, x):
        self._jacobian(x)
        return self.J_T @ self.J_T.T

    def _initialize_problem(self):
        if ~self.is_reduced:
            self.degree_degeneration()
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
    
    def solve_trust_regions(self, initial_guess=None, method='trust-ncg', \
                        display=False, maxiter=1000):
        self._initialize_problem()
        if initial_guess is None:
            #self.r_x = self.r_dseq_rows * self.r_dseq_rows / self.n_edges
            #self.r_y = self.r_dseq_cols * self.r_dseq_cols / self.n_edges
            self.r_x = self.r_dseq_rows / np.sqrt(self.n_edges)
            self.r_y = self.r_dseq_cols / np.sqrt(self.n_edges)
            
            x0 = np.concatenate((self.r_x, self.r_y)).astype(np.float64)
        elif initial_guess == 'random':
            x0 = np.random.rand(self.r_dim).astype(np.float64)
        else:
            x0 = initial_guess.astype(np.float64)
        opz = {'disp':display, 'maxiter':maxiter}
        if method == 'trust-ncg' or method == 'trust-krylov':
            res = opt.minimize(self._cost_gradient, x0, method=method, \
                               jac=True, hessp=self._hessian_vector_product, \
                               options=opz)
        elif method == 'trust-exact':
            res = opt.minimize(self._cost_gradient, x0, method=method, \
                               jac=True, hess=self._hessian, \
                               options=opz)
        self._set_solved_problem(res)
        del(res)
        self._clean_problem()
    
    def solve_root(self, initial_guess=None, method='hybr', scale=None):
        self._initialize_problem()
        # The preselected initial guess works best usually. The suggestion is, if this does not work, trying with random initial conditions several times.
        # If there is a more precise approximation, initial_guess can be passed as a vector.
        if initial_guess is None:
            self.r_x = self.r_dseq_rows / (np.sqrt(self.n_edges) + 1) # This +1 increases the stability of the solutions.
            self.r_y = self.r_dseq_cols / (np.sqrt(self.n_edges) + 1)
            x0 = np.concatenate((self.r_x, self.r_y)).astype(np.float64)
        elif initial_guess == 'random':
            x0 = np.random.rand(self.r_dim).astype(np.float64)
        elif initial_guess == 'uniform':
            x0 = np.ones(self.r_dim, dtype=np.float64) # All probabilities will be 1/2 initially
        elif initial_guess == 'degrees':
            x0 = np.concatenate((self.r_dseq_rows, self.r_dseq_cols)).astype(np.float64)
        else:
            x0 = initial_guess.astype(np.float64)
        opz = {'col_deriv':True, 'diag':scale}
        res = opt.root(self._residuals_jacobian, x0, method=method, jac=True, \
                        options=opz)
        self._set_solved_problem(res)
        del(res)
        self._clean_problem()

    def solve_least_squares(self, initial_guess=None, method='trf', \
                            scale=1.0, tr_solver='lsmr', disp=False):
        self._initialize_problem()
        if initial_guess is None:
            self.r_x = self.r_dseq_rows / (np.sqrt(self.n_edges) + 1)
            self.r_y = self.r_dseq_cols / (np.sqrt(self.n_edges) + 1)
            x0 = np.concatenate((self.r_x, self.r_y)).astype(np.float64)
        elif initial_guess == 'random':
            x0 = np.random.rand(self.r_dim).astype(np.float64)
        else:
            x0 = initial_guess.astype(np.float64)
        res = opt.least_squares(self._equations_rvalue, x0, method=method, \
                                jac=self._jacobian_rvalue, x_scale=scale, \
                                tr_solver=tr_solver, verbose=disp)
        self._set_solved_problem(res)
        del(res)
        self._clean_problem()
    
    def solve_iterative(self, initial_guess=None, tolerance=1e-8, max_iter=5000, print_counter=False):
        # I didn't finish implementing initial_guess, even if this simple one is working good
        self._initialize_problem()
        self.r_x, self.r_y = solve_iterations(self.n_edges, self.r_dseq_rows, self.r_dseq_cols, self.rows_multiplicity, self.cols_multiplicity)
        self.x = self.r_x[self.r_invert_dseq_rows]
        self.y = self.r_y[self.r_invert_dseq_cols]

    def solve_partial(self, constrained_layer=0):
        if ~self.is_reduced:
            self.degree_degeneration()
        if constrained_layer == 0:
            self.r_xy = np.concatenate((self.r_dseq_rows / self.n_cols, \
                                        np.ones(self.r_n_cols)))
            self.r_x = self.r_xy[:self.r_n_rows]
            self.r_y = self.r_xy[self.r_n_rows:]
            self.x = self.r_x[self.r_invert_dseq_rows]
            self.y = self.r_y[self.r_invert_dseq_cols]
        else:
            self.r_xy = np.concatenate((np.ones(self.r_n_rows), \
                                        self.r_dseq_cols / self.n_rows))
            self.r_x = self.r_xy[:self.r_n_rows]
            self.r_y = self.r_xy[self.r_n_rows:]
            self.x = self.r_x[self.r_invert_dseq_rows]
            self.y = self.r_y[self.r_invert_dseq_cols]