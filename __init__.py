"""
BiCM
========

BiCM is a Python package for computing the Bipartite Configuration Model
of a bipartite network and its statistical projection on one layer.
The paper of reference is at https://www.nature.com/articles/srep10595?origin=ppub

Github repository::

    https://github.com/mat701/BiCM_beta

Author: Matteo Bruno
"""

from .functions import (
    bicm_calculator,
    bicm_light,
    bicm_from_fitnesses,
    check_sol,
    check_sol_light,
    p_val_reconstruction,
    projection_calculator,
    indexes_edgelist,
    pvals_validator,
    projection_calculator_light
)

from .BiCM_class import BiCM_class as BiCM

__version__ = "0.0.1"
__author__ = """Matteo Bruno (matteo.bruno@imtlucca.it)"""