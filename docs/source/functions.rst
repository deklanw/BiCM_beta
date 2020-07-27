Functions
=========

The BiCM package uses and provides many useful functions for bipartite networks.

.. code-block:: python
    
    sample_bicm(avg_mat)
    sample_bicm_edgelist(x, y)

These are methods for sampling a network from a matrix of probabilities or from the fitnesses of the BiCM model. The first one returns a biadjacency matrix (a numpy array), the second one returns an edgelist (list).

.. code-block:: python

    edgelist_from_biadjacency(biadjacency)
    biadjacency_from_edgelist(edgelist, fmt='array')
    edgelist_from_edgelist(edgelist)

These methods can switch from one representation of a network to another. The third method takes in input an edgelist with nodes with custom names, and relabels the nodes with integers in the order of their appearance, yielding the new edgelist (lighter than the first one), the rows degree sequence, the columns degree sequence, a dictionary of the conversion of the row nodes and the same dictionary for the column nodes.

.. code-block:: python
    
    check_sol(biad_mat, avg_bicm, return_error=False, in_place=False)
    check_sol_light(x, y, rows_deg, cols_deg, return_error=False)

These two methods can check if a probability matrix is the solution of the bicm of a biadjacency matrix (first), or if two fitness vectors are solutions for two degree sequences (second). They print the errors, but don't return anything except when return_error is set to True, in which case they output 0 if the inputs are solutions, 1 if there are any errors.