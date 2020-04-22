import numpy as np
import sys
sys.path.append('..')
from functions import bicm_calculator, projection_calculator, bicm_light, projection_calculator_light, indexes_edgelist

def main():
    user_dict_dtype = np.dtype([('user_id', np.int64), ('user_name', 'U40'), ('verified', '?')])
    user_list = np.loadtxt('user_dict.csv',  dtype=user_dict_dtype, delimiter=',')
    user_name_dict = dict(zip(user_list['user_id'], user_list['user_name']))
    g_bip_vunv = np.loadtxt('global_v_vunv.csv', dtype=np.dtype([('v_user_id', float), ('unv_user_id', float)]))
    g_bip_vunv = g_bip_vunv.astype(np.dtype([('v_user_id', np.uint64), ('unv_user_id', np.uint64)]))
    vus, k_vus = np.unique(g_bip_vunv['v_user_id'], return_counts=True)
    unvus, k_unvus = np.unique(g_bip_vunv['unv_user_id'], return_counts=True)
    
###     Using the fast methods
    edgelist, rows_degs, cols_degs, invert_rows_dict, invert_cols_dict  = indexes_edgelist(g_bip_vunv)
    x, y = bicm_light(rows_degs, cols_degs, method='newton')
    val_couples = projection_calculator_light(edgelist, x, y)
    
###     Using the matrix methods
    biad_mat = np.zeros((len(vus), len(unvus)), dtype=int)
    for link in g_bip_vunv:
        biad_mat[np.where(vus == link['v_user_id'])[0][0], np.where(unvus == link['unv_user_id'])[0][0]] = 1
    avg_mat = bicm_calculator(biad_mat)
    val_couples = projection_calculator(biad_mat, avg_mat)
    
###     Printing the results
    projection_edge_list = []
    for couple in val_couples:
        edge_names = (user_name_dict[vus[couple[0]]], user_name_dict[vus[couple[1]]])
        projection_edge_list.append((edge_names[0], edge_names[1]))
    print(projection_edge_list)


if __name__ == "__main__":
    main()