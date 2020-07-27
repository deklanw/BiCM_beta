import numpy as np
import sys 
sys.path.append('..')
from functions import bicm_calculator, projection_calculator, bicm_light, projection_calculator_light, indexes_edgelist
from BipartiteGraph import *
import datetime as dt

user_dict_dtype = np.dtype([('user_id', np.int64), ('user_name', 'U40'), ('verified', '?')])
user_list = np.loadtxt('user_dict.csv',  dtype=user_dict_dtype, delimiter=',')
user_name_dict = dict(zip(user_list['user_id'], user_list['user_name']))

g_bip_vunv = np.loadtxt('global_v_vunv.csv', dtype=np.dtype([('v_user_id', float), ('unv_user_id', float)]))
g_bip_vunv = g_bip_vunv.astype(np.dtype([('v_user_id', np.uint64), ('unv_user_id', np.uint64)]))

vus, k_vus = np.unique(g_bip_vunv['v_user_id'], return_counts=True)
unvus, k_unvus = np.unique(g_bip_vunv['unv_user_id'], return_counts=True)

biad_mat = np.zeros((len(vus), len(unvus)), dtype=int)
for link in g_bip_vunv:
    biad_mat[np.where(vus == link['v_user_id'])[0][0], np.where(unvus == link['unv_user_id'])[0][0]] = 1
    
my_graph = BipartiteGraph(edgelist=g_bip_vunv)
my_bicm = my_graph.get_bicm_matrix()

if __name__ == '__main__':
    print(dt.datetime.now())
    my_cols_projection = my_graph.get_cols_projection()
    print(dt.datetime.now())