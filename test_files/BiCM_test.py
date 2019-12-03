import numpy as np
import sys
sys.path.append('..')
from functions import bicm_calculator, projection_calculator

def main():
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
    avg_mat = bicm_calculator(biad_mat)
    val_couples = projection_calculator(biad_mat, avg_mat)
    edge_list = []
    connected_users = []
    for couple in val_couples:
        try:
            edge_names = (user_name_dict[vus[couple[0]]], user_name_dict[vus[couple[1]]])
        except:
            try:
                edge_names = (user_name_dict[vus[couple[0]]], vus[couple[1]])
                print(repr(vus[couple[1]]) + ' not found in dict')
            except:
                try:
                    edge_names = (vus[couple[0]], user_name_dict[vus[couple[1]]])
                    print(repr(vus[couple[0]]) + ' not found in dict')
                except:
                    edge_names = (vus[couple[0]], vus[couple[1]])
                    print(repr(vus[couple[0]]) + ' and ' + repr(vus[couple[1]]) + ' not found in dict')
        edge_list.append((edge_names[0], edge_names[1]))
        if edge_names[0] not in connected_users:
            connected_users.append(edge_names[0])
        if edge_names[1] not in connected_users:
            connected_users.append(edge_names[1])
    np.savetxt('projection_edge_list.csv', edge_list, delimiter=',', fmt=('%s', '%s'))
    all_all_names = [(name, name) for name in connected_users]
    np.savetxt('all_names.csv', all_all_names, delimiter=',', fmt=('%s', '%s'),
                  header='id,label', comments='')

if __name__ == "__main__":
    main()