
# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.sparse as tsp
from sklearn.cluster import SpectralClustering

from tqdm import tqdm


def factorize_and_recover(H, niter=10, k=5):
    """
    H: torch sparse 稀疏矩阵
    """
    m, n = H.shape
    q = min(6, m, n)
    U, S, V = torch.svd_lowrank(H, q=q, niter=niter)
    S = torch.diag(S)
    H_hat = U @ S @ V.T
    v_row, i_row = torch.topk(H_hat, k=k, dim=0)
    v_col, i_col = torch.topk(H_hat, k=k, dim=1)
    row_indices = torch.vstack([i_row.flatten(), torch.arange(n).repeat(k)])
    col_indices = torch.vstack([torch.arange(m).reshape(-1, 1).repeat(1, k).flatten(), i_col.flatten()])
    indices = torch.hstack([row_indices, col_indices])
    values = torch.ones_like(indices).float()[0]
    H_filter = torch.sparse_coo_tensor(indices, values, size=H.shape).coalesce()
    indices = H_filter.indices()
    values = torch.ones_like(H_filter.values())

    H_filter = torch.sparse_coo_tensor(indices, values, size=H.shape).coalesce()
    return H_filter

def construct_graphs(data_train, nums_dict):
    name_lst = ['diag', 'proc', 'med']


    # 这里要注意，因为模型的embedding有pad token end token在，这里要注意n_diag的实际值
    # 构建诊断和手术联合超图

    coo_dict = {
        name: [] for name in name_lst
    }
    visit_num = 0
    for u, visits in enumerate(data_train):
        for t, v in enumerate(visits):
            # 这里v是[诊断，手术，药物]

            set_dict = {
                name_lst[i]: v[i]
                for i in range(3)
            }

            hyper_edge = {
                # name: sorted(set_dict[name]) for name in name_lst
                name: set_dict[name] for name in name_lst
            }

            for n in name_lst:
                for idx, item in enumerate(hyper_edge[n]):
                    coo_dict[n].append([item, visit_num])
                    # coo_dict[n].append([item, visit_num, idx + 1])   # 这里把顺序加进来
                # coo_dict[n].append(hyper_edge[n])

            visit_num += 1

    # 将稀疏邻接矩阵转换成numpy array
    # 超图
    coo_dict = {
        n: np.array(coo_dict[n], dtype=np.int64).T
        for n in name_lst
    }

    H_dict = {}
    cluster_dict = {}

    for n in name_lst:
        H_i = torch.from_numpy(coo_dict[n])
        H_v = torch.ones(H_i.shape[1])
        # H_i = torch.from_numpy(coo_dict[n])[:2, :]
        # H_v = torch.from_numpy(coo_dict[n])[2, :]
        H = torch.sparse_coo_tensor(
            indices=H_i,
            values=H_v,
            size=(nums_dict[n], int(coo_dict[n].max(1)[1]) + 1)
        ).coalesce()
        # H = factorize_and_recover(H)
        H_dict[n] = H.float()

    H_dict = {k: v.coalesce() for k, v in H_dict.items()}
    return H_dict


def construct_normal_graph(H_dict):
    normal_graph = {}

    for name, H in H_dict.items():
        indices = H.indices()
        n_nodes = H.shape[0]
        adjacency_matrix = torch.zeros((n_nodes, n_nodes), dtype=int)

        # 遍历每个超边
        for hyperedge_id in tqdm(torch.unique(indices[1])):
            nodes_in_hyperedge = indices[0, indices[1] == hyperedge_id]

            # 记录所有在这个超边中的节点对
            for node1 in nodes_in_hyperedge:
                for node2 in nodes_in_hyperedge:
                    if node1 != node2 and adjacency_matrix[node1, node2] == 0:
                        adjacency_matrix[node1, node2] = 1

        # 将邻接矩阵转换为edge index的形式
        edge_indices = np.array(np.nonzero(adjacency_matrix))
        normal_graph[name] = torch.tensor(edge_indices, dtype=torch.long)

    return normal_graph

