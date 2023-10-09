"""https://github.com/vijaydwivedi75/gnn-lspe/blob/main/data/molecules.py"""
import os

import torch
import torch.nn as nn
import torch.sparse as tsp

from .icd_category import first_level_code_category


class HypergraphStructureEncoding(nn.Module):
    """
    考虑每个节点所处的超边数量,以及这些超边中包含的节点数量的统计信息
    """

    def __init__(self, H, se_dim, cache_pth, device, name, statistics_name=('max', 'min', 'mean', 'mode', 'var')):
        super(HypergraphStructureEncoding, self).__init__()
        self.H = H
        self.zero_one_H = torch.sparse_coo_tensor(H.indices(), torch.ones_like(H.values()), size=H.shape).coalesce()
        self.n_nodes, self.n_edges = H.shape
        self.statistics_name = statistics_name
        self.name = name

        if os.path.exists(cache_pth):
            self.encodings = torch.load(cache_pth)
        else:
            encodings = self.compute_structure_encodings()
            torch.save(encodings, cache_pth)
            self.encodings = encodings
        self.encodings = self.encodings.to(device)

        self.norm = nn.LayerNorm(len(statistics_name) + 1)
        self.proj = nn.Linear(len(statistics_name) + 1, se_dim)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.proj.reset_parameters()

    def compute_node_degree(self):
        node_degree = tsp.sum(self.zero_one_H, dim=1).to_dense()
        return node_degree

    def compute_edge_degree(self):
        edge_degree = tsp.sum(self.zero_one_H, dim=0).to_dense()
        return edge_degree

    def compute_structure_encodings(self):
        # 首先计算所有超边的度
        edge_degree = self.compute_edge_degree()
        node_degree = self.compute_node_degree()
        # 然后找到每个节点所处的超边idx,就是H的indices
        # 可以直接用indices中超边的维度先去切片,切出来超边的度
        node_idx, edge_idx = self.zero_one_H.indices()
        edge_sizes = edge_degree[edge_idx]
        # 现在需要计算统计信息
        ## 首先计算全局信息
        # global_statistics = {
        #     n: getattr(torch, n)(edge_sizes)
        #     if n != 'mode' else getattr(torch, n)(edge_sizes)[0]
        #     for n in self.statistics_name
        # }
        global_statistics = {}
        for n in self.statistics_name:
            if n == 'mode':
                value = getattr(torch, n)(edge_sizes)[0]
            elif n == 'var' and len(edge_sizes) == 1:
                value = torch.zeros_like(getattr(torch, n)(edge_sizes))
            else:
                value = getattr(torch, n)(edge_sizes)
            global_statistics[n] = value

        n_nodes = max(node_idx)
        structure_encodings = torch.zeros(self.n_nodes, len(self.statistics_name) + 1)  # 6是6个统计量
        empty_idx = []
        for i in range(n_nodes + 1):
            mask = node_idx == i
            local_sizes = edge_sizes[mask]
            if len(local_sizes) == 0:
                empty_idx.append(i)
                continue
            edge_num = node_degree[i]
            # relative_statistics = {
            #     n: getattr(torch, n)(local_sizes) - global_statistics[n]
            #     if n != 'mode' else getattr(torch, n)(local_sizes)[0] - global_statistics[n]
            #     for n in self.statistics_name
            # }
            relative_statistics = {}
            for n in self.statistics_name:
                if n == 'mode':
                    value = getattr(torch, n)(local_sizes)[0] - global_statistics[n]
                elif n == 'var' and len(local_sizes) == 1:
                    var = getattr(torch, n)(local_sizes)  # nan当0
                    value = -global_statistics[n]
                else:
                    value = getattr(torch, n)(local_sizes) - global_statistics[n]
                relative_statistics[n] = value
            structure_rep = torch.hstack([edge_num] + [relative_statistics[n] for n in self.statistics_name])
            structure_encodings[i] = structure_rep
        if len(empty_idx) > 0:
            empty_idx = torch.tensor(empty_idx)
            structure_encodings[empty_idx] = structure_encodings[~empty_idx].mean(1)
        return structure_encodings

    def forward(self):
        return self.proj(self.norm(self.encodings))


class HypergraphKnowledgeEncoding(nn.Module):
    """
    基于ICD coding的树上距离计算node-pair之间的距离,每个距离对应一个可学习的标量
    wiki上针对前三位数字就已经包含了二级分类,因此需要对前三位代码进行二次编码,变成一个两位代码,用来表示在这两级分类中分别
    属于哪一类
    """

    def __init__(self, idx2word, ke_dim, cache_pth, device, name):
        super(HypergraphKnowledgeEncoding, self).__init__()
        self.device = device
        self.idx2word = idx2word
        self.n_nodes = len(self.idx2word)
        self.ke_dim = ke_dim
        if 'cache_4' in cache_pth:
            self.enabled = False
        else:
            self.enabled = True
            self.name = name
            if self.name == 'med':
                self.first_category = None
            else:
                self.first_category = first_level_code_category()  # 包含了一级代码到修改过的分层代码的映射

            if os.path.exists(cache_pth):
                self.ke_dist_matrix = torch.load(cache_pth)
            else:
                ke_dist_matrix = self.compute_knowledge_encoding()
                torch.save(ke_dist_matrix, cache_pth)
                self.ke_dist_matrix = ke_dist_matrix
            self.ke_dist_matrix = self.ke_dist_matrix.long().to(device)

            max_dist = torch.max(self.ke_dist_matrix).long().item()
            self.encoding = nn.Embedding(max_dist + 1, 1)
            self.norm = nn.LayerNorm(self.n_nodes)
            self.proj = nn.Linear(self.n_nodes, ke_dim)

    def reset_parameters(self):
        self.encoding.reset_parameters()
        self.norm.reset_parameters()
        self.proj.reset_parameters()

    def icd_dist(self, u, v):
        u_icd, v_icd = self.idx2word[u], self.idx2word[v]

        # ICD9编码,前3位数字,如果有字母就是字母+前三位数字算一个大类
        # 参照wiki https://zh.wikipedia.org/zh-hans/ICD-9%E7%BC%96%E7%A0%81%E5%88%97%E8%A1%A8
        def icd_transform(icd_code):
            split_lst = []
            if self.name != 'med':
                # 第一级
                first_level_idx = 3
                if icd_code[0] in 'E':  # 外伤及补充分类,E后面3位数字,V两位
                    first_level_idx = 4
                first_level_icd = icd_code[:first_level_idx]
                transformed_first_icd = self.first_category[first_level_icd]
                split_lst += transformed_first_icd.split()  # 第一级可以细分为两级
                split_lst.append(first_level_icd)  # 第三级
                split_lst += icd_code[first_level_idx:].split()  # 小数点后,一般有一到两位数字
            else:
                split_lst += [icd_code[0], icd_code[1:3], icd_code[3]]
            return split_lst

        # 转换后的代码,每一位代表一个层级,长度可能不一样
        u_icd, v_icd = icd_transform(u_icd), icd_transform(v_icd)  # 列表

        l_u = len(u_icd)
        l_v = len(v_icd)
        min_l = min(l_u, l_v)  # 较短的列表长度

        common_l = 0  # 公共部分的长度

        for i in range(min_l):
            if u_icd[i] == v_icd[i]:
                common_l += 1
            else:
                break  # 一旦找到不匹配的元素，退出循环

        # pos就是两个节点分流的位置,那pos-1就是两个节点的最近公共祖先
        # 两个节点在树上的距离就是pos-1到两个节点的距离之和,也就是两个列表的剩余长度之和
        dist = l_u - common_l + l_v - common_l

        return dist

    def compute_knowledge_encoding(self):
        # 需要两两计算
        ## 首先构建一个n×n的矩阵
        icd_dist_matrix = torch.zeros(self.n_nodes, self.n_nodes)
        idx_lst = list(self.idx2word.keys())
        ## 接着遍历idx2word,计算距离,因为距离是相对的,所以要注意避免重复计算
        for i in range(len(idx_lst)):
            for j in range(i + 1, len(idx_lst)):
                u, v = idx_lst[i], idx_lst[j]
                icd_dist_matrix[u, v] = self.icd_dist(u, v)

        # # 构建一个dist到bias的嵌入
        # max_dist = max(icd_dist_matrix)
        # bias = torch.randn(max_dist)

        return icd_dist_matrix

    def forward(self):
        if self.enabled:
            ke_bias = self.encoding(self.ke_dist_matrix).reshape(self.n_nodes, self.n_nodes)
            return self.proj(self.norm(ke_bias)), ke_bias
        else:
            return torch.zeros(self.n_nodes, self.ke_dim).to(self.device), torch.zeros(self.n_nodes, self.n_nodes).to(self.device)


class HypergraphPositionEncoding(nn.Module):
    """
    对拉普拉斯矩阵做svd分解
    """

    def __init__(self, H, pe_dim, cache_pth, device, name):
        super(HypergraphPositionEncoding, self).__init__()
        self.encodings = None
        self.H = H
        self.n_nodes = H.shape[0]
        self.pe_dim = pe_dim
        self.name = name

        if os.path.exists(cache_pth):
            self.encodings = torch.load(cache_pth)
        else:
            encodings = self.svd_decomposition()
            torch.save(encodings, cache_pth)
            self.encodings = encodings
        self.encodings = self.encodings.to(device)

        self.norm = nn.LayerNorm(self.pe_dim * 2)
        self.proj = nn.Linear(self.pe_dim * 2, self.pe_dim)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.proj.reset_parameters()

    def compute_laplacian_matrix(self):
        """

        Args:
            H: 超图邻接矩阵,值是权重

        Returns:

        """
        H = self.H
        n, e = H.shape
        adj_v = torch.ones_like(H.values())
        adj = torch.sparse_coo_tensor(H.indices(), adj_v, size=H.shape)
        D_v_inv_sqrt = torch.diag((tsp.sum(adj, dim=1) ** -1 / 2).to_dense())
        assert D_v_inv_sqrt.shape == (n, n)
        D_e_inv = torch.diag((tsp.sum(adj, dim=0) ** -1).to_dense())
        assert D_e_inv.shape == (e, e)
        # 这里需要思考一个计算边权的方法,姑且用等价的
        W = torch.eye(e)
        assert W.shape == (e, e)

        L = torch.eye(n) - torch.matmul(D_v_inv_sqrt, H @ W @ torch.matmul(D_e_inv, H.T @ D_v_inv_sqrt))
        return L

    def svd_decomposition(self):
        L = self.compute_laplacian_matrix()
        U, S, V = torch.svd(L)
        # 选取最大的r个奇异值与奇异向量
        r = self.pe_dim
        U, S, V = U[:, :r], S[:r], V[:, :r]
        S_sqrt = torch.sqrt(S + 1e-8).reshape(1, r)
        U_hat = U * S_sqrt
        V_hat = V * S_sqrt
        pe_encoding = torch.cat([U_hat, V_hat], dim=-1)  # pe_dim * 2
        return pe_encoding

    def forward(self):
        # 随机翻转编码的符号
        signs = torch.tensor([-1, 1])
        idx = torch.randint(0, 2, (self.n_nodes,))
        signs_idx = signs[idx].reshape(self.n_nodes, 1).to(self.encodings.device)
        return self.proj(self.norm(self.encodings * signs_idx))


class FeatureEncoder(nn.Module):
    """
    用来整合位置编码,结构编码,知识编码
    """

    def __init__(self, H, idx2word, se_dim, pe_dim, ke_dim, cache_dir, device, name):
        super(FeatureEncoder, self).__init__()
        self.se_encoding = HypergraphStructureEncoding(H, se_dim, os.path.join(cache_dir, 'se_encoding.pth'), device,
                                                       name)
        self.pe_encoding = HypergraphPositionEncoding(H, pe_dim, os.path.join(cache_dir, 'pe_encoding.pth'), device,
                                                      name)
        self.ke_encoding = HypergraphKnowledgeEncoding(idx2word, ke_dim, os.path.join(cache_dir, 'ke_encoding.pth'),
                                                       device, name)
        self.name = name

    def reset_parameters(self):
        self.se_encoding.reset_parameters()
        self.pe_encoding.reset_parameters()
        if self.ke_encoding.enabled:
            self.ke_encoding.reset_parameters()

    def forward(self):
        se_encoding = self.se_encoding()
        ke_encoding, ke_bias = self.ke_encoding()
        pe_encoding = self.pe_encoding()

        encodings = {
            'se': se_encoding,
            'ke': ke_encoding,
            'pe': pe_encoding
        }
        return encodings, ke_bias


