
"""
负责基于图进行预训练,用来做embedding初始化
考虑将diag和proc合并成一张图
"""

import torch
import torch.nn as nn
import torch.sparse as tsp
import torch.nn.functional as F

from torch import Tensor
from typing import Optional


class TriCL(nn.Module):
    def __init__(self, encoder, embedding_dim, proj_dim: int, num_nodes, num_edges, device):
        super(TriCL, self).__init__()
        self.device = device
        self.encoder = encoder

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_dim = embedding_dim
        self.edge_dim = embedding_dim

        self.node_embedding = nn.Embedding(self.num_nodes, self.node_dim)
        self.edge_embedding = nn.Embedding(self.num_edges + self.num_nodes, self.edge_dim)  # 加上自连边

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)

        self.disc = nn.Bilinear(self.node_dim, self.edge_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.node_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.disc.reset_parameters()

    def get_features(self, adj):
        node_idx = torch.arange(self.num_nodes, device=self.device)
        node_features = self.node_embedding(node_idx)

        edge_idx = torch.arange(self.num_edges + self.num_nodes, device=self.device)
        agg_edge_feat = self.get_hyperedge_representation(node_features, adj)

        edge_features = self.edge_embedding(edge_idx) + torch.cat([agg_edge_feat, node_features], dim=0)
        return node_features, edge_features

    @staticmethod
    def get_hyperedge_representation(embed, adj):
        """
        获取超边的表示，通过聚合当前超边下所有item的embedding
        实际上就是乘以H(n_edges, n_items)
        Args:
            embed:
            adj:

        Returns:

        """

        # embed: n_items, dim
        n_items, n_edges = adj.shape
        if adj.is_sparse:
            norm_factor = (tsp.sum(adj, dim=0) ** -1).to_dense().reshape(n_edges, -1)
            assert norm_factor.shape == (n_edges, 1)
            E = norm_factor * tsp.mm(adj.T, embed)
        else:
            norm_factor = (torch.sum(adj, dim=0) ** -1).reshape(n_edges, -1)
            assert norm_factor.shape == (n_edges, 1)
            E = norm_factor * torch.mm(adj.T, embed)

        return E

    def forward(self, x: Tensor, y: Tensor, hyperedge_index: Tensor):
        """

        Args:
            x: 节点特征
            y: 边特征
            hyperedge_index:

        Returns:

        """
        num_nodes, num_edges = self.num_nodes, self.num_edges

        # if num_nodes is None:
        #     num_nodes = int(hyperedge_index[0].max()) + 1
        # if num_edges is None:
        #     num_edges = int(hyperedge_index[1].max()) + 1

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        H = torch.sparse_coo_tensor(
            indices=self_loop_hyperedge_index,
            values=torch.ones_like(self_loop_hyperedge_index[0, :]),
            size=(num_nodes, num_edges + num_nodes)
        ).coalesce().float()
        n, e = self.encoder(x, y, H)
        return n, e[:num_edges]

    def without_selfloop(self, x: Tensor, hyperedge_index: Tensor, node_mask: Optional[Tensor] = None,
                         num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        if node_mask is not None:
            node_idx = torch.where(~node_mask)[0]
            edge_idx = torch.arange(num_edges, num_edges + len(node_idx), device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + len(node_idx))
            return n, e[:num_edges]
        else:
            return self.encoder(x, hyperedge_index, num_nodes, num_edges)

    def f(self, x, tau):
        return torch.exp(x / tau)

    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))

    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))

    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)

            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int],
               num_negs: Optional[int], mean: bool):
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    def node_level_loss(self, n1: Tensor, n2: Tensor, node_tau: float,
                        batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                        mean: bool = True):
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss

    def group_level_loss(self, e1: Tensor, e2: Tensor, edge_tau: float,
                         batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                         mean: bool = True):
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss

    def membership_level_loss(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
                              batch_size: Optional[int] = None, mean: bool = True):
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]
        if batch_size is None:
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)

            loss_n = -torch.log(pos / (pos + neg_n))
            loss_e = -torch.log(pos / (pos + neg_e))
        else:
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)

            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]

                pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)
                neg_n = self.f(
                    self.disc_similarity(n[hyperedge_index[:, mask][0]], e_perm[hyperedge_index[:, mask][1]]), tau)
                neg_e = self.f(
                    self.disc_similarity(n_perm[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)

                aggr_pos.append(pos)
                aggr_neg_n.append(neg_n)
                aggr_neg_e.append(neg_e)
            aggr_pos = torch.concat(aggr_pos)
            aggr_neg_n = torch.concat(aggr_neg_n)
            aggr_neg_e = torch.concat(aggr_neg_e)

            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]
        loss = loss_n + loss_e
        loss = loss.mean() if mean else loss.sum()
        return loss
