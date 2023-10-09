import torch.nn as nn
import torch.sparse as tsp

from .HypergraphConv import HypergraphConv


class HyperGraphEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout):
        super(HyperGraphEncoderLayer, self).__init__()
        self.graph_encoder = HypergraphConv(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            use_attention=True,
            heads=n_heads, dropout=dropout,
            concat=False
        )

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU()
        )

    def get_hyperedge_representation(self, embed, adj):
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
        norm_factor = (tsp.sum(adj, dim=0) ** -1).to_dense().reshape(n_edges, -1)

        assert norm_factor.shape == (n_edges, 1)
        omega = norm_factor * tsp.mm(adj.T, embed)
        return omega

    def forward(self, x, adj):
        """
        进行超图编码
        Args:
            x:
            adj:

        Returns:
            x:
        """
        adj_index = adj.indices()
        hyperedge_attr = self.get_hyperedge_representation(x, adj)
        hyperedge_weight = adj.values()
        x = self.norm_1(x + self.graph_encoder(x, adj_index, hyperedge_attr=hyperedge_attr, hyperedge_weight=hyperedge_weight))
        x = self.norm_2(x + self.ffn(x))

        return x


class HyperGraphEncoder(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout, n_layers):
        super(HyperGraphEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(n_layers):
            self.encoder.append(HyperGraphEncoderLayer(embedding_dim, n_heads, dropout))


    def forward(self, x, adj):
        """
        进行超图编码
        Args:
            x:
            adj:

        Returns:
            x:
        """
        output = [x]
        for i in range(self.n_layers):
            x = self.encoder[i](x, adj) + x
            output.append(x)
        output = sum(output) / (self.n_layers + 1)
        return output