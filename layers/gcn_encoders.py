
import os.path

import torch
from torch_geometric.nn import GCNConv, TransformerConv
import torch.nn as nn
import torch.sparse as tsp

# from position_encoding import HypergraphKnowledgeEncoding

class GCNLayer(nn.Module):
    def __init__(self, embed_dim, dropout, act=nn.LeakyReLU):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = act()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, X, A):
        X = self.conv(X, A)
        X = self.act(X)
        X = self.dropout(X)
        return X


class GCNEncoder(nn.Module):
    def __init__(self, embed_dim, dropout, n_layers, device, name):
        super(GCNEncoder, self).__init__()
        self.n_layers = n_layers
        self.name = name
        self.embed_dim = embed_dim
        self.device = device

        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(GCNLayer(embed_dim, dropout))
        
        self.reset_parameters()

    
    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
    
    def forward(self, X, A):
        X_lst = [X]
        for i in range(self.n_layers):
            X = self.encoders[i](X, A)
            X_lst.append(X)
        X = sum(X_lst) / (self.n_layers + 1)
        return X
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, heads=4):
        super(TransformerEncoderLayer, self).__init__()
        self.conv = TransformerConv(embed_dim, embed_dim, heads=heads, dropout=dropout, concat=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, X, A):
        X = self.conv(X, A)
        X = self.act(X)
        X = self.dropout(X)
        return X

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dropout, n_layers, device, name):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.name = name
        self.embed_dim = embed_dim
        self.device = device

        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(TransformerEncoderLayer(embed_dim, heads=4, dropout=dropout))
        
        self.reset_parameters()

    
    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
    
    def forward(self, X, A):
        X_lst = [X]
        for i in range(self.n_layers):
            X = self.encoders[i](X, A)
            X_lst.append(X)
        X = sum(X_lst) / (self.n_layers + 1)
        return X
    
class LocalMessagePassingNetwork(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super(LocalMessagePassingNetwork, self).__init__()
        self.conv = GCNConv(embed_dim, embed_dim, heads=heads, dropout=dropout, use_attention=True, concat=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, X, A):
        X = self.conv(X, A)
        X = self.act(X)
        X = self.dropout(X)
        return X

class KnowledgeAugmentedGlobalAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super(KnowledgeAugmentedGlobalAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

    def reset_parameters(self):
        self.self_attn._reset_parameters()

    def forward(self, X, ke_bias=None):
        X_attn = self._sa_block(X, ke_bias, None)  # 这里attn_mask如果是float类型,可以直接加到attn_weights上面
        return X_attn

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        # Requires PyTorch v1.11+ to support `average_attn_weights=False`
        # option to return attention weights of individual heads.
        x, A = self.self_attn(x, x, x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=True,
                              average_attn_weights=False)
        # self.attn_weights = A.detach().cpu()
        return x

class KAGTEncoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, heads=4):
        super(KAGTEncoderLayer, self).__init__()
        self.conv = LocalMessagePassingNetwork(embed_dim, heads=heads, dropout=dropout)
        self.global_attn = KnowledgeAugmentedGlobalAttention(embed_dim, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        self.input_ln = nn.LayerNorm(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.output_ln = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.global_attn.reset_parameters()
        self.ln.reset_parameters()

    def forward(self, X, A, ke_bias=None):
        X = self.input_ln(X)
        local_X = self.conv(X, A)
        glocal_X = self.global_attn(X, ke_bias)
        X = self.act(self.ln(local_X + glocal_X))
        X = self.dropout(X)
        X = self.output_ln(X)
        return X

class KAGTEncoder(nn.Module):
    def __init__(self, embed_dim, dropout, n_layers, device, name):
        super(KAGTEncoder, self).__init__()
        self.n_layers = n_layers
        self.name = name
        self.embed_dim = embed_dim
        self.device = device

        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(KAGTEncoderLayer(embed_dim, heads=4, dropout=dropout))
        
        self.reset_parameters()

        # self.ke, self.ke_bias = HypergraphKnowledgeEncoding(idx2word, embed_dim, cache_pth, device, name)

    
    def reset_parameters(self):
        for encoder in self.encoders:
            encoder.reset_parameters()
    
    def forward(self, X, A, ke_bias=None):
        X_lst = [X]
        for i in range(self.n_layers):
            X = self.encoders[i](X, A, ke_bias)
            X_lst.append(X)
        X = sum(X_lst) / (self.n_layers + 1)
        return X
    
        
    


        