import torch
import torch.nn as nn
import faiss

class EHRMemoryAttention(nn.Module):
    """
    这里做一个简易的attention+ffn,用的transformer架构
    """
    def __init__(self, embedding_dim, n_heads, dropout, top_n=10, act=nn.LeakyReLU):
        super(EHRMemoryAttention, self).__init__()
        self.visit_mem_attn = nn.MultiheadAttention(
            # embed_dim=embedding_dim * 3,
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        d_model = embedding_dim
        dim_feedforward = embedding_dim
        self.top_n = top_n


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = act()

        self.res = faiss.StandardGpuResources()

    
    def neighbour_search(self, visit_rep, E_mem_patient_rep, k=10):
        d = visit_rep.shape[1]
        index = faiss.GpuIndexFlatL2(self.res, d)
        index.add(E_mem_patient_rep.detach().cpu())

        D, I = index.search(visit_rep.detach().cpu(), k)
        # print(D, I)
        return D, I

    def forward(self, visit_rep, E_mem_patient_rep, E_mem_med_rep):
        """

        Args:
            visit_rep: 当前的visit的表示
            E_mem: ehr中的超边经过聚类得到的聚类中心,代表典型病例

        Returns:

        """
        x = visit_rep.unsqueeze(1)  # 调整x的维度为匹配的三维张量
        k = E_mem_patient_rep
        v = E_mem_med_rep
        D, I = self.neighbour_search(visit_rep, E_mem_patient_rep, k=self.top_n)
        k = E_mem_patient_rep[I, :]
        v = E_mem_med_rep[I, :]
        
        # print(x.shape, k.shape, v.shape)
        
        x = self.norm1(x + self._att_block(x, k, v, attn_mask=None))
        x = self.norm2(x + self._ff_block(x))
        # x = self._att_block(x, k, v)
        # print(x.shape)
        return x.squeeze(1)  # 将x的维度调整回原来的二维张量


    def _att_block(self, q, k, v, attn_mask=None):
        x, attn = self.visit_mem_attn(q, k, v,
                           need_weights=True, attn_mask=attn_mask)
        # x = x.squeeze(1)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class HistoryAttention(nn.Module):
    """
    这里做一个简易的attention+ffn,用的transformer架构
    """
    def __init__(self, embedding_dim, n_heads, dropout, act=nn.LeakyReLU):
        super(HistoryAttention, self).__init__()
        self.visit_mem_attn = nn.MultiheadAttention(
            # embed_dim=embedding_dim * 3,
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Implementation of Feedforward model
        # d_model = embedding_dim * 3
        # dim_feedforward = embedding_dim * 3
        d_model = embedding_dim
        dim_feedforward = embedding_dim
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = act()

    def forward(self, x, attn_mask):
        """

        Args:
            visit_rep: 当前的visit的表示
            E_mem: ehr中的超边经过聚类得到的聚类中心,代表典型病例

        Returns:

        """
        x = self.norm1(x + self._att_block(x, x, x, attn_mask))
        x = self.norm2(x + self._ff_block(x))
        # x = self._att_block(x, k, v)

        return x

    # self-attention block
    def _att_block(self, x, k, v, attn_mask):
        x, attn = self.visit_mem_attn(x, k, v, attn_mask=attn_mask,
                           need_weights=True)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)