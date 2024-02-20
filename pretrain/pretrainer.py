# -*- coding: utf-8 -*-
import sys

# sys.path.append("..")
from layers.pretrain_model import TriCL, GCNCL, HGCNCL
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import trange
import torch
from .utils import drop_incidence, drop_features, valid_node_edge_mask, hyperedge_index_masking

from layers.gcn_encoders import GCNEncoder, TransformerEncoder, KAGTEncoder
from layers.hgcn_encoders import HGCNEncoder, HKAGTEncoder
from layers.hgt_encoder import HGTEncoder


import GCL.augmentors as A
import GCL.losses as L
from GCL.models.contrast_model import WithinEmbedContrast


class BasePretrainer:
    def __init__(self):
        super(BasePretrainer, self).__init__()

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def pretrain(self, *args, **kwargs):
        raise NotImplementedError

    def get_encoded_embedding(self, *args, **kwargs):
        raise NotImplementedError


class HGTPretrainer(BasePretrainer):
    def __init__(
            self,
            args,
            num_dict,
            num_edges,
            adj_dict,
            device,
            idx2word_dict,
            cache_dir,

    ):
        super(HGTPretrainer, self).__init__()
        pretrain_epoch = args.pretrain_epoch
        pretrain_lr = args.pretrain_lr
        pretrain_weight_decay = args.pretrain_weight_decay
        self.name_lst = ['diag', 'proc', 'med']
        self.num_negs = None
        self.params = {
            'drop_incidence_rate': args.drop_incidence_rate,
            'drop_feature_rate': args.drop_feature_rate,
            'tau_n': args.tau_n,
            'tau_g': args.tau_g,
            'tau_m': args.tau_m,
            'tau_c': args.tau_c,
            'batch_size_1': args.batch_size_1,
            'batch_size_2': args.batch_size_2,
            'w_g': args.w_g,
            'w_m': args.w_m,
        }

        self.num_dict = num_dict
        self.num_edges = num_edges

        self.adj_dict = {n: adj_dict[n].to(device) for n in self.name_lst}

        self.device = device
        self.pretrain_epoch = pretrain_epoch
        self.model_dict = {
            n: self.build_model(
                args, num_dict[n], self.num_edges,
                adj=adj_dict[n],
                idx2word=idx2word_dict[n],
                cache_dir=cache_dir,
                device=device,
                name=n,
            )
            for n in self.name_lst
        }

        self.optimizer_dict = {
            n: self.build_optimizer(self.model_dict[n], lr=pretrain_lr, weight_decay=pretrain_weight_decay)
            for n in self.name_lst
        }
        # self.optimizer = Adam([
        #     {'params': self.model_dict['diag'].parameters()},
        #     {'params': self.model_dict['proc'].parameters()},
        #     {'params': self.model_dict['med'].parameters()}
        # ], lr=pretrain_lr, weight_decay=pretrain_weight_decay)

    @staticmethod
    def build_model(args, num_nodes, num_edges, adj, idx2word, cache_dir, device, name):
        encoder = HGTEncoder(
            embed_dim=args.dim,
            n_heads=args.n_heads,
            dropout=args.dropout,
            n_layers=args.n_layers,
            H=adj,
            idx2word=idx2word,
            cache_dir=cache_dir,
            device=device,
            name=name,
        ).to(device)

        model = TriCL(
            encoder,
            embedding_dim=args.dim,
            proj_dim=args.proj_dim,
            num_nodes=num_nodes,
            num_edges=num_edges,
            device=device
        ).to(device)
        return model

    @staticmethod
    def build_optimizer(encoder, **kwargs):
        lr = kwargs['lr']
        weight_decay = kwargs['weight_decay']
        optimizer = AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    @staticmethod
    def get_raw_node_edge_representation(model, adj):
        node_features, edge_features = model.get_features(adj)
        return node_features, edge_features

    # def single_domain_step(self, model, adj, num_nodes):
    def single_domain_step(self, model, optimizer, adj, num_nodes):
        hyperedge_index = adj.indices()
        num_edges = self.num_edges

        params = self.params
        num_negs = self.num_negs
        model.train()
        # optimizer.zero_grad(set_to_none=True)

        # Hypergraph Augmentation
        hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
        hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_rate'])
        node_features, edge_features = model.get_features(adj)
        x1 = drop_features(node_features, params['drop_feature_rate'])
        x2 = drop_features(node_features, params['drop_feature_rate'])
        y1 = drop_features(edge_features, params['drop_feature_rate'])
        y2 = drop_features(edge_features, params['drop_feature_rate'])

        node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
        node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
        node_mask = node_mask1 & node_mask2
        edge_mask = edge_mask1 & edge_mask2

        # Encoder
        n1, e1 = model(x1, y1, hyperedge_index1)
        n2, e2 = model(x2, y2, hyperedge_index2)

        # Projection Head
        n1, n2 = model.node_projection(n1), model.node_projection(n2)
        e1, e2 = model.edge_projection(e1), model.edge_projection(e2)

        loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
        loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], params['tau_g'],
                                        batch_size=params['batch_size_1'], num_negs=num_negs)

        masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask1)
        masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask2)
        loss_m1 = model.membership_level_loss(n1, e2[edge_mask2], masked_index2, params['tau_m'],
                                              batch_size=params['batch_size_2'])
        loss_m2 = model.membership_level_loss(n2, e1[edge_mask1], masked_index1, params['tau_m'],
                                              batch_size=params['batch_size_2'])
        loss_m = (loss_m1 + loss_m2) * 0.5

        loss = loss_n + params['w_g'] * loss_g + params['w_m'] * loss_m
        # return loss, e1
        loss.backward()
        optimizer.step()
        return loss.item()

    def cross_domain_step(self, edges_dict, tau):
        # proc2diag
        diag = F.normalize(edges_dict['diag'])
        proc = F.normalize(edges_dict['proc'])
        med = F.normalize(edges_dict['med'])

        proc2diag_sim = torch.exp((torch.mm(proc, diag.t()) / tau))
        med2diag_sim = torch.exp((torch.mm(med, diag.t()) / tau))
        loss = -torch.log(proc2diag_sim.diag() / proc2diag_sim.sum(1)) + -torch.log(med2diag_sim.diag() / med2diag_sim.sum(1))
        return loss.mean()

    def pretrain(self):
        for i in trange(self.pretrain_epoch):
            loss_dict = {}
            edges_dict = {}
            # self.optimizer.zero_grad(set_to_none=True)
            for n in self.name_lst:
                loss_dict[n] = self.single_domain_step(self.model_dict[n], self.optimizer_dict[n],
                                                                      self.adj_dict[n],
                                                                      self.num_dict[n])
                # loss_dict[n], edges_dict[n] = self.single_domain_step(self.model_dict[n],
                #                                                       self.adj_dict[n],
                #                                                       self.num_dict[n])
            # single_domain_loss = loss_dict['diag'] + loss_dict['proc'] + loss_dict['med']
            # cross_domain_loss = self.cross_domain_step(edges_dict, self.params['tau_c'])
            # loss = single_domain_loss + cross_domain_loss
            # loss.backward()
            # self.optimizer.step()
            # print(f'epoch_{i}: single-{single_domain_loss}\tcross-{cross_domain_loss}\n')
            print(f'epoch_{i}: diag-{loss_dict["diag"]}\tproc-{loss_dict["proc"]}\tmed-{loss_dict["med"]}\n')

    def get_encoded_embedding(self, model, adj):
        with torch.no_grad():
            model = model
            model.eval()
            node_features, edge_features = model.get_features(adj)
            hyperedge_index = adj.indices()
            X_hat, E_hat = model(node_features, edge_features, hyperedge_index)
            res = {
                'X': X_hat,
                'E': E_hat,
            }
        return res

class GCNPretrainer(BasePretrainer):
    def __init__(
        self,
        args,
        num_dict,
        adj_dict,
        device,
        **kwargs
    ):
        super(GCNPretrainer, self).__init__()
        pretrain_epoch = args.pretrain_epoch
        pretrain_lr = args.pretrain_lr
        pretrain_weight_decay = args.pretrain_weight_decay
        self.name_lst = ['diag', 'proc', 'med']
        self.save_step = args.save_step
        self.num_dict = num_dict
        if 'num_edges' in kwargs:
            self.num_edges = kwargs['num_edges']

        self.adj_dict = {n: adj_dict[n].to(device) for n in self.name_lst}
        print(self.adj_dict)

        self.device = device
        self.pretrain_epoch = pretrain_epoch
        self.model_dict = {
            n: self.build_model(
                args, num_dict[n],
                adj=adj_dict[n],
                device=device,
                name=n,
                **kwargs
            )
            for n in self.name_lst
        }


        self.optimizer_dict = {
            n: self.build_optimizer(self.model_dict[n], lr=pretrain_lr, weight_decay=pretrain_weight_decay)
            for n in self.name_lst
        }

        self.aug = A.Compose([A.EdgeRemoving(pe=args.drop_incidence_rate), A.FeatureDropout(pf=args.drop_feature_rate), A.NodeDropping(pn=args.drop_node_rate)])
    
        self.contrast_model = self.build_contrast_model(args=args, device=device, **kwargs)

    def build_contrast_model(self, args, device, **kwargs):
        contrast_model = {}
        for n in self.name_lst:
            k = self.num_dict[n]
            pos_mask = torch.eye(k).to(device)
            neg_mask = (torch.ones((k, k)) - torch.eye(k)).to(device)
            contrast_model[n] = WithinEmbedContrast(loss=L.InfoNCE(tau=args.tau_n), 
                                                         pos_mask=pos_mask, neg_mask=neg_mask).to(device)
        return contrast_model


    @staticmethod
    def build_model(args, num_nodes, adj, device, name, **kwargs):
        encoder = GCNEncoder(
            embed_dim=args.dim,
            dropout=args.dropout,
            n_layers=args.n_layers,
            device=device,
            name=name,
        ).to(device)

        model = GCNCL(
            encoder,
            embedding_dim=args.dim,
            num_nodes=num_nodes,
            device=device,
            name=name
        ).to(device)
        return model

    @staticmethod
    def build_optimizer(encoder, **kwargs):
        lr = kwargs['lr']
        weight_decay = kwargs['weight_decay']
        optimizer = AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
    
    
    def single_domain_step(self, model, optimizer, contrast_model, adj):
        if adj.shape[0] != 2:
            edge_index = adj.T
        else:
            edge_index = adj
        model.train()
        # optimizer.zero_grad(set_to_none=True)
        # Apply the augmentation to the data object
        node_features = model.get_features()
        x1, edge_index1, _ = self.aug(node_features, edge_index, None)
        x2, edge_index2, _ = self.aug(node_features, edge_index, None)

        # Encoder
        n1 = model(x1, edge_index1)
        n2 = model(x2, edge_index2)

        # loss

        loss = contrast_model(n1, n2)

        loss.backward()
        optimizer.step()
        return loss.item()
    
    def pretrain(self):
        for i in trange(self.pretrain_epoch):
            loss_dict = {}
            for n in self.name_lst:
                loss_dict[n] = self.single_domain_step(self.model_dict[n], self.optimizer_dict[n], self.constrast_model[n],
                                                                      self.adj_dict[n])
            print(f'epoch_{i}: diag-{loss_dict["diag"]}\tproc-{loss_dict["proc"]}\tmed-{loss_dict["med"]}\n')
    
    def get_encoded_embedding(self, model, adj):
        with torch.no_grad():
            model = model
            model.eval()
            node_features = model.get_features()
            if adj.shape[0] != 2:
                edge_index = adj.T
            else:
                edge_index = adj
            X_hat = model(node_features, edge_index)
            res = {
                'X': X_hat,
            }
        return res

class TransformerConvPretrainer(GCNPretrainer):
    @staticmethod
    def build_model(args, num_nodes, adj, device, name):
        encoder = TransformerEncoder(
            embed_dim=args.dim,
            dropout=args.dropout,
            n_layers=args.n_layers,
            device=device,
            name=name,
        ).to(device)

        model = GCNCL(
            encoder,
            embedding_dim=args.dim,
            num_nodes=num_nodes,
            device=device
        ).to(device)
        return model
    

class KAGTPretrainer(GCNPretrainer):
    @staticmethod
    def build_model(args, num_nodes, adj, device, name, **kwargs):
        encoder = KAGTEncoder(
            embed_dim=args.dim,
            dropout=args.dropout,
            n_layers=args.n_layers,
            device=device,
            name=name,
        ).to(device)

        model = GCNCL(
            encoder,
            embedding_dim=args.dim,
            num_nodes=num_nodes,
            device=device,
            name=name,
            idx2word=kwargs['voc_dict'][name].idx2word,
            cache_pth=kwargs['cache_pth'],
        ).to(device)
        return model
    
    def build_contrast_model(self, args, device, **kwargs):
        contrast_model = {}
        for n in self.name_lst:
            contrast_model[n] = {}
            k = self.num_dict[n]
            pos_mask = torch.eye(k).to(device)
            neg_mask = (torch.ones((k, k)) - torch.eye(k)).to(device)
            contrast_model[n]['n2n'] = WithinEmbedContrast(loss=L.InfoNCE(tau=1), 
                                                         pos_mask=pos_mask, neg_mask=neg_mask).to(device)

            pos_mask = (kwargs['icd_mask'][n] < 4).to(device)
            neg_mask = (kwargs['icd_mask'][n] > 5).to(device)
            contrast_model[n]['icd'] = WithinEmbedContrast(loss=L.InfoNCE(tau=1),
                                                    pos_mask=pos_mask, neg_mask=neg_mask).to(device)
            
        
        return contrast_model
    
    def single_domain_step(self, model, optimizer, contrast_model, adj):
        if adj.shape[0] != 2:
            edge_index = adj.T
        else:
            edge_index = adj
        model.train()
        # optimizer.zero_grad(set_to_none=True)
        # Apply the augmentation to the data object
        node_features = model.get_features()
        x1, edge_index1, _ = self.aug(node_features, edge_index, None)
        x2, edge_index2, _ = self.aug(node_features, edge_index, None)

        # Encoder
        n1 = model(x1, edge_index1)
        n2 = model(x2, edge_index2)

        # loss
        icd_loss = contrast_model['icd'](n1, n2)
        n2n_loss = contrast_model['n2n'](n1, n2)
        # sementic_loss = (contrast_model['n2n'](n1, x2) + contrast_model['n2n'](x1, n2)) / 2
        loss = icd_loss + n2n_loss

        loss.backward()
        optimizer.step()
        return loss.item()

class HGCNPretrainer(GCNPretrainer):
    @staticmethod
    def build_model(args, num_nodes, adj, device, name, **kwargs):
        encoder = KAGTEncoder(
            embed_dim=args.dim,
            dropout=args.dropout,
            n_layers=args.n_layers,
            device=device,
            name=name,
        ).to(device)

        model = HGCNCL(
            encoder,
            embedding_dim=args.dim,
            num_nodes=num_nodes,
            device=device
        ).to(device)
        return model

class HGCLPretrainer(HGCNPretrainer):
    @staticmethod
    def build_model(args, num_nodes, adj, device, name, **kwargs):
        encoder = HGCNEncoder(
            embed_dim=args.dim,
            dropout=args.dropout,
            n_layers=args.n_layers,
            device=device,
            name=name,
        ).to(device)

        model = HGCNCL(
            encoder,
            embedding_dim=args.dim,
            num_nodes=num_nodes,
            device=device,
            name=name,
            num_edges=kwargs['num_edges'],
        ).to(device)
        return model

    def build_contrast_model(self, args, device, **kwargs):
        contrast_model = {}
        for n in self.name_lst:
            contrast_model[n] = {}
            k = self.num_dict[n]
            pos_mask = torch.eye(k).to(device)
            neg_mask = (torch.ones((k, k)) - torch.eye(k)).to(device)
            contrast_model[n]['n2n'] = WithinEmbedContrast(loss=L.InfoNCE(tau=args.tau_n), 
                                                         pos_mask=pos_mask, neg_mask=neg_mask).to(device)
            pos_mask = kwargs['raw_adj_dict'][n].to_dense().bool().to(device)
            neg_mask = ~pos_mask.to(device)
            contrast_model[n]['n2e'] = WithinEmbedContrast(loss=L.InfoNCE(tau=args.tau_m),
                                                    pos_mask=pos_mask, neg_mask=neg_mask).to(device)
            # pos_mask = kwargs['raw_adj_dict'][n].to_dense().bool().T.to(device)
            # neg_mask = ~pos_mask.to(device)
            # contrast_model[n]['e2n'] = WithinEmbedContrast(loss=L.InfoNCE(tau=args.tau_m),
            #                                         pos_mask=pos_mask, neg_mask=neg_mask).to(device)
            # pos_mask = kwargs['icd_mask'][n].to(device)
            # neg_mask = ~pos_mask.to(device)
            # contrast_model[n]['icd'] = WithinEmbedContrast(loss=L.InfoNCE(tau=1),
            #                                         pos_mask=pos_mask, neg_mask=neg_mask).to(device)
            

        k = self.num_edges
        pos_mask = torch.eye(k).to(device)
        neg_mask = (torch.ones((k, k)) - torch.eye(k)).to(device)
        contrast_model['edge'] = WithinEmbedContrast(loss=L.InfoNCE(tau=args.tau_g), 
                                                        pos_mask=pos_mask, neg_mask=neg_mask).to(device)
        
        return contrast_model
    
    def single_domain_step(self, model, optimizer, contrast_model, adj):
        if adj.shape[0] != 2:
            edge_index = adj.T
        else:
            edge_index = adj
        model.train()
        # optimizer.zero_grad(set_to_none=True)
        # Apply the augmentation to the data object
        node_features = model.get_features()
        x1, edge_index1, _ = self.aug(node_features, edge_index, None)
        x2, edge_index2, _ = self.aug(node_features, edge_index, None)
        # icd loss
        # icd_loss = contrast_model['icd'](node_features, node_features)

        # Encoder
        n1, e1 = model(x1, edge_index1)
        n2, e2 = model(x2, edge_index2)

        # loss
        n2n_loss = contrast_model['n2n'](n1, n2)
        e2e_loss = self.contrast_model['edge'](e1, e2)
        n2e_loss = (contrast_model['n2e'](n1, e2) + contrast_model['n2e'](n2, e1)) * (1 / 2)


        # loss = ((n2n_loss + e2e_loss + n2e_loss) / 3) + icd_loss * 0
        loss = (n2n_loss + e2e_loss + n2e_loss) / 3
        # loss = icd_loss
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # def ehr_step(self, e_diag, e_proc, adj):
    #     e_dp = e_diag + e_proc
    #     # loss
    #     ehr_loss = self.contrast_model['ehr'](e_dp, e_dp)
    #     loss = ehr_loss
    #     loss.backward()
    #     optimizer.step()
    #     return loss.item()

    def get_encoded_embedding(self, model, adj):
        with torch.no_grad():
            model = model
            model.eval()
            node_features = model.get_features()
            if adj.shape[0] != 2:
                edge_index = adj.T
            else:
                edge_index = adj
            X_hat, _ = model(node_features, edge_index)
            res = {
                'X': X_hat,
            }
        return res
    
    def pretrain(self):
        for i in trange(self.pretrain_epoch):
            loss_dict = {}
            for n in self.name_lst:
                loss_dict[n] = self.single_domain_step(self.model_dict[n], self.optimizer_dict[n],
                                                                        self.adj_dict[n])
            print(f'epoch_{i}: diag-{loss_dict["diag"]}\tproc-{loss_dict["proc"]}\tmed-{loss_dict["med"]}\n')