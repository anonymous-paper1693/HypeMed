import torch
from pretrain.pretrainer import GCNPretrainer, TransformerConvPretrainer, KAGTPretrainer, HGTPretrainer, HGCLPretrainer
from tqdm import trange
import os
from util import seed_torch
from graph_construction import construct_graphs, construct_normal_graph
from config import parse_args
from HypeMed import dill_load
import numpy as np


def save_embedding(pretrainer, name_lst, mimic, encoder_name, epoch, dir='./pretrain/embed'):
    res_X = {}
    res_E = {}
    for n in name_lst:
        model = pretrainer.model_dict[n]
        adj = pretrainer.adj_dict[n]
        res = pretrainer.get_encoded_embedding(model, adj)
        res_X[n] = res['X'].to('cpu')
        if 'E' in res:
            res_E[n] = res['X'].to('cpu')
    if not os.path.exists(os.path.join(dir, encoder_name)):
        os.makedirs(os.path.join(dir, encoder_name))
    save_pth = os.path.join(dir, encoder_name, f'{encoder_name}_embed_mimic_{mimic}_{epoch}.pkl')

    embed_dict = {}
    if len(res_E) > 0:
        embed_dict['E'] = res_E
    embed_dict['X'] = res_X
    torch.save(
        embed_dict
        , save_pth
    )

def gcn_pretrain(args, voc_size_dict, adj_dict, name_lst, encoder_name='gcn', voc_dict=None, cache_pth=None):
    device = torch.device("cuda:{}".format(args.cuda))
    if encoder_name == 'gcn':
        pretrainer = GCNPretrainer(
            args,
            num_dict=voc_size_dict,
            adj_dict=adj_dict,
            device=device,
        )
    elif encoder_name == 'transformerconv':
        pretrainer = TransformerConvPretrainer(
            args,
            num_dict=voc_size_dict,
            adj_dict=adj_dict,
            device=device,
        )
    elif encoder_name == 'kagt':
        print(adj_dict)
        icd_mask = {}
        for n in name_lst:
            dist_pth = f'./pretrain/embed/ke/mimiciii/{n}_ke.pt'
            dist_matrix = torch.load(dist_pth)
            dist_matrix = dist_matrix.T + dist_matrix
            dist_matrix += torch.eye(dist_matrix.shape[0], dtype=torch.int32) * 6
            icd_mask[n] = dist_matrix
        pretrainer = KAGTPretrainer(
            args,
            num_dict=voc_size_dict,
            adj_dict=adj_dict,
            device=device,
            voc_dict=voc_dict,
            cache_pth=cache_pth,
            icd_mask=icd_mask
        )
    elif encoder_name == 'hgcn':
        indices_adj_dict = {k: v.coalesce().indices() for k, v in adj_dict.items()}
        print(adj_dict)
        icd_mask = {}
        for n in name_lst:
            dist_pth = f'./pretrain/embed/ke/mimiciii/{n}_ke.pt'
            dist_matrix = torch.load(dist_pth)
            dist_matrix = dist_matrix.T + dist_matrix
            dist_matrix += torch.eye(dist_matrix.shape[0], dtype=torch.int32) * 6
            icd_mask[n] = dist_matrix
        if args.mimic == 3:
            num_edges = 10489
        else:
            num_edges = 13763
        pretrainer = HGCLPretrainer(
            args,
            num_dict=voc_size_dict,
            num_edges=num_edges,
            adj_dict=indices_adj_dict,
            device=device,
            raw_adj_dict=adj_dict,
            icd_mask=icd_mask
        )
        
    else:
        raise ValueError('Wrong Encoder Name.!!!!')


    # pretrainer.pretrain()
    for i in trange(args.pretrain_epoch):
        loss_dict = {}
        if i % args.save_step == 0:
            save_embedding(pretrainer, name_lst, args.mimic, encoder_name, i)
            # print(pretrainer.model_dict['diag'].knowledge_encoding.encoding.weight.data.reshape(-1))

        for n in name_lst:
            loss_dict[n] = pretrainer.single_domain_step(pretrainer.model_dict[n], 
                                                         pretrainer.optimizer_dict[n], 
                                                         pretrainer.contrast_model[n],
                                                        pretrainer.adj_dict[n])
        print(f'epoch_{i}: diag-{loss_dict["diag"]}\tproc-{loss_dict["proc"]}\tmed-{loss_dict["med"]}\n')
    save_embedding(pretrainer, name_lst, args.mimic, encoder_name, args.pretrain_epoch)
    return


def main():
    args = parse_args()

    seed_torch(args.seed)

    if args.mimic == 3:
        # load old_data
        data_path = './data/records_final.pkl'
        voc_path = './data/voc_final.pkl'
        ddi_adj_path = './data/ddi_A_final.pkl'
    elif args.mimic == 4:
        # load old_data
        data_path = './data/records_final_4.pkl'
        voc_path = './data/voc_final_4.pkl'
        ddi_adj_path = './data/ddi_A_final_4.pkl'
    else:
        raise ValueError('Wrong Dataset Arg.!!!!!')

    if args.mimic == 3:
        cache_dir = './data/cache'
    else:
        cache_dir = './data/cache_4'

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_path = cache_dir

    ddi_adj = dill_load(ddi_adj_path, "rb")
    data = dill_load(data_path, "rb")
    assert len(data) > 100
    voc = dill_load(voc_path, "rb")

    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    # 这里数据划分好像有点问题，不是按照每个病人划分的，也没有shuffle
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    if not args.debug:
        assert eval_len > 100
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    voc_size_dict = {
        'diag': voc_size[0],
        'proc': voc_size[1],
        'med': voc_size[2]
    }
    voc_dict = {
        'diag': diag_voc,
        'proc': pro_voc,
        'med': med_voc
    }



    refresh = True
    if refresh:
        adj_dict = construct_graphs(data_train, nums_dict=voc_size_dict)
        if args.pretrain_model in ['gcn', 'transformerconv', 'kagt']:
            adj_dict = construct_normal_graph(adj_dict)
            torch.save(adj_dict, f'./pretrain/embed/graph_cache/ord_adj_dict_{args.mimic}.pt')
        else:
            torch.save(adj_dict, f'./pretrain/embed/graph_cache/hyper_adj_dict_{args.mimic}.pt')
    else:
        if args.pretrain_model in ['gcn', 'transformerconv', 'kagt']:
            adj_dict = torch.load(f'./pretrain/embed/graph_cache/ord_adj_dict_{args.mimic}.pt')
        else:
            adj_dict = torch.load(f'./pretrain/embed/graph_cache/hyper_adj_dict_{args.mimic}.pt')

    n_ehr_edges = adj_dict['diag'].shape[1]
    # print(adj_dict['diag'])

    # # ddi
    # adj_dict['ddi_adj'] = torch.FloatTensor(ddi_adj)

    name_lst = ['diag', 'proc', 'med']
    cache_pth = './pretrain/embed/ke/mimiciii'
    gcn_pretrain(args, voc_size_dict, adj_dict, name_lst, encoder_name=args.pretrain_model, voc_dict=voc_dict, cache_pth=cache_pth)

if __name__ == '__main__':
    main()