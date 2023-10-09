import os
import random
import time
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from config import parse_args
from dataloader import collate_fn, MIMICDataset
from graph_construction import construct_graphs

from layers import HGTDecoder
from pretrain import HGTPretrainer
from util import llprint, multi_label_metric, ddi_rate_score, replace_with_padding_woken, multihot2idx

SEED = 8895


def seed_torch(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(SEED)


def _init_fn(worker_id):
    np.random.seed(int(SEED) + worker_id)


# torch.set_num_threads(30)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# wandb
def init_wandb(args):
    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="MLHC",
            group=args.model_name,
            tags=[f'mimic{args.mimic}'],
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "architecture": args.model_name,
                "dataset": f"MIMIC-{args.mimic}",
                "epochs": args.epoch,
                "batch_size": args.bsz,
                "emb_dim": args.dim,
                "seed": args.seed,
                'n_layers': args.n_layers,
                'win_sz': args.win_sz,
                'n_heads': args.n_heads,
                'dropout': args.dropout,
            },

            name=f'{args.model_name}_{args.name}_drop_{args.dropout}_lr_{args.lr}_w_decay_{args.weight_decay}_win_sz_{args.win_sz}_layers_{args.n_layers}',

            # dir
            dir='./saved'
        )


def log_and_eval(model, eval_dataloader, voc_size, epoch, padding_dict, tic, history, args, bce_loss_lst,
                 multi_loss_lst, ddi_loss_lst, ssl_loss_lst, total_loss_lst, best_ja, best_epoch, ddi_adj_path):
    print(
        f'\nLoss: {np.mean(total_loss_lst):.4f}\t'
        f'BCE Loss: {np.mean(bce_loss_lst):.4f}\t'
        f'Multi Loss: {np.mean(multi_loss_lst):.4f}\t'
        f'DDI Loss: {np.mean(ddi_loss_lst):.4f}\t'
        f'SSL Loss: {np.mean(ssl_loss_lst):.4f}\t')
    tic2 = time.time()
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
        model, eval_dataloader, voc_size, epoch, padding_dict, ddi_adj_path
    )
    print(
        "training time: {}, test time: {}".format(
            time.time() - tic, time.time() - tic2
        )
    )

    history["ja"].append(ja)
    history["ddi_rate"].append(ddi_rate)
    history["avg_p"].append(avg_p)
    history["avg_r"].append(avg_r)
    history["avg_f1"].append(avg_f1)
    history["prauc"].append(prauc)
    history["med"].append(avg_med)

    if args.wandb or args.search:
        wandb.log({
            'ja': ja,
            'ddi_rate': ddi_rate,
            'avg_p': avg_p,
            'avg_r': avg_r,
            'avg_f1': avg_f1,
            'prauc': prauc,
            'med': avg_med,
            'bce_loss': np.mean(bce_loss_lst),
            'multi_loss': np.mean(multi_loss_lst),
            'ddi_loss': np.mean(ddi_loss_lst),
            'ssl_loss': np.mean(ssl_loss_lst),
            'total_loss': np.mean(total_loss_lst),
        })

    if epoch >= 5:
        print(
            "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                np.mean(history["ddi_rate"][-5:]),
                np.mean(history["med"][-5:]),
                np.mean(history["ja"][-5:]),
                np.mean(history["avg_f1"][-5:]),
                np.mean(history["prauc"][-5:]),
            )
        )

    if epoch != 0 and best_ja < ja:
        best_epoch = epoch
        best_ja = ja
        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    f"best_{args.name}_{args.dropout}_{args.lr}_{args.weight_decay}.model",
                ),
                "wb",
            ),
        )

    print("best_epoch: {}".format(best_epoch))
    return best_epoch, best_ja, ja


def compute_metric(targets, result, bsz, max_visit, masks):
    padding_mask = (masks['key_padding_mask'] == False).reshape(bsz * max_visit).detach().cpu().numpy()
    true_visit_idx = np.where(padding_mask == True)[0]
    y_gt = targets['loss_bce_target'].detach().cpu().numpy()

    y_pred_prob = F.sigmoid(result).detach().cpu().numpy()

    y_pred = y_pred_prob.copy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    y_pred_label = multihot2idx(y_pred)

    y_gt = y_gt.reshape(bsz * max_visit, -1)[true_visit_idx]
    y_pred_prob = y_pred_prob
    adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
        np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
    )
    med_cnt = y_pred.sum()
    visit_cnt = len(y_pred_label)
    return adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1, med_cnt / visit_cnt


# evaluate
def eval(model, eval_data_loader, voc_size, epoch, padding_dict, ddi_adj_path, threshold=0.5, test=False):
    with torch.no_grad():
        model.eval()

        # smm_record = []
        ja, prauc, avg_p, avg_r, avg_f1, avg_ddi = [[] for _ in range(6)]
        avg_length = []
        visit_num_lst = []
        med_cnt, visit_cnt = 0, 0

        gates_lst = []
        # model.cache_in_eval()

        for step, (records, masks, targets, visit2edge) in enumerate(eval_data_loader):
            records = {k: replace_with_padding_woken(v, -1, padding_dict[k]).to(model.device) for k, v in
                       records.items()}
            masks = {k: v.to(model.device) for k, v in masks.items()}
            targets = {k: v.to(model.device) for k, v in targets.items()}
            visit2edge = visit2edge.to(model.device)

            # y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

            bsz, max_visit, med_size = targets['loss_bce_target'].shape

            padding_mask = (masks['key_padding_mask'] == False).reshape(bsz * max_visit).detach().cpu().numpy()
            true_visit_idx = np.where(padding_mask == True)[0]
            # if test:
            #     target_output, _, gate = model(records, masks, torch.from_numpy(padding_mask == True).to(model.device),
            #                              visit2edge)
            #     gates_lst.extend(gate)
            # else:
            #     target_output, _, = model(records, masks, torch.from_numpy(padding_mask == True).to(model.device), visit2edge)
            target_output, _, = model(records, masks, torch.from_numpy(padding_mask == True).to(model.device),
                                      visit2edge)

            y_gt = targets['loss_bce_target'].detach().cpu().numpy()

            y_pred_prob = F.sigmoid(target_output).detach().cpu().numpy()
            # threshold = 0
            # y_pred_prob = target_output.detach().cpu().numpy()

            y_pred = y_pred_prob.copy()
            y_pred[y_pred >= threshold] = 1
            y_pred[y_pred < threshold] = 0

            y_pred_label = multihot2idx(y_pred)
            # smm_record.append(y_pred_label)

            y_gt = y_gt.reshape(bsz * max_visit, -1)[true_visit_idx]
            y_pred_prob = y_pred_prob
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
            )

            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            # ddi rate
            ddi_rate = ddi_rate_score([y_pred_label], path=ddi_adj_path)
            avg_ddi.append(ddi_rate)

            med_cnt += y_pred.sum()
            visit_cnt += len(y_pred_label)

            avg_length.append(y_pred.sum() / len(y_pred_label))
            visit_num_lst.append(len(y_pred_label))

            llprint("\rtest step: {} / {}".format(step, len(eval_data_loader)))

        llprint(
            "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                np.mean(avg_ddi),
                np.mean(ja),
                np.mean(prauc),
                np.mean(avg_p),
                np.mean(avg_r),
                np.mean(avg_f1),
                med_cnt / visit_cnt,
            )
        )

        return (
            np.mean(avg_ddi),
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def eval_v2(model, eval_data_loader, voc_size, epoch, padding_dict, ddi_adj_path, threshold=0.5, test=False):
    with torch.no_grad():
        model.eval()

        # smm_record = []
        ja, prauc, avg_p, avg_r, avg_f1, avg_ddi = [[] for _ in range(6)]
        avg_length = []
        visit_num_lst = []
        med_cnt, visit_cnt = 0, 0

        gates_lst = []
        # model.cache_in_eval()

        for step, (records, masks, targets, visit2edge) in enumerate(eval_data_loader):
            records = {k: replace_with_padding_woken(v, -1, padding_dict[k]).to(model.device) for k, v in records.items()}
            masks = {k: v.to(model.device) for k, v in masks.items()}
            targets = {k: v.to(model.device) for k, v in targets.items()}
            visit2edge = visit2edge.to(model.device)

            # y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

            bsz, max_visit, med_size = targets['loss_bce_target'].shape

            padding_mask = (masks['key_padding_mask'] == False).reshape(bsz * max_visit).detach().cpu().numpy()
            true_visit_idx = np.where(padding_mask == True)[0]
            # if test:
            #     target_output, _, gate = model(records, masks, torch.from_numpy(padding_mask == True).to(model.device),
            #                              visit2edge)
            #     gates_lst.extend(gate)
            # else:
            #     target_output, _, = model(records, masks, torch.from_numpy(padding_mask == True).to(model.device), visit2edge)
            target_output, _, = model(records, masks, torch.from_numpy(padding_mask == True).to(model.device), visit2edge)

            y_gt = targets['loss_bce_target'].detach().cpu().numpy()

            y_pred_prob = F.sigmoid(target_output).detach().cpu().numpy()
            # threshold = 0
            # y_pred_prob = target_output.detach().cpu().numpy()

            y_pred = y_pred_prob.copy()
            y_pred[y_pred >= threshold] = 1
            y_pred[y_pred < threshold] = 0

            y_pred_label = multihot2idx(y_pred)
            # smm_record.append(y_pred_label)

            y_gt = y_gt.reshape(bsz * max_visit, -1)[true_visit_idx]
            y_pred_prob = y_pred_prob

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
            )

            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            # ddi rate
            ddi_rate = ddi_rate_score([y_pred_label], path=ddi_adj_path)
            avg_ddi.append(ddi_rate)

            med_cnt += y_pred.sum()
            visit_cnt += len(y_pred_label)

            avg_length.append(y_pred.sum() / len(y_pred_label))
            visit_num_lst.append(len(y_pred_label))

            llprint("\rtest step: {} / {}".format(step, len(eval_data_loader)))

        llprint(
            "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                np.mean(avg_ddi),
                np.mean(ja),
                np.mean(prauc),
                np.mean(avg_p),
                np.mean(avg_r),
                np.mean(avg_f1),
                med_cnt / visit_cnt,
            )
        )

        if test:
            return (
                np.array(avg_ddi),
                np.array(ja),
                np.array(prauc),
                np.array(avg_p),
                np.array(avg_r),
                np.array(avg_f1),
                np.array(avg_length),
                np.array(visit_num_lst)
                )
        else:
            return (
                np.mean(avg_ddi),
                np.mean(ja),
                np.mean(prauc),
                np.mean(avg_p),
                np.mean(avg_r),
                np.mean(avg_f1),
                med_cnt / visit_cnt,
            )

def test_model_v2(model, args, device, data_test, voc_size, best_epoch, padding_dict, ddi_adj_path):
    # model.load_state_dict(torch.load(open(args.resume_path, "rb")))

    resume_pth = os.path.join(args.resume_path, f'best_{args.name}_{args.dropout}_{args.lr}_{args.weight_decay}.model')

    model.load_state_dict(
        torch.load(open(resume_pth, "rb")))
    # torch.load(open(os.path.join(args.resume_path, f'best_35_0.3_0.001_1e-05.model'), "rb")))
    model.to(device=device)
    tic = time.time()

    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    test_set = MIMICDataset(data_test)
    test_dataloader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn, shuffle=False,
                                 pin_memory=True, worker_init_fn=_init_fn)
    total_ddi, total_ja, total_prauc, total_p, total_r, total_f1, avg_length, visit_num_lst = eval_v2(
        model, test_dataloader, voc_size, best_epoch, padding_dict, ddi_adj_path, test=True
    )

    result = []
    for _ in range(10):
        idx = np.arange(len(data_test))
        test_sample_idx = np.random.choice(idx, round(len(data_test) * 0.8), replace=True)
        # test_sample = [data_test[i] for i in test_sample_idx]
        ja = np.mean(total_ja[test_sample_idx])
        avg_f1 = np.mean(total_f1[test_sample_idx])
        prauc = np.mean(total_prauc[test_sample_idx])
        ddi_rate = np.mean(total_ddi[test_sample_idx])
        avg_med = np.mean(avg_length[test_sample_idx])

        # ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
        #     model, test_sample, voc_size, 0
        # )
        result.append([ja, avg_f1, prauc, ddi_rate, avg_med])
        print([ja, avg_f1, prauc, ddi_rate, avg_med])
    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)
    # metric = ['ja', 'avg_f1', 'prauc', 'ddi_rate', 'avg_med']
    # metric_str = ''
    # metric_name_str = ''
    # for i in range(len(mean)):
    #     m, s = mean[i], std[i]
    #     metric_name_str += metric[i] + '\t'
    #     metric_str += f'{m:.4f}±{s:.4f}\t'
    # print(metric_str)

    log_hyperparam_lst = [
        'model_name',
        'name',
        'lr',
        'weight_decay',
        'bsz',
        'win_sz',
        'n_layers',
        'dim',
        'n_heads',
        'dropout',
        'multi_weight',
        'ddi_weight',
        'ssl_weight'
    ]
    hyper_param_str = ''
    hyper_param_name = ''
    for arg in vars(args):
        if arg in log_hyperparam_lst:
            hyper_param_name += f'{arg}\t'
            hyper_param_str += f'{getattr(args, arg)}\t'
        # print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    log_pth = os.path.join('./log', args.model_name)
    if not os.path.exists(log_pth):
        os.makedirs(log_pth)
    log_fn = f'ja_{mean[1]:.4f}+-{std[1]:.4f}.txt'
    with open(os.path.join(log_pth, log_fn), mode='w') as fw:
        fw.write(hyper_param_name + '\n')
        fw.write(hyper_param_str + '\n')
    print(hyper_param_name)
    print(hyper_param_str)

    metric = ['ja', 'avg_f1', 'prauc', 'ddi_rate', 'avg_med']
    metric_str = ''
    metric_name_str = ''
    for i in range(len(mean)):
        m, s = mean[i], std[i]
        metric_name_str += metric[i] + '\t'
        metric_str += f'{m:.4f}±{s:.4f}\t'
    with open(os.path.join(log_pth, log_fn), mode='a') as fw:
        fw.write(metric_name_str + '\n')
        fw.write(metric_str + '\n')
    print(metric_name_str)
    print(metric_str)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

    print(outstring)

    print("test time: {}".format(time.time() - tic))

    test_metric = {
        'test_ja': mean[0],
        'test_f1': mean[1],
        'test_prauc': mean[2],
        'test_ddi_rate': mean[3],
        'test_avg_med': mean[4],

        'test_ja_std': std[0],
        'test_f1_std': std[1],
        'test_prauc_std': std[2],
        'test_ddi_rate_std': std[3],
        'test_avg_med_std': std[4],
    }

    return test_metric

def test_model(model, args, device, data_test, voc_size, best_epoch, padding_dict, ddi_adj_path):
    # model.load_state_dict(torch.load(open(args.resume_path, "rb")))

    resume_pth = os.path.join(args.resume_path, f'best_{args.name}_{args.dropout}_{args.lr}_{args.weight_decay}.model')

    model.load_state_dict(
        torch.load(open(resume_pth, "rb")))
    # torch.load(open(os.path.join(args.resume_path, f'best_35_0.3_0.001_1e-05.model'), "rb")))
    model.to(device=device)
    tic = time.time()

    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []

    result = []
    total_gates_lst = []
    for _ in range(10):
        idx = np.arange(len(data_test))
        test_sample_idx = np.random.choice(idx, round(len(data_test) * 0.8), replace=True)
        test_sample = [data_test[i] for i in test_sample_idx]
        test_set = MIMICDataset(test_sample)
        test_dataloader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn, shuffle=False,
                                     pin_memory=True, worker_init_fn=_init_fn)
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
            model, test_dataloader, voc_size, best_epoch, padding_dict, ddi_adj_path, test=True
        )
        # ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med, gates_lst_single_run = eval(
        #     model, test_dataloader, voc_size, best_epoch, padding_dict, test=True
        # )
        # total_gates_lst.extend(gates_lst_single_run)
        # ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
        #     model, test_sample, voc_size, 0
        # )
        result.append([ja, avg_f1, prauc, ddi_rate, avg_med])
    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)

    log_hyperparam_lst = [
        'model_name',
        'name',
        'lr',
        'weight_decay',
        'bsz',
        'win_sz',
        'n_layers',
        'dim',
        'n_heads',
        'dropout',
        'multi_weight',
        'ddi_weight',
        'ssl_weight'
    ]
    hyper_param_str = ''
    hyper_param_name = ''
    for arg in vars(args):
        if arg in log_hyperparam_lst:
            hyper_param_name += f'{arg}\t'
            hyper_param_str += f'{getattr(args, arg)}\t'
        # print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    log_pth = os.path.join('./log', args.model_name)
    if not os.path.exists(log_pth):
        os.makedirs(log_pth)
    log_fn = f'ja_{mean[1]:.4f}+-{std[1]:.4f}.txt'
    with open(os.path.join(log_pth, log_fn), mode='w') as fw:
        fw.write(hyper_param_name + '\n')
        fw.write(hyper_param_str + '\n')
    print(hyper_param_name)
    print(hyper_param_str)

    metric = ['ja', 'avg_f1', 'prauc', 'ddi_rate', 'avg_med']
    metric_str = ''
    metric_name_str = ''
    for i in range(len(mean)):
        m, s = mean[i], std[i]
        metric_name_str += metric[i] + '\t'
        metric_str += f'{m:.4f}±{s:.4f}\t'
    with open(os.path.join(log_pth, log_fn), mode='a') as fw:
        fw.write(metric_name_str + '\n')
        fw.write(metric_str + '\n')
    print(metric_name_str)
    print(metric_str)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

    print(outstring)

    print("test time: {}".format(time.time() - tic))

    test_metric = {
        'test_ja': mean[0],
        'test_f1': mean[1],
        'test_prauc': mean[2],
        'test_ddi_rate': mean[3],
        'test_avg_med': mean[4],

        'test_ja_std': std[0],
        'test_f1_std': std[1],
        'test_prauc_std': std[2],
        'test_ddi_rate_std': std[3],
        'test_avg_med_std': std[4],
    }

    return test_metric

def dill_load(pth, mode='rb'):
    with open(pth, mode) as fr:
        file = dill.load(fr)
    return file


def main():
    args = parse_args()
    # print(args.search)
    if args.search:
        wandb.init(config=None)
        config = wandb.config

        args.name = config['name']
        args.pretrain_epoch = config['pretrain_epoch']
        args.eval_bsz = config['eval_bsz']
        args.cuda = config['cuda']
        args.mimic = config['mimic']
        args.epoch = config['epoch']
        args.seed = config['seed']
        args.search = config['search']

        args.top_n = config['top_n']

        args.lr = config['lr']
        args.weight_decay = config['weight_decay']
        args.dropout = config['dropout']
        args.multi_weight = config['multi_weight']
        args.ddi_weight = config['ddi_weight']
        args.ssl_weight = config['ssl_weight']
        args.target_ddi = config['target_ddi']
        # args.kp = config['kp']

    if not args.Test:
        init_wandb(args)

    seed_torch(args.seed)

    if not os.path.exists(os.path.join("saved", args.model_name)):
        os.makedirs(os.path.join("saved", args.model_name))

    if args.mimic == 3:
        # load old_data
        data_path = '../data/records_final.pkl'
        voc_path = '../data/voc_final.pkl'

        # ehr_adj_path = '../data/ehr_adj_final.pkl'
        ddi_adj_path = '../data/ddi_A_final.pkl'
        # ddi_mask_path = '../data/ddi_mask_H.pkl'
        # molecule_path = '../data/idx2drug.pkl'
    elif args.mimic == 4:
        # load old_data
        data_path = '../data/records_final_4.pkl'
        voc_path = '../data/voc_final_4.pkl'

        # ehr_adj_path = '../data/ehr_adj_final.pkl'
        ddi_adj_path = '../data/ddi_A_final_4.pkl'
        # ddi_mask_path = '../data/ddi_mask_H.pkl'
        # molecule_path = '../data/idx2drug.pkl'
    else:
        raise ValueError('Wrong Dataset Arg.!!!!!')

    if args.mimic == 3:
        cache_dir = '../data/cache'
    else:
        cache_dir = '../data/cache_4'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = cache_dir
    # desc_dict_path = '../data/desc_dict.pkl'

    ddi_adj = dill_load(ddi_adj_path, "rb")
    # ddi_mask_H = dill_load(ddi_mask_path, "rb")
    data = dill_load(data_path, "rb")
    assert len(data) > 100
    # molecule = dill_load(molecule_path, "rb")
    # desc_dict = dill_load(desc_dict_path, 'rb')
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

    adj_dict = construct_graphs(data_train, nums_dict=voc_size_dict)
    n_ehr_edges = adj_dict['diag'].shape[1]

    # ddi
    adj_dict['ddi_adj'] = torch.FloatTensor(ddi_adj)

    if args.debug:
        data_train = data_train[:100]
        data_eval = data_train[:100]
        data_test = data_train[:10]

    train_set = MIMICDataset(data_train)
    eval_set = MIMICDataset(data_eval)

    padding_dict = {
        'diag': voc_size_dict['diag'],
        'proc': voc_size_dict['proc'],
        'med': voc_size_dict['med'],
    }

    # model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
    name_lst = ['diag', 'proc', 'med']
    if args.pretrain:
        device = torch.device("cuda:{}".format(args.cuda))
        pretrainer = HGTPretrainer(
            args,
            num_dict=voc_size_dict,
            num_edges=n_ehr_edges,
            adj_dict=adj_dict,
            device=device,
            voc_dict=voc_dict,
            cache_dir=cache_dir,
        )

        pretrainer.pretrain()

        res_X = {}
        res_E = {}
        for n in name_lst:
            model = pretrainer.model_dict[n]
            adj = pretrainer.adj_dict[n]
            res = pretrainer.get_encoded_embedding(model, adj)
            X_hat = res['X']
            E_hat = res['E']
            res_X[n] = X_hat.to('cpu')
            res_E[n] = E_hat.to('cpu')

        torch.save(
            {
                'X': res_X,
                'E': res_E
            }
            , f'embed_mimic_{args.mimic}_{args.pretrain_epoch}.pkl'
        )
        return
    else:
        if args.mimic == 3:
            res = torch.load(f'embed_mimic_{args.mimic}_{args.pretrain_epoch}.pkl')
        else:
            res = torch.load(f'embed_mimic_{args.mimic}_{args.pretrain_epoch}.pkl')
        # res = torch.load('tmp.pkl')
        X_hat = res['X']
        E_mem = res['E']

    train_sampler = None

    device = torch.device("cuda:{}".format(args.cuda))
    model = HGTDecoder(embedding_dim=args.dim, n_heads=args.n_heads, dropout=args.dropout, n_ehr_edges=n_ehr_edges,
                       voc_size_dict=voc_size_dict,
                       padding_dict=padding_dict, device=device, X_hat=X_hat, E_mem=E_mem,
                       ddi_adj=adj_dict['ddi_adj'], channel_ablation=args.channel_ablation,
                       embed_ablation=args.embed_ablation, top_n=args.top_n, act=args.act)
    model.to(device=device)

    train_dataloader = DataLoader(train_set, batch_size=args.bsz, collate_fn=collate_fn,
                                  shuffle=False, pin_memory=True, sampler=train_sampler, worker_init_fn=_init_fn)
    eval_dataloader = DataLoader(eval_set, batch_size=args.eval_bsz, collate_fn=collate_fn, shuffle=False,
                                 pin_memory=True, worker_init_fn=_init_fn)

    if args.Test:
        test_model_v2(model, args, device, data_test, voc_size, 0, padding_dict, ddi_adj_path)
        return

    model.to(device=device)
    print('parameters', get_n_params(model))

    if args.optim == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_sche == 'CosWarm':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.T_0, T_mult=args.T_mult,
                                               eta_min=args.eta_min, verbose=True)
    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = args.epoch
    if args.debug:
        EPOCH = min(EPOCH, 3)

    for epoch in range(EPOCH):
        tic = time.time()

        print('\nepoch {} --------------------------'.format(epoch))

        model.train()
        bce_loss_lst, multi_loss_lst, ddi_loss_lst, ssl_loss_lst, total_loss_lst = [[] for _ in range(5)]
        adm_ja_lst, adm_prauc_lst, adm_avg_p_lst, adm_avg_r_lst, adm_avg_f1_lst, med_num_lst = [[] for _ in range(6)]
        for step, (records, masks, targets, visit2edge) in enumerate(train_dataloader):
            # 做点yu
            records = {k: replace_with_padding_woken(v, -1, padding_dict[k]).to(device) for k, v in records.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            visit2edge = visit2edge.to(device)

            bsz, max_visit, _ = records['diag'].shape
            bce_loss_mask = (masks['key_padding_mask'] == False).unsqueeze(-1).repeat(1, 1,
                                                                                      voc_size_dict['med']).reshape(
                bsz * max_visit, -1)
            true_visit_idx = bce_loss_mask.sum(-1) != 0

            result, side_loss = model(records, masks, true_visit_idx, visit2edge)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1, med_num = compute_metric(targets, result, bsz,
                                                                                          max_visit, masks)
            adm_ja_lst.append(adm_ja)
            adm_prauc_lst.append(adm_prauc)
            adm_avg_p_lst.append(adm_avg_p)
            adm_avg_r_lst.append(adm_avg_r)
            adm_avg_f1_lst.append(adm_avg_f1)
            med_num_lst.append(med_num)

            loss_ddi = side_loss['ddi']
            loss_ssl = side_loss['ssl']

            loss_bce = F.binary_cross_entropy_with_logits(
                result, targets['loss_bce_target'].reshape(bsz * max_visit, -1)[true_visit_idx],
                reduction='none'
            ).mean()

            # loss_bce = multilabel_categorical_crossentropy(targets['loss_bce_target'].reshape(bsz * max_visit, -1)[true_visit_idx], result)

            # multi_loss_mask = (masks['key_padding_mask'] == False).reshape(bsz * max_visit)
            loss_multi = F.multilabel_margin_loss(
                F.sigmoid(result),
                targets['loss_multi_target'].reshape(bsz * max_visit, -1)[true_visit_idx], reduction='none'
            ).mean()

            result_binary = F.sigmoid(result).detach().cpu().numpy()
            result_binary[result_binary >= 0.5] = 1
            result_binary[result_binary < 0.5] = 0
            y_label = multihot2idx(result_binary)
            current_ddi_rate = ddi_rate_score(
                [y_label], path=ddi_adj_path
            )

            multi_weight = args.multi_weight
            loss_multi = loss_multi * multi_weight
            ssl_weight = args.ssl_weight
            loss_ssl = ssl_weight * loss_ssl

            if current_ddi_rate <= args.target_ddi:
                loss_ddi = 0 * loss_ddi
                loss = loss_bce + loss_multi + loss_ssl
            else:
                ddi_weight = ((current_ddi_rate - args.target_ddi) / args.kp) * args.ddi_weight
                loss_ddi = ddi_weight * loss_ddi
                loss = loss_bce + loss_multi + loss_ssl + loss_ddi

            bce_loss_lst.append(loss_bce.item())
            multi_loss_lst.append(loss_multi.item())
            ddi_loss_lst.append(loss_ddi.item())
            ssl_loss_lst.append(loss_ssl.item())
            total_loss_lst.append(loss.item())

            optimizer.zero_grad()
            # print(loss_bce.item())
            loss.backward()
            optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(train_dataloader)))

        print(
            "\nTrain: Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                np.mean(adm_ja_lst),
                np.mean(adm_prauc_lst),
                np.mean(adm_avg_p_lst),
                np.mean(adm_avg_r_lst),
                np.mean(adm_avg_f1_lst),
                np.mean(med_num_lst),
            ))
        best_epoch, best_ja, cur_ja = log_and_eval(model, eval_dataloader, voc_size, epoch, padding_dict, tic, history,
                                                   args, bce_loss_lst,
                                                   multi_loss_lst, ddi_loss_lst, ssl_loss_lst, total_loss_lst,
                                                   best_ja,
                                                   best_epoch, ddi_adj_path)
        if args.lr_sche == 'CosWarm':
            lr_scheduler.step()

    # 测试
    test_metric = test_model_v2(model, args, device, data_test, voc_size, best_epoch, padding_dict, ddi_adj_path)

    if args.wandb or args.search:
        wandb.log(test_metric)
        wandb.finish()

    dill.dump(
        history,
        open(
            os.path.join(
                "saved", args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )



if __name__ == "__main__":
    main()
