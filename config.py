
import argparse

# setting
model_name = "HypeMed"

resume_path = f'saved/{model_name}/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Test", action="store_true", default=False, help="test mode")
    parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
    parser.add_argument("--search", type=int, default=0, help="search is 1, single is 0")
    parser.add_argument("--project_name", type=str, default='', help="project name")
    parser.add_argument("--seed", type=int, default=424724, help="random seed")

    parser.add_argument("--mimic", type=int, default=4, help="choose mimic dataset")
    parser.add_argument("--channel_ablation", type=str, default=None, help="channel ablation study; only_ehr or only_his")
    parser.add_argument("--embed_ablation", type=str, default=None, help="embed ablation study; random or fix")

    parser.add_argument("--pretrain", action="store_true", default=False, help="pretrain mode")
    parser.add_argument("--pretrain_model", type=str, default='hgt', help="pretrain model name")
    
    parser.add_argument("--pretrain_epoch", type=int, default=1500, help="pretrain_epoch")
    parser.add_argument("--save_step", type=int, default=100, help="save step")
    parser.add_argument("--pretrain_lr", type=float, default=1e-3, help="pretrain_learning rate")
    parser.add_argument("--pretrain_weight_decay", type=float, default=1e-5, help="pretrain_weight decay")
    parser.add_argument("--batch_size_1", type=int, default=None, help="pretrain batch size node ssl")
    parser.add_argument("--batch_size_2", type=int, default=None, help="pretrain batch size membership ssl")
    parser.add_argument("--proj_dim", type=int, default=64, help="projection dimension")
    parser.add_argument("--tau_n", type=float, default=0.5, help="node tau")
    parser.add_argument("--tau_g", type=float, default=2, help="node group")
    parser.add_argument("--tau_m", type=float, default=1, help="node membership")
    parser.add_argument("--tau_c", type=float, default=1, help="cross domain")
    parser.add_argument("--drop_feature_rate", type=float, default=0.4, help="drop feature ratio")
    parser.add_argument("--drop_node_rate", type=float, default=0.4, help="drop node ratio")
    parser.add_argument("--drop_incidence_rate", type=float, default=0.4, help="drop incidence ratio")
    parser.add_argument("--w_g", type=float, default=1, help="group level weight")
    parser.add_argument("--w_m", type=float, default=1, help="membership level weight")
    parser.add_argument("--n_layers", type=int, default=2, help="num of gnn layers")

    parser.add_argument("--wandb", action="store_true", default=False, help="use wandb to log")
    parser.add_argument("--model_name", type=str, default=model_name, help="model name")
    parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
    parser.add_argument("--multi_weight", type=float, default=0.05, help="multi loss weight")
    parser.add_argument("--ddi_weight", type=float, default=0.5, help="ddi loss weight")
    parser.add_argument("--ssl_weight", type=float, default=0.01, help="ssl loss weight")

    parser.add_argument("--optim", type=str, default='Adam', help="optimizer")
    parser.add_argument("--lr_sche", type=str, default='CosWarm', help="lr scheduler")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--T_0", type=int, default=25, help="lr scheduler T_0")
    parser.add_argument("--T_mult", type=int, default=2, help="lr scheduler T_mult")
    parser.add_argument("--eta_min", type=float, default=0., help="minimum lr")

    parser.add_argument("--bsz", type=int, default=16, help="batch size")
    parser.add_argument("--epoch", type=int, default=75, help="epoch")
    parser.add_argument("--eval_bsz", type=int, default=1, help="eval batch size")
    parser.add_argument("--win_sz", type=int, default=3, help="seq window size")
    parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
    parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
    parser.add_argument("--dim", type=int, default=64, help="dimension")
    parser.add_argument("--med_sz", type=int, default=131, help="med size")
    parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
    parser.add_argument("--top_n", type=int, default=10, help="num of ehr neighbours selected")
    parser.add_argument("--act", type=str, default='relu', help="activation")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout ratio")
    parser.add_argument("--cuda", type=int, default=0, help="which cuda")
    parser.add_argument('--name', type=str, default='', help='name to save model')
    return parser.parse_args()
