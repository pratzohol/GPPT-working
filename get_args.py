import argparse
from dgl.data import register_data_args
def get_my_args():
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--lr_c", type=float, default=0.01, help="lamda for orthogonality")
    parser.add_argument("--seed", type=int, default=100, help="random seed")
    parser.add_argument("--n-hidden", type=int, default=128, help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, default=2, help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--sample-list', type=list, default=[4,4])
    parser.add_argument("--n-epochs", type=int, default=30, help="number of training epochs")
    parser.add_argument("--file-id", type=str, default='128GCN')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--half', type=bool, default=False)
    parser.add_argument('--mask_rate', type=float, default=0)
    parser.add_argument('--center_num', type=int, default=7, help="number of clusters M")

    # dataset -
    # cora, citeseer, pubmed, reddit, Fraud_yelp, Fraud_amazon, CoraFull, AmazonCoBuyComputer, AmazonCoBuyPhoto, CoauthorCS, ogbn-arxiv
    args = parser.parse_args(args=['--dataset','cora'])
    print(args)
    return args