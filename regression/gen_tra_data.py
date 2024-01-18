# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse
import random
import pickle
import ot
from torch.utils.data import TensorDataset, DataLoader

from utils import dataset_preparation, make_noise, cal_dist, AverageMeter, ProgressMeter, accuracy
from model import LinearExtractor, SingleLayerClassifier
from ot_func import cal_w_transport_reg

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
if not os.path.isdir('logs'):
    os.makedirs('logs')
log_file = 'logs/log_{}.log'.format(datetime.datetime.now().strftime("%Y_%B_%d_%I-%M-%S%p"))
open(log_file, 'a').close()

# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)


def log(str): logger.info(str)


log('Is GPU available? {}'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="TDG")

# Data args
datasets = ['Energy', 'HousePrice']
parser.add_argument("--dataset", default="HousePrice", type=str, help="one of: {}".format(", ".join(sorted(datasets))))
parser.add_argument("--input_dim", default=2, type=int, help="input feature dimension")
parser.add_argument("--output_dim", default=1, type=int, help="output dimension")
parser.add_argument("--feature_dim", default=50, type=int, help="featurizer output dimension")
parser.add_argument("--num_tasks", default=10, type=int, help="total time stamps(including test time)")
parser.add_argument("--num_workers", default=0, type=int, help="the number of threads for loading data.")

# Model args
parser.add_argument("--num_layers", default=2, type=int, help="the number of layers in MLP")
parser.add_argument("--mlp_dropout", default=0.0, type=float, help="dropout rate of MLP")

# Training args
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
parser.add_argument("--epochs", default=300, type=int, help="total epochs")
parser.add_argument("--iterations", default=100, type=int, help="iterations per epoch")
parser.add_argument("--print_freq", default=20, type=int)
parser.add_argument("--seed", default=42, type=int)
# Regression args
parser.add_argument("--alpha", default=1.0, type=float, help="balance for d(x1,x2) and d(y1,y2)")

args = parser.parse_args()

def main(args):
    model_directory = 'models/pretrain_models-{}'.format(args.dataset)
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)
    file_path = model_directory + '/' + 'bestmodel.pt'
    print("Loading pretrained model from ", file_path)
    checkpoint = torch.load(file_path, map_location='cpu')
    pretrained_featurizer_dict = checkpoint['featurizer_state_dict']

    log('use {} data'.format(args.dataset))
    log('-' * 40)

    if args.dataset == 'HousePrice':
        args.num_tasks = 7
        args.input_dim = 30
        args.feature_dim = 128
        args.num_layers = 2
        num_instances = None
    elif args.dataset == 'Energy':
        args.num_tasks = 9
        args.input_dim = 27
        args.feature_dim = 128
        args.num_layers = 2
        num_instances = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Defining dataloaders
    _, _, vanilla_loaders, source_datasets = dataset_preparation(args, args.num_tasks, num_instances)
    source_domain = len(source_datasets)

    # Define models
    featurizer = LinearExtractor(args.input_dim, args.feature_dim, args.num_layers, args.mlp_dropout)

    featurizer.load_state_dict(pretrained_featurizer_dict)
    Z_list, Y_list = [], []
    with torch.no_grad():
        for t in range(source_domain):
            cur_z = []
            cur_y = []
            for x, y, _ in vanilla_loaders[t]:
                x = x.float()
                y = y.float()
                z = featurizer(x)
                cur_z.append(z)
                cur_y.append(y)
            Z_list.append(torch.cat(cur_z, dim=0))
            Y_list.append(torch.cat(cur_y))
            print(Z_list[0].shape)
            print(Y_list[0].shape)

        max_sample = 0
        for t in range(source_domain):
            if Z_list[t].shape[0]>max_sample:
                max_sample = Z_list[t].shape[0]
                max_domain = t
        print("max sample", max_sample)
        print("max domain is domain", max_domain)

        next_max_values, next_indices = [], []
        for i in range(max_domain, source_domain - 1):
            T = cal_w_transport_reg(Z_list[i], Y_list[i], Z_list[i + 1], Y_list[i + 1], args)
            mv, id = T.max(dim=1)
            next_max_values.append(mv)
            next_indices.append(id)

        pre_max_values, pre_indices = [], []
        for i in range(max_domain, 0, -1):
            T = cal_w_transport_reg(Z_list[i], Y_list[i], Z_list[i - 1], Y_list[i - 1], args)
            mv, id = T.max(dim=1)
            pre_max_values.append(mv)
            pre_indices.append(id)

        zs_list, conf_list = [],  []
        for i in range(Z_list[max_domain].shape[0]):
            pre = i
            conf = torch.tensor([1.0])
            cur_trajectory = [torch.cat((Z_list[max_domain][i], Y_list[max_domain][i].unsqueeze(0)),dim=0)]
            for t in range(len(next_indices)):
                j = next_indices[t][pre]
                cur_trajectory.append(torch.cat((Z_list[max_domain+t+1][j], Y_list[max_domain+t+1][j].unsqueeze(0)),dim=0))
                conf *= (next_max_values[t][pre] * Z_list[t].shape[0])
                pre = j

            pre = i
            for t in range(len(pre_indices)):
                j = pre_indices[t][pre]
                cur_trajectory.insert(0, torch.cat((Z_list[max_domain-t-1][j], Y_list[max_domain-t-1][j].unsqueeze(0)),dim=0))
                conf *= (pre_max_values[t][pre] * Z_list[t].shape[0])
                pre = j

            zs_list.append(torch.stack(cur_trajectory, dim=0))
            conf_list.append(conf)
    print(torch.stack(zs_list, dim=0).shape)
    data = {'Z': torch.stack(zs_list, dim=0),  'conf': torch.stack(conf_list)}
    data_path = 'data/{}'.format(args.dataset)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    torch.save(data, data_path + '/' + 'trajectory_alpha{}.pt'.format(args.alpha))
    print("save trajectory data in ", data_path + '/' + 'trajectory_alpha{}.pt'.format(args.alpha))

if __name__ == "__main__":
    print("Generating Trajectory Data...")

    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    main(args)
