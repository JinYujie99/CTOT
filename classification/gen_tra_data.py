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

from utils import dataset_preparation, make_noise, cal_dist_matrix, AverageMeter, ProgressMeter, accuracy
from model import LinearExtractor, ConvExtractor, SingleLayerClassifier, ImageClassifier
from ot_func import cal_w_transport

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
datasets = ['ONP', 'Moons', 'MNIST', 'Elec2', 'Shuttle']
parser.add_argument("--dataset", default="Moons", type=str, help="one of: {}".format(", ".join(sorted(datasets))))
parser.add_argument("--input_dim", default=2, type=int, help="input feature dimension")
parser.add_argument("--output_dim", default=2, type=int, help="output dimension")
parser.add_argument("--feature_dim", default=50, type=int, help="featurizer output dimension")
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--num_tasks", default=10, type=int, help="total time stamps(including test time)")
parser.add_argument("--num_workers", default=0, type=int, help="the number of threads for loading data.")

# Model args
parser.add_argument("--num_layers", default=2, type=int, help="the number of layers in MLP")
parser.add_argument("--mlp_dropout", default=0.0, type=float, help="dropout rate of MLP")

# Training args
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
parser.add_argument("--epochs", default=30, type=int, help="total epochs")
parser.add_argument("--iterations", default=100, type=int, help="iterations per epoch")
parser.add_argument("--print_freq", default=5, type=int)

parser.add_argument("--seed", default=42, type=int)

args = parser.parse_args()

def main(args):
    model_directory = 'models/pretrain_models-{}'.format(args.dataset)
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)
    file_path = model_directory + '/' + 'bestmodel.pt'
    print("Loading pretrained model from ", file_path)
    checkpoint = torch.load(file_path, map_location='cpu')
    pretrained_featurizer_dict = checkpoint['featurizer_state_dict']

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log('use {} data'.format(args.dataset))
    log('-' * 40)

    if args.dataset == 'Moons':
        args.num_tasks = 10
        args.input_dim = 2
        args.feature_dim = 50
        args.num_layers = 2
        num_instances = 200
        args.num_classes = 2
    elif args.dataset == 'MNIST':
        args.num_tasks = 5
        num_instances = 1000
        args.num_classes = 10
    elif args.dataset == 'ONP':
        args.num_tasks = 6
        args.input_dim = 58
        args.feature_dim = 200
        args.num_layers = 2
        num_instances = None
        args.num_classes = 2
    elif args.dataset == 'Elec2':
        args.num_tasks = 41
        args.input_dim = 8
        args.feature_dim = 128
        args.num_layers = 2
        num_instances = None
        args.num_classes = 2
    elif args.dataset == 'Shuttle':
        args.num_tasks = 8
        args.input_dim = 9
        args.feature_dim = 128
        args.num_layers = 3
        num_instances = 7250
        args.num_classes = 2

    # Defining dataloaders
    _, _, vanilla_loaders, source_datasets = dataset_preparation(args, args.num_tasks, num_instances)
    source_domain = len(source_datasets)

    # Define models
    if args.dataset == 'MNIST':
        featurizer = ConvExtractor()
    elif args.dataset in ['ONP', 'Elec2', 'Shuttle','Moons']:
        featurizer = LinearExtractor(args.input_dim, args.feature_dim, args.num_layers, args.mlp_dropout)

    featurizer.load_state_dict(pretrained_featurizer_dict)
    Z_list, Y_list = [], []
    with torch.no_grad():
        for t in range(source_domain):
            cur_z = []
            cur_y = []
            for x, y, _ in vanilla_loaders[t]:
                x = x.float()
                z = featurizer(x)
                cur_z.append(z)
                cur_y.append(y)
            Z = torch.cat(cur_z, dim=0)
            Y = torch.cat(cur_y)
            Z_list.append(Z)
            Y_list.append(Y)

        max_values, indices = [], []
        for i in range(source_domain - 1):
            T = cal_w_transport(Z_list[i], Y_list[i], Z_list[i + 1], Y_list[i + 1])
            mv, id = T.max(dim=1)
            max_values.append(mv)
            indices.append(id)

        zs_list, y_list, conf_list = [], [], []
        for i in range(Z_list[0].shape[0]):
            pre = i
            conf = torch.tensor([1.0])
            y_list.append(Y_list[0][i])
            cur_trajectory = [Z_list[0][i]]
            for t in range(source_domain - 1):
                j = indices[t][pre]
                cur_trajectory.append(Z_list[t + 1][j])
                conf *= (max_values[t][pre] * Z_list[t].shape[0])
                pre = j
            zs_list.append(torch.stack(cur_trajectory, dim=0))
            conf_list.append(conf)
    data = {'Z': torch.stack(zs_list, dim=0), 'Y': torch.stack(y_list), 'conf': torch.stack(conf_list)}
    data_path = 'data/{}'.format(args.dataset)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    torch.save(data, data_path + '/' + 'trajectory.pt')
    print("save trajectory data in ", data_path + '/' + 'trajectory.pt')

if __name__ == "__main__":
    print("Generating Trajectory Data...")

    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    main(args)
