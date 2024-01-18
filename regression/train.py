# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import numpy as np

import os
import random
import logging
import time
import datetime
from tqdm import tqdm
import argparse
import pickle
import ot
import torchsde

from utils import dataset_preparation, make_noise, cal_dist, AverageMeter, ProgressMeter, InfiniteDataLoader, accuracy
from latentmodel import LatentModel
from model import LinearExtractor, SingleLayerClassifier

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
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
parser.add_argument("--epochs", default=300, type=int, help="total epochs")
parser.add_argument("--clip", default=0.05, type=float, help="gradient clipping")
parser.add_argument("--print_freq", default=10, type=int)
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--lr_gamma', default=0.1, type=float)

# Latent Model args
parser.add_argument("--latent_size", default=64, type=int, help="dimension of latent state")
parser.add_argument("--context_size", default=16, type=int, help="dimension of encoder context")
parser.add_argument("--hidden_size", default=128, type=int, help="hidden dimension of gru and mlp in latentmodel")
parser.add_argument("--kl_weight", default=0.02, type=float, help="kl weight")
parser.add_argument("--interp_weight", default=0.1, type=float)
parser.add_argument("--gen_batch", default=1000, type=int)
parser.add_argument("--interp_num", default=1, type=int)

# Regression args
parser.add_argument("--alpha", default=1.0, type=float, help="balance for d(x1,x2) and d(y1,y2)")
parser.add_argument("--lambda_value", default=0.01, type=float, help="L2 regularization weight of Ridge Regression")
args = parser.parse_args()

def train(train_loader, featurizer, classifier, latentmodel, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':3.5f')
    recon_losses = AverageMeter('Recon Loss', ':3.5f')
    kl_losses = AverageMeter('KL Loss', ':3.5f')
    interp_losses = AverageMeter('Interp Loss', ':3.5f')

    featurizer.train()
    classifier.train()
    latentmodel.train()

    for all_x, all_conf in train_loader:
        x = all_x.float().to(device)
        if x.dim()==2:
            x = x.unsqueeze(0)
        x = x.transpose(0, 1)
        Ns = x.shape[0]
        ext_ts = torch.linspace(0, 1, 2*Ns+3).to(device)
        ts = ext_ts[0::2][1:-1]

        if x.numel()==0:
            continue
        recon_loss, kl_loss, interp_x = latentmodel(x, ts, ext_ts, method="euler")

        interp_loss = torch.tensor([0.0]).to(device)
        for t in range(Ns - 1):
            interp_input = interp_x[t,:,:-1]
            interp_labels = interp_x[t,:,-1]
            pred = 0.5 * (classifier(interp_input, t) + classifier(interp_input, t + 1))
            interp_loss += F.mse_loss(pred.squeeze(-1), interp_labels)
        interp_loss = interp_loss / (Ns - 1)

        loss = recon_loss + args.kl_weight * kl_loss + args.interp_weight * interp_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), all_x.size(0))
        recon_losses.update(recon_loss.item(), all_x.size(0))
        kl_losses.update(kl_loss.item(), all_x.size(0))
        interp_losses.update(interp_loss.item(), all_x.size(0))

    print('Total loss:{:.3f}, recon_loss:{:.3f}, kl_loss:{:.3f}, interp_loss:{:.3f}'.format(
        losses.avg, recon_losses.avg, kl_losses.avg, interp_losses.avg))

def test(test_loader, featurizer, latentmodel, args):
    featurizer.eval()
    latentmodel.eval()

    Ns = args.num_tasks - 1  # train domains
    ext_ts = torch.linspace(0, 1, 2 * Ns + 3).to(device)
    total_samples = 0
    total_mae = 0
    with torch.no_grad():
        xhat = latentmodel.sample(batch_size=args.gen_batch, ts=ext_ts)
        Z = xhat[-1][:,:-1]
        Y = xhat[-1][:,-1]
        Z_with_bias = torch.cat([Z, torch.ones((args.gen_batch, 1)).to(device)], dim=1)
        Z_transpose = torch.transpose(Z_with_bias, 0, 1)
        ZtZ = torch.matmul(Z_transpose, Z_with_bias)
        ZtZ = ZtZ + args.lambda_value * torch.eye(ZtZ.size(0)).to(device)
        ZtY = torch.matmul(Z_transpose, Y)
        ZtZ_inv = torch.inverse(ZtZ)
        W = torch.matmul(ZtZ_inv, ZtY)
        W = W[:-1]
        Y_mean = torch.mean(Y)
        Z_mean = torch.mean(Z, dim=0)
        b = Y_mean - torch.matmul(W, Z_mean)

        for x, y, _ in test_loader:
            x, y = x.float().to(device), y.to(device)
            z = featurizer(x)
            pred = torch.matmul(z, W.view(-1,1)) + b
            mae = F.l1_loss(pred.squeeze(-1), y, reduction='sum').item()
            total_mae += mae
            total_samples += y.size(0)

    mae = total_mae / total_samples
    return mae


def main(args):
    output_directory = 'results/outputs-{}'.format(args.dataset)
    ptmodel_directory = 'models/pretrain_models-{}'.format(args.dataset)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    print(args)
    # Defining dataloaders
    train_iterator, test_loader, _, _ = dataset_preparation(args, args.num_tasks, num_instances)

    data_path = 'data/{}'.format(args.dataset)
    data = torch.load(data_path + '/trajectory_alpha{}.pt'.format(args.alpha))
    print("Loading Trajectory Data from ", data_path + '/trajectory_alpha{}.pt'.format(args.alpha))
    trajectory_dataset = TensorDataset(data['Z'], data['conf'])

    # Define models
    featurizer = LinearExtractor(args.input_dim, args.feature_dim, args.num_layers, args.mlp_dropout)
    classifier = SingleLayerClassifier(args.feature_dim, 1, args.num_tasks - 1)

    featurizer.to(device)
    classifier.to(device)
    latentmodel = LatentModel(data_size=args.feature_dim+1, latent_size=args.latent_size, context_size=args.context_size,
                              hidden_size=args.hidden_size).to(device)

    # loading checkpoint
    checkpoint = torch.load(os.path.join(ptmodel_directory, f'bestmodel.pt'))
    featurizer.load_state_dict(checkpoint['featurizer_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    for param in featurizer.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = False

    # Define optimizers
    optimizer = torch.optim.Adam(latentmodel.parameters(), lr=args.lr, weight_decay=args.wd)

    train_loader = DataLoader(dataset=trajectory_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    for epoch in range(args.epochs):
        starting_time = time.time()
        train(train_loader, featurizer, classifier, latentmodel, optimizer, epoch, args)
        ending_time = time.time()
        print("Traing time for epoch {}: {}".format(epoch, ending_time - starting_time))

    mae = test(test_loader, featurizer, latentmodel, args)
    print("finish training...")
    print("test_mae = {:3.6f}".format(mae))

if __name__ == "__main__":
    print("Start Training...")

    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    main(args)
