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

from utils import dataset_preparation, make_noise, cal_dist_matrix, AverageMeter, ProgressMeter, InfiniteDataLoader, accuracy
from latentmodel import LatentModel
from model import LinearExtractor, ConvExtractor, SingleLayerClassifier, ImageClassifier

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
parser.add_argument("--lr", default=2e-3, type=float)
parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
parser.add_argument("--epochs", default=350, type=int, help="total epochs")
parser.add_argument("--clip", default=0.05, type=float, help="gradient clipping")
parser.add_argument("--print_freq", default=10, type=int)
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--lr_gamma', default=0.1, type=float)

# Latent Model args
parser.add_argument("--latent_size", default=32, type=int, help="dimension of latent state")
parser.add_argument("--context_size", default=16, type=int, help="dimension of encoder context")
parser.add_argument("--hidden_size", default=32, type=int, help="hidden dimension of gru and mlp in latentmodel")
parser.add_argument("--kl_weight", default=0.02, type=float, help="kl weight")
parser.add_argument("--interp_weight", default=0.1, type=float)
parser.add_argument("--gen_batch", default=32, type=int)
parser.add_argument("--interp_num", default=1, type=int)
parser.add_argument("--epsilon", default=0.5, type=float)
parser.add_argument("--sigma", default=1.0, type=float)

args = parser.parse_args()

def train(train_loader, featurizer, classifier, latentmodel, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':3.5f')
    recon_losses = AverageMeter('Recon Loss', ':3.5f')
    kl_losses = AverageMeter('KL Loss', ':3.5f')
    interp_losses = AverageMeter('Interp Loss', ':3.5f')

    featurizer.train()
    classifier.train()
    latentmodel.train()

    for all_x, all_y, all_conf in train_loader:
        all_x, all_y = all_x.float().to(device), all_y.to(device)
        x_by_label, y_by_label = [], []
        for label in range(args.num_classes):
            label_indices = (all_y == label).nonzero().squeeze()
            x_by_label.append(all_x[label_indices])
            y_by_label.append(all_y[label_indices])

        recon_loss = 0.0
        kl_loss = 0.0
        interp_loss = 0.0
        for label in range(args.num_classes):
            x = x_by_label[label]
            if x.dim()==2:
                x = x.unsqueeze(0)
            x = x.transpose(0, 1)
            Ns = x.shape[0]

            ts = torch.linspace(0, 1, Ns+2).to(device)
            interp_ts = ts + 1/(Ns+1) * args.epsilon
            interp_ts = interp_ts[:-1]
            ext_ts = ts[0:1]
            for i in range(len(interp_ts)):
                ext_ts = torch.cat([ext_ts, interp_ts[i].view(1), ts[i + 1:i + 2]])
            ts = ext_ts[0::2][1:-1]

            if x.numel()==0:
                continue
            recon, kl, interp_x = latentmodel[label](x, ts, ext_ts, adjoint=False, method="euler")
            recon_loss += recon
            kl_loss += kl

            cur_interp_loss = 0.0
            for t in range(Ns - 1):
                pred = (1-args.epsilon) * classifier(interp_x[t], t) + args.epsilon * classifier(interp_x[t], t + 1)
                interp_y = torch.full((interp_x.shape[1],), label, dtype=torch.long).to(device)
                cur_interp_loss += F.cross_entropy(pred, interp_y)
            interp_loss += cur_interp_loss / (Ns - 1)

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
    ts = torch.linspace(0, 1, Ns + 2).to(device)
    interp_ts = ts + 1 / (Ns + 1) * args.epsilon
    interp_ts = interp_ts[:-1]
    ext_ts = ts[0:1]
    for i in range(len(interp_ts)):
        ext_ts = torch.cat([ext_ts, interp_ts[i].view(1), ts[i + 1:i + 2]])

    total_samples = 0
    total_correct = 0
    with torch.no_grad():
        zhats = []
        for label in range(args.num_classes):
            xhat = latentmodel[label].sample(batch_size=args.gen_batch, ts=ext_ts)
            zhats.append(xhat[-1])
        for x, y, _ in test_loader:
            x, y = x.float().to(device), y.to(device)
            z = featurizer(x)
            bs = x.shape[0]
            sim = torch.zeros(bs, args.num_classes).to(device)
            for label in range(args.num_classes):
                dist_matrix = cal_dist_matrix(z, zhats[label])
                weights = torch.exp(dist_matrix / (2 * args.sigma ** 2))
                weights = weights.sum(dim=1)
                sim[:, label] = weights

            correct = (sim.argmax(dim=1) == y).sum().item()
            total_correct += correct
            total_samples += y.size(0)


    accuracy = total_correct / total_samples
    return accuracy


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

    print(args)
    # Defining dataloaders
    train_iterator, test_loader, vanilla_loaders, _ = dataset_preparation(args, args.num_tasks, num_instances)

    data_path = 'data/{}'.format(args.dataset)
    data = torch.load(data_path + '/trajectory.pt')
    print("Loading Trajectory Data from ", data_path + '/trajectory.pt')
    trajectory_dataset = TensorDataset(data['Z'], data['Y'], data['conf'])

    # Define models
    if args.dataset == 'MNIST':
        featurizer = ConvExtractor()
        classifier = ImageClassifier(args.num_tasks - 1, 256, 10)
        args.feature_dim = 128
    elif args.dataset in ['ONP', 'Elec2', 'Shuttle','Moons']:
        featurizer = LinearExtractor(args.input_dim, args.feature_dim, args.num_layers, args.mlp_dropout)
        classifier = SingleLayerClassifier(args.feature_dim, args.output_dim, args.num_tasks - 1)

    featurizer.to(device)
    classifier.to(device)
    latentmodel = nn.ModuleList([LatentModel(data_size=args.feature_dim, latent_size=args.latent_size, context_size=args.context_size,
                                             hidden_size=args.hidden_size) for label in range(args.num_classes)]).to(device)
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
    # Training
    for epoch in range(args.epochs):
        starting_time = time.time()
        train(train_loader, featurizer, classifier, latentmodel, optimizer, epoch, args)
        ending_time = time.time()
        print("Traing time for epoch {}: {}".format(epoch, ending_time - starting_time))

    print("finish training...")
    print("testing...")
    acc = test(test_loader, featurizer, latentmodel, args)
    print("test_acc = {:3.6f}".format(acc))

if __name__ == "__main__":
    print("Start Training...")

    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    main(args)
