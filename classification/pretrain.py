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
import pickle

from utils import dataset_preparation, make_noise, AverageMeter, ProgressMeter, accuracy
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
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
parser.add_argument("--epochs", default=30, type=int, help="total epochs")
parser.add_argument("--iterations", default=100, type=int, help="iterations per epoch")
parser.add_argument("--print_freq", default=50, type=int)

args = parser.parse_args()


def train(train_iterator, featurizer, classifier, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':3.5f')
    cls_accs = AverageMeter('Cls Acc', ':3.3f')

    progress = ProgressMeter(
        args.iterations,
        [losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    featurizer.train()
    classifier.train()

    for iter in range(args.iterations):
        minibatches = [(x.float().to(device), y.to(device), idx.to(device)) for x, y, idx in next(train_iterator)]
        T = len(minibatches)  # train domains
        bs = args.batch_size
        all_x = torch.cat([x for x, y, idx in minibatches])
        all_y = torch.cat([y for x, y, idx in minibatches])
        all_z = featurizer(all_x)
        all_z = all_z.view(T, bs, -1)
        cls_loss = 0.0
        cls_acc = 0.0
        for i in range(T):
            pred = classifier(all_z[i], i)
            cls_loss += F.cross_entropy(pred, all_y[i*bs:(i+1)*bs])
            cls_acc += accuracy(pred, all_y[i*bs:(i+1)*bs])[0]
        cls_loss /= T
        cls_acc /= T

        loss = cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), all_x.size(0))
        cls_accs.update(cls_acc.item(), all_x.size(0))

        if iter % args.print_freq == 0:
            progress.display(iter)

def validate(vanilla_loaders, featurizer, classifier, args):
    featurizer.eval()
    classifier.eval()
    T = len(vanilla_loaders)
    total_samples = 0
    total_correct = 0
    for i in range(T):
        for x, y, _ in vanilla_loaders[i]:
            x,y = x.float().to(device), y.to(device)
            z = featurizer(x)
            pred = classifier(z,i)
            correct = (pred.argmax(dim=1) == y).sum().item()
            total_correct += correct
            total_samples += y.size(0)
    accuracy = total_correct / total_samples
    return accuracy

def main(args):
    output_directory = 'results/pretrain_outputs-{}'.format(args.dataset)
    model_directory = 'models/pretrain_models-{}'.format(args.dataset)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)

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
    train_iterator, test_loader, vanilla_loaders, _ = dataset_preparation(args, args.num_tasks, num_instances)

    # Define models
    if args.dataset == 'MNIST':
        featurizer = ConvExtractor()
        classifier = ImageClassifier(args.num_tasks - 1, 256, 10)
    elif args.dataset in ['Moons','ONP', 'Elec2', 'Shuttle']:
        featurizer = LinearExtractor(args.input_dim, args.feature_dim, args.num_layers, args.mlp_dropout)
        classifier = SingleLayerClassifier(args.feature_dim, args.num_classes, args.num_tasks - 1)

    featurizer.to(device)
    classifier.to(device)

    # Define optimizers
    params = list(featurizer.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    # Training
    best_acc = 0.0
    for epoch in range(args.epochs):
        starting_time = time.time()
        train(train_iterator, featurizer, classifier, optimizer, epoch, args)
        acc = validate(vanilla_loaders, featurizer, classifier, args)
        print("current epoch {}, acc {}".format(epoch, acc))
        if acc >= best_acc:
            best_acc = acc
            torch.save({
                'featurizer_state_dict': featurizer.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
            }, os.path.join(model_directory, f'bestmodel.pt'))
        print("current best = {:3.5f}".format(best_acc))
        ending_time = time.time()
        print("Traing time for epoch {}: {}".format(epoch, ending_time - starting_time))

    print("finish pretraining...")
    print("pretrained model saved in ", model_directory)
    print("best_acc = {:3.5f}".format(best_acc))


if __name__ == "__main__":
    print("Start Pre-Training...")

    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    main(args)


