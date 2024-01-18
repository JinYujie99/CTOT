# -*- coding: utf-8 -*-
import sys
import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

sys.dont_write_bytecode = True


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == 'Uniform':
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise


def dataset_preparation(args, num_tasks=10, num_instance=220):

    if args.dataset in ['Elec2','HousePrice','M5Hobby','M5Household','Energy','Shuttle']:
        # A = np.load('data/{}/processed/A.npy'.format(args.dataset))
        # U = np.load('data/{}/processed/U.npy'.format(args.dataset))
        X = np.load('data/{}/X.npy'.format(args.dataset))
        Y = np.load('data/{}/Y.npy'.format(args.dataset))
    else:
        # A = np.load('data/{}/processed/A.npy'.format(args.dataset))
        # U = np.load('data/{}/processed/U.npy'.format(args.dataset))
        X = np.load('data/{}/processed/X.npy'.format(args.dataset))
        Y = np.load('data/{}/processed/Y.npy'.format(args.dataset))

    if args.dataset == 'Shuttle':
        # # sort them by the first feature (time)
        # concatenated_array = np.concatenate([X, Y[:, np.newaxis]], axis=1)  # Y[:, np.newaxis] makes Y a column vector
        # sorted_indices = np.argsort(concatenated_array[:, 0])
        # sorted_array = concatenated_array[sorted_indices]
        # X = sorted_array[:, :-1]
        # Y = sorted_array[:, -1]
        Y[Y != 0] = 1
    
    train_loaders = []
    vanilla_loaders = []
    source_domains = []

    if args.dataset == 'Moons':
        intervals = np.arange(num_tasks+1)*num_instance
    elif args.dataset == 'MNIST':
        intervals = np.array([0, 1000, 2000, 3000, 4000, 5000])
    elif args.dataset == 'ONP':
        intervals = np.array([0,7049,13001,18725,25081,32415,39644])
    elif args.dataset == 'Elec2':
        intervals = np.array([0,670,1342,2014,2686,3357,4029,4701,5373,6045,6717,7389,8061,8733,
            9405,10077,10749,11421,12093,12765,13437,14109,14781,15453,16125,16797,17469,18141,18813,
            19485,20157,20829,21501,22173,22845,23517,24189,24861,25533,26205,26877,27549])
    elif args.dataset == 'HousePrice':
        intervals = np.array([0,2119,4982,8630,12538,17079,20937,22322])
    elif args.dataset == 'M5Hobby':
        intervals = np.array([0,323390,323390*2,323390*3,997636])
    elif args.dataset == 'M5Household':
        intervals = np.array([0,124100,124100*2,124100*3,382840])
    elif args.dataset == 'Energy':
        intervals = np.array([0,2058,2058+2160,2058+2*2160,2058+3*2160,2058+4*2160,2058+5*2160,2058+6*2160,2058+7*2160,19735])
    elif args.dataset == 'Shuttle':
        start = 7250
        span = 7250
        intervals = np.array([0, start, start + span, start + 2*span, start + 3*span, start + 4*span, start + 5*span, start + 6*span, start + 7*span])

    for i in range(len(intervals)-1):
        temp_X = X[intervals[i]:intervals[i+1]]
        temp_Y = Y[intervals[i]:intervals[i+1]]
        temp_idx = [i+1] * (intervals[i+1]-intervals[i]) #start from 0
        domain_dataset = DomainDataset(temp_X, temp_Y, temp_idx) # create dataset for each domain
        if i != len(intervals)-2:
            temp_dataloader = InfiniteDataLoader(domain_dataset, weights=None, batch_size=args.batch_size, num_workers=args.num_workers)
            vanilla_dataloader = DataLoader(domain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            train_loaders.append(temp_dataloader)
            vanilla_loaders.append(vanilla_dataloader)
            source_domains.append(domain_dataset)
        else:
            test_loader = DataLoader(domain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_minibatches_iterator = zip(*train_loaders)

    return train_minibatches_iterator, test_loader, vanilla_loaders, source_domains



class DomainDataset(Dataset):
    """ Customized dataset for each domain"""
    def __init__(self, X, Y, domain_idx):
        self.X = X                           # set data
        self.Y = Y                           # set lables
        self.domain_idx = domain_idx

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx], self.domain_idx[idx]]    # return list of batch data [data, labels]


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                                                             replacement=True,
                                                             num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                                                     replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""

    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length

def cal_dist(x, y):
    m = x.size(0)
    d = x.size(1)
    n = y.size(0)

    x = x.unsqueeze(1).expand(m, n, d)
    y = y.unsqueeze(0).expand(m, n, d)
    return -torch.pow(x - y, 2).sum(2)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res