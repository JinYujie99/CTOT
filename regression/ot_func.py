import torch
import numpy as np
import ot
import scipy.spatial.distance

def cal_w_transport(X1, Y1, X2, Y2):
    a = ot.unif(X1.shape[0], type_as=X1)
    b = ot.unif(X2.shape[0], type_as=X1)
    M = ot.dist(X1, X2, metric='sqeuclidean') + label_distance(Y1, Y2)
    T = ot.emd(a, b, M, numItermax=1000000)
    return T

def cal_w_transport_reg(X1, Y1, X2, Y2, args):
    a = ot.unif(X1.shape[0], type_as=X1)
    b = ot.unif(X2.shape[0], type_as=X1)
    M = ot.dist(X1, X2, metric='sqeuclidean') + args.alpha * ot.dist(Y1.unsqueeze(-1), Y2.unsqueeze(-1), metric='sqeuclidean')
    T = ot.emd(a, b, M, numItermax=1000000)
    return T

def label_distance(Y1, Y2):
    C = torch.eq(Y1.view(-1, 1), Y2.view(1, -1)).int()
    C = (1-C)*100000
    return C






