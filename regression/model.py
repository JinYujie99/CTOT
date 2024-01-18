import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class LinearExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, mlp_dropout):
        super(LinearExtractor, self).__init__()
        self.num_layers = num_layers
        self.input = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(mlp_dropout)
        if num_layers > 1:
            self.hiddens = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)])

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        if self.num_layers > 1:
            for hidden in self.hiddens:
                x = hidden(x)
                x = self.dropout(x)
                x = F.relu(x)
        return x


class SingleLayerClassifier(nn.Module):
    def __init__(self, feature_dim, output_dim, num_domains):
        super(SingleLayerClassifier, self).__init__()
        self.fcs = nn.ModuleList([nn.Linear(feature_dim, output_dim) for _ in range(num_domains)])

    def forward(self, x, domain_idx):
        return self.fcs[domain_idx](x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(MLP, self).__init__()
        self.num_layers = args.mlp_layer
        self.hidden_dim = args.hidden_dim
        self.output_dim = output_dim

        if self.num_layers > 1:
            self.input = nn.Linear(input_dim, args.hidden_dim)
            self.dropout = nn.Dropout(args.mlp_dropout)
            self.hiddens = nn.ModuleList([
                nn.Linear(args.hidden_dim, args.hidden_dim)
                for _ in range(args.mlp_layer-2)])
            self.output = nn.Linear(args.hidden_dim, output_dim)
        else:
            self.input = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.input(x)
        if self.num_layers > 1:
            x = self.dropout(x)
            x = F.relu(x)
            for hidden in self.hiddens:
                x = hidden(x)
                x = self.dropout(x)
                x = F.relu(x)
            x = self.output(x)
        if self.output_dim > 1:
            x = F.softmax(x, dim=-1)
        else:
            x = F.sigmoid(x)
        return x
