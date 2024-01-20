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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):

    def __init__(self, block, layers, **kwargs):
        super(Bottleneck, self).__init__()

        self.in_channels = 16
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)

        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1])
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.layer4 = self.make_layer(block, 128, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = F.relu(out)

        out = self.dropout(out)
        out = self.layer2(out)
        out = F.relu(out)

        out = self.dropout(out)
        out = self.layer3(out)
        out = F.relu(out)

        out = self.dropout(out)
        out = self.layer4(out)
        out = F.relu(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

def ConvExtractor(**kwargs):
    return Bottleneck(ResidualBlock, [2,2,2,2], **kwargs)

class ImageClassifier(nn.Module):
    def __init__(self, num_domains, latent_dim=256, output_dim=10):
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.ModuleList([nn.Linear(128, latent_dim) for _ in range(num_domains)])
        self.fc2 = nn.ModuleList([nn.Linear(latent_dim, output_dim) for _ in range(num_domains)])

    def forward(self, x, domain_idx):
        out = self.fc1[domain_idx](x)
        out = F.relu(out)
        out = self.fc2[domain_idx](out)
        return out



