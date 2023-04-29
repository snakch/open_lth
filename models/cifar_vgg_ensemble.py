import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """
    A model which wraps an ensemble of VGG-style neural network designed for CIFAR-10.
    """

    class ConvModule(nn.Module):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters):
            super(Model.ConvModule, self).__init__()
            self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_filters)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    def __init__(self, plan, initializer, outputs=10):
        super(Model, self).__init__()

        layers = []
        filters = 3

        for spec in plan:
            if spec == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Model.ConvModule(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, outputs)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)
