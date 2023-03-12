import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data as data

import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

class FullyConnected(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FullyConnected, self).__init__()
        self.classifier =  nn.Sequential(
            *linear_relu(input_size, 256),
            *linear_relu(256, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y.to(device)




