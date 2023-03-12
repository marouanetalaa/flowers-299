
import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data as data

import torch.optim as optim
import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms to apply to images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset from disk
dataset = datasets.ImageFolder('./data_flowers/', transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)




val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
