import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data as data

import torch.optim as optim
import torch.nn as nn





def validate(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        val_loss = criterion(outputs, labels).item()

    model.train()
    return val_acc, val_loss
