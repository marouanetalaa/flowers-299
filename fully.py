
import torch
import torch.nn as nn


import torch.optim as optim
import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter
import os

from validation import validate
from model import FullyConnected

from data import train_loader, val_loader




# Train the model
if __name__ == '__main__':
    log_dir = os.path.join("logs")
    writer = SummaryWriter(log_dir)


# Check if CUDA device is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    num_epochs = 10
    num_classes = 5

# Create a FullyConnected model with input size of 3*224*224 and output size of number of classes
    model = FullyConnected(3*224*224, num_classes).to(device)


# Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        # Train the model on the training set
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss /= train_total

        # Evaluate the model on the validation set
        val_acc, val_loss = validate(model, val_loader, criterion)

        # Log metrics to TensorBoard

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        print('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
              .format(epoch+1, num_epochs, val_loss, val_acc))
        # Save the best model based on validation accuracy
        if epoch == 0 or val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    writer.flush()
