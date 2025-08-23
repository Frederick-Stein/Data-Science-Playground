import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms
from  pathlib import Path
# from torchmetrics.classification import Accuracy
from timeit import default_timer as timer

# 1) Transforms (tensor + normalize to MNIST stats)
generator1 = torch.Generator().manual_seed(42)
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),
                              ])

# 2) Datasets (downloaded to ./data)
train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
class_name = train_data.classes

print(f"Train data shape: {train_data}")

# 3) Split data into train and validation
train_data, val_data = random_split(train_data, [55000, 5000], generator=torch.Generator().manual_seed(41))

# 4) Set device and loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
pin = (device.type == "cuda")

train_loader = DataLoader(train_data, batch_size = 128, shuffle = True, pin_memory = pin)
val_loader = DataLoader(val_data, batch_size = 128, shuffle = True, pin_memory = pin)
test_loader = DataLoader(test_data, batch_size = 128, shuffle = True, pin_memory = pin)

## Define a timer
def train_time(start, end, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")


# class ClassificationNN(nn.Module):

#     def __init__(self, sizes):

#         super().__init__()

#         self.num_layers = len(sizes)
#         self.sizes = sizes

#         self.weights = nn.ParameterList([nn.Parameter(torch.randn(y, x)) for x, y in zip(sizes[:-1], sizes[1:])])
#         self.biases = nn.ParameterList([nn.Parameter(torch.randn(y)) for y in sizes[1:]])

#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         for i in range(self.num_layers - 1):
#             y = torch.mm(x, self.weights[i].T) + self.biases[i]
#             if i != self.num_layers - 2:
#                 # Sigmoid activation for all layers
#                 x = torch.sigmoid(y)

#                 # ReLU activation for all layers except the last one
#                 # x = F.relu(y)
#             else:
#                 x = y

#         return x

# class ClassificationNN(nn.Module):

#     def __init__(self, sizes):

#         super().__init__()

#         self.num_layers = len(sizes)
#         self.sizes = sizes

#         self.layers = nn.ModuleList([nn.Linear(x, y) for x, y in zip(sizes[:-1], sizes[1:])])
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         for layer in self.layers[:-1]:
#             # x = torch.sigmoid(layer(x))  # hidden layers
#             # x = self.relu(layer(x))
#             x = self.tanh(layer(x))

#         x = self.layers[-1](x)  # last layer (logits)
#         return x


class ClassificationNN(nn.Module):

    def __init__(self, sizes):

        super().__init__()

        self.activiation = nn.ReLU()
        # self.activiation = nn.Sigmoid()
        # self.activiation = nn.Tanh()

        layers = [nn.Flatten()]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i != len(sizes) - 2:
                layers.append(self.activiation)
        self.layer_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer_stack(x)

## test
X_0, y_0 = next(iter(train_loader))
print(X_0.shape, y_0)
device
model = ClassificationNN([784, 30, 10]).to(device)
X_0= X_0.to(device)
model(X_0)

## Move the model to the device
model = ClassificationNN([784, 30, len(class_name)]).to(device)

## Loss function
loss_fcn = nn.CrossEntropyLoss()

## Optimization method
# optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay= 1e-4)


## Metric setup
# accuracy_metric = Accuracy(task="multiclass", num_classes=10).to(device)

epochs = 20
train_error = []
test_error = []
train_accuracy = []
test_accuracy = []
epoch_range = []

train_time_start = timer()

for epoch in range(1, epochs+1):

    model.train()
    train_loss = 0
    correct_train = 0
    # accuracy_metric.reset()

    for X_batch, y_batch in train_loader:

        # Move data to device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Feedforward
        y_pred = model(X_batch)

        # Compute loss
        loss = loss_fcn(y_pred, y_batch)

        # Initialize optimizer
        optimizer.zero_grad()

        # Back-propogation
        loss.backward()

        # Gradient descent
        optimizer.step()

        train_loss += loss.detach().item() * X_batch.size(0)
        predicted = y_pred.argmax(dim=1)
        correct_train += (predicted == y_batch).sum().item()
        # accuracy_metric.update(predicted, y_batch)

    train_loss /= len(train_data)
    train_acc = correct_train / len(train_data) * 100 # train accuracy
    # train_acc = accuracy_metric.compute().item() * 100

    model.eval()
    with torch.inference_mode():
        correct_test = 0 # accuracy
        test_loss = 0
        # accuracy_metric.reset()


        for X_batch, y_batch in test_loader:
            # Move data to device
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Training process
            y_pred = model(X_batch)

            loss = loss_fcn(y_pred, y_batch)
            test_loss += loss.item() * X_batch.size(0)

            # Calculate accuracy
            predicted = y_pred.argmax(dim=1)
            correct_test += (predicted == y_batch).sum().item()
            # accuracy_metric.update(predicted, y_batch)

        test_loss /= len(test_data)
        test_acc = correct_test / len(test_data) * 100
        # test_acc = accuracy_metric.compute().item() * 100

        train_error.append(train_loss)
        test_error.append(test_loss)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        epoch_range.append(epoch)

        if epoch % 5 == 0:
            print(f"Epoch: {epoch} | Train loss: {train_loss: .4f} | Test loss: {test_loss: .4f} | Test accuracy: {test_acc:.2f}%")

train_time_end = timer()
train_time(train_time_start, train_time_end, device)


# Plot error
plt.plot(epoch_range, train_error, label = "Train error")
plt.plot(epoch_range, test_error, label = "Test error")
plt.legend()

# Plot accuracy
plt.plot(epoch_range, train_accuracy, label = "Train accuracy")
plt.plot(epoch_range, test_accuracy, label = "Test accuracy")
plt.legend()
