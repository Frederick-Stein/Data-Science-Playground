import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from  pathlib import Path
from timeit import default_timer as timer
from tqdm import tqdm ## progress bar


generator1 = torch.Generator().manual_seed(42)
# 1) Datasets
train_data = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

image, label = next(iter(test_data))
print(f"Sample size of Train data: {len(train_data)}, image: {image.size()}, label: {label}")
plt.imshow(image.squeeze(), cmap="gray")
class_name = train_data.classes
print(f"Class names: {class_name}")

# 2) Split data
split_ratio = 0.8
train_size = int(split_ratio * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size], generator=generator1)

# 3) Set device and loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pin = (device.type == 'cuda')
print(f'Device: {device}')

Batch_size = 64

train_loader = DataLoader(train_data, batch_size = Batch_size, shuffle = True, pin_memory = pin)
test_loader = DataLoader(test_data, batch_size = Batch_size, shuffle = False, pin_memory = pin)


## Define a timer
def train_time(start, end, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")


## Construct CNN
class ConvelutionalNN(nn.Module):

    def __init__(self, in_ch, hidden_ch, out_ch):

        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )

        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_ch * 7 * 7, out_ch)
        )

    def forward(self, x):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

## test model
model = ConvelutionalNN(1, 16, 10)
# Get a batch of data from the train loader
image, label = next(iter(train_data))
image = image.unsqueeze(0)
model(image)


## train model
model = ConvelutionalNN(1, 12, 10).to(device)

loss_fcn = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 1e-4)
optimizer = optim.Adam(model.parameters(), weight_decay = 1e-5)

epochs = 10

train_error = []
test_error = []
train_accuracy = []
test_accuracy = []
start = timer()

for epoch in tqdm(range(1, epochs + 1)):

    model.train()
    train_correct = 0
    train_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        loss = loss_fcn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        predicted = y_pred.argmax(dim = 1)
        train_correct += (predicted == y_batch).sum().item()

    train_loss = train_loss / len(train_data)
    train_acc = train_correct / len(train_data) * 100

    model.eval()
    with torch.inference_mode():

        test_loss = 0
        test_correct = 0

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            loss = loss_fcn(y_pred, y_batch)
            test_loss += loss.item() * len(X_batch)

            predicted = y_pred.argmax(dim = 1)
            test_correct += (predicted == y_batch).sum().item()

        test_loss = test_loss / len(test_data)
        test_acc = test_correct / len(test_data) * 100

        train_error.append(train_loss)
        test_error.append(test_loss)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

        if epoch % 1 == 0:
            print(f"Epoch: {epoch} | Train loss: {train_loss: .4f} | Test loss: {test_loss: .4f} | Test accuracy: {test_acc:.2f}%")


end = timer()
train_time(start, end, device)


## plot error
x = np.arange(1, epochs +1)
plt.plot(x, train_error, label = "Train Error")
plt.plot(x, test_error, label = "Test Error")
plt.legend()


## plot accuracy
plt.plot(x, train_accuracy, label = "Train Error")
plt.plot(x, test_accuracy, label = "Test Error")
plt.legend()
