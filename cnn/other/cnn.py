import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

import numpy as np

import os
import glob


def imshow(inp):
    """Imshow the Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

# imshow(valid[50][0])


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 2
batch_size = 1
learning_rate = 0.001
num_workers = 2

# Transforms
simple_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# Dataset
train_dataset = datasets.ImageFolder('static/train/', simple_transform)
valid_dataset = datasets.ImageFolder('static/valid/', simple_transform)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers)


# Convolutional neural network (ResNet18)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(
    model_ft.parameters(), lr=learning_rate, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, labels = data

        # if is_cuda:
        #     inputs = Variable(inputs.cuda())
        #     labels = Variable(labels.cuda())
        # else:
        #     inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # Forward pass
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer_ft.step()

        if (i+1) % 100 == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, i+1, total_step, loss.item()))
