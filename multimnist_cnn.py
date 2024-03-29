#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from matplotlib import pyplot as plt

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs")
opt = parser.parse_args()


x = np.load("data_60/xtrain32.npy")
y = np.load("data_60/ytrain.npy")
cuda = True if torch.cuda.is_available() else False
x = torch.Tensor(x).to(torch.int8)
y = torch.Tensor(y).to(torch.int8)

# shuffle x and y
random_perm = torch.randperm(x.shape[0])
x = x[random_perm]
y = y[random_perm]

x_train = x[:300000]
y_train = y[:300000]
x_test = x[300000:]
y_test = y[300000:]

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

num_classes = 60
learning_rate = 0.001
img_size = 32


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        x = Image.fromarray(x.numpy(), mode='L')
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class ConvNet(nn.Module):
    def __init__(self, num_classes=50):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4*4*64, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out 


model = ConvNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if cuda:
    model.cuda()
    criterion.cuda()

# Configure data loader
train_loader = torch.utils.data.DataLoader(
    CustomTensorDataset(tensors = (x_train, y_train), transform = transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
       )),
    batch_size=opt.batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    CustomTensorDataset(tensors = (x_test, y_test), transform = transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
       )),
    batch_size=opt.batch_size,
    shuffle=True,
)

total_step = len(train_loader)
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        batch_size = imgs.shape[0]

        imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, opt.n_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for imgs, labels in test_loader:

        imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 50000 test images: {} %'.format(100 * correct / total))


torch.save(model.state_dict(), 'model_60.ckpt')





