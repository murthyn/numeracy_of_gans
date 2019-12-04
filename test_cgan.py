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
parser.add_argument("--name", type=str, default="None", help="name of training (refer to cgan_all.py)")
parser.add_argument("--sample", type=bool, default=True, help="whether to sample images from generator")
opt = parser.parse_args()

# get training conditions
n_epochs, batch_size, lr, n_discriminator, loss, n_classes = opt.name.split("_")
n_epochs, batch_size, lr, n_discriminator, n_classes = int(n_epochs), int(batch_size), float(lr), int(n_discriminator), int(n_classes)

latent_dim = 100 # NEED TO MANUALLY CHANGE
img_size = 32
channels = 1
img_shape = (channels, img_size, img_size) if n_classes == 10 else (channels, img_size, img_size*2)
embedding_size = 50

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Linear(embedding_size, embedding_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + embedding_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Linear(embedding_size, embedding_size)

        self.model = nn.Sequential(
            nn.Linear(embedding_size + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

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

# load generator, discriminator, CNN and digit embeddings
generator = torch.load("images/" + str(opt.name) + "/generator.pt")
discriminator = torch.load("images/" + str(opt.name) + "/discriminator.pt")
digit_embeddings = np.load("digit_embeddings.npy")
net = ConvNet()
net.load_state_dict(torch.load('model.ckpt'))

# set both models to eval mode
generator.eval()
discriminator.eval()
net.eval()

# if cuda is available, use it
if cuda:
    generator.cuda()
    discriminator.cuda()
    net.cuda()

def inception_score(images, batch_size=10):
    scores = []
    for i in range(int(math.ceil(float(len(images)) / float(batch_size)))):
        batch = Variable(torch.cat(images[i * batch_size: (i + 1) * batch_size], 0))
        s = net(batch)  # skipping aux logits
        scores.append(s)
    p_yx = F.softmax(torch.cat(scores, 0), 1)
    p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
    KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
    final_score = KL_d.mean()
    return final_score

def sample_images(numbers):
    """Saves a grid of generated digits in numbers"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (10*10, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(10) for num in numbers])
    gen_labels = Variable(FloatTensor(digit_embeddings[labels]))
    gen_imgs = generator(z, gen_labels)
    if opt.sample:
        save_image(gen_imgs.data, "images/" + str(opt.name) + "/test_" + str(numbers) + ".png", nrow=10, normalize=True)
    return gen_imgs

total_numbers = [i for i in range(0,70)]
for i in range(7):
    gen_imgs = sample_images(total_numbers[10*i:10*i+10])
    print("inception score for " + str(total_numbers[10*i:10*i+10]) + " is ", inception_score(gen_imgs))

    # SAMPLE MORE THAN TEN PER SCORE






