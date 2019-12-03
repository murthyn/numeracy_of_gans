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
opt = parser.parse_args()


# load generator, discriminator and digit embeddings
generator = torch.load("images/" + str(opt.name) + "/generator.pt")
discriminator = torch.save("images/" + str(opt.name) + "/discriminator.pt")
digit_embeddings = np.load("digit_embeddings.npy")

# set both models to eval mode
generator.eval()
discriminator.eval()

# if cuda is available, use it
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    discriminator.cuda()

# get training conditions
n_epochs, batch_size, lr, n_discriminator, loss, n_classes = opt.name.split("_")

latent_dim = 100 # NEED TO MANUALLY CHANGE

def sample_images(numbers):
    """Saves a grid of generated digits in numbers"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (10*10, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(10) for num in numbers])
    gen_labels = Variable(FloatTensor(digit_embeddings[labels]))
    gen_imgs = generator(z, gen_labels)
    save_image(gen_imgs.data, "images/" + "test_" + str(numbers) + ".png", nrow=10, normalize=True)

total_numbers = [i for i in range(0,70)]

for i in range(7):
    sample_images(total_numbers[10*i:10*i+10])

