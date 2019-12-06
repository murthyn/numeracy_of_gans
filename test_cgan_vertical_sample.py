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
parser.add_argument("--name", type=str, default="we_trained", help="name of training (refer to cgan_updated.py)")
parser.add_argument("--sample", type=bool, default=True, help="whether to sample images from generator")
parser.add_argument("--inception_batch_size", type=int, default=10, help="inception batch size")

parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=50, help="number of classes for dataset (if use_10, then will be set to 10)")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--embedding_size", type=int, default=50, help="size of embedding layer")
parser.add_argument("--n_discriminator", type=int, default=1, help="train discriminator every n_discriminator iterations")
parser.add_argument("--n_generator", type=int, default=1, help="train generator every n_discriminator iterations")
parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "Wasserstein"], help="adversarial loss: MSE or Wasserstein")
parser.add_argument("--use_10", type=bool, default=False, help="use 10 classes or test on all 50")
parser.add_argument("--clip_value", type=float, default=0.01, help="clipping discriminator weights for Wassertein loss")
parser.add_argument("--interpolate", type=str, default="True", choices=["True", "False"], help="whether to dropout 5 of the labels 0-49 from training, otherwise use all of 0-49, set to False if use_10=True")
parser.add_argument("--use_word_embedding", type=str, default="True", choices=["True", "False"], help="whether to use word embeddings or one-hot encoding of the word")
opt = parser.parse_args()

# get training conditions
#n_epochs, batch_size, lr, n_discriminator, loss, n_classes = opt.name.split("_")
#n_epochs, batch_size, lr, n_discriminator, n_classes = int(n_epochs), int(batch_size), float(lr), int(n_discriminator), int(n_classes)

opt.interpolate = opt.interpolate == "True"
opt.use_word_embedding = opt.use_word_embedding == "True"


img_shape = (opt.channels, opt.img_size, opt.img_size) if opt.n_classes == 10 else (opt.channels, opt.img_size, opt.img_size*2)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if opt.use_word_embedding:
    digit_embeddings = np.load("digit_embeddings.npy")
else:
    digit_embeddings = np.eye(100)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        if opt.use_word_embedding:
            self.label_emb = nn.Linear(50, opt.embedding_size)
        else:
            self.label_emb = nn.Linear(100, opt.embedding_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.embedding_size, 128, normalize=False),
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
        
        if opt.use_word_embedding:
            self.label_embedding = nn.Linear(50, opt.embedding_size)
        else:
            self.label_embedding = nn.Linear(100, opt.embedding_size)

        self.model = nn.Sequential(
            nn.Linear(opt.embedding_size + int(np.prod(img_shape)), 512),
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

# load generator, discriminator, CNN and digit embeddings
generator = torch.load("images/" + str(opt.name) + "/generator.pt")
discriminator = torch.load("images/" + str(opt.name) + "/discriminator.pt")


# set both models to eval mode
generator.eval()
discriminator.eval()

# if cuda is available, use it
if cuda:
    generator.cuda()
    discriminator.cuda() 

def sample_images(numbers, times=1):
    """Saves a grid of generated digits in numbers"""
    gen_imgs_total = None
    for i in range(times):
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (15, opt.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([i for i in numbers for _ in range(3)])
        gen_labels = Variable(FloatTensor(digit_embeddings[labels]))
        gen_imgs = generator(z, gen_labels)
        
        if opt.sample and i == 0:
            save_image(gen_imgs.data, "images/" + str(opt.name) + "/test_" + str(numbers) + ".png", nrow=3, padding=3, pad_value =1,normalize=True)
        
        if gen_imgs_total is None:
            gen_imgs_total = gen_imgs
        else:
            gen_imgs_total = torch.cat((gen_imgs_total, gen_imgs), 0)

    return gen_imgs_total

total_numbers = [i for i in range(0,60)]
#for i in range(6):

sample_images([0,13,25,40,47])
sample_images([2,24,27,45,48])
sample_images([50,51,52,53,54])




