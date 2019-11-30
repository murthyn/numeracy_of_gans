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
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=50, help="number of classes for dataset (if n_classes, then will be set to 10)")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--embedding_size", type=int, default=50, help="size of embedding layer")
parser.add_argument("--n_discriminator", type=int, default=1, help="train discriminator every n_discriminator iterations")
parser.add_argument("--loss", type=str, default="MSE", help="adversarial loss: MSE or Wasserstein")
parser.add_argument("--use_10", type=bool, default=False, help="use 10 classes or test on all 50")
opt = parser.parse_args()


x = np.load("data/xtrain32.npy")
y = np.load("data/ytrain.npy")
digit_embeddings = np.load("digit_embeddings.npy")

cuda = True if torch.cuda.is_available() else False

if opt.use_10:
    x = torch.Tensor(x[:60000, :, 28:]).to(torch.int8)
    y = torch.Tensor(y[:60000]).to(torch.int8)
    opt.n_classes = 10
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    numbers = [1, 22, 23, 26, 29, 33, 35, 37, 42, 48]
else:
    x = torch.Tensor(x)
    y = torch.Tensor(y).to(torch.int8)
    img_shape = (opt.channels, opt.img_size, opt.img_size*2)
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


name = str(opt.n_epochs)+"_"+str(opt.batch_size)+"_"+str(opt.lr)+"_"+str(opt.n_discriminator)+"_"+str(opt.loss)+"_"+str(opt.n_classes)
os.makedirs(os.path.join("images/", str(name)), exist_ok=True)


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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Linear(opt.embedding_size, opt.embedding_size)

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

        self.label_embedding = nn.Linear(opt.embedding_size, opt.embedding_size)

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


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    CustomTensorDataset(tensors = (x, y), transform = transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
       )),
    batch_size=opt.batch_size,
    shuffle=True,
)

total_d_loss = 0.0
total_g_loss = 0.0

# Optimizers
if opt.loss == "MSE":
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
else:
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)#, betas=(opt.b1, opt.b2))
    
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    """does not use n_row"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (10*10, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(10) for num in numbers])
    gen_labels = Variable(FloatTensor(digit_embeddings[labels]))
    gen_imgs = generator(z, gen_labels)
    save_image(gen_imgs.data, "images/" + str(name) + "/%d.png" % batches_done, nrow=10, normalize=True)


# ----------
#  Training
# ----------

loss_file = open("images/" + str(name) + "/loss_file.txt","w")

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(FloatTensor(digit_embeddings[labels]))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(FloatTensor(digit_embeddings[np.random.randint(0, opt.n_classes, batch_size)]))
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        if opt.loss == "MSE":
            g_loss = adversarial_loss(validity, valid)
        else:
            g_loss = -torch.mean(validity)
            
        total_g_loss += g_loss.item()
        
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if i%opt.n_discriminator==0:
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            if opt.loss == "MSE":
                d_loss = (d_real_loss + d_fake_loss) / 2
            else:
                d_loss = -torch.mean(validity_real)+torch.mean(validity_fake) 
            total_d_loss += d_loss.item()

            d_loss.backward()
            optimizer_D.step()
            
        if i%1000==0:
            print_stmt = "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader), total_d_loss/(1000/opt.n_discriminator), total_g_loss/1000)
            print(print_stmt)
            loss_file.write(print_stmt + "\n")
            total_d_loss = 0.0
            total_g_loss = 0.0

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=20, batches_done=batches_done)
            
loss_file.close()