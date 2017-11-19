import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import os

# Hyper Parameters

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 100
X_dim = 784
y_dim = 10
h_dim = 128
cnt = 0
d_step = 3
lr = 1e-3
m = 5

# Load Data

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=mb_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=mb_size,
                                          shuffle=False)

# CNN-Based EBGAN

ngpu = 1
ngf = 4  # ngf = 64
nz = z_dim  # 100 --> batchsize, 100, 1, 1
nc = 1
# noise.data.resize_(batch_size, nz, 1, 1)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # b, nz, 1, 1
            nn.ConvTranspose2d(nz, 28 * 28, 1, stride=1, padding=0, bias=False),
            # b, 28*28, 1, 1
            nn.BatchNorm2d(28 * 28),
            nn.ReLU(True),
            nn.ConvTranspose2d(28 * 28, 14 * 14, 2, stride=2, padding=0, bias=False),
            # b, 14*14, 2, 2
            nn.BatchNorm2d(14 * 14),
            nn.ReLU(True),
            nn.ConvTranspose2d(14 * 14, 7 * 7, 2, stride=2, padding=0, bias=False),
            # b, 7*7, 4, 4
            nn.BatchNorm2d(7 * 7),
            nn.ReLU(True),
            nn.ConvTranspose2d(7 * 7, 1, 7, stride=7, padding=0, bias=False),
            # b. 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # b, nc, 28, 28
            nn.Conv2d(nc, ngf, 3, stride=1, padding=0),
            # b, ngf, 28, 28
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # b, ngf, 14, 14
            nn.Conv2d(ngf, ngf * 2, 3, stride=1, padding=0),
            # b, ngf*2, 14, 14
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # b, nfg*2, 7, 7
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=1, padding=0),
            # b, ngf*4, 7, 7
            nn.ReLU(True),
            # nn.MaxPool2d(7, stride=7),
            # b, ngf*4, 1, 1
            nn.Conv2d(ngf * 4, nz, 3, stride=1, padding=0),
            # b, nz, 7, 7
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            # b, nz, 7, 7
            nn.ConvTranspose2d(nz, ngf * 4, 1, 1, bias=False),
            # b, ngf*4, 7, 7
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 1, 1, bias=False),
            # b, ngf*2, 7, 7
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # b. ngf, 14, 14
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # b. nc, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Energy is the MSE of autoencoder


def D(X):
    X_recon = D_(X)
    return torch.mean(torch.sum((X - X_recon)**2, 1))


def reset_grad():
    G.zero_grad()
    D_.zero_grad()


D_ = _netD(ngpu)
G = _netG(ngpu)

G.cuda()
D_.cuda()

G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D_.parameters(), lr=lr)

# Training

# noise = torch.FloatTensor(mb_size, nz, 1, 1)
# noise.resize_(mb_size, nz, 1, 1).normal_(0, 1)
iter_train = ((i, images, labels)
              for i, (images, labels) in enumerate(train_loader))
for it in range(10000):
    try:
        (i, images, labels) = next(iter_train)
        # print(i, it)
    except StopIteration:
        iter_train = ((i, images, labels)
                      for i, (images, labels) in enumerate(train_loader))
        continue

    # Sample data
    Z = Variable(torch.randn(mb_size, z_dim, 1, 1)).cuda()
    #X = Variable(images.view(-1, 28*28)).cuda()
    X = Variable(images).cuda()  # 3d data (-1, 28, 28)

    # Dicriminator
    # Z = z.data.resize_(mb_size, nz, 1, 1)
    G_sample = G(Z)
    D_real = D(X)
    D_fake = D(G_sample)

    # EBGAN D loss. D_real and D_fake is energy, i.e. a number
    D_loss = D_real + F.relu(m - D_fake)

    # Reuse D_fake for generator loss
    D_loss.backward()
    D_solver.step()
    reset_grad()

    # Generator
    G_sample = G(Z)
    D_fake = D(G_sample)

    G_loss = D_fake

    G_loss.backward()
    G_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 100 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0]))

        samples = G(Z).cpu().data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        dir_n = 'out_cnn_ebgan/'
        if not os.path.exists(dir_n):
            os.makedirs(dir_n)

        plt.savefig(dir_n + '{}.png'.format(str(cnt).zfill(3)),
                    bbox_inches='tight')
        cnt += 1
        plt.close(fig)
