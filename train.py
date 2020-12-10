#!/usr/bin/env python
# coding: utf-8

# In[30]:


import os
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# In[2]:


seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None, type=str, required=True)
parser.add_argument("--sample_path", default=None, type=str, required=True)
parser.add_argument("--model_path", default=None, type=str, required=True)
parser.add_argument("--dataset", default=None, type=str, required=True)
parser.add_argument("--batch_size", type=int, required=False, default=32)
parser.add_argument("--image_size", type=int, required=False, default=28)
parser.add_argument("--sample_size", type=int, required=False, default=100)
parser.add_argument("--epochs", type=int, required=False, default=100)
parser.add_argument("--log_step", type=int, required=False, default=50)
parser.add_argument("--sample_step", type=int, required=False, default=100)
parser.add_argument("--b1", type=float, required=False, default=0.5)
parser.add_argument("--b2", type=float, required=False, default=0.999)
parser.add_argument("--lr_D", type=float, required=False, default=1e-3)
parser.add_argument("--lr_G", type=float, required=False, default=2e-4)
parser.add_argument("--coeff", type=float, required=False, default=1.0)
parser.add_argument("--continuous_weight", type=float, required=False, default=0.5)

args = parser.parse_args()
eps=1e-10

if args.dataset == "CelebA":
    args.image_size = 64
    args.epochs = 4
    args.continuous_weight = 1.0

transform = transforms.Compose([
    transforms.Resize(args.image_size), 
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor()])
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if args.dataset == "MNIST":
    dataset = datasets.MNIST(os.path.join(args.data_dir, args.dataset), train="train", download=True, transform=transform)
else:
    dataset = datasets.ImageFolder(os.path.join(args.data_dir, args.dataset), transform=transform)
    
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

#### DEFINE ####
# dim_cat_code = dimension of the categorical codes
# n_cat_code = number of the categorical codes
# n_conti_code = number of the continuous latent codes
# n_z = number of noise variables

dim_cat_code = 10
if args.dataset == "MNIST":
    n_cat_code = 1
    n_conti_code = 2
    n_z = 62 
elif args.dataset == "SVHN":
    n_cat_code = 4
    cn_conti_code = 4
    n_z = 124
elif args.dataset == "CelebA":
    n_cat_code = 10
    n_conti_code = 10
    n_z = 128

def init_weights(model):
    if type(model) == nn.ConvTranspose2d or type(model) == nn.Conv2d:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif type(model) == nn.BatchNorm2d:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def gaussian_logli(x1, x2):
    std = torch.std(x2)
    m = torch.mean(x2)
    e = (x1 - m)/ (std + eps)
    pi = torch.acos(torch.zeros(1)) * 2
    result = torch.sum(-0.5*torch.log(2*pi.to(device)) - torch.log(std.to(device)+eps)-0.5*(e.to(device)**2),1)
    return result

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def get_continuous_code(batch_size, n_conti_code):
    #return torch.rand(batch_size, n_conti_code)*2.0 - 1.0
    return torch.rand(batch_size, n_conti_code)

def get_categorical_code(batch_size, n_cat_code, dim_cat_code):
    categorical_code = torch.zeros(batch_size, n_cat_code, dim_cat_code)
    for i in range(n_cat_code):
        idx = torch.randint(dim_cat_code, size=(batch_size,)) # [{0~dim_cat_code}] * batch_size
        categorical_code[torch.arange(0, batch_size), i, idx] = 1.0
    categorical_code = categorical_code.view(batch_size, -1)
    return categorical_code

from models import Generator, Discriminator
generator = Generator(n_z, n_cat_code, n_conti_code, dim_cat_code).to(device)
generator.apply(init_weights)
print(generator)
discriminator = Discriminator(n_z, n_cat_code, n_conti_code, dim_cat_code).to(device)
discriminator.apply(init_weights)
print(discriminator)

g_optimizer = optim.Adam(generator.parameters(), args.lr_D, [args.b1, args.b2])
d_optimizer = optim.Adam(discriminator.parameters(), args.lr_G, [args.b2, args.b2])

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
noise_fixed = torch.zeros((args.sample_size, n_z)).to(device)
total_steps = len(dataloader)
loss_dict = {"G":[], "D":[], "I_cat":[], "I_conti":[], "L1":[]}
cross_ent = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    for i, images in enumerate(dataloader):
        if args.dataset == "MNIST":
            images = images[0].to(device)
        else:
            images = images.to(device)
        
        b = images.shape[0]
        noise = torch.randn(b, n_z).to(device)
        conti_code = get_continuous_code(b, n_conti_code).to(device)
        cat_code = get_categorical_code(b, n_cat_code, dim_cat_code).to(device)
        fake_images = generator(torch.cat((noise, conti_code, cat_code), 1).to(device)) # G(z)
        d_output_real = discriminator(images) # D(x)
        d_output_fake = discriminator(fake_images) # D(G(z))
        
        # V(D,G) = E(logD(x)) + E(log(1-D(G(z))))
        d_loss_ = -torch.mean(torch.log(d_output_real[:,0] + eps) + torch.log(1. - d_output_fake[:,0] + eps))
        g_loss_ = -torch.mean(torch.log(d_output_fake[:,0] + eps))
 
        output_conti_code = d_output_fake[:, -n_conti_code:]
        output_cat_code = d_output_fake[:, n_cat_code:n_cat_code*dim_cat_code+1]
        
        H_cat = torch.mean(-torch.sum(cat_code*torch.log(cat_code + eps), 1)) # logQ(c')
        H_cat_G = torch.mean(-torch.sum(cat_code*torch.log(output_cat_code + eps), 1)) # logQ(c'|x)
        I_cat_G = H_cat - H_cat_G
        
        # CrossEntropy between Gaussian distribution.
        #H_conti = torch.mean(-torch.sum(conti_code*torch.log(conti_code + eps), 1))
        H_conti = torch.mean(-gaussian_logli(conti_code, conti_code))
        #H_conti_G = torch.mean(-torch.sum(conti_code*torch.log(output_conti_code + eps), 1))
        H_conti_G = torch.mean(-gaussian_logli(conti_code, output_conti_code))
        I_conti_G = H_conti - H_conti_G

        #I_c_G = I_cat_G + args.continuous_weight * I_conti_G
        I_c_G = I_cat_G + I_conti_G

        d_loss = d_loss_ - (args.coeff * I_c_G)
        discriminator.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        g_loss = g_loss_ - (args.coeff * I_c_G)
        generator.zero_grad()
        g_loss.backward(retain_graph=True)
        g_optimizer.step()
        
        # print the log info
        if (i + 1) % args.log_step == 0:
            print('Epoch [%d/%d], Step[%d/%d], D_loss: %.4f, G_loss: %.4f, GAN_D_loss: %.4f, GAN_G_loss:%.4f, I_c_G: %.4f, I_cat_G: %.4f, I_conti_G:%.4f' % (epoch + 1, args.epochs, i + 1, total_steps, d_loss.item(), g_loss.item(), d_loss_.item(), g_loss_.item(), I_c_G.item(), I_cat_G.item(), I_conti_G.item()))
            loss_dict["G"].append(g_loss.item())
            loss_dict["D"].append(d_loss.item())
            loss_dict["I_cat"].append(I_cat_G.item())
            loss_dict["I_conti"].append(I_conti_G.item())
            loss_dict["L1"].append((args.coeff * I_c_G).item())
            
        if (i + 1) % args.sample_step == 0:
            conti_code = get_continuous_code(args.batch_size, n_conti_code).to(device)
            cat_code = get_categorical_code(args.batch_size, n_cat_code, dim_cat_code).to(device)
            
            fake_images = generator(torch.cat((noise, conti_code, cat_code), 1).to(device))
            torchvision.utils.save_image(fake_images.data,
                                         os.path.join(args.sample_path,
                                                      'generated-%d-%d.png' % (epoch + 1, i + 1)), nrow=10)

# save the model parameters for each epoch
g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
torch.save(generator.state_dict(), g_path)

import json
f = open(os.path.join(args.model_path, "loss.json"), "w")
json.dump(loss_dict, f)
