import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class Discriminator(nn.Module):
    def _init_(self,img_dim):
        super()._init_()
        self.disc = nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid() 
        )

    def forward(self,x):
        return self.disc(x)


class Generator(nn.Module):
    def _init_(self,z_dim,img_dim):
        super()._init_()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim),
            nn.Tanh() 
        )

    def forward(self,x):
        return self.gen(x)
    


