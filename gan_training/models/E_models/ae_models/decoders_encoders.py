import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np


class MnistEncoder(nn.Module):
    def __init__(self, nc, img_sz, enc_output_dim, hidden_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(nc * img_sz * img_sz, 256),
            nn.ReLU(),
            nn.Linear(256, enc_output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x shape: (batch_sz, nc, img_sz, img_sz)
        x = x.view(x.size(0), -1)
        output = self.encoder(x)  # shape: (batch_sz, enc_output_dim)
        return output


class MnistDecoder(nn.Module):
    def __init__(self, nc, img_sz, latent_dim, enc_output_dim, hidden_dims):
        super().__init__()
        self.img_sz = img_sz
        self.nc = nc
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, nc * img_sz * img_sz),
        )
    
    def forward(self, x):
        # x shape: (batch_sz, latent_dim)
        x = self.decoder(x)
        x = x.view(x.size(0), self.nc , self.img_sz, self.img_sz)
        return x  # shape: (batch_sz, nc, img_sz, img_sz)


class FCEncoder(nn.Module):
    
    def __init__(self, nc, img_sz, enc_output_dim, hidden_dims):
        super(FCEncoder, self).__init__()
        self.input_dim = nc * img_sz * img_sz
        self.hidden_dims = hidden_dims
        self.hidden_dims.append(enc_output_dim)
        
        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {'linear0': nn.Linear(self.input_dim, hidden_dim),
                     'activation0': nn.ReLU()
                    })
            else:
                layers.update(
                    {'linear{}'.format(idx): nn.Linear(
                        self.hidden_dims[idx-1], hidden_dim),
                     'activation{}'.format(idx): nn.ReLU(),
                     'bn{}'.format(idx): nn.BatchNorm1d(self.hidden_dims[idx])
                    })
        self.encoder = nn.Sequential(layers)
    
    def forward(self, x):
        # x shape: (batch_sz, nc, img_sz, img_sz)
        x = x.view(x.size(0), -1)
        output = self.encoder(x)  # shape: (batch_sz, enc_output_dim)
        return output


class FCDecoder(nn.Module):
    
    def __init__(self, nc, img_sz, latent_dim, enc_output_dim, hidden_dims):
        super(FCDecoder, self).__init__()
        self.img_sz = img_sz
        self.nc = nc
        self.output_dim = nc * img_sz * img_sz
        self.hidden_dims = hidden_dims
        self.hidden_dims.append(latent_dim)
        self.dims_list = (hidden_dims + hidden_dims[:-1][::-1])  # mirrored structure, for the default values we get: [500, 500, 2000, 128, 2000, 500, 500]
        
        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]  # for list lst, lst[::-1] returns the mirror list of lst
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {'linear{}'.format(idx): nn.Linear(
                        hidden_dim, self.output_dim),
                    })
            else:
                layers.update(
                    {'linear{}'.format(idx): nn.Linear(
                        hidden_dim, tmp_hidden_dims[idx+1]),
                     'activation{}'.format(idx): nn.ReLU(),
                     'bn{}'.format(idx): nn.BatchNorm1d(tmp_hidden_dims[idx+1])
                    })
        self.decoder = nn.Sequential(layers)
    
    def forward(self, x):
        # x shape: (batch_sz, latent_dim)
        x = self.decoder(x)
        x = x.view(x.size(0), self.nc , self.img_sz, self.img_sz)
        return x  # shape: (batch_sz, nc, img_sz, img_sz)


class ConvEncoder(nn.Module):
    def __init__(self, nc, img_sz, enc_output_dim, hidden_dims=None):
        super().__init__()

        self.img_sz = img_sz

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 100, 3, 1, 1),  # (100,32,32)
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, 1, 1), # (100,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # (100,16,16)
            nn.Conv2d(100, 100, 3, 1, 1), # (100,16,16)
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, 1, 1), # (100,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # (100,8,8)
            nn.Conv2d(100, 100, 3, 1, 1), # (100,8,8)
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),           # (100,4,4)      
        )
        self.fc_layer = nn.Linear(100 * self.img_sz//8 * self.img_sz//8, enc_output_dim)
    
    def forward(self, x):
        # x shape: (batch_sz, nc, img_sz, img_sz)
        x = self.encoder(x)
        x = x.view(x.size(0), 100 * self.img_sz//8 * self.img_sz//8)
        output = self.fc_layer(x)  # shape: (batch_sz, enc_output_dim)
        return output


class ConvDecoder(nn.Module):
    def __init__(self, nc, img_sz, latent_dim, enc_output_dim, hidden_dims):
        super().__init__()

        self.img_sz = img_sz
        self.init_conv_shape_flat = 100 * self.img_sz//8 * self.img_sz//8

        self.latent = nn.Linear(latent_dim, enc_output_dim)
        self.enc_layer = nn.Linear(enc_output_dim, self.init_conv_shape_flat)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 2, 1, 0),  # (100, 5, 5)
            nn.ReLU(),
            nn.ConvTranspose2d(100, 100, 2, 1, 0),  # (100, 6, 6)
            nn.ReLU(),
            # # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(100, 100, 2, 1, 0),  # (100, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(100, 100, 2, 1, 0),  # (100, 8, 8)
            nn.ReLU(),
            # # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(100, 100, 2, 2, 0),  # (100, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(100, 100, 2, 2, 0),  # (100, 32, 32)
            nn.ReLU(),
            # # nn.Upsample(scale_factor=2),
            nn.Conv2d(100, nc, 1, 1, 0))         # (3, 32, 32)
    
    def forward(self, x):
        # x shape: (batch_sz, latent_dim)
        x= self.latent(x)
        x = self.enc_layer(x)
        x = x.view(x.size(0), 100, self.img_sz//8, self.img_sz//8)
        output = self.decoder(x)
        return output   # shape: (batch_sz, nc, img_sz, img_sz)


class ConvDecoder2(nn.Module):
    def __init__(self, nc, img_sz, latent_dim, enc_output_dim, hidden_dims):
        super().__init__()

        self.img_sz = img_sz
        self.init_conv_shape_flat = 100 * self.img_sz//8 * self.img_sz//8

        self.latent = nn.Linear(latent_dim, enc_output_dim)
        self.enc_layer = nn.Linear(enc_output_dim, self.init_conv_shape_flat)

        self.decoder = nn.Sequential(  # starts with input: (100, img_sz//8, img_sz//8)
            nn.Conv2d(100, 100, 3, 1, 1),   # (100, img_sz//8, img_sz//8)
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, 1, 1),   # (100, img_sz//8, img_sz//8)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),   # (100, img_sz//4, img_sz//4)
            nn.Conv2d(100, 100, 3, 1, 1),  # (100, img_sz//4, img_sz//4)
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, 1, 1),  # (100, img_sz//4, img_sz//4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # (100, img_sz//2, img_sz//2)
            nn.Conv2d(100, nc, 3, 1, 1),   # (3, img_sz//2, img_sz//2)
            nn.Upsample(scale_factor=2))  # (3, img_sz, img_sz)
        
    def forward(self,x):
        # x shape: (batch_sz, latent_dim)
        x= self.latent(x)
        x = self.enc_layer(x)
        x = x.view(x.size(0), 100, self.img_sz//8, self.img_sz//8)
        output = self.decoder(x)
        return output   # shape: (batch_sz, nc, img_sz, img_sz)
