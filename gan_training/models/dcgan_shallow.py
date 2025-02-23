import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from gan_training.models import blocks
import numpy as np



class Generator(nn.Module):
    def __init__(self,
                 nlabels,
                 device,
                 size=32,
                 conditioning='embedding',
                 z_dim=128,
                 nc=3,
                 ngf=64,
                 embed_dim=256,
                 **kwargs):
        super(Generator, self).__init__()

        self.img_sz = size   # in mnist img_sz = 32
        self.nlabels = nlabels
        self.init_cov_shape = (ngf * 8, self.img_sz//8, self.img_sz//8)  # The shape of the feature map that starts the transposed conv until it reaches the image shape.
        self.init_cov_shape_flat = int(np.prod(self.init_cov_shape))

        assert conditioning != 'unconditional' or nlabels == 1

        if conditioning == 'embedding':
            self.get_latent = blocks.LatentEmbeddingConcat(nlabels, embed_dim)
            self.fc1 = nn.Sequential(nn.Linear(z_dim + embed_dim, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(0.2, inplace=True))
        elif conditioning == 'unconditional':
            self.get_latent = blocks.Identity()
            self.fc1 = nn.Sequential(nn.Linear(z_dim, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(0.2, inplace=True))
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for generator")

        self.fc2 = nn.Sequential(nn.Linear(1024, self.init_cov_shape_flat),
                                 nn.BatchNorm1d(self.init_cov_shape_flat),
                                 nn.LeakyReLU(0.2, inplace=True))

        self.reshape = blocks.Reshape(self.init_cov_shape)
        
        self.convTranspose1 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                                            nn.BatchNorm2d(ngf * 4),
                                            nn.LeakyReLU(0.2, inplace=True))

        self.convTranspose2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                                            nn.BatchNorm2d(ngf * 2),
                                            nn.LeakyReLU(0.2, inplace=True))

        self.convTranspose3 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                                            nn.BatchNorm2d(ngf),
                                            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv_out = nn.Sequential(nn.Conv2d(ngf, nc, 3, 1, 1), nn.Tanh())

    def forward(self, input, y, get_features=False):
        y = y.clamp(None, self.nlabels - 1)  # (batch_sz,)
        out = self.get_latent(input, y)      # (batch_sz, z_dim + embed_dim)
        out = self.fc1(out)                  # (batch_sz, 1024)
        out = self.fc2(out)                  # (batch_sz, ngf * 8 * self.img_sz//8 * self.img_sz//8)
        out = self.reshape(out)              # (batch_sz, ngf * 8, self.img_sz//8, self.img_sz//8)
        out = self.convTranspose1(out)       # (batch_sz, ngf * 4, img_sz/4, img_sz/4)
        out = self.convTranspose2(out)       # (batch_sz, ngf * 2, img_sz/2, img_sz/2)
        out = self.convTranspose3(out)       # (batch_sz, ngf, img_sz, img_sz)
        x_gen = self.conv_out(out)           # (batch_sz, 3, img_sz, img_sz)
        return x_gen
        


# Use LeakyReLU activation in the discriminator for all layers.
class Discriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 conditioning,
                 device,
                 size=32,
                 features='penultimate',
                 pack_size=1,
                 nc=3,
                 ndf=64,
                 y_embed_dim=256,
                 **kwargs):
        super(Discriminator, self).__init__()
        self.device=device
        self.img_sz = size   # in mnist img_sz = 32
        self.nc = nc
        self.nlabels = nlabels

        assert conditioning != 'unconditional' or nlabels == 1

        self.embed_y = blocks.LatentEmbedding(nlabels, y_embed_dim)
        self.fc_y = nn.Linear(y_embed_dim, self.nc * self.img_sz * self.img_sz)
        self.reshape_y = blocks.Reshape((self.nc, self.img_sz, self.img_sz))

        self.conv1 = nn.Sequential(nn.Conv2d(nc * pack_size * 2, ndf, 4, 2, 1),
                                   nn.BatchNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 4),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True))

        if conditioning == 'mask':
            self.fc_out = blocks.LinearConditionalMaskLogits(ndf * 8 * (self.img_sz//16) * (self.img_sz//16) , nlabels, self.device)
        elif conditioning == 'unconditional':
            self.fc_out = blocks.LinearUnconditionalLogits(ndf * 8 * (self.img_sz//16) * (self.img_sz//16))
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for discriminator")

        self.pack_size = pack_size
        self.features = features
        print(f'Getting features from {self.features}')

    def stack(self, x):
        #pacgan
        nc = self.pack_size
        if nc == 1:
            return x
        x_new = []
        for i in range(x.size(0) // nc):
            imgs_to_stack = x[i * nc:(i + 1) * nc]
            x_new.append(torch.cat([t for t in imgs_to_stack], dim=0))
        return torch.stack(x_new)

    def forward(self, input_img, y):
        # Insert y (of shape (batch_sz,1)) through fc layers:
        y_embd = self.embed_y(y)              # (batch_sz, y_embed_dim)
        y_emb_fc = self.fc_y(y_embd)          # (batch_sz, 3 * self.img_sz * self.img_sz)
        y_img_sz = self.reshape_y(y_emb_fc)   # (batch_sz, 3, self.img_sz, self.img_sz)
        
        input_img = self.stack(input_img)  # (batch_sz, 3, img_sz, img_sz) = (?, 3, 32, 32)   # stack function does anything (as pack_size is always 1)

        # Concat input_img and y:
        concat_img_and_y = torch.cat((input_img, y_img_sz), dim=1)  # (batch_sz, 6, img_sz, img_sz) = (?, 6, 32, 32)

        out = self.conv1(concat_img_and_y)  # (batch_sz, ndf, img_sz/2, img_sz/2) = (?, 64, 16, 16)
        out = self.conv2(out)      # (batch_sz, ndf * 2, img_sz/4, img_sz/4) = (?, 128, 8, 8)
        out = self.conv3(out)      # (batch_sz, ndf * 4, img_sz/8, img_sz/8) = (?, 256, 4, 4)
        out = self.conv4(out)      # (batch_sz, ndf * 8, img_sz/16, img_sz/16) = (?, 512, 2, 2)
        ftrs = out.view(out.size(0), -1)  # shape: (batch_size, ndf * 8 * img_sz/16 * img_sz/16) = (?, 2048)

        y = y.clamp(None, self.nlabels - 1)
        D_result = self.fc_out(ftrs, y)  # shape: batch_sz x 1
        assert (len(D_result.shape) == 1)
 
        return D_result


if __name__ == '__main__':
    z = torch.zeros((1, 128))
    g = Generator()
    x = torch.zeros((1, 3, 32, 32))
    d = Discriminator()

    g(z)
    d(g(z))
    d(x)
