# coding: utf-8
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from neptune.new.types import File
from gan_training.contrastiveLoss import ContrastiveLoss
import os

# Notes:
# model.train(): sets the mode to "train", but it does NOT actually train the model. It just tells the params to behave in a certain way that fits the training process and not the test process.
# optimizer.zero_grad(): explicitly sets the gradients to zero before starting to do backpropragation (i.e., updating the Weights and biases) because PyTorch accumulates the gradients on subsequent backward passes. 
# loss.backward(): computes the gradient of current tensor w.r.t. graph leaves (it computes dloss/dx for every parameter x which has requires_grad=True).
# optimizer.step(): performs a parameter update based on the current gradient and the update rule.
# requires_grad(): if we want to freeze part of the network and train the rest, we can set the parameter's attribute requires_grad_ to False. (this is done in the "toggle_grad" function). The default is True.


class Trainer(object):
    def __init__(self,
                 generator,
                 discriminator,
                 encoder,
                 g_optimizer,
                 d_optimizer,
                 encoder_optimizer,
                 gan_type,
                 reg_type,
                 reg_param,
                 rec_lambda,
                 contrastive_lambda,
                 D_lambda,
                 kl_lambda,
                 real_fake_lambda,
                 g_variety_lambda,
                 var_for_kl,
                 run,
                 device,
                 batch_size):

        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.encoder_optimizer = encoder_optimizer
        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.criterion = nn.MSELoss(reduction='mean')  # MSE with reduction 'mean': compute (y_i-y_i')^2 for each data point i, we get (N,D) matrix, and then perform sum / mean on ALL N*D elements in this matrix.
        self.rec_lambda = rec_lambda
        self.contrastive_lambda = contrastive_lambda
        self.D_lambda = D_lambda
        self.kl_lambda = kl_lambda
        self.real_fake_lambda = real_fake_lambda
        self.g_variety_lambda = g_variety_lambda
        self.var_for_kl = var_for_kl
        self.run = run
        self.device = device
        self.contrastive_loss = ContrastiveLoss(batch_size, device)
        self.c1 = 0
        self.c2 = 0

        print('D reg gamma', self.reg_param)

    def generator_and_encoder_trainstep(self, y, z, centroids, x_real):
        # y: (batch_sz,)
        # z: (batch_sz, z_dim)
        # centroids: (batch_sz, encoder_latent_dim)
        # x_real: (batch_sz, 3, img_Sz, img_sz). This is the real-data batch that was used to predict y

        assert (y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        toggle_grad(self.encoder, True)
        toggle_grad(self.discriminator, False)

        self.generator.train()
        self.encoder.train()
        self.discriminator.train()

        self.g_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        x_emb = self.encoder(x_fake, is_augment=False)
        x_emb_real = self.encoder(x_real, is_augment=False)
        d_fake = self.discriminator(x_fake, y)
        
        # Compute losses:
        g_loss = self.compute_GAN_loss(d_fake, 1)
        kl_loss = self.compute_kl_loss(x_emb, torch.tensor(self.var_for_kl).to(self.device), centroids, self.var_for_kl)
        real_fake_loss = self.compute_real_fake_loss(x_emb, x_emb_real)
        
        gloss_total_loss = g_loss + real_fake_loss + kl_loss
        gloss_total_loss.backward()

        self.encoder_optimizer.step()
        self.g_optimizer.step()

        return gloss_total_loss.item(), g_loss.item(), real_fake_loss.item(), kl_loss.item()

    def generator_trainstep(self, y, z):
        assert (y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        toggle_grad(self.encoder, False)
        toggle_grad(self.discriminator, False)

        self.generator.train()
        self.encoder.train()
        self.discriminator.train()

        self.g_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y)

        # Compute loss:
        g_loss = self.compute_GAN_loss(d_fake, 1)
        g_loss.backward()

        self.g_optimizer.step()

        return g_loss.item()

    def encoder_trainstep(self, x, centroids):
        toggle_grad(self.generator, False)
        toggle_grad(self.encoder, True)
        toggle_grad(self.discriminator, False)

        self.generator.train()
        self.encoder.train()
        self.discriminator.train()

        self.encoder_optimizer.zero_grad()

        x.requires_grad_()
        x_emb_1, x_emb_2 = self.encoder(x)

        # Compute losses:
        contrastive_loss = self.compute_contrastive_loss(x_emb_1, x_emb_2)
        kl_loss_1 = self.compute_kl_loss(x_emb_1, torch.tensor(self.var_for_kl).to(self.device), centroids, self.var_for_kl)
        kl_loss_2 = self.compute_kl_loss(x_emb_2, torch.tensor(self.var_for_kl).to(self.device), centroids, self.var_for_kl)
        kl_loss = (kl_loss_1 + kl_loss_2) / 2
        encoder_total_loss = contrastive_loss + kl_loss

        encoder_total_loss.backward()

        self.encoder_optimizer.step()

        return encoder_total_loss.item(), contrastive_loss.item(), kl_loss.item() 


    def discriminator_trainstep(self, x_real, y, z):
        ''' y: (batch_sz x 1), contains the labels for each data point.'''
        ''' centroids: (batch_sz x D), contains the centroids for each data point.'''
        
        toggle_grad(self.generator, False)
        toggle_grad(self.encoder, False)
        toggle_grad(self.discriminator, True)

        self.generator.train()
        self.encoder.train()
        self.discriminator.train()
        
        self.d_optimizer.zero_grad()

        # --- On real data ----
        x_real.requires_grad_()
        d_real = self.discriminator(x_real, y)
        dloss_real = self.compute_GAN_loss(d_real, 1)

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # ---- On fake data ----
        with torch.no_grad():
            x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake, y)
        dloss_fake = self.compute_GAN_loss(d_fake, 0)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(
                x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        dloss = (dloss_real + dloss_fake)
        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def compute_GAN_loss(self, d_out, target):
        # d_out (N,1) and target (int)
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            d_loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            d_loss = (2 * target - 1) * d_out.mean()
        else:
            raise NotImplementedError
        
        return self.D_lambda * d_loss

    def compute_contrastive_loss(self, x_emb_1, x_emb_2):
        # x_emb_1 and x_emb_2:, in shape (N, encoder_latent_dim)
        # Here we compute constrastive loss of 2 batches that are 2 augmentations of the same images. And we contrast each row with the rest of teh rows (which are different images).
        contrastive_loss = self.contrastive_loss(x_emb_1, x_emb_2)
        return self.contrastive_lambda * contrastive_loss

    def compute_reconstruction_loss(self, x, x_rec):
        # x: shape (N, 3 * img_sz * img_sz). This is the augmented real image / fake image that we inserted to VAE.
        # x_rec: shape (N, 3 * img_sz * img_sz), This is the reconstructed image that we got from VAE.
        rec_loss = self.criterion(x, x_rec)
        return self.rec_lambda * rec_loss

    def compute_kl_loss(self, mu, log_var, target_mu, target_var):
        # mu: (batch_sz, encoder_latent_dim)
        # log_var: (batch_sz, encoder_latent_dim)
        # centroids: (batch_sz, encoder_latent_dim)

        var = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, var)
        p = torch.distributions.Normal(target_mu, target_var)
        kl_loss = torch.distributions.kl.kl_divergence(q, p).mean() / mu.size(0)
        kl_loss = kl_loss * self.kl_lambda 
        return kl_loss

    def compute_real_fake_loss(self, v_fake, v_real):
        # v_fake, v_real could be either: x_emb, x_emb_real: (batch_sz, encoder_latent_dim) or mu, mu_real: (batch_sz, encoder_latent_dim)
        real_fake_loss = F.mse_loss(v_fake, v_real).float()
        real_fake_loss = real_fake_loss * self.real_fake_lambda 
        return real_fake_loss


    def compute_G_variety_loss(self, x_emb_1, x_emb_2):
        # x_emb_1 and x_emb_2:, in shape (N, encoder_latent_dim)
        # Here we compute the contrastive loss between the 2 embeddings of the 2 fake images, in order to force G to generate various images.
        
        # # Cosine loss 1:
        # target = -1.0
        # targets = x_emb_1.new_full(size=x_emb_1.size(0), fill_value=target)  # (batch_sz, 1)
        # cos_loss = torch.nn.CosineEmbeddingLoss()
        # g_variety_loss = cos_loss(x_emb_1, x_emb_2, targets).float()

        # Cosine loss 2:
        cos = torch.nn.CosineSimilarity(dim=1)
        cos_sim = cos(x_emb_1, x_emb_2)  # shape (N,1)
        d = (cos_sim + 1.0)/2.0  # shape (N,1)
        m = 0
        loss_tensor = torch.clamp(d - m, min=0.0)
        g_variety_loss = torch.mean(loss_tensor)
        
        # # DEBUG
        # if self.c2 % 500 == 0:
        #     print('--loss_tensor--:')
        #     print(loss_tensor)
        
        # self.c2 = self.c2 + 1
        # # END DEBUG

        # # MSE loss:
        # d = F.mse_loss(x_emb_1, x_emb_2, reduction='none').float()
        # m = 10.0
        # loss_tensor = torch.mean(torch.clamp(m - d, min=0.0), dim=1)  # shape (N,1)

        # if self.c2 % 500 == 0:
        #     print('--loss_tensor--:')
        #     print(loss_tensor)
        
        # self.c2 = self.c2 + 1

        # g_variety_loss = torch.mean(loss_tensor)
        
        return self.g_variety_lambda * g_variety_loss



    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

    def save_ftrs_for_real_imgs(self, X, it, ftrs_dir):
        # Get ftrs from AE:
        toggle_grad(self.generator, False)
        toggle_grad(self.encoder, False)
        toggle_grad(self.discriminator, False)
        x_emb = self.encoder(X, get_features=True)

        ftrs_path = os.path.join(ftrs_dir, '%04d_real.npy' % it)
        np.save(ftrs_path, x_emb.detach().cpu())  # save it as (N, AE_latent_dim=128)


    def save_ftrs_for_fake_imgs(self, z, y, it, ftrs_dir):
        # Get ftrs from AE:
        toggle_grad(self.generator, False)
        toggle_grad(self.encoder, False)
        toggle_grad(self.discriminator, False)
        x_fake = self.generator(z, y)
        x_emb = self.encoder(x_fake, get_features=True)

        ftrs_path = os.path.join(ftrs_dir, '%04d_fake.npy' % it)
        np.save(ftrs_path, x_emb.detach().cpu())  # save it as (N, AE_latent_dim=128)
        return x_fake

    def save_gt_labels(self, y, ftrs_dir):
        labels_path = os.path.join(ftrs_dir, 'gt_labels.npy')
        np.save(labels_path, y.detach().cpu())

    def save_predicted_labels(self, y, it, ftrs_dir):
        labels_path = os.path.join(ftrs_dir, '%04d_predicted_labels.npy' % it)
        np.save(labels_path, y.detach().cpu())

    def save_real_images(self, X, ftrs_dir):
        images_path = os.path.join(ftrs_dir, 'images_real.npy')
        np.save(images_path, X.detach().cpu())

    def save_fake_images(self, X, it, ftrs_dir):
        images_path = os.path.join(ftrs_dir, 'images_fake_%04d.npy' % it)
        np.save(images_path, X.detach().cpu())


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)
