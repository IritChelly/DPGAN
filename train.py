# -------------------- use only this device: ------------------------------
default_device = "1"
# -------------------------------------------------------------------------

import os
import argparse
import copy
import pprint
from os import path
import shutil

import torch
import numpy as np
from torch import nn
import time

from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, get_clusterer, build_models, build_optimizers)
from seeing.pidfile import exit_if_job_done, mark_job_done
import neptune.new as neptune
from neptune.new.types import File

torch.backends.cudnn.benchmark = True

# dpgan_AE, pre-train of 50, kmeans, num_clusters=100, recluster every

# Initialize Neptune and create new Neptune Run
run = neptune.init(
    project="itohamy/G-E-and-D",
    tags="code_18, cifar, loss on real-fake mu distance, with KL loss when training G+E",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZWU1MWE2ZS0yZjEwLTQzMTItODdlYi1kN2I4ODgzMDA4M2IifQ==",
    source_files=["*.py", "configs/*", "clusterers/*", "gan_training/*", "utils/*"]
)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--outdir', type=str, help='used to override outdir (useful for multiple runs)')
parser.add_argument('--nepochs', type=int, default=250, help='number of epochs to run before terminating')
parser.add_argument('--model_it', type=int, default=-1, help='which model iteration to load from, -1 loads the most recent model')
parser.add_argument('--devices', nargs='+', type=str, default=['0'], help='devices to use')  # this arg is ignored, we use the default_device set above.

args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')
out_dir = config['training']['out_dir'] if args.outdir is None else args.outdir


def main():
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint({
        'data': config['data'],
        'generator': config['generator'],
        'discriminator': config['discriminator'],
        'clusterer': config['clusterer'],
        'training': config['training']
    })

    # Set the device:
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + default_device)
    #device = torch.device("cuda:" + default_device if is_cuda else "cpu")  # This will be used in the program.
    devices = [int(x) for x in [default_device]]  #[int(x) for x in args.devices]   # Used for G and D.

    # More parameters;
    is_pca = False
    pca_dim = 128  # The target dimention in case we perform PCA on the features before clustering
    args.nepochs = 200 #350
    nepoch_pretrain_encoder = 1  #200 #200
    nepoch_pretrain_gan = 0 #100
    num_clusters = 100  # "num_clusters" is the initial K when running DP
    d_iters = 1 #10  # number if iterations for D (the discriniator will do d_iters more iterations than the generator)
    g_iters = 1
    # "max_K" is the maximum k value that we can reach to, when runnin DP (as k changes during the process)
    # config['generator']['nlabels'] and config['discriminator']['nlabels']

    # Short hands
    batch_size = config['training']['batch_size']
    encoder_latent_dim = config['encoder_args']['encoder_output_dim']
    log_every = config['training']['log_every']
    inception_every = config['training']['inception_every']
    backup_every = config['training']['backup_every']
    sample_nlabels = config['training']['sample_nlabels']  # Not used in practice
    nlabels = config['data']['nlabels']   # Not used in practice
    sample_nlabels = min(nlabels, sample_nlabels)  # Not used in practice

    D_lambda = config['training']['D_lambda']
    contrastive_lambda = config['training']['contrastive_lambda']
    rec_lambda = config['training']['rec_lambda']

    run["config/params/num_clusters"] = num_clusters
    run["config/params/data_type"] = config['data']['type']
    run["config/params/is_pca"] = is_pca
    run["config/params/pca_dim"] = pca_dim
    run["config/params/nepocs"] = args.nepochs
    run["config/params/nepoch_pretrain_encoder"] = nepoch_pretrain_encoder
    run["config/params/nepoch_pretrain_gan"] = nepoch_pretrain_gan
    run["config/params/D_lambda"] = D_lambda
    run["config/params/contrastive_lambda"] = contrastive_lambda
    run["config/params/rec_lambda"] = rec_lambda

    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Logger
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

    train_dataset, _ = get_dataset(
        name=config['data']['type'],
        data_dir=config['data']['train_dir'],
        size=config['data']['img_size'],
        deterministic=config['data']['deterministic'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    # Create models
    generator, discriminator, encoder = build_models(config, device)

    # Put models on gpu if needed
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    encoder = encoder.to(device)

    for name, module in discriminator.named_modules():
        if isinstance(module, nn.Sigmoid):
            print('Found sigmoid layer in discriminator; not compatible with BCE with logits')
            exit()

    g_optimizer, d_optimizer, encoder_optimizer = build_optimizers(generator, discriminator, encoder, config)

    generator = nn.DataParallel(generator, device_ids=devices)
    discriminator = nn.DataParallel(discriminator, device_ids=devices)

    # Register modules to checkpoint
    checkpoint_io.register_modules(generator=generator,
                                   discriminator=discriminator,
                                   encoder=encoder,
                                   g_optimizer=g_optimizer,
                                   d_optimizer=d_optimizer,
                                   encoder_optimizer=encoder_optimizer)

    # Logger
    logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    monitoring=config['training']['monitoring'],
                    monitoring_dir=path.join(out_dir, 'monitoring'))

    # Distributions
    ydist = get_ydist(nlabels, device=device)   # Not used in practice
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)  # Usually Gaussian dist with mu=0 and var=1, in dim=128

    ntest = config['training']['ntest']
    x_test, y_test = utils.get_nsamples(train_loader, ntest)  # Used only here, for the gt images
    x_test, y_test = x_test.to(device), y_test.to(device)
    utils.save_images(x_test, path.join(out_dir, 'real.png'))
    logger.add_imgs(x_test, 'gt', 0)
    x_cluster, y_cluster = utils.get_nsamples(train_loader, config['clusterer']['nimgs'])
    z_test = zdist.sample((ntest, ))  # Usually it's in the shape of: (128,)

    # Test generator
    if config['training']['take_model_average']:
        print('Taking model average')
        bad_modules = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        for model in [generator, discriminator]:
            for name, module in model.named_modules():
                for bad_module in bad_modules:
                    if isinstance(module, bad_module):
                        print('Batch norm in discriminator not compatible with exponential moving average')
                        exit()
        generator_test = copy.deepcopy(generator)
        checkpoint_io.register_modules(generator_test=generator_test)
    else:
        generator_test = generator

    # Load checkpoint if it exists
    it = utils.get_most_recent(checkpoint_dir, 'model') if args.model_it == -1 else args.model_it
    it, epoch_idx, loaded_clusterer = checkpoint_io.load_models(it=it, load_samples='supervised' != config['clusterer']['name'])

    # Trainer
    trainer = Trainer(generator,
                      discriminator,
                      encoder,
                      g_optimizer,
                      d_optimizer,
                      encoder_optimizer,
                      gan_type=config['training']['gan_type'],
                      reg_type=config['training']['reg_type'],
                      reg_param=config['training']['reg_param'],
                      rec_lambda=config['training']['rec_lambda'],
                      contrastive_lambda=config['training']['contrastive_lambda'],
                      D_lambda=config['training']['D_lambda'],
                      kl_lambda=config['training']['kl_lambda'],
                      real_fake_lambda=config['training']['real_fake_lambda'],
                      g_variety_lambda=config['training']['g_variety_lambda'],
                      var_for_kl=config['training']['var_for_kl'],
                      run=run,
                      device=device,
                      batch_size=batch_size)


    # Pre-train the VAE first on real images:
    print('\nStart: Encoder pre-training, on' , nepoch_pretrain_encoder, 'epochs.')
    it_pretrain_encoder = 0
    epoch_idx_pretrain_encoder = 0
    while it_pretrain_encoder < nepoch_pretrain_encoder * len(train_loader):
        epoch_idx_pretrain_encoder += 1

        for x_real, y in train_loader:
            it_pretrain_encoder += 1

            x_real = x_real.to(device)
            centroids = torch.zeros(batch_size, encoder_latent_dim).to(device)  # (batch_sz, encoder_latent_dim)

            # Encoder update:
            encoder_total_loss, contrastive_loss, kl_loss  = trainer.encoder_trainstep(x_real, centroids)

            # Log losses:
            logger.add('losses', 'encoder_total_loss', encoder_total_loss, it=it_pretrain_encoder)
            logger.add('losses', 'encoder_contrastive_loss', contrastive_loss, it=it_pretrain_encoder)
            logger.add('losses', 'encoder_kl_loss', kl_loss, it=it_pretrain_encoder)

        # Print and save in neptune every epoc:
        run["results/loss/E_total_loss"].log(encoder_total_loss)
        run["results/loss/E_contrastive_loss"].log(contrastive_loss)
        run["results/loss/E_kl_loss"].log(kl_loss)
        print('[epoch %0d, it %4d]: Encoder_total_loss = %.4f, Encoder_contrastive_loss = %.4f, Encoder_kl_loss = %.4f'
                 % (epoch_idx_pretrain_encoder, it_pretrain_encoder, encoder_total_loss, contrastive_loss, kl_loss))

    print('\nDone: Encoder pre-training\n')

    # DEBUG 
    # Debug real and fake distributions:
    n = 10000
    ftrs_dir = os.path.join('/vilsrv-storage/tohamy/BNP/Experiments/trials/trial_KL_div/cifar/ftrs_E_on_G/')
    makedir(ftrs_dir)
    x_test2, y_test2 = utils.get_nsamples(train_loader, n)
    #trainer.save_real_images(x_test2.detach().cpu(), ftrs_dir)
    trainer.save_gt_labels(y_test2, ftrs_dir)
    x_test2 = x_test2.to(device)
    z_test2 = zdist.sample((n, )) 
    # END DEBUG

    # Clusterer:
    clusterer = get_clusterer(config)(encoder=encoder,
                                      k_value=num_clusters,
                                      x_cluster=x_cluster,
                                      y_cluster=y_cluster,
                                      device=device,
                                      is_pca=is_pca,
                                      new_pca_dim=pca_dim)

    if loaded_clusterer is None:
        print('Initializing new clusterer. The first clustering can be quite slow.')
        clusterer.recluster(encoder=encoder, run=run)
        checkpoint_io.save_clusterer(clusterer, it=0)
        np.savez(os.path.join(checkpoint_dir, 'cluster_samples.npz'), x=x_cluster)
    else:
        print('Using loaded clusterer')
        clusterer = loaded_clusterer


    # Evaluator
    evaluator = Evaluator(
        generator_test,
        zdist,
        ydist,
        train_loader=train_loader,
        clusterer=clusterer,
        batch_size=batch_size,
        device=device,
        inception_nsamples=config['training']['inception_nsamples'])


    # # Pre-train the GAN:
    # print('\nStart: GAN pre-training, on' ,nepoch_pretrain_gan, 'epochs.')
    # it_pretrain_gan = 0
    # epoch_idx_pretrain_gan = 0
    # while it_pretrain_gan < nepoch_pretrain_gan * len(train_loader):
    #     epoch_idx_pretrain_gan += 1

    #     for x_real, y in train_loader:
    #         it_pretrain_gan += 1

    #         x_real = x_real.to(device)
    #         y, centroids = clusterer.predict(x_real)   # returns labels (N,1) and centroids (N,D)
    #         z = zdist.sample((batch_size, ))

    #         # Discriminator update:
    #         for _ in range(0, d_iters):
    #             z = zdist.sample((batch_size, ))
    #             dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
    #             logger.add('losses', 'discriminator', dloss, it=it_pretrain_gan)
    #             logger.add('losses', 'regularizer', reg, it=it_pretrain_gan)

    #         # Generators update:
    #         for _ in range(0, g_iters):
    #             g_loss = trainer.generator_trainstep(y, z)
    #             logger.add('losses', 'generator', g_loss, it=it_pretrain_gan)
    #             z = zdist.sample((batch_size, ))

    #     print('[epoch %0d, it %4d]: g_loss = %.4f, d_loss = %.4f' % (epoch_idx_pretrain_gan, it_pretrain_gan, g_loss, dloss))

    # print('\nDone: GAN pre-training\n')


    # Training loop
    print('\nStart main training...')
    print('\nNumber of data points:', len(train_loader) * batch_size)
    print('Number of iterations per epoch:', len(train_loader))
    print('Number of total iterations:', len(train_loader) * args.nepochs)
    print('Starting from iteration:', it)

    while it < args.nepochs * len(train_loader):
        epoch_idx += 1

        for x_real, y_gt in train_loader:
            it += 1

            x_real, y_gt = x_real.to(device), y_gt.to(device)   # x_real: (batch_sz, 3, img_sz, img_sz)
            y, centroids = clusterer.predict(x_real)   # y: (batch_sz,),  centroids: (batch_sz, D)
            z = zdist.sample((batch_size, ))  # z: (batch_sz, z_dim)

            # Discriminator update:
            dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
            logger.add('losses', 'discriminator', dloss, it=it)
            logger.add('losses', 'regularizer', reg, it=it)

            # Generator + Encoder update:            
            gloss_total_loss, g_loss, real_fake_loss, kl_loss = trainer.generator_and_encoder_trainstep(y, z, centroids, x_real)
            logger.add('losses', 'gloss_total_loss', gloss_total_loss, it=it)
            logger.add('losses', 'generator_G', g_loss, it=it)
            logger.add('losses', 'generator_kl', kl_loss, it=it)
            logger.add('losses', 'generator_real_fake_loss', real_fake_loss, it=it)

            # Encoder update:
            encoder_total_loss, contrastive_loss, kl_loss = trainer.encoder_trainstep(x_real, centroids)
            logger.add('losses', 'encoder_total_loss', encoder_total_loss, it=it_pretrain_encoder)
            logger.add('losses', 'encoder_contrastive_loss', contrastive_loss, it=it_pretrain_encoder)
            logger.add('losses', 'encoder_kl_loss', kl_loss, it=it_pretrain_encoder)

            # DEBUG: save ftrs for real images right before training the GAN:
            # For each saved iteration: we save x_test embeddings (real images) --> get the y_pred from clusterer (based on x_test) --> use y_pred to get x_fake embeddings.
            if it % 5000 == 0:
                x_test2 = x_test2.to(device)
                y_pred2, _ = clusterer.predict(x_test2)
                trainer.save_ftrs_for_real_imgs(x_test2, it, ftrs_dir)
                trainer.save_predicted_labels(y_pred2, it, ftrs_dir)
                x_fake = trainer.save_ftrs_for_fake_imgs(z_test2, y_pred2, it, ftrs_dir)
                if it % 30000 == 0:
                    trainer.save_fake_images(x_fake, it, ftrs_dir)
            # END DEBUG

            if config['training']['take_model_average']:
                update_average(generator_test, generator, beta=config['training']['model_average_beta'])

            # Print stats and save in neptune:
            if it % log_every == 0:
                g_loss_last = logger.get_last('losses', 'generator_G')
                g_kl_loss_last = logger.get_last('losses', 'generator_kl')
                g_real_fake_loss_last = logger.get_last('losses', 'generator_real_fake_loss')
                d_loss_last = logger.get_last('losses', 'discriminator')
                encoder_constrastive_last = logger.get_last('losses', 'encoder_contrastive_loss')
                encoder_kl_loss_last = logger.get_last('losses', 'encoder_kl_loss')

                run["results/loss/G_loss"].log(g_loss_last)
                run["results/loss/G_kl_loss"].log(g_kl_loss_last)
                run["results/loss/G_real_fake_loss"].log(g_real_fake_loss_last)
                run["results/loss/D_loss"].log(d_loss_last)
                run["results/loss/E_contrastive_loss"].log(encoder_constrastive_last)
                run["results/loss/E_kl_loss"].log(encoder_kl_loss_last)

                print('[epoch %0d, it %4d] Losses: G = %.4f, G_real_fake = %.4f, G_kl = %.4f, D = %.4f, E_cntrs = %.4f, E_kl = %.4f'
                     % (epoch_idx, it, g_loss_last, g_real_fake_loss_last, g_kl_loss_last, d_loss_last, encoder_constrastive_last, encoder_kl_loss_last))


            # Re-cluster when needed:
            if it % config['training']['recluster_every'] == 0 and it > config['training']['burnin_time']:
                # print cluster distribution for online methods
                if it % 100 == 0 and config['training']['recluster_every'] <= 100:
                    print(f'[epoch {epoch_idx}, it {it}], distribution: {clusterer.get_label_distribution()}')
                    #print(f'[epoch {epoch_idx}, it {it}]')
                clusterer.recluster(encoder=encoder, run=run)

            # (i) Sample if necessary
            if it % config['training']['sample_every'] == 0:
                # --------- Sample a batch of x, predict their y, and then run G(z, y): -------------
                print('Creating samples...')
                x_test, _ = utils.get_nsamples(train_loader, ntest)
                y_test, centroid_test = clusterer.predict(x_test.to(device))  # returns labels (N,1) and centroids (N,D)
                x = evaluator.create_samples(z_test, y_test)
                logger.add_imgs(x, 'all_generated', it)
                run['results/visualize_generated/' + str(it) + '/'].log(File(os.path.join(out_dir, 'imgs/all_generated', '%08d.png' % it)))

                # --------- Run on the labels list (the latest predicted y for x_cluster), and then run G(z, y) for each y in the list: -------------
                labels_list = clusterer.get_labels_list()
                #print('Save imgs per label, for labels', labels_list)
                for y_inst in labels_list:
                    y_inst = int(y_inst)
                    x = evaluator.create_samples(z_test, y_inst)
                    logger.add_imgs(x, '%04d' % y_inst, it)

            # (ii) Compute inception if necessary
            if it % inception_every == 0 and it > 0 and config['data']['type'] != 'mnist':
                print('PyTorch Inception score...')
                inception_mean, inception_std = evaluator.compute_inception_score()
                logger.add('metrics', 'pt_inception_mean', inception_mean, it=it)
                logger.add('metrics', 'pt_inception_stddev', inception_std, it=it)
                print(f'[epoch {epoch_idx}, it {it}] pt_inception_mean: {inception_mean}, pt_inception_stddev: {inception_std}')
                run["results/pt_inception_mean"].log(inception_mean)
                
            # (iii) Backup if necessary
            if it % backup_every == 0:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it)
                checkpoint_io.save_clusterer(clusterer, int(it))
                logger.save_stats('stats_%08d.p' % it)

                if it > 0:
                    checkpoint_io.save('model.pt', it=it)


# Delete is exist, and create the dir in the path:
def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass



if __name__ == '__main__':
    time_start = time.time()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    shutil.rmtree(out_dir)
    exit_if_job_done(out_dir)
    main()
    mark_job_done(out_dir)
    elapsed = time.time() - time_start
    print('\nFinish CGAN in:', int(elapsed), 'seconds.\n')
