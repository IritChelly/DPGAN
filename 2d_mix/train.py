# -------------------- use only this device: ------------------------------
default_device = "1"
# -------------------------------------------------------------------------

import os
import argparse
import sys
import shutil
import time

import torch
from torch import optim
from torch import distributions
from torch import nn
import torch.nn.functional as F
import numpy as np

import evaluation
import inputs

from config import get_models, get_optimizers, get_test, get_dataset, sample_new_x_test_y_test, get_nsamples
from visualizations import (visualize_generated, visualize_clusters)

sys.path.append('../')
from clusterers import clusterer_dict
from gan_training.train import Trainer

sys.path.append('../seeing/')
import pidfile

parser = argparse.ArgumentParser(description='2d dataset experiments')
parser.add_argument('--clusterer', help='type of clusterer to use. cluster specifies selfcondgan')
parser.add_argument('--data_type', help='either grid or ring')
parser.add_argument('--recluster_every', type=int, default=5000, help='how frequently to recluster')  # 5000
parser.add_argument('--nruns', type=int, default=1, help='number of trials to do')
parser.add_argument('--burnin_time', type=int, default=0, help='wait this amount of iterations before clustering')

parser.add_argument('--variance', type=float, default=None, help='variance of the gaussians')
parser.add_argument('--model_type', type=str, default='standard', help='model architecture')
parser.add_argument('--num_clusters', type=int, default=50, help='number of clusters to use for selfcondgan')
parser.add_argument('--z_dim', type=int, default=2, help='G latent dim')
parser.add_argument('--d_act_dim', type=int, default=200, help='hidden layer width')
parser.add_argument('--npts', type=int, default=100000, help='number of points to use in dataset')  # 100000
parser.add_argument('--train_batch_size', type=int, default=100, help='training time batch size')
parser.add_argument('--test_batch_size', type=int, default=50000, help='number of examples to get metrics with')  # 50000
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to run')
parser.add_argument('--outdir', default='output')
args = parser.parse_args()


# Set the device:
device = torch.device("cuda:" + default_device)
#device = torch.device("cuda:" + default_device if torch.cuda.is_available() else "cpu")

data_type = args.data_type
k_value = 8 if data_type == 'ring' else 25
num_clusters = k_value if args.clusterer == 'supervised' else args.num_clusters

# "num_clusters" is the initial K when running DP:
num_clusters = 50

# "max_K" is the maximum k value that we can reach to, when runnin DP (as k changes during the process):
max_K = 60 # num_clusters + 1

# The target dimention in case we perform PCA on the features before clustering:
pca_dim = 100


exp_name = f'{args.data_type}_{args.clusterer}_{args.recluster_every}_{num_clusters}/'
if args.model_type != 'standard':
    exp_name = f'{args.model_type}_{exp_name}'
if args.variance is not None:
    exp_name = f'{args.variance}_{exp_name}'

if args.variance is None:
    variance = 0.0025 if data_type == 'grid' else 0.0001
else:
    variance = args.variance
nepochs = args.nepochs
z_dim = args.z_dim
test_batch_size = args.test_batch_size
train_batch_size = args.train_batch_size
npts = args.npts


def main(outdir):
    for subdir in ['all', 'snapshots', 'clusters']:
        if not os.path.exists(os.path.join(outdir, subdir)):
            os.makedirs(os.path.join(outdir, subdir), exist_ok=True)

    if data_type == 'grid':
        get_data = inputs.get_data_grid_dirichlet   # choose from: [inputs.get_data_grid_dirichlet, inputs.get_data_grid]
        percent_good = evaluation.percent_good_grid_dirichlet  # choose from: [evaluation.percent_good_grid_dirichlet, evaluation.percent_good_grid]
    elif data_type == 'ring':
        get_data = inputs.get_data_ring
        percent_good = evaluation.percent_good_ring
    else:
        raise NotImplementedError()

    zdist = distributions.Normal(torch.zeros(z_dim, device=device),
                                 torch.ones(z_dim, device=device))
    z_test = zdist.sample((test_batch_size, ))

    # x_test, y_test = get_test(get_data=get_data,
    #                           batch_size=test_batch_size,
    #                           variance=variance,
    #                           k_value=k_value,
    #                           device=device,
    #                           ftrs_dim=args.d_act_dim)

    # x_cluster, y_cluster = get_test(get_data=get_data,
    #                         batch_size=10000,
    #                         variance=variance,
    #                         k_value=k_value,
    #                         device=device,
    #                         ftrs_dim=args.d_act_dim)

    # Here we extract the data that will be used in the clusterer (and only this data):
    # k_value is not in use.
    train_loader = get_dataset(get_data=get_data,
                               batch_size=train_batch_size,
                               npts=npts,
                               variance=variance,
                               k_value=k_value,
                               ftrs_dim=args.d_act_dim)

    x_cluster, y_cluster = get_nsamples(train_loader, npts)


    def train(trainer, g, d, clusterer, exp_dir, conditioning):
        it = 0
        if os.path.exists(os.path.join(exp_dir, 'log.txt')):
            os.remove(os.path.join(exp_dir, 'log.txt'))

        for epoch in range(nepochs):
            for x_real, y in train_loader:
                z = zdist.sample((train_batch_size, ))
                x_real = x_real.to(device)
                y = clusterer.predict(x_real)

                dloss, _ = trainer.discriminator_trainstep(x_real, y, z)
                gloss = trainer.generator_trainstep(y, z)

                if it % args.recluster_every == 0 and args.clusterer != 'supervised':
                    if args.clusterer != 'burnin' or it >= args.burnin_time:
                        clusterer.recluster(discriminator, x_batch=x_real)  # x_batch is not used.

                # ------- Visualization: -------
                if it % 1000 == 0:
                    # get x_test and y_text in the size of "test_batch_size":
                    x_test, _ = get_nsamples(train_loader, test_batch_size)  # Used only here, for the gt images
                    y_test = clusterer.predict(x_test.to(device))

                    x_fake = g(z_test, y_test).detach().cpu().numpy()

                    # y_test is not in use:
                    visualize_generated(x_fake,
                                        x_test.detach().cpu().numpy(), y_test, it,
                                        exp_dir)

                    #print("\nVisualize_clusters")
                    #visualize_clusters(x_test.detach().cpu().numpy(),
                    #                   y_test.detach().cpu().numpy(),
                    #                   it, exp_dir)
                    #print("Done: Visualize_clusters\n")

                    torch.save(
                        {
                            'generator': g.state_dict(),
                            'discriminator': d.state_dict(),
                            'g_optimizer': g_optimizer.state_dict(),
                            'd_optimizer': d_optimizer.state_dict()
                        },
                        os.path.join(exp_dir, 'snapshots', 'model_%d.pt' % it))

                # ------ Metrics: --------
                if it % 1000 == 0:
                    g.eval()
                    d.eval()
                    
                    x_test, _ = get_nsamples(train_loader, test_batch_size)  # Used only here, for the gt images
                    y_test = clusterer.predict(x_test.to(device))

                    x_fake = g(z_test, y_test).detach().cpu().numpy()
                    
                    percent, modes, kl = percent_good(x_fake, var=variance)
                    log_message = f'[epoch {epoch} it {it}] dloss = {dloss}, gloss = {gloss}, prop_real = {percent}, modes = {modes}, kl = {kl}'
                    with open(os.path.join(exp_dir, 'log.txt'), 'a+') as f:
                        f.write(log_message + '\n')
                    print(log_message)
                    
                it += 1

    # train a G/D from scratch
    conditioning = 'conditional'   # choose from: ['conditional', 'unconditional', 'conditioned_by_centroids']
    generator, discriminator = get_models(args.model_type, conditioning, max_K, args.d_act_dim, device)
    g_optimizer, d_optimizer = get_optimizers(generator, discriminator)
    trainer = Trainer(generator, discriminator, g_optimizer, d_optimizer, gan_type='standard', reg_type='none', reg_param=0)
    clusterer = clusterer_dict[args.clusterer](discriminator=discriminator,
                                               k_value=num_clusters,
                                               x_cluster=x_cluster,
                                               device=device,
                                               is_pca=True,
                                               new_pca_dim=pca_dim)
    clusterer.recluster(discriminator=discriminator)
    train(trainer, generator, discriminator, clusterer, os.path.join(outdir), conditioning)


if __name__ == '__main__':
    time_start = time.time()
    outdir = os.path.join(args.outdir, exp_name)
    shutil.rmtree(args.outdir)
    pidfile.exit_if_job_done(outdir)
    for run_number in range(args.nruns):
        run_dir = f'{outdir}_run_{run_number}' if args.nruns > 1 else outdir
        main(run_dir)
    pidfile.mark_job_done(outdir)
    elapsed = time.time() - time_start
    print('\nFinish CGAN in:', int(elapsed), 'seconds.\n')
