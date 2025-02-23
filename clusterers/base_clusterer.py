import copy

import torch
import numpy as np

# Note: x_cluster and y_cluster are not in use
# self.k is used only for the kmeans when extracting initial labels for the first time.
class BaseClusterer():
    def __init__(self,
                 encoder,
                 k_value=-1,
                 x_cluster=None,
                 y_cluster=None,
                 batch_size=100,
                 device=None,
                 is_pca=False,
                 new_pca_dim=100,
                 **kwargs):
        ''' requires that self.x is not on the gpu, or else it hogs too much gpu memory ''' 
        self.cluster_counts = [0] * k_value
        self.encoder = copy.deepcopy(encoder)
        self.encoder.eval()
        self.k = k_value
        self.kmeans = None
        self.x = x_cluster
        self.y = np.array(y_cluster.detach().cpu())
        self.x_labels = None
        self.variance = None
        self.batch_size = batch_size
        self.alpha = 10.0  # for DP
        self.prior = None
        self.dp = None
        self.device = device
        self.is_pca = is_pca
        self.new_d = new_pca_dim
        self.pca = None
        self.clustering_num = 0
        self.max_k = 0

    def recluster(self, encoder, run, **kwargs):
        return

    def get_batch_features(self):
        ''' returns the encoder features for the input batch as a numpy array '''
        with torch.no_grad():
            outputs = []
            x = self.x
            for batch in range(x.size(0) // self.batch_size):
                x_batch = x[batch * self.batch_size:(batch + 1) * self.batch_size].to(self.device)
                outputs.append(self.get_features(x_batch).detach().cpu())
            if (x.size(0) % self.batch_size != 0):
                x_batch = x[x.size(0) // self.batch_size * self.batch_size:].to(self.device)
                outputs.append(self.get_features(x_batch).detach().cpu())
            result = torch.cat(outputs, dim=0).numpy()
            return result

    def get_features(self, x):
        ''' by default gets encoder, but you can use other things '''
        return self.get_encoder_output(x)
    
    def get_encoder_output(self, x):
        '''returns encoder latent features'''
        self.encoder.eval()
        with torch.no_grad():
            x_emb = self.encoder(x, get_features=True)  # each in the shape of: (N, encoder_latent_dim)
            return x_emb

    # def get_discriminator_output(self, x):
    #     '''returns discriminator features'''
    #     self.generator.eval()
    #     self.encoder.eval()
    #     with torch.no_grad():
    #         return self.discriminator(x, get_features=True)

    def get_label_distribution(self, x=None):
        '''returns the empirical distributon of clustering'''
        y = self.x_labels if x is None else self.predict(x, None)
        
        if self.dp is None:  # if the clusterer is kmeans:
            counts = [0] * self.k
        else:
            counts = [0] * (self.max_k + 1)
        
        for yi in y:
            counts[yi] += 1
        return counts

    def sample_y(self, batch_size):
        '''samples y according to the empirical distribution (not sure if used anymore)'''
        distribution = self.get_label_distribution()
        distribution = [i / sum(distribution) for i in distribution]
        m = torch.distributions.Multinomial(batch_size,
                                            torch.tensor(distribution))
        return m.sample()

    def print_label_distribution(self, x=None):
        print(self.get_label_distribution(x))

    def get_max_k(self):

        if self.dp is None:  # if the clusterer is kmeans:
            return self.k
        else:
            return self.max_k + 1


    # def get_cluster_batch_features(self):
    #     ''' returns the discriminator features for the batch self.x as a numpy array '''
    #     with torch.no_grad():
    #         outputs = []
    #         x = self.x
    #         for batch in range(x.size(0) // self.batch_size):
    #             x_batch = x[batch * self.batch_size:(batch + 1) * self.batch_size].cuda()
    #             outputs.append(self.get_features(x_batch).detach().cpu())
    #         if (x.size(0) % self.batch_size != 0):
    #             x_batch = x[x.size(0) // self.batch_size * self.batch_size:].cuda()
    #             outputs.append(self.get_features(x_batch).detach().cpu())
    #         result = torch.cat(outputs, dim=0).numpy()
    #         return result