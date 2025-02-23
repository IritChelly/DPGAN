import copy, random

import os
import torch
import numpy as np
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

from clusterers import kmeans

#from julia.api import Julia
#jl = Julia(compiled_modules=False)
#from dpmmpython.priors import niw
#from dpmmpython.dpmmwrapper import DPMMPython
#from dpmmpython.dpmmwrapper import DPMModel
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from neptune.new.types import File


class Clusterer(kmeans.Clusterer):
    def __init__(self, initialization=True, matching=True, **kwargs):
        self.initialization = initialization
        self.matching = matching

        super().__init__(**kwargs)

    def recluster(self, encoder, run):
        self.encoder = copy.deepcopy(encoder)
        self.fit_means(run)

    def fit_means(self, run):
        # Get learned features for the whole dataset:
        features_net = self.get_batch_features()  # shape: (N,D)
        features_net.astype(np.float64)

        # Reduce D by running PCA if needed:
        print('Original data shape:', features_net.shape)
        if self.is_pca:
            self.pca = PCA(n_components=self.new_d)
            self.pca.fit(features_net)
            features = self.pca.transform(features_net)
        else:
            features = features_net
        print('Data shape after PCA:', features.shape)
        N, D = features.shape

        # ------------------------------------------------------------------------
        #                       Cluster using kmeans:      
        # ------------------------------------------------------------------------

        # if clustered already, use old assignments for the cluster mean
        if self.x_labels is not None and self.initialization:
            print('Initializing k-means with previous cluster assignments')
            initialization = self.get_initialization(features, self.x_labels)
        else:
            initialization = 'k-means++'

        new_classes = self.kmeans_fit_predict(features, init=initialization)

        # we've clustered already, so compute the permutation
        if self.x_labels is not None and self.matching:
            print('Doing cluster matching')
            matching = self.hungarian_match(new_classes, self.x_labels, self.k, self.k)
            self.mapping = [int(j) for i, j in sorted(matching)]

        # recompute the fixed labels
        self.x_labels = np.array([self.mapping[x] for x in new_classes])  # shape: (N,1)

        # Compute NMI:
        nmi_kmeans = nmi(self.y, self.x_labels)
        print('NMI (kmeans):', nmi_kmeans)

        # Compute weights:
        counts = np.zeros(self.k)
        for lbl in self.x_labels:
            counts[lbl] = counts[lbl] + 1
    
        weights_kmeans = counts / np.sum(counts)
        print('Weights:', weights_kmeans, '\n')

        run["results/avg_weights_kmeans"].log(np.mean(weights_kmeans))
        run["results/nmi_kmeans"].log(nmi_kmeans)

        # Compute variance of the latent ftrs of real images in each cluster:
        lbl_unique = np.unique(self.x_labels)
        self.variance = np.ones((self.k, D))
        for lbl in lbl_unique:
            data = features[self.x_labels == lbl]  # shape: (n, D) where n is the number of data points that got label = lbl
            self.variance[lbl, :] = np.var(data, axis=0)   # shape: (D,)  (where self.variance is of shape (K, D))


        # # ------ Save (locally) imgs, gt_labels and features in each clustering:
        # ftrs_dir = os.path.join('/vilsrv-storage/tohamy/BNP/Experiments/trials/trial_clustering_dpgan/stl/ftrs_ours')
        # if not os.path.exists(ftrs_dir):
        #     os.makedirs(ftrs_dir)

        # ftrs_path = os.path.join(ftrs_dir, '%04d.npy' % self.clustering_num)
        # #imgs_path = os.path.join(ftrs_dir, 'imgs.npy')
        # gt_labels_path = os.path.join(ftrs_dir, 'gt_labels.npy')
        # np.save(ftrs_path, features.T)  # save it as (D,N)
        # #np.save(imgs_path, self.x)  # save the images
        # np.save(gt_labels_path, self.y)  # save the images
        # self.clustering_num = self.clustering_num + 1
        # # ------------------------------------------------------------------------



        # ------------------------------------------------------------------------
        #                       Cluster using DP:      
        # ------------------------------------------------------------------------

        # print('\nRunning DP on', features.shape)

        # # Get the initial labels for the DP process:
        # # If this is the first recluster run, use kmeans initialization, otherwise, use x_labels as initial labels for dp.
        # init_labels = None
        # if self.x_labels is None:
        #     init_labels = self.kmeans_fit_predict(features, init='k-means++')
        #     #init_labels = np.random.randint(0, self.k , features_net.shape[0])
        #     # Convert to 1-based values:
        #     init_labels = init_labels + 1
        # else:
        #     init_labels = np.array(self.x_labels.detach().cpu())

        # # --- hyper params:
        # N, D = features.shape
        # m = np.mean(features, axis=0)  #np.zeros(D)
        # k = 1  #np.unique(init_labels).size
        # nu_addition = 100
        # nu = D + nu_addition  #+2 #(2000*N)  # should be > D
        # psi_multiplier = 1.0
        # psi = np.cov(features.T)*psi_multiplier  #*0.00001  # *0.01 # np.eye(D)*0.005 # insert data in teh shape of (D,N) and get covariance in the shape of (D,D) 
        # psi_addition = 0.0000001  # Avoid non-SPD martix
        # psi = psi + (np.eye(D)*psi_addition)
        # hyper_prior = niw(k, m, nu, psi)
        # alpha = 100.
        # iters = 100 #200

        # dp_params = {'m': 'data_mean',
        #             'k': k,
        #             'nu_addition': nu_addition,
        #             'psi': 'identity matrix',
        #             'psi_multiplier': psi_multiplier,
        #             'psi_addition': psi_addition,
        #             'alpha': alpha,
        #             'iters': iters,
        #             'init_labels': 'kmeans 100'}
        # run["config/params/dp_params"] = dp_params

        # print(np.all(np.linalg.eigvals(psi) > 0))

        # #  ------ Save (locally) imgs, gt_labels and features in each clustering:
        # # ftrs_dir = os.path.join('/vilsrv-storage/tohamy/BNP/GAN_DP/code_10_DP/dp-gan/ftrs')
        # # if not os.path.exists(ftrs_dir):
        # #     os.makedirs(ftrs_dir)

        # # ftrs_path = os.path.join(ftrs_dir, '%04d.npy' % self.clustering_num)
        # # imgs_path = os.path.join(ftrs_dir, 'imgs.npy')
        # # gt_labels_path = os.path.join(ftrs_dir, 'gt_labels.npy')
        # # np.save(ftrs_path, features.T)  # save it as (D,N)
        # # np.save(imgs_path, self.x)  # save the images
        # # np.save(gt_labels_path, self.y)  # save the images
        # # self.clustering_num = self.clustering_num + 1
        # # ------------------------------------------------------------------------

        # # -- Run DP: (data should be D,N)
        # #dp_results = DPMMPython.fit(features.T, alpha, prior = hyper_prior, iterations=iters, outlier_params=init_labels, verbose=False)
        # # -- Run DP: This call runs the fit function of DPMM and also provides the "predict" function for later:
        # self.dp = DPMModel(features.T, alpha, prior = hyper_prior, iterations=iters, outlier_params=init_labels, verbose=False, gt = self.y)
        
        # nmi_dp = nmi(self.y, self.dp._labels)
        # num_of_learned_clusters = len(np.unique(self.dp._labels))
        # print('Lables learned (dp):', num_of_learned_clusters, np.unique(self.dp._labels))
        # print('Weights learned (dp):', self.dp._weights)
        # print('NMI (dp):', nmi_dp, '\n')

        # run["results/avg_weights_dp"].log(np.mean(self.dp._weights))
        # run["results/num_of_learned_clusters_dp"].log(num_of_learned_clusters)
        # run["results/nmi_dp"].log(nmi_dp)

        # # Save the labels as torch:
        # self.x_labels = torch.from_numpy(self.dp._labels).long().to(self.device)  # shape: (N,1)

        # self.max_k = np.max(self.dp._labels)

        # print('Done: Running DP\n')

        # # --------- Run kmeans for comparison: ------------
        # k_kmeans = 100
        # print('Run Kmeans..')
        # kmeans = KMeans(init='k-means++', n_clusters=k_kmeans, n_init=10).fit(features)
        # labels_kmeans = kmeans.predict(features)

        # # Compute kmeans weights:
        # counts = np.zeros(k_kmeans)
        # for lbl in labels_kmeans:
        #     counts[lbl] = counts[lbl] + 1
    
        # weights_kmeans = counts / np.sum(counts)
        # print('Weights learned (kmeans):', weights_kmeans)
        # nmi_kmeans = nmi(self.y, labels_kmeans)
        # print('Done Kmeans. NMI (kmeans):', nmi_kmeans, '\n')

        # run["results/avg_weights_kmeans"].log(np.mean(weights_kmeans))
        # run["results/nmi_kmeans"].log(nmi_kmeans)
        # # -------- Done running kmeans ---------------------


    # Predict using kmeans:
    def predict(self, X):
        # Get X, in shape (N,D), in cuda, and return predicted labels (N,1), in cuda, according to the dp_results from DPMM
        X_features_net = self.get_features(X).detach().cpu().numpy()
            
        # Reduce D by running PCA is needed:
        if self.is_pca:
            X_features = self.pca.transform(X_features_net)
        else:
            X_features = X_features_net

        np_prediction = self.kmeans.predict(X_features)   # N x 1
        permuted_prediction = np.array([self.mapping[x] for x in np_prediction])

        # Get the predicted centroid:
        centroids = self.kmeans.cluster_centers_     # k x D
        centroids_predictions = np.array([centroids[l] for l in np_prediction])  # N x D

        return torch.from_numpy(permuted_prediction).to(self.device), torch.from_numpy(centroids_predictions).to(self.device)


    def get_emp_variance(self, label):
        lbl = int(label.detach().cpu().numpy())
        relevant_variance = self.variance[lbl]
        return torch.from_numpy(relevant_variance).to(self.device)

    # Predict using DP:
    def predict_DP(self, X):
        # Get X, in shape (N,D), in cuda, and return predicted labels (N,1), in cuda, according to the dp_results from DPMM
        X_features_net = self.get_features(X).detach().cpu().numpy()
            
        # Reduce D by running PCA is needed:
        if self.is_pca:
            X_features = self.pca.transform(X_features_net)
        else:
            X_features = X_features_net

        # Get label predictions: (N,1) and centroids predictions: (N,D):
        predictions, centroids_predictions = self.dp.predict(X_features)

        return torch.from_numpy(np.array(predictions)).to(self.device), torch.from_numpy(np.array(centroids_predictions)).to(self.device)

    # Get labels using kmeans:
    def get_labels_list(self):
        return np.unique(self.x_labels)

    # Get labels using DP:
    def get_labels_list_dp(self):
        return np.unique(np.array(self.x_labels.detach().cpu()))

    def get_initialization(self, features, labels):
        '''given points (from new discriminator) and their old assignments as np arrays, compute the induced means as a np array'''
        means = []
        for i in range(self.k):
            mask = (labels == i)
            mean = np.zeros(features[0].shape)
            numels = mask.astype(int).sum()
            if numels > 0:
                for index, equal in enumerate(mask):
                    if equal: mean += features[index]
                means.append(mean / numels)
            else:
                # use kmeans++ init if cluster is starved
                rand_point = random.randint(0, features.shape[0] - 1)
                means.append(features[rand_point])
        result = np.array(means)
        return result

    def hungarian_match(self, flat_preds, flat_targets, preds_k, targets_k):
        '''takes in np arrays flat_preds, flat_targets of integers'''
        num_samples = flat_targets.shape[0]

        assert (preds_k == targets_k)  # one to one
        num_k = preds_k
        num_correct = np.zeros((num_k, num_k))

        for c1 in range(num_k):
            for c2 in range(num_k):
                votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
                num_correct[c1, c2] = votes

        # num_correct is small
        match = linear_assignment(num_samples - num_correct)

        # return as list of tuples, out_c to gt_c
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))

        return res

    # def fit_means_old(self):
    #     features = self.get_cluster_batch_features()

    #     # Use old assignments for the cluster mean if exist:
    #     if self.x_labels is not None and self.initialization:
    #         print('Initializing k-means with previous cluster assignments')
    #         initialization = self.get_initialization(features, self.x_labels)
    #     else:
    #         initialization = 'k-means++'

    #     # Learn / update the kmeans model:
    #     new_classes = self.kmeans_fit_predict(features, init=initialization)

    #     # Compute the permutation in case we need it:
    #     if self.x_labels is not None and self.matching:
    #         print('Doing cluster matching')
    #         matching = self.hungarian_match(new_classes, self.x_labels, self.k,
    #                                         self.k)
    #         self.mapping = [int(j) for i, j in sorted(matching)]

    #     # Recompute the fixed labels
    #     self.x_labels = np.array([self.mapping[x] for x in new_classes])