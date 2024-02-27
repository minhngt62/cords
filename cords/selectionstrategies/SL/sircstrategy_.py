import math
import random
import time
from typing import Any
import torch
import torch.nn.functional as F
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import numpy as np

class OODCal:
    """_summary_

    Returns:
        _type_: _description_
    """
    def __init__(self, model, labels, logits, features, W, b, trained_indices, device) -> None:
        self.labels = labels
        self.logits = logits
        self.features = features
        self.W = W
        self.b = b
        self.trained_indices = trained_indices
        self.device = device
        self.model = model
    
    @staticmethod
    def kl(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    @staticmethod
    def generalized_entropy(softmax_id_val, gamma=0.1, M=100):
        probs =  softmax_id_val 
        probs_sorted,_ = torch.sort(probs, axis=1)
        probs_sorted = probs_sorted[:,-M:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
        return -scores 
    
    @staticmethod
    def shannon_entropy(softmax_id_val):
        probs =  softmax_id_val
        scores = torch.sum(probs * torch.log(probs), axis=-1)   
        return scores 

    @staticmethod
    def gradnorm(x, w, b):
        fc = torch.nn.Linear(*w.shape[::-1])
        fc.weight.data[...] = torch.from_numpy(w)
        fc.bias.data[...] = torch.from_numpy(b)
        fc.cuda()
        x = torch.from_numpy(x).float().cuda()
        logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
        confs = []
        for i in x:
            targets = torch.ones((1, 1000)).cuda()
            fc.zero_grad()
            loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
            loss.backward()
            layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data)).cpu().numpy()
            confs.append(layer_grad_norm)
        return np.array(confs)
    
    @staticmethod
    def flatten_grads(m, numpy_output=False, bias=True, only_linear=False):
        total_grads = []
        for name, param in m.named_parameters():
            if only_linear:
                if (bias or not 'bias' in name) and 'linear' in name:
                    total_grads.append(param.grad.detach().view(-1))
            else:
                if (bias or not 'bias' in name) and not 'bn' in name and not 'IC' in name:
                    try:
                        total_grads.append(param.grad.detach().view(-1))
                    except AttributeError:
                        pass
        total_grads = torch.cat(total_grads)
        if numpy_output:
            return total_grads.cpu().detach().numpy()
        return total_grads
    
    def compute_and_flatten_example_grads(self, m, data, target):
        _eg = []
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        m.eval()
        m.zero_grad()
        pred = m(data)
        loss = criterion(pred, target)
        for idx in range(data.shape[0]):
            loss[idx].backward(retain_graph=True)
            _g = self.flatten_grads(m, numpy_output=False)
            _eg.append(torch.Tensor(_g))
            m.zero_grad()
        return torch.stack(_eg)
    
    @staticmethod
    def ocs(g, eg, tau, ref_grads=None):
        ng = torch.norm(g) 
        neg = torch.norm(eg, dim=1)
        mean_sim = torch.matmul(g,eg.t()) / torch.maximum(ng*neg, torch.ones_like(neg)*1e-6)
        negd = torch.unsqueeze(neg, 1)
        # print(g.shape, eg.shape, ref_grads.shape, ng.shape, neg.shape) # [5130] [45000, 5130] [5130] [] [45000]
        cross_div = torch.matmul(eg,eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd)*1e-6)
        mean_div = torch.mean(cross_div, 0)

        coreset_aff = 0.
        if ref_grads is not None:
            ref_ng = torch.norm(ref_grads)
            coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng*neg, torch.ones_like(neg)*1e-6)

        measure = mean_sim - mean_div + tau * coreset_aff
        # _, u_idx = torch.sort(measure, descending=True)
        return measure
    
    def sirc(self, s1, s2, s1_max=1.0):
        "Combine 2 confidence metrics with SIRC."
        # use logarithm for stability
        s1 = s1_max - s1
        s1 = torch.clip(s1, min=1e-64)
        soft = s1.log() # <--
        s1_mean = torch.mean(s1)
        s1_std = torch.std(s1)
        s2_mean = torch.mean(s2)
        s2_std = torch.std(s2)
        s2 = s1_mean + (s2-s2_mean)*s1_std/s2_std
        # s2 = (s2-s2_mean-3*s2_std)/s2_std
        additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), s2)
        assert torch.isnan(soft).any()==False and torch.isinf(soft).any()==False
        assert torch.isnan(additional).any()==False and torch.isinf(additional).any()==False
        
        return soft, additional # return as confidence default -soft -->
    
    def sf(self, s1, s2, d=10, s1_max=1.0):
        s1 = s1_max - s1
        s1 = torch.clip(s1, min=1e-64)
        soft = -s1.log() # <--
        additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), s2)
        soft_max = torch.max(soft)
        additional_max = torch.max(additional)
        s_max = torch.maximum(soft_max, additional_max)
        d = (soft + additional).mean()
        return ((d / soft_max) * soft) + (((s_max - d) / additional_max) * additional)

    def __call__(self, **kwargs) -> Any:
        method = kwargs['method']
        print(f'\n{method}')
        
        probs = self.logits.softmax(dim=-1)
        
        if method == 'MSP':
            score = probs.max(axis=-1)

        # ---------------------------------------
        
        elif method == 'MaxLogit':
            score = self.logits.max(axis=-1)

        # ---------------------------------------
        
        elif method == 'Energy':
            score = torch.logsumexp(self.logits, axis=-1)
        
        # ---------------------------------------
        
        elif method == 'Energy + React':
            clip_quantile = 0.99
            clip = torch.quantile(self.features[self.trained_indices], clip_quantile)
            logit_clip = torch.clip(self.features, min=None, max=clip) @ self.W.T + self.b
            score = torch.logsumexp(logit_clip, axis=-1)
    
        # ---------------------------------------
        
        elif method == 'ViM':
            from sklearn.covariance import EmpiricalCovariance
            from numpy.linalg import norm, pinv
            from scipy.special import logsumexp
            DIM = 1000 if self.features.shape[-1] > 1500 else 512 
            if self.features.shape[-1] <= 512:
                DIM = self.features.shape[-1]//2
            u = -np.matmul(pinv(self.W), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(self.features[self.trained_indices] - u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

            vlogit_train = norm(np.matmul(self.features[self.trained_indices] - u, NS), axis=-1)
            alpha = self.logits[self.trained_indices].max(axis=-1).mean() / vlogit_train.mean()

            vlogit_val = norm(np.matmul(self.features - u, NS), axis=-1) * alpha
            energy_val = logsumexp(self.logits, axis=-1)
            score = -vlogit_val + energy_val

        # ---------------------------------------
        
        elif method == 'Shannon entropy':
            score = self.shannon_entropy(probs)

        # ---------------------------------------
        
        elif method == 'Generalized entropy (GEN)':
            score = self.generalized_entropy(probs, gamma=0.1, M=100)

        # -------------------------------------------
        
        elif method == 'GEN + Residual':
            from sklearn.covariance import EmpiricalCovariance
            from numpy.linalg import norm, pinv
            from scipy.special import logsumexp
            DIM = 1000 if self.features.shape[-1] > 1500 else 512 
            if self.features.shape[-1] <= 512:
                DIM = self.features.shape[-1]//2
            u = -np.matmul(pinv(self.W), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(self.features[self.trained_indices] - u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

            residual_score = norm(np.matmul(self.features - u, NS), axis=-1)
            GEN_score = self.generalized_entropy(probs, gamma=0.1, M=2)
            score =  residual_score * GEN_score
        
        # ---------------------------------------
        
        elif method == 'GradNorm':
            score = self.gradnorm(self.features, self.W, self.b)

        # ---------------------------------------
        
        elif method == 'KL-Matching':
            from sklearn.metrics import pairwise_distances_argmin_min
            probs = probs.numpy()
            probs_train = probs[self.trained_indices]
            pred_labels = np.argmax(probs_train, axis=-1)
            mean_softmax = [probs_train[pred_labels==i].mean(axis=0) for i in range(10)]
            score = -pairwise_distances_argmin_min(probs, np.array(mean_softmax), metric=self.kl)[1]
        
        # ---------------------------------------
        
        elif method == 'SIRC':
            labels, logits, features = self.labels.to(self.device), self.logits.to(self.device), self.features.to(self.device)
            W, b = self.W.to(self.device), self.b.to(self.device)
            confs = probs.max(dim=-1).values.to(self.device)
            # confs = -torch.sum(probs * torch.log(probs + 1e-20), axis=-1).to(self.device)
            u = -torch.linalg.pinv(W) @ b
            D = 1000 if features.shape[-1] > 1500 else 512 
            if features.shape[-1] <= 512:
                D = features.shape[-1]//2
                
            # self.trained_indices = list(range(features.shape[0]))
            centered_feats_t = (features[self.trained_indices] - u)
            U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
            R = U[:,D:] # eigenvectors in columns
            assert R.shape[0] == features.shape[-1]
            vlogits_t = torch.norm((R.T @ centered_feats_t.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            centered_feats = (features - u)
            vlogits = torch.norm((R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            alpha = logits[self.trained_indices].max(dim=-1).values.mean() / vlogits_t.mean()
            vlogits = vlogits * alpha
            vlogits[self.trained_indices] += vlogits.mean()
            assert torch.isnan(vlogits).any()==False and torch.isinf(vlogits).any()==False
            soft, additional = self.sirc(confs, vlogits, s1_max=1.0)
            score = -soft + additional
            # return -soft, additional
            
        elif method == 'SIRC-':
            labels, logits, features = self.labels.to(self.device), self.logits.to(self.device), self.features.to(self.device)
            W, b = self.W.to(self.device), self.b.to(self.device)
            confs = probs.max(dim=-1).values.to(self.device)
            # confs = -torch.sum(probs * torch.log(probs + 1e-20), axis=-1).to(self.device)
            u = -torch.linalg.pinv(W) @ b
            D = 1000 if features.shape[-1] > 1500 else 512 
            if features.shape[-1] <= 512:
                D = features.shape[-1]//2
            # self.trained_indices = list(range(features.shape[0]))
            centered_feats_t = (features[self.trained_indices] - u)
            U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
            R = U[:,D:] # eigenvectors in columns
            assert R.shape[0] == features.shape[-1]
            vlogits_t = torch.norm((R.T @ centered_feats_t.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            centered_feats = (features - u)
            vlogits = torch.norm((R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            alpha = logits[self.trained_indices].max(dim=-1).values.mean() / vlogits_t.mean()
            vlogits = vlogits * alpha
            vlogits[self.trained_indices] += vlogits.mean()
            assert torch.isnan(vlogits).any()==False and torch.isinf(vlogits).any()==False
            soft, additional = self.sirc(confs, vlogits, s1_max=1.0)
            score = -soft - additional
            # score -= score.min()
            
        elif method == 'OCS':
            labels, features = self.labels.to(self.device), self.features.to(self.device)
            import copy
            ref_grads = copy.deepcopy(self.flatten_grads(self.model))
            _eg = self.compute_and_flatten_example_grads(self.model, features, labels.view(-1))
            _g = torch.mean(_eg, 0)
            score = -self.ocs(g=_g, eg=_eg, tau=10, ref_grads=ref_grads)
    
        elif method == 'EXP':
            labels, logits, features = self.labels.to(self.device), self.logits.to(self.device), self.features.to(self.device)
            W, b = self.W.to(self.device), self.b.to(self.device)
            confs = probs.max(dim=-1).values.to(self.device)
            # confs = -torch.sum(probs * torch.log(probs + 1e-20), axis=-1).to(self.device)
            u = -torch.linalg.pinv(W) @ b
            D = 1000 if features.shape[-1] > 1500 else 512 
            if features.shape[-1] <= 512:
                D = features.shape[-1]//2
            # self.trained_indices = list(range(features.shape[0]))
            centered_feats_t = (features[self.trained_indices] - u)
            U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
            R = U[:,D:] # eigenvectors in columns
            assert R.shape[0] == features.shape[-1]
            vlogits_t = torch.norm((R.T @ centered_feats_t.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            centered_feats = (features - u)
            vlogits = torch.norm((R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            alpha = probs[self.trained_indices].max(dim=-1).values.mean() / vlogits_t.mean()
            vlogits = vlogits * alpha # * self.generalized_entropy(probs.to(self.device), gamma=0.1, M=3)
            # vlogits[self.trained_indices] += vlogits.mean()
            assert torch.isnan(vlogits).any()==False and torch.isinf(vlogits).any()==False
            soft = (1.0 - confs).log()
            # soft = torch.sum(probs * torch.log(probs), axis=1)
            # soft = -self.shannon_entropy(probs)
            # soft = -self.generalized_entropy(probs, M=3)
            additional = torch.logaddexp(torch.zeros(len(vlogits), device=self.device), vlogits)
            return -soft, additional
            
        elif method == 'EXP2':
            labels, logits, features = self.labels.numpy(), self.logits.numpy(), self.features.numpy()
            W, b = self.W.numpy(), self.b.numpy()
            confs = probs.max(dim=-1).values.to(self.device)
            from sklearn.covariance import EmpiricalCovariance
            from numpy.linalg import norm, pinv
            from scipy.special import logsumexp
            DIM = 1000 if features.shape[-1] > 1500 else 512 
            if features.shape[-1] <= 512:
                DIM = features.shape[-1]//2
            u = -torch.linalg.pinv(W) @ b
            D = 1000 if features.shape[-1] > 1500 else 512 
            if self.features.shape[-1] <= 512:
                D = self.features.shape[-1]//2
            non_trained_mask = np.ones(self.features.shape[0], dtype=bool)
            non_trained_mask[self.trained_indices] = False
                
            centered_feats_t = (self.features[self.trained_indices] - u)
            U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
            R = U[:,D:] # eigenvectors in columns
            assert R.shape[0] == self.features.shape[-1]
            vlogits = torch.zeros(self.features.shape[0], D, dtype=self.features.dtype)
            vlogits_t = (R.T @ centered_feats_t.unsqueeze(dim=-1)).squeeze(dim=-1)
            vlogits[self.trained_indices] = vlogits_t
            
            centered_feats = (self.features[non_trained_mask] - u)
            # vlogits_t = vlogits[self.trained_indices]
            vlogits[non_trained_mask] = (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1)
            vlogits_norm = torch.norm(vlogits, p=2, dim=-1)
            assert torch.isnan(vlogits_norm).any()==False and torch.isinf(vlogits_norm).any()==False
            soft, additional = self.sirc(confs, vlogits_norm, s1_max=1.0)
            score = -soft + additional
            
        return score
    
class SIRCStrategy(DataSelectionStrategy):
    """
    Implementation of SIRCStrategy:
    """

    def __init__(self, trainloader, valloader, model, 
                loss_func, eta, device, num_classes, 
                linear_layer, selection_type, logger):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss_func, device, logger)
        self.eta = eta  # step size for the one step gradient update
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type

    def get_outputs_labels(self):
        if isinstance(self.trainloader.dataset[0], dict):
            for batch_idx, batch in enumerate(self.trainloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out, l1 = self.model(**batch, last=True, freeze=True)
                if batch_idx == 0:
                    trn_lbls = batch['labels'].view(-1, 1)
                    trn_logits = out
                    trn_features = l1
                else:
                    trn_lbls = torch.cat((trn_lbls, batch['labels'].view(-1, 1)), dim=0)
                    trn_logits = torch.cat((trn_logits, out), dim=0)
                    trn_features = torch.cat((trn_features, l1), dim=0)
        else:
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                out, l1 = self.model(inputs, last=True, freeze=True)
                if batch_idx == 0:
                    trn_lbls = targets.view(-1, 1)
                    trn_logits = out
                    trn_features = l1
                else:
                    trn_lbls = torch.cat((trn_lbls, targets.view(-1, 1)), dim=0)
                    trn_logits = torch.cat((trn_logits, out), dim=0)
                    trn_features = torch.cat((trn_features, l1), dim=0)
        return trn_lbls, trn_logits, trn_features
    
    @staticmethod
    def gram_schmidt(vectors):
        num_vectors, vector_size = vectors.shape
        basis = torch.zeros_like(vectors)
        for i in range(num_vectors):
            # Orthogonalize the current vector
            basis[i] = vectors[i].clone()
            for j in range(i):
                basis[i] -= torch.dot(vectors[i], basis[j]) / torch.dot(basis[j], basis[j]) * basis[j]
            # Normalize the orthogonalized vector
            basis[i] /= torch.norm(basis[i])

        return basis
    
    @staticmethod
    def vector_projection(vectors, basis):
        # v is the vector to be projected
        # basis is a matrix where each column is a basis vector of the subspace
        # Initialize the projected vector to zero
        projections = torch.zeros_like(vectors)
        for i, v in enumerate(vectors):
            for u in basis:  # Iterate over basis vectors
                projections[i] += torch.dot(v, u) / torch.dot(u, u) * u
        return projections
    
    @staticmethod
    def angle_between_vectors(vectors, projections):
        # Compute the angle in radians between two vectors        
        angles = torch.zeros([vectors.shape[0]])
        for i in range(vectors.shape[0]):
            dot_product = torch.dot(vectors[i], projections[i])
            norm_v1 = torch.norm(vectors[i])
            norm_v2 = torch.norm(projections[i])
            cosine_similarity = dot_product / (norm_v1 * norm_v2)
            angles[i] = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))
        return angles
    
    @staticmethod
    def knn_score(x, y, indexes, k):
        from sklearn.neighbors import KNeighborsClassifier
        def distance(x, y, **kwargs):
            return (((x-y)**2).sum())**0.5
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        model = KNeighborsClassifier(# metric=distance, p=2,
                                    metric='minkowski', p=2,
                                    # metric_params=dict(w=weights)
                                    )
        model.fit(x[indexes], y[indexes])
        idxs = model.kneighbors(x, k, False)
        distances = np.zeros(x.shape[0])
        for i, idx in enumerate(idxs):
            covering_score = (y[idx]==y[i]).sum() / k
            distance_score = np.dot(x[idx], x[i]).mean()
            distances[i] = (covering_score + distance_score) / 2
            assert covering_score <= 1.0 and distance_score <= 1.0
        return distances
    
    @staticmethod
    def kmeans_score(x, y, weights, k):
        from sklearn.cluster import KMeans
        from collections import Counter
        
        # x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        x = F.normalize(x.view(x.shape[0],-1), p=2, dim=1).detach()
        cluster = KMeans(n_clusters=k)
        cluster.fit(x, sample_weight=weights)
        distances = np.zeros(y.shape)
        for i in range(k):
            inds_per_cluster = (cluster.labels_==i)
            # count number of gt per cluster
            unique, counts = np.unique(y[inds_per_cluster], return_counts=True)
            # efficiency mapping gt to covering
            mapping_covering = np.zeros(unique.max()+1,dtype=counts.dtype)
            mapping_covering[unique] = counts
            covering_score = mapping_covering[y[inds_per_cluster]]*1.0/counts.sum()
            # calculate distance of sample to its centroid from L2 distance -> Cosine distance
            distance_to_centroids = 1 - cluster.transform(x[inds_per_cluster]) / 2
            distance_to_its_centroid = np.array([distance_to_centroids[i,j] for i, j in enumerate(cluster.labels_[inds_per_cluster])])
            max_distance_to_centroids = distance_to_centroids.max(axis=-1)
            distances[inds_per_cluster] = covering_score
        return distances
    
    @staticmethod
    def clusters_sample(n, feats):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import pairwise_distances_argmin_min
        feats = F.normalize(feats.view(feats.shape[0],-1), p=2, dim=1).detach().numpy()
        # feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        cluster = AgglomerativeClustering(n_clusters=n)
        labels = cluster.fit_predict(feats)
        nearest_sample_idxs = []
        for cluster_label in range(n):
            cluster_points = feats[labels==cluster_label]
            center = np.mean(cluster_points, axis=0)
            # Find the index of the sample nearest to the center
            nearest_index = pairwise_distances_argmin_min(center.reshape(1, -1), cluster_points)[0][0]
            nearest_sample_idxs.append(np.arange(feats.shape[0])[labels == cluster_label][nearest_index])
        return np.array(nearest_sample_idxs)
        
    def uncertainties(self):
        # get params
        labels, logits, features = self.get_outputs_labels()
        self.labels, self.logits, self.features = labels.detach().cpu().clone(), logits.detach().cpu().clone().type(torch.DoubleTensor), features.detach().cpu().clone()
        # get final fc layer 
        W = self.model.state_dict()["fc.weight"].detach().cpu().clone()
        b = self.model.state_dict()["fc.bias"].detach().cpu().clone()
        assert torch.isnan(W).any()==False and torch.isinf(W).any()==False
        oodCal = OODCal(model=self.train_model.fc, labels=self.labels, logits=self.logits, features=self.features, W=W, b=b, trained_indices=self.trained_indices, device=self.device)
        score = oodCal(method='SIRC')
        # score = oodCal(method='EXP2')
        # score_1, score_2 = oodCal(method='EXP')
        # if self.cur_epoch>200:
        #     score = score_1
        # else:
        #     score = score_1 + score_2
        # score -= score.min()
        # return score_1, score_2 # as minimum as choosen
        return score
    
    def select(self, budget, model_params):
        """
        Apply naive greedy method for data selection

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters

        Returns
        ----------
        greedySet: list
            List containing indices of the best datapoints,
        budget: Tensor
            Tensor containing gradients of datapoints present in greedySet
        """
        sirc_start_time = time.time()
        self.update_model(model_params)
        if self.selection_type == 'PerClass':
            self.get_labels(valid=True)
            idxs = []
            gammas = []
            pass
        elif self.selection_type == 'PerBatch':
            idxs = []
            gammas = []
            pass
        else:
            # score = self.uncertainties()
            
            # get params
            labels, logits, features = self.get_outputs_labels()
            self.labels, self.logits, self.features = labels.detach().cpu().clone(), logits.detach().cpu().clone().type(torch.DoubleTensor), features.detach().cpu().clone()
            # # get final fc layer 
            # W = self.model.state_dict()["fc.weight"].detach().cpu().clone()
            # b = self.model.state_dict()["fc.bias"].detach().cpu().clone()
            # assert torch.isnan(W).any()==False and torch.isinf(W).any()==False
            # oodCal = OODCal(model=self.train_model.fc, labels=self.labels, logits=self.logits, features=self.features, W=W, b=b, trained_indices=self.trained_indices, device=self.device)
                        
            # Gradients angle
            # self.grads_per_elem = self.calc_gradient(model_params)
            # self.compute_gradients()
            # centered_feats_t = self.grads_per_elem[self.trained_indices]
            # self.grads_per_elem = self.grads_per_elem / torch.norm(self.grads_per_elem, dim=1, keepdim=True)            
            # centered_feats_t = self.grads_per_elem[self.trained_indices]
            # U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
            # U,_,_ = torch.svd(centered_feats_t.T)
            # R = torch.nn.init.orthogonal_(U) # [:,D:] # eigenvectors in columns
            # assert R.shape[0] == self.grads_per_elem.shape[-1]
            # centered_feats = self.grads_per_elem
            # projections = self.vector_projection(centered_feats, R)
            # vlogits = self.angle_between_vectors(centered_feats, projections)
            # score = oodCal.sf(confs, vlogits, d=5, s1_max=1.0)
            
            # Confidence score
            confs = self.logits.softmax(dim=-1).max(dim=-1).values#.to(self.device)
            # soft = -confs.log()
            
            # Flexmatch
            thr = 0.95
            # l_thr = 0.5
            probs = self.logits.softmax(dim=-1)
            # learning effect (le)
            pseudo_mask = probs >= torch.max(probs, dim=1).values[:, None] # m x c
            lr_eff = (pseudo_mask * (probs >= thr)).sum(axis=0) # c
            # normalize le
            B = lr_eff / max(lr_eff.max(), len(probs) - lr_eff.sum()) # c
            # min_probs = (1-pseudo_mask.float()+probs).min(dim=0).values
            # mean_probs = np.nanmean(np.where((pseudo_mask*probs)==0, np.nan, pseudo_mask*probs), axis=0)
            # std_probs = np.nanstd(np.where((pseudo_mask*probs)==0, np.nan, pseudo_mask*probs), axis=0)
            T = (1.0 - B) * thr # c
            # L = (1.0 - B) * l_thr # c
            # T = torch.where(T>min_probs, T, torch.FloatTensor(mean_probs-3*std_probs))
            candidate = ~(((pseudo_mask * probs) >= T).sum(axis=1) >= 1)
            # candidate = torch.logical_and(~(((pseudo_mask * probs) >= T).sum(axis=1) >= 1), (((pseudo_mask * probs) >= L).sum(axis=1) >= 1))
            # if candidate.sum()==0: candidate = ~(((pseudo_mask * probs) >= T+min_probs+3*std_probs).sum(axis=1) >= 1)
            self.logger.debug('Flexmatch: %d, %s, %s', int(candidate.sum()), str(T.tolist()), str(lr_eff.tolist()))
            # print((((pseudo_mask * probs) <= T).sum(axis=1) >= 1)[:10], (((pseudo_mask * probs) >= L).sum(axis=1) >= 1)[:10], candidate[:10])
            # ViM
            # u = -torch.linalg.pinv(W) @ b
            # D = 1000 if features.shape[-1] > 1500 else 512 
            # if self.features.shape[-1] <= 512:
            #     D = self.features.shape[-1]//2
            # non_trained_mask = np.ones(self.features.shape[0], dtype=bool)
            # non_trained_mask[self.trained_indices] = False
            # centered_feats_t = (self.features[self.trained_indices] - u)
            # U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
            # R = U[:,D:] # eigenvectors in columns
            # assert R.shape[0] == self.features.shape[-1]
            # vlogits_t = torch.norm((R.T @ centered_feats_t.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            # centered_feats = (self.features - u)
            # vlogits = (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1)
            # vlogits_norm = torch.norm(vlogits, p=2, dim=-1)
            # alpha = self.logits[self.trained_indices].max(dim=-1).values.mean() / vlogits_t.mean()
            # vlogits_norm[non_trained_mask] = vlogits_norm[non_trained_mask] * alpha
            # vlogits_norm[self.trained_indices] += vlogits_norm.mean()
            
            # ViM_o
            # from sklearn.covariance import EmpiricalCovariance
            # from numpy.linalg import norm, pinv
            # from scipy.special import logsumexp
            # DIM = 1000 if self.features.shape[-1] > 1500 else 512 
            # if self.features.shape[-1] <= 512:
            #     DIM = self.features.shape[-1]//2
            # u = -np.matmul(pinv(W), b)
            # ec = EmpiricalCovariance(assume_centered=True)
            # ec.fit(self.features[self.trained_indices] - u)
            # eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            # NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
            # vlogits = torch.zeros(self.features.shape[0], DIM, dtype=self.features.dtype)
            # non_trained_mask = np.ones(self.features.shape[0], dtype=bool)
            # non_trained_mask[self.trained_indices] = False
            # vlogits[self.trained_indices] = torch.FloatTensor(np.matmul(self.features[self.trained_indices] - u, NS))
            # vlogits[non_trained_mask] = torch.FloatTensor(np.matmul(self.features[non_trained_mask] - u, NS))
            # vlogits_norm = torch.norm(vlogits, p=2, dim=-1)
            # # alpha = self.logits[self.trained_indices].max(dim=-1).values.mean() / vlogits[self.trained_indices].mean()
            # # vlogits_norm[non_trained_mask] = vlogits_norm[non_trained_mask] * alpha
            # # vlogits_norm[self.trained_indices] += vlogits_norm.mean()
            # vlogits_norm = (vlogits_norm - vlogits_norm[self.trained_indices].min()) / \
            #                 (vlogits_norm.max() - vlogits_norm[self.trained_indices].min())
            # # vlogits_norm -= vlogits_norm[self.trained_indices].min()
            # vlogits_norm = torch.clip(vlogits_norm, min=0.0)
            
            # ViM_x
            # u = -torch.linalg.pinv(W) @ b
            # D = 1000 if features.shape[-1] > 1500 else 512 
            # if self.features.shape[-1] <= 512:
            #     D = self.features.shape[-1]//2
            # non_trained_mask = np.ones(self.features.shape[0], dtype=bool)
            # non_trained_mask[self.trained_indices] = False
                
            # centered_feats_t = (self.features[self.trained_indices] - u)
            # U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
            # R = U[:,D:] # eigenvectors in columns
            # assert R.shape[0] == self.features.shape[-1]
            # vlogits = torch.zeros(self.features.shape[0], D, dtype=self.features.dtype)
            # vlogits_t = (R.T @ centered_feats_t.unsqueeze(dim=-1)).squeeze(dim=-1)
            # vlogits[self.trained_indices] = vlogits_t
            
            # centered_feats = (self.features[non_trained_mask] - u)
            # # vlogits_t = vlogits[self.trained_indices]
            # vlogits[non_trained_mask] = (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1)
            # vlogits_norm = torch.norm(vlogits, p=2, dim=-1)
            # alpha = self.logits[self.trained_indices].max(dim=-1).values.mean() / vlogits_norm[self.trained_indices].mean()
            # vlogits_norm = -vlogits_norm * alpha + torch.logsumexp(self.logits, axis=-1)
            # # # vlogits_norm[self.trained_indices] += vlogits_norm.mean()
            # # # print(vlogits[self.trained_indices].max(), vlogits[self.trained_indices].min(), vlogits.max(), vlogits.min())
            # # k_score = self.kmeans_score(self.features.squeeze(), self.labels.squeeze(), None, 10)
            # k_score = self.knn_score(vlogits.squeeze(), self.labels.squeeze(), self.trained_indices, 10)
            # k_score = torch.FloatTensor(1-k_score)
             
            # SIRC
            s1 = 1.0 - confs.to(self.device)
            # s2 = vlogits_norm.to(self.device)
            # s3 = k_score.to(self.device)
            s1 = torch.clip(s1, min=1e-64)
            soft = -s1.log() # <--
            # s1_mean = torch.mean(s1)
            # s1_std = torch.std(s1)
            # s2_mean = torch.mean(s2)
            # s2_std = torch.std(s2)
            # # s3_mean = torch.mean(s3[self.trained_indices])
            # # s3_std = torch.std(s3[self.trained_indices])
            # s2 = s1_mean + (s2-s2_mean)*s1_std/s2_std
            # s3 = s1_mean + (s3-s3_mean)*s1_std/s3_std
            # s2 = (s2-s2_mean-3*s2_std)/s2_std
            # s3 = (s3-s3_mean-3*s3_std)/s3_std
            # additional = - s3.log() # torch.logaddexp(torch.zeros(len(s2), device=self.device), s2) #+ torch.logaddexp(torch.zeros(len(s3), device=self.device), s3)
            # additional = -s3.log() #+ vlogits_norm.log()
            # soft, additional = oodCal.sirc(confs.to(self.device), vlogits_norm.to(self.device), s1_max=1.0)
            
            # if self.cur_epoch<50:
            #     score = soft
            #     additional = torch.zeros(len(s2), device=self.device)
            # elif self.cur_epoch%20!=0:
            #     additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), -s2)
            #     score = soft + additional
            # else:
            #     additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), s2)
            #     score = soft + additional
            # additional = -s2.log()
            # additional = (s2 - s2.min()) / (s2.max() - s2.min()) * (soft.max() - soft.min()) + soft.min()
            
            # additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), s2)
            additional = torch.zeros(len(s1), device=self.device)
            score = soft #+ additional # (soft + additional) / 2 # additional # soft #  
            # print(soft[:10], additional[:10])
            # soft, additional = oodCal.sirc(confs.to(self.device), vlogits.to(self.device), s1_max=1.0)
            # score = -soft + additional
            # score = torch.logaddexp(torch.zeros(vlogits.shape[0], device=self.device), vlogits.to(self.device))
            # idxs_ood = torch.where(vlogits>vlogits_t.min())[0]
            # idxs_ood = torch.IntTensor(list((set(idxs_ood).difference(set(self.trained_indices)))))
            # sorted, indices = torch.sort(score, descending=True)
            # idxs = indices[:budget].tolist()
            # gammas = vlogits[idxs].tolist()

            # score = soft[idxs_ood]
            # labels = self.labels.to(self.device)
            # _, indices = torch.sort(score.view(-1), descending=True)
            # budget_per_class = budget // self.num_classes
            # idxs, gammas = [], []
            # selected_idxs = [[] for i in range(self.num_classes)]
            # selected_count = 0
            # i = 0
            # while selected_count<budget:
            #     index = indices[i]
            #     label = labels[index]
            #     i += 1
            #     if len(selected_idxs[label])<budget_per_class: 
            #         selected_idxs[label].append(index)
            #         selected_count += 1
            # idxs = idxs_ood[[int(i) for q in selected_idxs for i in q]]
            # gammas = soft[idxs].tolist()
            
            if isinstance(score, np.ndarray): score = torch.FloatTensor(score)
            
            # # visualization
            # import matplotlib.pyplot as plt
            # _, indices = torch.sort(score.view(-1), descending=False) 
            # x = np.arange(score.shape[0]).reshape(-1)
            # plt.plot(soft[indices].view(-1).detach().cpu().clone().numpy(), label = "soft", alpha=0.5) 
            # plt.plot(additional[indices].view(-1).detach().cpu().clone().numpy(), label = "additional", alpha=0.5) 
            # plt.plot(score[indices].view(-1).detach().cpu().clone().numpy(), label = "score", alpha=0.5)
            # plt.legend() 
            # plt.savefig(f'exp_{self.cur_epoch:3d}.png')
            # plt.close()
            
            # if self.cur_epoch>=0:
            # select by balance candidate
            idxs = []
            candidate_numb = candidate.sum()
            if candidate.sum()<budget:
                idxs += [int(i) for i, c in enumerate(candidate) if c==1]
                candidate = ~candidate
                print('idxs: ', len(idxs))
                
            if candidate_numb>budget*5:
                # # select by bin
                # self.logger.debug('select by bin')
                # n_bin = 10
                # budget_bin = math.ceil((budget-len(idxs))/n_bin)
                # selected = []
                # s = score[candidate]
                # _, indices_bin = torch.sort(s.view(-1), descending=False)
                # splited_bin = torch.split(indices_bin, int(s.shape[0]/n_bin))
                # for idxs_bin in splited_bin:
                #     sorted_s, indices_s = torch.sort(s.view(-1)[idxs_bin], descending=False)
                #     selected += idxs_bin[indices_s][:budget_bin].tolist()
                # idxs += np.arange(candidate.shape[0])[candidate][selected].tolist()
                # gammas = score[idxs].tolist()
                # # select by cluster
                # selected = self.clusters_sample(budget-len(idxs), feats=self.features[candidate]) # clustering
                # idxs += np.arange(candidate.shape[0])[candidate][selected].tolist()
                # gammas = score[idxs].tolist()
                # select by cluster bin
                self.logger.debug('select by cluster bin')
                n_bin = 200
                budget_bin = math.ceil((budget-len(idxs))/n_bin)
                selected = []
                features = self.features[candidate]
                s = score[candidate]
                from sklearn.cluster import KMeans
                features = F.normalize(features.view(features.shape[0],-1), p=2, dim=1).detach()
                cluster = KMeans(n_clusters=n_bin)
                cluster.fit(features)
                for i in range(n_bin):
                    inds_per_cluster = (cluster.labels_==i)
                    sorted_s, indices_s = torch.sort(s[inds_per_cluster].view(-1), descending=False)
                    selected += np.arange(candidate.shape[0])[candidate][inds_per_cluster][indices_s.cpu().numpy()][:budget_bin].tolist()
                idxs += selected
                gammas = score[idxs].tolist()
            else:
                # # select by sort
                # score[~candidate] += score.mean()
                # sorted_gains, indices = torch.sort(score.view(-1), descending=False)
                # idxs, gammas = indices[:budget].tolist(), sorted_gains[:budget].tolist()
                # select by balance score
                self.logger.debug('select by balance score')
                total_scores = [0 for i in range(self.num_classes)]
                selected_idxs = [[] for i in range(self.num_classes)]
                queue_idxs = [[] for i in range(self.num_classes)]
                sorted_gains, indices = torch.sort(score.view(-1), descending=False)
                selected_count = 0
                i = 0
                while i<score.shape[0]:
                    idx = indices[i]
                    i += 1
                    if candidate[idx]==0: continue
                    label = self.labels[idx]
                    queue_idxs[label].append(idx)
                    
                while selected_count<budget-len(idxs):
                    queue_idxs_mask = [True if len(q)>0 else False for q in queue_idxs]
                    min_total_score = min([total_scores[i] for i, m in enumerate(queue_idxs_mask) if m==True])
                    min_score_label = [i for i, s in enumerate(total_scores) if (queue_idxs_mask[i]==True and s==min_total_score)][0]
                    selected_idx = queue_idxs[min_score_label].pop(0)
                    selected_idxs[min_score_label].append(selected_idx)
                    total_scores[min_score_label] += score[selected_idx]
                    selected_count += 1
                idxs += [int(i) for q in selected_idxs for i in q]
                gammas = score[idxs].tolist()
            
            self.logger.debug('Number: %s', len(list(set(idxs))))
            # # select by cluster  
            # selected = self.clusters_sample(budget, feats=self.features[candidate]) # clustering
            # idxs += np.arange(candidate.shape[0])[candidate][selected].tolist()
            # gammas = score[idxs].tolist()
            
            # # select by cascade
            # s2, s1 = score
            # _, indices_s1 = torch.sort(s1.view(-1), descending=False)
            # budget_s1 = int(0.4 * s1.shape[0])
            # idxs_s1 = indices_s1[:budget_s1]
            # s2 = s2[idxs_s1]
            # sorted_s2, indices_s2 = torch.sort(s2.view(-1), descending=False)
            # idxs, gammas = idxs_s1[indices_s2][:budget].tolist(), sorted_s2[:budget].tolist()
            
            # # select by bin
            # n_bin = 10
            # budget_bin = int(budget/n_bin)
            # idxs, gammas = [], []
            
            # s2, s1 = score
            # _, indices_s1 = torch.sort(s1.view(-1), descending=False)
            # splited_s1 = torch.split(indices_s1, int(self.N_trn/n_bin))
            
            # for idxs_bin in splited_s1:
            #     sorted_s2, indices_s2 = torch.sort(s2.view(-1)[idxs_bin], descending=False)
            #     idxs += idxs_bin[indices_s2][:budget_bin].tolist()
            # gammas = (s1 + s2)[idxs].tolist()
            
            # # select by balance quantity
            # labels = self.labels.to(self.device)
            # _, indices = torch.sort(score.view(-1), descending=False)
            # budget_per_class = budget // self.num_classes
            # idxs, gammas = [], []
            # selected_idxs = [[] for i in range(self.num_classes)]
            # selected_count = 0
            # i = 0
            # while selected_count<budget:
            #     index = indices[i]
            #     label = labels[index]
            #     i += 1
            #     if len(selected_idxs[label])<budget_per_class: 
            #         selected_idxs[label].append(index)
            #         selected_count += 1
            # idxs = [int(i) for q in selected_idxs for i in q]
            # gammas = score[idxs].tolist()
            # assert len(idxs)==budget
            
        # for i in range(self.num_classes):
        #     class_idxs = torch.where(self.labels.view(-1)==i)[0].tolist()
        #     inter_idxs = list(set(idxs).intersection(set(class_idxs)))
        #     self.logger.debug('Class %d: %d %.4f %.2f %.2f', i, len(inter_idxs), score[inter_idxs].sum(),\
        #                         soft[inter_idxs].sum(), additional[inter_idxs].sum())
        sirc_end_time = time.time()
        self.logger.debug("SIRC algorithm Subset Selection time is: %.4f", sirc_end_time - sirc_start_time)
        return idxs, torch.FloatTensor(gammas)
    
    def set_trained_indices(self, subset_indices):
        self.trained_indices = subset_indices
        
    def set_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch
        
    def set_train_model(self, train_model):
        self.train_model = train_model