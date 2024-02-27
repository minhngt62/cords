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
    def __init__(self, labels, logits, features, W, b, trained_indices) -> None:
        self.labels = labels
        self.logits = logits
        self.features = features
        self.W = W
        self.b = b
        self.trained_indices = trained_indices
    
    @staticmethod
    def kl(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    @staticmethod
    def generalized_entropy(softmax_id_val, gamma, M):
            probs =  softmax_id_val 
            probs_sorted = np.sort(probs, axis=1)[:,-M:]
            scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
            return -scores 
    @staticmethod
    def shannon_entropy(softmax_id_val):
            probs =  softmax_id_val 
            scores = np.sum(probs* np.log(probs), axis=1)   
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
    def sirc(s1, s2, s1_max=1):
        "Combine 2 confidence metrics with SIRC."
        # use logarithm for stability
        soft = (s1_max - s1).log() # <--
        s2_mean = torch.mean(s2)
        s2_std = torch.std(s2)
        # s2 = s2-(s2-s2_mean).min()/s2_std
        # additional = s2.log()
        s2 = (s2-s2_mean-(s2-s2_mean).max())/s2_std
        additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), s2)
        # additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), -1/s2_std * (-s2-(-s2_mean-3*s2_std))) 
        assert torch.isnan(soft).any()==False and torch.isinf(soft).any()==False
        assert torch.isnan(additional).any()==False and torch.isinf(additional).any()==False
        
        return -soft + additional # return as confidence default -soft -->

    def __call__(self, **kwargs) -> Any:
        method = kwargs['method']
        print(f'\n{method}')
        
        probs = self.logits.softmax(dim=-1)
        confs = probs.max(dim=-1).values
        
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
            logit_clip = torch.clip(self.features, a_min=None, a_max=clip) @ self.W.T + self.b
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
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(self.features[self.trained_indices] - u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

            residual_score = norm(np.matmul(self.features - u, NS), axis=-1)
            GEN_score = self.generalized_entropy(probs, gamma=0.1, M=100)
            score =  residual_score * GEN_score
        
        # ---------------------------------------
        
        elif method == 'GradNorm':
            score = self.gradnorm(self.features, self.W, self.b)

        # ---------------------------------------
        
        elif method == 'KL-Matching':
            from sklearn.metrics import pairwise_distances_argmin_min
            pred_labels = np.argmax(probs, axis=-1)
            probs_train = probs[self.trained_indices]
            mean_softmax = [probs_train[pred_labels==i].mean(axis=0) for i in range(self.num_classes)]
            score = -pairwise_distances_argmin_min(probs, np.array(mean_softmax), metric=kl)[1]
        
        # ---------------------------------------
        
        elif method == 'SIRC':
            u = -torch.linalg.pinv(self.W) @ self.b
            D = 1000 if self.features.shape[-1] > 1500 else 512 
            if self.features.shape[-1] <= 512:
                D = self.features.shape[-1]//2
            centered_feats_t = (self.features[self.trained_indices] - u)
            U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
            R = U[:,D:] # eigenvectors in columns
            assert R.shape[0] == self.features.shape[-1]
            vlogits_t = torch.norm((R.T @ centered_feats_t.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            centered_feats = (self.features - u)
            vlogits = torch.norm((R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
            alpha = self.logits[self.trained_indices].max(dim=-1).values.mean() / vlogits_t.mean()
            vlogits = vlogits * alpha
            assert torch.isnan(vlogits).any()==False and torch.isinf(vlogits).any()==False
            score = self.sirc(confs, vlogits, s1_max=1+1e-7)
            score -= score.min()
        
        return score
    
class SIRCStrategy(DataSelectionStrategy):
    """
    Implementation of GLISTER-ONLINE Strategy from the paper :footcite:`killamsetty2021glister`  for supervised learning frameworks.
    GLISTER-ONLINE methods tries to solve the  bi-level optimization problem given below:
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

        # Active FT
        self.temperature = 0.07
        self.balance = 1.0

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
        
    def uncertainties(self):
        def sirc(s1, s2, s1_max=1):
            "Combine 2 confidence metrics with SIRC."
            # use logarithm for stability
            soft = (s1_max - s1).log() # <--
            s2_mean = torch.mean(s2)
            s2_std = torch.std(s2)
            # s2 = s2-(s2-s2_mean).min()/s2_std
            # additional = s2.log()
            s2 = (s2-s2_mean-(s2-s2_mean).max())/s2_std
            additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), s2)
            # additional = torch.logaddexp(torch.zeros(len(s2), device=self.device), -1/s2_std * (-s2-(-s2_mean-3*s2_std))) 
            assert torch.isnan(soft).any()==False and torch.isinf(soft).any()==False
            assert torch.isnan(additional).any()==False and torch.isinf(additional).any()==False
            
            return -soft + additional # return as confidence default -soft -->
        
        # get params
        labels, logits, features = self.get_outputs_labels()
        self.labels, self.logits, self.features = labels.to(self.device), logits.to(self.device), features.to(self.device)
        # get final fc layer 
        W = self.model.state_dict()["fc.weight"].detach().clone()
        b = self.model.state_dict()["fc.bias"].detach().clone()
        assert torch.isnan(W).any()==False and torch.isinf(W).any()==False
        
        # logits = logits.type(torch.DoubleTensor).to(self.device)
        probs = self.logits.softmax(dim=-1)
        confs = probs.max(dim=-1).values
        # sims = self.get_sim()
        
        # vlogits
        u = -torch.linalg.pinv(W) @ b
        # size of subspace
        # pretty much just a heuristic
        D = 1000 if self.features.shape[-1] > 1500 else 512 
        if self.features.shape[-1] <= 512:
            D = self.features.shape[-1]//2
        centered_feats_t = (self.features[self.trained_indices] - u)
        # L, Q = torch.linalg.eigh(centered_feats_t.T@centered_feats_t)
        # U = Q[torch.argsort(L)]
        U = torch.linalg.eigh(centered_feats_t.T@centered_feats_t).eigenvectors.flip(-1)
        R = U[:,D:] # eigenvectors in columns
        assert R.shape[0] == self.features.shape[-1]
        vlogits_t = torch.norm((R.T @ centered_feats_t.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
        
        centered_feats = (self.features - u)
        # L, Q = torch.linalg.eigh(centered_feats.T@centered_feats)
        # U = Q[torch.argsort(L)]
        # U = torch.linalg.eigh(centered_feats.T@centered_feats).eigenvectors.flip(-1)
        # R = U[:,D:] # eigenvectors in columns
        vlogits = torch.norm((R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), p=2, dim=-1)
        alpha = self.logits[self.trained_indices].max(dim=-1).values.mean() / vlogits_t.mean()
        vlogits = vlogits * alpha
        assert torch.isnan(vlogits).any()==False and torch.isinf(vlogits).any()==False
        # vlogits = (vlogits-vlogits.min())/(vlogits.max()-vlogits.min())
        
        # grads
        # grads = self.get_grads().sum(dim=-1)
        # grads = (grads-grads.min())/(grads.max()-grads.min())
        # grads = grads * self.logits.max(dim=-1).values.mean() / grads.mean() - torch.logsumexp(self.logits, dim=-1)
        # grads = -grads
        # grads -= grads.min()
        # vlograds
        # centered_feats = (self.features - vim_params["u"]).unsqueeze(dim=-1)   
        # vlogits = torch.norm((vim_params["R"].T @ centered_feats).squeeze(dim=-1), p=2, dim=-1)
        # import matplotlib.pyplot as plt
        # print(grads.shape, vlogits.shape)
        # plt.plot(vlogits.clone().cpu().detach().numpy(), color="red", alpha=0.5)
        # plt.plot(grads.clone().cpu().detach().numpy(), color="green", alpha=0.5)
        # plt.savefig('vlograd.png')
        # vlograds = vlogits * grads
        # # vlograds = torch.log(vlograds)
        # vlograds = vlograds * self.logits.max(dim=-1).values.mean() / vlograds.mean() - torch.logsumexp(self.logits, dim=-1)
        # get_sirc_params
        vlogits_sirc = sirc(confs, vlogits, s1_max=1+1e-7) # <--
        vlogits_sirc -= vlogits_sirc.min()
        # vlogits_sirc[self.trained_indices] = vlogits_sirc.max()
        # Debug indexes
        # vidxs = torch.sort(grads.view(-1), descending=True)[1][:4500].tolist()
        # gidxs = torch.sort(vlogits_sirc.view(-1), descending=False)[1][:4500].tolist()
        # self.logger.debug('Debug idxs: %5d', len(set(vidxs).intersection(set(gidxs))))
        return vlogits_sirc # as minimum as choosen
    
    # ActiveFT
    def get_sim(self):
        features = F.normalize(self.features, dim=1)
        centroids = self.features[self.trained_indices].clone()
        centroids = F.normalize(centroids, dim=1)
        prod = torch.matmul(features, centroids.transpose(1, 0))  # (n, k)
        prod = prod / self.temperature
        prod_exp = torch.exp(prod)
        prod_exp_pos, pos_k = torch.max(prod_exp, dim=1)  # (n, )

        cent_prod = torch.matmul(centroids.detach(), centroids.transpose(1, 0))  # (k, k)
        cent_prod = cent_prod / self.temperature
        cent_prod_exp = torch.exp(cent_prod)
        cent_prob_exp_sum = torch.sum(cent_prod_exp, dim=0)  # (k, )

        sim = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        return -sim
    
    # Grads
    def get_grads(self):
        W = self.model.state_dict()["fc.weight"].detach().clone()
        b = self.model.state_dict()["fc.bias"].detach().clone()
        logits = (self.features @ W.T + b).clone().detach().requires_grad_(True)
        probs = logits.softmax(dim=-1)
        labels = probs.max(dim=-1)[-1]
        idxs = torch.where(probs.max(dim=-1).values<0.5)[0]
        labels[idxs] = self.labels.view(-1)[idxs]
        # print(labels.view(-1), self.labels.view(-1))
        loss = self.loss(probs, labels.view(-1)).mean() # GT or psudo-label
        grads = torch.autograd.grad(loss, logits)[0]
        return torch.abs(grads)
    
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
            # vim_params_dict = self.get_params()
            score = OODCal()
            # select by balance score
            # total_scores = [0 for i in range(self.num_classes)]
            # selected_idxs = [[] for i in range(self.num_classes)]
            # queue_idxs = [[] for i in range(self.num_classes)]
            # sorted_gains, indices = torch.sort(vlogits.view(-1), descending=False)
            # selected_count = 0
            # i = 0
            # while selected_count<budget and i<vlogits.shape[0]:
            #     # print(i, selected_count)
            #     idx = indices[i]
            #     label = self.labels[idx]
            #     queue_idxs[label].append(idx)
                
            #     min_score_label = total_scores.index(min(total_scores))
            #     i += 1
            #     if len(queue_idxs[min_score_label])>0:
            #         selected_idx = queue_idxs[min_score_label].pop(0)
            #         selected_idxs[min_score_label].append(selected_idx)
            #         total_scores[min_score_label] += vlogits[selected_idx]
            #         selected_count += 1
            # idxs = [int(i) for q in selected_idxs for i in q]
            # gammas = vlogits[idxs].tolist()
            # select by sort
            sorted_gains, indices = torch.sort(vlogits.view(-1), descending=False)
            idxs, gammas = indices[:budget].tolist(), sorted_gains[:budget].tolist()
            # gammas = [max(gammas)-g for g in gammas]
            for i in range(self.num_classes):
                class_idxs = torch.where(self.labels.view(-1)==i)[0].tolist()
                inter_idxs = list(set(idxs).intersection(set(class_idxs)))
                self.logger.debug('Class %d: %d %.4f', i, len(inter_idxs), vlogits[inter_idxs].sum())
        sirc_end_time = time.time()
        self.logger.debug("SIRC algorithm Subset Selection time is: %.4f", sirc_end_time - sirc_start_time)
        return idxs, torch.FloatTensor(gammas)

# import math
# import random
# import time
# import torch
# import torch.nn.functional as F
# from .dataselectionstrategy import DataSelectionStrategy
# from torch.utils.data import Subset, DataLoader
# import numpy as np


# class SIRCStrategy(DataSelectionStrategy):
#     """
#     Implementation of GLISTER-ONLINE Strategy from the paper :footcite:`killamsetty2021glister`  for supervised learning frameworks.
#     GLISTER-ONLINE methods tries to solve the  bi-level optimization problem given below:

#     .. math::
#         \\overbrace{\\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}} L_V(\\underbrace{\\underset{\\theta}{\\operatorname{argmin\\hspace{0.7mm}}} L_T( \\theta, S)}_{inner-level}, {\\mathcal V})}^{outer-level}

#     In the above equation, :math:`\\mathcal{U}` denotes the training set, :math:`\\mathcal{V}` denotes the validation set that guides the subset selection process, :math:`L_T` denotes the
#     training loss, :math:`L_V` denotes the validation loss, :math:`S` denotes the data subset selected at each round,  and :math:`k` is the budget for the subset.

#     Since, solving the complete inner-optimization is expensive, GLISTER-ONLINE adopts a online one-step meta approximation where we approximate the solution to inner problem
#     by taking a single gradient step.

#     The optimization problem after the approximation is as follows:

#     .. math::
#         \\overbrace{\\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}} L_V(\\underbrace{\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S)}_{inner-level}, {\\mathcal V})}^{outer-level}

#     In the above equation, :math:`\\eta` denotes the step-size used for one-step gradient update.

#     GLISTER-ONLINE also makes an additional approximation called Taylor-Series approximation to easily solve the outer problem using a greedy selection algorithm.
#     The Taylor series approximation is as follows:

#     .. math::
#         L_V(\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S), {\\mathcal V}) \\approx L_V(\\theta) - \\eta {\\nabla_{\\theta}L_T(\\theta, S)}^T \\nabla_{\\theta}L_V(\\theta, {\\mathcal V})

#     The Optimization problem after the Taylor series approximation is as follows:

#     .. math::
#         \\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}}L_V(\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S), {\\mathcal V}) \\approx L_V(\\theta) - \\eta {\\nabla_{\\theta}L_T(\\theta, S)}^T \\nabla_{\\theta}L_V(\\theta, {\\mathcal V})

#     Taylor's series approximation reduces the time complexity by reducing the need of calculating the validation loss for each element during greedy selection step which
#     means reducing the number of forward passes required.

#     GLISTER-ONLINE is an adaptive subset selection algorithm that tries to select a subset every :math:`L` epochs and the parameter `L` can be set in the original training loop.

#     Parameters
# 	----------
#     trainloader: class
#         Loading the training data using pytorch DataLoader
#     valloader: class
#         Loading the validation data using pytorch DataLoader
#     model: class
#         Model architecture used for training
#     loss_func: object
#         Loss function object
#     eta: float
#         Learning rate. Step size for the one step gradient update
#     device: str
#         The device being utilized - cpu | cuda
#     num_classes: int
#         The number of target classes in the dataset
#     linear_layer: bool
#         If True, we use the last fc layer weights and biases gradients
#         If False, we use the last fc layer biases gradients
#     selection_type: str
#         Type of selection algorithm -
#         - 'PerBatch' : PerBatch method is where GLISTER algorithm is applied on each minibatch data points.
#         - 'PerClass' : PerClass method is where GLISTER algorithm is applied on each class data points seperately.
#         - 'Supervised' : Supervised method is where GLISTER algorithm is applied on entire training data.
#     greedy: str
#         Type of greedy selection algorithm -
#         - 'RGreedy' : RGreedy Selection method is a variant of naive greedy where we just perform r rounds of greedy selection by choosing k/r points in each round.
#         - 'Stochastic' : Stochastic greedy selection method is based on the algorithm presented in this paper :footcite:`mirzasoleiman2014lazier`
#         - 'Naive' : Normal naive greedy selection method that selects a single best element every step until the budget is fulfilled
#     logger: class
#         logger class for logging the information
#     r : int, optional
#         Number of greedy selection rounds when selection method is RGreedy (default: 15)
#     """

#     def __init__(self, trainloader, valloader, model, 
#                 loss_func, eta, device, num_classes, 
#                 linear_layer, selection_type, logger):
#         """
#         Constructor method
#         """
#         super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss_func, device, logger)
#         self.eta = eta  # step size for the one step gradient update
#         self.init_out = list()
#         self.init_l1 = list()
#         self.selection_type = selection_type

#     def get_outputs_labels(self, valid=False):
#         if isinstance(self.trainloader.dataset[0], dict):
#             for batch_idx, batch in enumerate(self.trainloader):
#                 batch = {k: v.to(self.device) for k, v in batch.items()}
#                 out, l1 = self.model(**batch, last=True, freeze=True)
#                 if batch_idx == 0:
#                     trn_lbls = batch['labels'].view(-1, 1)
#                     trn_logits = out
#                     trn_features = l1
#                 else:
#                     trn_lbls = torch.cat((trn_lbls, batch['labels'].view(-1, 1)), dim=0)
#                     trn_logits = torch.cat((trn_logits, out), dim=0)
#                     trn_features = torch.cat((trn_features, l1), dim=0)
#         else:
#             for batch_idx, (inputs, targets) in enumerate(self.trainloader):
#                 inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
#                 out, l1 = self.model(inputs, last=True, freeze=True)
#                 if batch_idx == 0:
#                     trn_lbls = targets.view(-1, 1)
#                     trn_logits = out
#                     trn_features = l1
#                 else:
#                     trn_lbls = torch.cat((trn_lbls, targets.view(-1, 1)), dim=0)
#                     trn_logits = torch.cat((trn_logits, out), dim=0)
#                     trn_features = torch.cat((trn_features, l1), dim=0)
#         return trn_lbls, trn_logits, trn_features
    
#     def get_grads(self, features, labels):
#         out = self.model.fc(features)
#         loss = self.loss(out, labels.view(-1)).sum() # GT or psudo-label
#         grads = torch.autograd.grad(loss, out)[0]
#         return grads
    
#     def get_params(self):
#         """Calculate params for vim and KL matching on training set."""

#         # this is being done on the training set this time
#         labels, logits, features = self.get_outputs_labels()
#         labels, logits, features = labels.to(self.device), logits.to(self.device), features.to(self.device)

#         # get final fc layer 
#         try:
#             W = self.model.state_dict()["fc.weight"].detach().clone()
#             b = self.model.state_dict()["fc.bias"].detach().clone()
#         except:
#             W = self.model.state_dict()["classifier.weight"].detach().clone()
#             b = self.model.state_dict()["classifier.bias"].detach().clone()

#         u = -torch.linalg.pinv(W) @ b

#         # size of subspace
#         # pretty much just a heuristic
#         D = 1000 if features.shape[-1] > 1500 else 512 
#         if features.shape[-1] <= 512:
#             D = features.shape[-1]//2
#         centered_feats = features - u
#         U = torch.linalg.eigh(
#             centered_feats.T@centered_feats
#         ).eigenvectors.flip(-1)
#         R = U[:,D:] # eigenvectors in columns
#         assert R.shape[0] == features.shape[-1]
#         vlogits = torch.norm(
#             (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), 
#             p=2, dim=-1
#         )
#         alpha = logits.max(dim=-1).values.mean() / vlogits.mean()
#         alpha = alpha.item()

#         vim_params_dict = {
#             "alpha": alpha,
#             "u":  u,
#             "R": R
#         }

#         centered_feats = F.normalize(centered_feats, dim=-1)
#         U = torch.linalg.eigh(
#             centered_feats.T@centered_feats
#         ).eigenvectors.flip(-1) # rev order to descending eigenvalue
#         R = U[:,D:] # eigenvectors in columns
#         assert R.shape[0] == features.shape[-1]
#         vlogits = torch.norm(
#             (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1), 
#             p=2, dim=-1
#         )
#         alpha = logits.max(dim=-1).values.mean() / vlogits.mean()
#         alpha = alpha.item()
#         vim_params_dict.update({
#             "norm_alpha": alpha,
#             "norm_R": R
#             })
#         return self.uncertainties(logits=logits, features=features, vim_params=vim_params_dict, labels=labels)
        
#     def uncertainties(self, logits=None, features=None, vim_params=None, labels=None):
#         def get_sirc_params(mean, std):
#             # remember that the values are negative
#             a = -mean - 3*std
#             # investigate effect of varying b
#             b = 1/std
#             return a, b
        
#         def sirc(s1, s2, a, b, s1_max=1):
#             "Combine 2 confidence metrics with SIRC."
#             # use logarithm for stability
#             soft = (s1_max - s1).log()
#             additional = torch.logaddexp(
#                 torch.zeros(len(s2), device=self.device),
#                 -b * (s2 - a) 
#             )
#             return soft - additional # return as confidence
        
#         # logits = logits.type(torch.DoubleTensor).to(self.device)
#         grads = self.get_grads(features, labels).sum(dim=-1)
#         probs = logits.softmax(dim=-1)
#         conf = probs.max(dim=-1).values
#         # for broadcasting
#         centered_feats = (features - vim_params["u"]).unsqueeze(dim=-1)   
#         vlogits = torch.norm((vim_params["R"].T @ centered_feats).squeeze(dim=-1), p=2, dim=-1) * vim_params["alpha"]
#         # metric_stats
#         vlogits_mean = torch.mean(vlogits)
#         vlogits_std = torch.std(vlogits)
#         # get_sirc_params
#         sirc_a, sirc_b = get_sirc_params(vlogits_mean, vlogits_std)
#         vlogits_sirc = -sirc(conf, -vlogits, sirc_a, sirc_b, s1_max=1+1e-5)
#         # return -(vlogits - torch.logsumexp(logits, dim=-1))
#         return vlogits_sirc
    
#     def select(self, budget, model_params):
#         """
#         Apply naive greedy method for data selection

#         Parameters
#         ----------
#         budget: int
#             The number of data points to be selected
#         model_params: OrderedDict
#             Python dictionary object containing models parameters

#         Returns
#         ----------
#         greedySet: list
#             List containing indices of the best datapoints,
#         budget: Tensor
#             Tensor containing gradients of datapoints present in greedySet
#         """
#         sirc_start_time = time.time()
#         self.update_model(model_params)
#         if self.selection_type == 'PerClass':
#             self.get_labels(valid=True)
#             idxs = []
#             gammas = []
#             pass
#         elif self.selection_type == 'PerBatch':
#             idxs = []
#             gammas = []
#             pass
#         else:
#             vlogits = self.get_params()
#             sorted_gains, indices = torch.sort(vlogits.view(-1), descending=False)
#             idxs, gammas = indices[:budget].tolist(), sorted_gains[:budget].tolist()
#         sirc_end_time = time.time()
#         idxs = [int(x) for x in idxs]
#         self.logger.debug("SIRC algorithm Subset Selection time is: %.4f", sirc_end_time - sirc_start_time)
#         return idxs, torch.FloatTensor(gammas)