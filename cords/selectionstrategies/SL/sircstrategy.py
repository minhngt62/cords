import math
import random
import time
from typing import Any
import torch
import torch.nn.functional as F
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import numpy as np

    
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
        start = time.time()

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
            # get params
            labels, logits, features = self.get_outputs_labels()
            self.labels, self.logits, self.features = labels.detach().cpu().clone(), logits.detach().cpu().clone().type(torch.DoubleTensor), features.detach().cpu().clone()
            
            # flexmatch ..........................
            flex_start = time.time()
            thr = 0.95
            probs = self.logits.softmax(dim=-1)
            
            # learning effect (le)
            pseudo_mask = probs >= torch.max(probs, dim=1).values[:, None]
            lr_eff = (pseudo_mask * (probs >= thr)).sum(axis=0)
            # normalize le
            B = lr_eff / max(lr_eff.max(), len(probs) - lr_eff.sum())
            T = (1.0 - B) * thr
            candidate = ~(((pseudo_mask * probs) >= T).sum(axis=1) >= 1) # samples <= upper-bound T
            
            self.logger.debug('Flexmatch: %d, %s, %s', int(candidate.sum()), str(T.tolist()), str(lr_eff.tolist()))

            print(f"FlexMatch: {time.time() - flex_start}s")
             
            # sirc ..........................
            sirc_start = time.time()
            s1 = 1.0 - probs.max(dim=-1).values.to(self.device)
            s1 = torch.clip(s1, min=1e-64)
            score = -s1.log() # <--
            if isinstance(score, np.ndarray): 
                score = torch.FloatTensor(score)
            print(f"SIRC: {time.time() - sirc_start}s")
            
            idxs = []
            n_candidates = candidate.sum()
            if n_candidates < budget:
                idxs += [int(i) for i, c in enumerate(candidate) if c==1] # <--
                candidate = ~candidate
                print('idxs: ', len(idxs))
                
            elif n_candidates > budget*5:
                cluster_start = time.time()
                # select by cluster bin
                self.logger.debug('select by cluster bin')
                n_bin = 200
                budget_bin = math.ceil((budget-len(idxs)) / n_bin)
                selected = []
                features = self.features[candidate]
                s = score[candidate]
                
                # clustering
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
                print(f"Clustering: {cluster_start}s")
            else:
                balance_start = time.time()
                self.logger.debug('select by balance score')
                total_scores = [0 for i in range(self.num_classes)]
                selected_idxs = [[] for i in range(self.num_classes)]
                queue_idxs = [[] for i in range(self.num_classes)]
                sorted_gains, indices = torch.sort(score.view(-1), descending=False)
                selected_count = 0
                i = 0
                while i < score.shape[0]:
                    idx = indices[i]
                    i += 1
                    if candidate[idx]==0: continue
                    label = self.labels[idx]
                    queue_idxs[label].append(idx)
                    
                while selected_count < budget-len(idxs):
                    queue_idxs_mask = [True if len(q)>0 else False for q in queue_idxs]
                    min_total_score = min([total_scores[i] for i, m in enumerate(queue_idxs_mask) if m==True])
                    min_score_label = [i for i, s in enumerate(total_scores) if (queue_idxs_mask[i]==True and s==min_total_score)][0]
                    selected_idx = queue_idxs[min_score_label].pop(0)
                    selected_idxs[min_score_label].append(selected_idx)
                    total_scores[min_score_label] += score[selected_idx]
                    selected_count += 1
                idxs += [int(i) for q in selected_idxs for i in q]
                gammas = score[idxs].tolist()
                print(f"Balance: {time.time() - balance_start}s")
            
            self.logger.debug('Number: %s', len(list(set(idxs))))

        self.logger.debug("SIRC algorithm Subset Selection time is: %.4f", time.time() - start)
        return idxs, torch.FloatTensor(gammas)
    
    def set_trained_indices(self, subset_indices):
        self.trained_indices = subset_indices
        
    def set_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch
        
    def set_train_model(self, train_model):
        self.train_model = train_model