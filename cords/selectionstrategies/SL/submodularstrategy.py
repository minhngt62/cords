import numpy as np
import time
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data.sampler import SubsetRandomSampler

class SubmodularStrategy(DataSelectionStrategy):
    """
    This class extends :class:`selectionstrategies.supervisedlearning.dataselectionstrategy.DataSelectionStrategy`
    to include submodular optmization functions using apricot for data selection.

    Parameters
    ----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    if_convex: bool
        If convex or not
    selection_type: str
        PerClass or Supervised
    submod_func_type: str
        The type of submodular optimization function. Must be one of
        'facility-location', 'graph-cut', 'sum-redundancy', 'saturated-coverage'
    """

    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer, if_convex, selection_type, submod_func_type, logger, optimizer):
        """
        Constructer method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device, logger)
        self.if_convex = if_convex
        self.selection_type = selection_type
        self.submod_func_type = submod_func_type
        self.logger = logger
        self.optimizer = optimizer
        
    def calc_gradient(self, model_params, idxs):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''
        trainset = self.trainloader.sampler.data_source
        if idxs is None:
            batch_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
                                                        pin_memory=True)
        else:
            batch_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
                                                        sampler=SubsetRandomSampler(idxs),
                                                        pin_memory=True)
        self.model.load_state_dict(model_params)
        self.model.eval()

        self.embedding_dim = self.model.get_embedding_dim()

        # Initialize a matrix to save gradients.
        gradients = []

        for i, (input, targets) in enumerate(batch_loader):
            outputs, l1 = self.model(input.to(self.device), freeze=True, last=True)
            loss = self.loss(torch.nn.functional.softmax(outputs.requires_grad_(True), dim=1),
                                  targets.to(self.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = l1.view(batch_num, 1,
                                        self.embedding_dim).repeat(1, self.num_classes, 1) *\
                                        bias_parameters_grads.view(batch_num, self.num_classes,
                                        1).repeat(1, 1, self.embedding_dim)
                gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
                                            dim=1))

        gradients = torch.concatenate(gradients, axis=0)
        return gradients

    @staticmethod
    def cossim_np(v1, v2):
        num = torch.matmul(v1, v2.T)
        denom = torch.linalg.norm(v1, axis=1).reshape(-1, 1) * torch.linalg.norm(v2, axis=1)
        res = num / denom
        res[torch.isneginf(res)] = 0.
        return 0.5 + 0.5 * res.cpu().numpy()
    
    def compute_gamma(self, dist_mat, idxs):
        """
        Compute the gamma values for the indices.

        Parameters
        ----------
        idxs: list
            The indices

        Returns
        ----------
        gamma: list
            Gradient values of the input indices
        """

        if self.selection_type == 'PerClass':
            gamma = [0 for i in range(len(idxs))]
            best = dist_mat[idxs]  # .to(self.device)
            rep = np.argmax(best, axis=0)
            for i in rep:
                gamma[i] += 1
        elif self.selection_type == 'Supervised':
            gamma = [0 for i in range(len(idxs))]
            best = dist_mat[idxs]  # .to(self.device)
            rep = np.argmax(best, axis=0)
            for i in range(rep.shape[1]):
                gamma[rep[0, i]] += 1
        return gamma

    def select(self, budget, model_params):
        """
        Data selection method using different submodular optimization
        functions.

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        optimizer: str
            The optimization approach for data selection. Must be one of
            'random', 'modular', 'naive', 'lazy', 'approximate-lazy', 'two-stage',
            'stochastic', 'sample', 'greedi', 'bidirectional'

        Returns
        ----------
        total_greedy_list: list
            List containing indices of the best datapoints
        gammas: list
            List containing gradients of datapoints present in greedySet
        """
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                x_trn, labels = inputs, targets
            else:
                tmp_inputs, tmp_target_i = inputs, targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        per_class_bud = int(budget / self.num_classes)
        gammas = []
        # Turn on the embedding recorder and the no_grad flag
        self.model.no_grad = True
        self.train_indx = np.arange(self.N_trn)

        if self.selection_type == 'PerClass':
            selection_result = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = torch.where(labels == c)[0]
                # Calculate gradients into a matrix
                gradients = self.calc_gradient(model_params=model_params, idxs=c_indx)
                # Instantiate a submodular function
                if self.submod_func_type=='GraphCut':
                    submod_function = GraphCut(index=c_indx, 
                                               similarity_kernel=lambda a, b: self.cossim_np(gradients[a], gradients[b]))
                if self.optimizer=='NaiveGreedy':
                    submod_optimizer = NaiveGreedy(index=c_indx, budget=per_class_bud, already_selected=[])

                c_selection_result, selected = submod_optimizer.select(gain_function=submod_function.calc_gain, 
                                                                       update_state=submod_function.update_state)
                selection_result = np.append(selection_result, c_selection_result)
                gammas += gradients[selected].tolist()
        else:
            # Calculate gradients into a matrix
            gradients = self.calc_gradient(model_params=model_params, idxs=None)
            # Instantiate a submodular function
            if self.submod_func_type=='GraphCut':
                submod_function = GraphCut(index=self.train_indx, 
                                           similarity_kernel=lambda a, b: self.cossim_np(gradients[a], gradients[b]))
            if self.optimizer=='NaiveGreedy':    
                submod_optimizer = NaiveGreedy(index=self.train_indx, budget=budget)
            selection_result, selected = submod_optimizer.select(gain_function=submod_function.calc_gain, 
                                                                 update_state=submod_function.update_state)
            gammas += gradients[selection_result].tolist()

        self.model.no_grad = False
        end_time = time.time()
        self.logger.debug("Submodular strategy data selection time is: %.4f", end_time-start_time)
        # return selection_result.tolist(), torch.FloatTensor(gammas).mean(dim=-1)
        return selection_result.tolist(), torch.ones(selection_result.shape)
    
class SubmodularFunction(object):
    def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        self.index = index
        self.n = len(index)

        self.already_selected = already_selected

        assert similarity_kernel is not None or similarity_matrix is not None

        # For the sample similarity matrix, the method supports two input modes, one is to input a pairwise similarity
        # matrix for the whole sample, and the other case allows the input of a similarity kernel to be used to
        # calculate similarities incrementally at a later time if required.
        if similarity_kernel is not None:
            assert callable(similarity_kernel)
            self.similarity_kernel = self._similarity_kernel(similarity_kernel)
        else:
            assert similarity_matrix.shape[0] == self.n and similarity_matrix.shape[1] == self.n
            self.similarity_matrix = similarity_matrix
            self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]

    def _similarity_kernel(self, similarity_kernel):
        return similarity_kernel


class GraphCut(SubmodularFunction):
    def __init__(self, lam: float = 1., **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

        if 'similarity_matrix' in kwargs:
            self.sim_matrix_cols_sum = np.sum(self.similarity_matrix, axis=0)
        self.all_idx = np.ones(self.n, dtype=bool)

    def _similarity_kernel(self, similarity_kernel):
        # Initialize a matrix to store similarity values of sample points.
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.sim_matrix_cols_sum = np.zeros(self.n, dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.sim_matrix_cols_sum[not_calculated] = np.sum(self.sim_matrix[:, not_calculated], axis=0)
                self.if_columns_calculated[not_calculated] = True
            return self.sim_matrix[np.ix_(a, b)]
        return _func

    def calc_gain(self, idx_gain, selected, **kwargs):

        gain = -2. * np.sum(self.similarity_kernel(selected, idx_gain), axis=0) + self.lam * self.sim_matrix_cols_sum[idx_gain]

        return gain

    def update_state(self, new_selection, total_selected, **kwargs):
        pass
    

class Optimizer(object):
    def __init__(self, index, budget:int, already_selected=[]):
        self.index = index

        if budget <= 0 or budget > index.__len__():
            raise ValueError("Illegal budget for optimizer.")

        self.n = len(index)
        self.budget = budget
        self.already_selected = already_selected


class NaiveGreedy(Optimizer):
    def __init__(self, index, budget:int, already_selected=[]):
        super(NaiveGreedy, self).__init__(index, budget, already_selected)

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        for i in range(sum(selected), self.budget):
            greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
            current_selection = greedy_gain.argmax()
            selected[current_selection] = True
            greedy_gain[current_selection] = -np.inf
            if update_state is not None:
                update_state(np.array([current_selection]), selected, **kwargs)
        return self.index[selected], selected


# import numpy as np
# import time
# import torch
# import torch.nn.functional as F
# from scipy.sparse import csr_matrix
# from .dataselectionstrategy import DataSelectionStrategy
# from torch.utils.data.sampler import SubsetRandomSampler

# class SubmodularStrategy(DataSelectionStrategy):
#     """
#     This class extends :class:`selectionstrategies.supervisedlearning.dataselectionstrategy.DataSelectionStrategy`
#     to include submodular optmization functions using apricot for data selection.

#     Parameters
#     ----------
#     trainloader: class
#         Loading the training data using pytorch DataLoader
#     valloader: class
#         Loading the validation data using pytorch DataLoader
#     model: class
#         Model architecture used for training
#     loss_type: class
#         The type of loss criterion
#     device: str
#         The device being utilized - cpu | cuda
#     num_classes: int
#         The number of target classes in the dataset
#     linear_layer: bool
#         Apply linear transformation to the data
#     if_convex: bool
#         If convex or not
#     selection_type: str
#         PerClass or Supervised
#     submod_func_type: str
#         The type of submodular optimization function. Must be one of
#         'facility-location', 'graph-cut', 'sum-redundancy', 'saturated-coverage'
#     """

#     def __init__(self, trainloader, valloader, model, loss,
#                  device, num_classes, linear_layer, if_convex, selection_type, submod_func_type, logger, optimizer):
#         """
#         Constructer method
#         """
#         super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device, logger)
#         self.if_convex = if_convex
#         self.selection_type = selection_type
#         self.submod_func_type = submod_func_type
#         self.logger = logger
#         self.optimizer = optimizer
        
#     def calc_gradient(self, model_params, idxs):
#         '''
#         Calculate gradients matrix on current network for specified training dataset.
#         '''
#         trainset = self.trainloader.sampler.data_source
#         batch_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
#                                                     sampler=SubsetRandomSampler(idxs),
#                                                     pin_memory=True)
#         self.model.load_state_dict(model_params)
#         self.model.eval()

#         self.embedding_dim = self.model.get_embedding_dim()

#         # Initialize a matrix to save gradients.
#         # (on cpu)
#         gradients = []

#         for i, (input, targets) in enumerate(batch_loader):
#             outputs, l1 = self.model(input.to(self.device), freeze=True, last=True)
#             loss = self.loss(torch.nn.functional.softmax(outputs.requires_grad_(True), dim=1),
#                                   targets.to(self.device)).sum()
#             batch_num = targets.shape[0]
#             with torch.no_grad():
#                 bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
#                 weight_parameters_grads = l1.view(batch_num, 1,
#                                         self.embedding_dim).repeat(1, self.num_classes, 1) *\
#                                         bias_parameters_grads.view(batch_num, self.num_classes,
#                                         1).repeat(1, 1, self.embedding_dim)
#                 gradients.append(torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)],
#                                             dim=1).cpu().numpy())

#         gradients = np.concatenate(gradients, axis=0)
#         return gradients

#     @staticmethod
#     def cossim_np(v1, v2):
#         num = np.dot(v1, v2.T)
#         denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)
#         res = num / denom
#         res[np.isneginf(res)] = 0.
#         return 0.5 + 0.5 * res

#     def select(self, budget, model_params):
#         """
#         Data selection method using different submodular optimization
#         functions.

#         Parameters
#         ----------
#         budget: int
#             The number of data points to be selected
#         model_params: OrderedDict
#             Python dictionary object containing models parameters
#         optimizer: str
#             The optimization approach for data selection. Must be one of
#             'random', 'modular', 'naive', 'lazy', 'approximate-lazy', 'two-stage',
#             'stochastic', 'sample', 'greedi', 'bidirectional'

#         Returns
#         ----------
#         total_greedy_list: list
#             List containing indices of the best datapoints
#         gammas: list
#             List containing gradients of datapoints present in greedySet
#         """
#         start_time = time.time()
#         for batch_idx, (inputs, targets) in enumerate(self.trainloader):
#             if batch_idx == 0:
#                 x_trn, labels = inputs, targets
#             else:
#                 tmp_inputs, tmp_target_i = inputs, targets
#                 labels = torch.cat((labels, tmp_target_i), dim=0)
#         per_class_bud = int(budget / self.num_classes)
#         gammas = []
#         # Turn on the embedding recorder and the no_grad flag
#         self.model.no_grad = True
#         self.train_indx = np.arange(self.N_trn)
        
#         # Calculate gradients into a matrix
#         gradients = self.calc_gradient(model_params=model_params, idxs=None)

#         if self.selection_type == 'PerClass':
#             selection_result = np.array([], dtype=np.int64)
#             for c in range(self.num_classes):
#                 c_indx = torch.where(labels == c)[0]
#                 # Calculate gradients into a matrix
#                 c_gradients = gradients[c_indx]
#                 # Instantiate a submodular function
#                 if self.submod_func_type=='GraphCut':
#                     submod_function = GraphCut(index=c_indx, 
#                                                similarity_kernel=lambda a, b: self.cossim_np(c_gradients[a], c_gradients[b]))
#                 if self.optimizer=='NaiveGreedy':
#                     submod_optimizer = NaiveGreedy(index=c_indx, budget=per_class_bud, already_selected=[])

#                 c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
#                                                                 update_state=submod_function.update_state)
#                 selection_result = np.append(selection_result, c_selection_result)
#         else:
#             # Instantiate a submodular function
#             if self.submod_func_type=='GraphCut':
#                 submod_function = GraphCut(index=self.train_indx, 
#                                            similarity_kernel=lambda a, b: self.cossim_np(gradients[a], gradients[b]))
#             if self.optimizer=='NaiveGreedy':    
#                 submod_optimizer = NaiveGreedy(index=self.train_indx, budget=budget)
#             selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
#                                                         update_state=submod_function.update_state)
#         gammas += gradients[selection_result].tolist()

#         self.model.no_grad = False
#         end_time = time.time()
#         self.logger.debug("Submodular strategy data selection time is: %.4f", end_time-start_time)
#         return selection_result.tolist(), torch.FloatTensor(gammas)
    
# class SubmodularFunction(object):
#     def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
#         self.index = index
#         self.n = len(index)

#         self.already_selected = already_selected

#         assert similarity_kernel is not None or similarity_matrix is not None

#         # For the sample similarity matrix, the method supports two input modes, one is to input a pairwise similarity
#         # matrix for the whole sample, and the other case allows the input of a similarity kernel to be used to
#         # calculate similarities incrementally at a later time if required.
#         if similarity_kernel is not None:
#             assert callable(similarity_kernel)
#             self.similarity_kernel = self._similarity_kernel(similarity_kernel)
#         else:
#             assert similarity_matrix.shape[0] == self.n and similarity_matrix.shape[1] == self.n
#             self.similarity_matrix = similarity_matrix
#             self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]

#     def _similarity_kernel(self, similarity_kernel):
#         return similarity_kernel


# class GraphCut(SubmodularFunction):
#     def __init__(self, lam: float = 1., **kwargs):
#         super().__init__(**kwargs)
#         self.lam = lam

#         if 'similarity_matrix' in kwargs:
#             self.sim_matrix_cols_sum = np.sum(self.similarity_matrix, axis=0)
#         self.all_idx = np.ones(self.n, dtype=bool)

#     def _similarity_kernel(self, similarity_kernel):
#         # Initialize a matrix to store similarity values of sample points.
#         self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
#         self.sim_matrix_cols_sum = np.zeros(self.n, dtype=np.float32)
#         self.if_columns_calculated = np.zeros(self.n, dtype=bool)

#         def _func(a, b):
#             if not np.all(self.if_columns_calculated[b]):
#                 if b.dtype != bool:
#                     temp = ~self.all_idx
#                     temp[b] = True
#                     b = temp
#                 not_calculated = b & ~self.if_columns_calculated
#                 self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
#                 self.sim_matrix_cols_sum[not_calculated] = np.sum(self.sim_matrix[:, not_calculated], axis=0)
#                 self.if_columns_calculated[not_calculated] = True
#             return self.sim_matrix[np.ix_(a, b)]
#         return _func

#     def calc_gain(self, idx_gain, selected, **kwargs):

#         gain = -2. * np.sum(self.similarity_kernel(selected, idx_gain), axis=0) + self.lam * self.sim_matrix_cols_sum[idx_gain]

#         return gain

#     def update_state(self, new_selection, total_selected, **kwargs):
#         pass
    

# class Optimizer(object):
#     def __init__(self, index, budget:int, already_selected=[]):
#         self.index = index

#         if budget <= 0 or budget > index.__len__():
#             raise ValueError("Illegal budget for optimizer.")

#         self.n = len(index)
#         self.budget = budget
#         self.already_selected = already_selected


# class NaiveGreedy(Optimizer):
#     def __init__(self, index, budget:int, already_selected=[]):
#         super(NaiveGreedy, self).__init__(index, budget, already_selected)

#     def select(self, gain_function, update_state=None, **kwargs):
#         assert callable(gain_function)
#         if update_state is not None:
#             assert callable(update_state)
#         selected = np.zeros(self.n, dtype=bool)
#         selected[self.already_selected] = True

#         greedy_gain = np.zeros(len(self.index))
#         for i in range(sum(selected), self.budget):
#             greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
#             current_selection = greedy_gain.argmax()
#             selected[current_selection] = True
#             greedy_gain[current_selection] = -np.inf
#             if update_state is not None:
#                 update_state(np.array([current_selection]), selected, **kwargs)
#         return self.index[selected]