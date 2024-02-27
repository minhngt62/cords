from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import SIRCStrategy
from torch.utils.data import DataLoader
import time, copy


# GLISTER
class SIRCDataLoader(AdaptiveDSSDataLoader):
    """
    Implements of GLISTERDataLoader that serves as the dataloader for the adaptive GLISTER subset selection strategy from the paper 
    :footcite:`killamsetty2021glister`.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary required for GLISTER subset selection strategy
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args, logger, *args, **kwargs):
        """
         Constructor function
        """

        # Arguments assertion check
        assert "model" in dss_args.keys(), "'model' is a compulsory argument. Include it as a key in dss_args"
        assert "loss" in dss_args.keys(), "'loss' is a compulsory argument. Include it as a key in dss_args"
        if dss_args.loss.reduction != "none":
            raise ValueError("Please set 'reduction' of loss function to 'none' for adaptive subset selection strategies")
        assert "eta" in dss_args.keys(), "'eta' is a compulsory argument. Include it as a key in dss_args"
        assert "num_classes" in dss_args.keys(), "'num_classes' is a compulsory argument for GLISTER. Include it as a key in dss_args"
        assert "linear_layer" in dss_args.keys(), "'linear_layer' is a compulsory argument for GLISTER. Include it as a key in dss_args"
        assert "selection_type" in dss_args.keys(), "'selection_type' is a compulsory argument for GLISTER. Include it as a key in dss_args"

        
        super(SIRCDataLoader, self).__init__(train_loader, val_loader, dss_args,
                                                logger, *args, **kwargs)
        
        self.strategy = SIRCStrategy(train_loader, val_loader, copy.deepcopy(dss_args.model), dss_args.loss, dss_args.eta, dss_args.device,
                                        dss_args.num_classes, dss_args.linear_layer, dss_args.selection_type, logger)
        self.train_model = dss_args.model    
        self.logger.debug('SIRC dataloader initialized. ')

    def __iter__(self):
        """
        Iter function that returns the iterator of full data loader or data subset loader or empty loader based on the 
        warmstart kappa value.
        """
        self.initialized = True
        if self.warmup_epochs < self.cur_epoch < self.select_after:
            self.logger.debug(
                "Skipping epoch {0:d} due to warm-start option. ".format(self.cur_epoch, self.warmup_epochs))
            loader = DataLoader([])
            
        elif self.cur_epoch < self.warmup_epochs:
            self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
            loader = self.wtdataloader
            self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        else:
            self.logger.debug('Epoch: {0:d}, reading dataloader... '.format(self.cur_epoch))
            if ((self.cur_epoch) % self.select_every == 0) and (self.cur_epoch > 1):
                self.resample()
            loader = self.subset_loader
            self.logger.debug('Epoch: {0:d}, finished reading dataloader. '.format(self.cur_epoch))
        
        self.cur_epoch += 1
        self.strategy.set_trained_indices(self.subset_indices)
        self.strategy.set_cur_epoch(self.cur_epoch)
        return loader.__iter__()

    def _resample_subset_indices(self):
        """
        Function that calls the GLISTER subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        self.strategy.set_train_model(self.train_model)
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        end = time.time()
        self.logger.info('Epoch: {0:d}, SIRC dataloader subset selection finished, takes {1:.4f}. '.format(self.cur_epoch, (end - start)))
        # debug
        # import pickle
        # labels, logits, features = self.strategy.get_outputs_labels()
        # log_features = {'labels': labels, 'logits':logits, 'features':features, 'idxs':subset_indices, 'gammas':subset_weights}
        # with open(f'debug/epoch_{self.cur_epoch}.pickle', 'wb') as f:
        #     pickle.dump(log_features, f)
        return subset_indices, subset_weights
    
    def update_model(self, model_params):
        """
        Update the models parameters

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)
