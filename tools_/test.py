import sys
sys.path.append('../')
from train_sl import TrainClassifier

fraction = 0.1
# glister_config_file = '../configs/SL/config_glister_cifar10.py'
glister_config_file = '../configs/SL/config_sirc_cifar10.py'
# glister_config_file = '../configs/SL/config_subml_cifar10.py'
# glister_config_file = '../configs/SL/config_full_cifar10.py'
# glister_config_file = '../configs/SL/config_craig_cifar10.py'

from cords.utils.config_utils import load_config_data

cfg = load_config_data(glister_config_file)
glister_trn = TrainClassifier(cfg)

glister_trn.cfg.scheduler.T_max = 200

glister_trn.cfg.dss_args.fraction = fraction
glister_trn.cfg.dss_args.select_every = 10
# glister_trn.cfg.dss_args.selection_type = 'Supervised'

glister_trn.cfg.train_args.device = 'cuda'
glister_trn.cfg.train_args.print_every = 10
glister_trn.cfg.train_args.num_epochs = 200
# glister_trn.cfg.train_args.print_args=["trn_loss", "trn_acc", "tst_loss", "tst_acc", "time"]

# Noise setting
# glister_trn.cfg.dataset.feature = 'noise'

# glister_trn.cfg.dataloader.batch_size = 20
# glister_trn.cfg.dataloader.shuffle = True

glister_trn.train()