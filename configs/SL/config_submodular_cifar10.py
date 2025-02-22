# Learning setting
config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="cifar10",
                           datadir="../data",
                           feature="dss",
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=128,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=10),
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),
              
              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.05,
                             weight_decay=5e-4,
                             nesterov=True),

              scheduler=dict(type="cosine_annealing",
                             T_max=300),

              dss_args=dict(type="Submodular",
                            fraction=0.1,
                            select_every=20,
                            kappa=0,
                            linear_layer=False,
                            if_convex=True,
                            optimizer='stochastic', # random, modular, naive, lazy, approximate-lazy, two-stage, stochastic, sample, greedi, bidirectional
                            selection_type='PerClass', # Supervised - PerClass
                            submod_func_type='graph-cut',
                            collate_fn = None
                            ),

              train_args=dict(num_epochs=300,
                              device="cuda",
                              print_every=10,
                              run=1,
                              results_dir='results/',
                              wandb=False,
                              print_args=["trn_loss", "trn_acc", "val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )
