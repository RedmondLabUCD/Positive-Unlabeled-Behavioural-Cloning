#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:21:17 2023

@author: qiang
"""


import models
import utils
import torch
from collections import defaultdict
import numpy as np
import os
import random
from utils import matrix
import gc

def set_seed(seed):
    utils.set_seed(seed)
    models.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PUTrainer():
    def __init__(self,
                 args):
        set_seed(args.seeds)
        self.args = args
        self.device = utils.device_handler(args)
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.train_logger = utils.TrainLogger(
            experiment_name=args.exp_name,
            save_metrics=True,
            root_dir=args.save_path,
            verbose=False,
            tensorboard_dir=1,
            with_timestamp=True,
        )
        self.conf_mat = matrix()
        self.turn = 0
        self.turn_step = 0
        self.total_step = 0
        self.model = None
        
    def reset(self):
        if self.model != None:
            del self.model
            gc.collect()
            
        self.model = []
        for i in range(self.args.models_in_ensemble):
            self.model.append(models.FilterNet(self.args,
                                              bias=True
                                              ).to(self.device))
            
        self.optimizer = []
        for j in self.model:
            self.optimizer.append(torch.optim.Adam(j.parameters(),
                                              lr=self.args.learning_rate,
                                              weight_decay=3e-3,
                                              betas=(0.9, 0.999),
                                              eps=1e-8,
                                              amsgrad=False
                                              ))

    def train(self, loaders, epochs, turn, validate_func):
        self.reset()
        for idx,loader in enumerate(loaders):
            self.model[idx].train()
            for epoch in range(1, epochs + 1):
                epoch_loss = defaultdict(list)
                self.turn_step += 1
                
                for [observations, actions, labels] in loader:
                    observations = torch.tensor(observations).to(
                        torch.float32).to(self.device)
                    actions = torch.tensor(actions).to(
                        torch.float32).to(self.device)
                    labels = torch.tensor(labels).to(
                        torch.float32).to(self.device)
                    
                    pred = self.model[idx](observations, actions)
                    loss = self.criterion(pred, labels)
    
                    self.optimizer[idx].zero_grad()
                    loss.backward()
                    self.optimizer[idx].step()
    
                    self.train_logger.add_metric(
                        f'loss-model={idx+1}', loss.cpu().detach().numpy())
                    epoch_loss['loss'].append(loss.cpu().detach().numpy())
                    
                    self.total_step += 1
                    
                    self.train_logger.commit(self.turn_step, self.total_step)
                
                print(f"Iteration: {turn+1}   Model: {idx}   Epoch: {epoch}   Loss: {np.array(epoch_loss['loss']).mean()}")

        amount, acc, count, probs = validate_func(self.model,
                                                        f'{self.args.save_path}/{self.train_logger._experiment_name}/adap_probs-iteration={turn+1}.jpg')
        
        import seaborn
        from matplotlib import pyplot as plt   
        seaborn.displot( 
          data=np.array(probs), 
          kind="hist", 
          aspect=1.4,
          bins=100,
        )
        np.save(f'{self.args.save_path}/{self.train_logger._experiment_name}/probs-iteration={turn+1}.npy', np.array(probs))
        plt.plot()
        plt.tight_layout()
        plt.xlabel("Summed probs")
        plt.ylabel("Amount")
        plt.savefig(f'{self.args.save_path}/{self.train_logger._experiment_name}/hist-iteration={turn+1}.jpg',
                    dpi=400,bbox_inches='tight')
            
        conf_log = [turn+1, acc]
        self.conf_mat.update(conf_log)
        self.conf_mat.save(f'{self.args.save_path}/{self.train_logger._experiment_name}/accuracy.csv')
        print(f"Iteration: {turn+1}   Final   Acc: {acc}   ")
        print("-"*50)
        
        self.save(
            self.turn+1, path=f'{self.args.save_path}/{self.train_logger._experiment_name}')
        self.turn += 1
                
    def save(self, epoch, path=None):
        for idx,model in enumerate(self.model):
            torch.save(model.state_dict(), f'{path}/ckpt-model={idx+1}-iteration={epoch}.pth')

    def load(self, validate_func=None):
        self.reset()
        for idx,model in enumerate(self.model):
            model.load_state_dict(torch.load(f"{self.args.trained_filter_path}/ckpt-model={idx+1}-iteration={self.args.ckpt_iterations}.pth", map_location=self.device))
        print('Model loaded!')
        print('Using the trained model to filter the mixed dataset.')
        if validate_func:
            amount, acc, count, probs = validate_func(self.model,
                                                            f'{self.args.save_path}/{self.train_logger._experiment_name}/adap_probs_{self.args.ckpt_iterations}.jpg')
            import seaborn
            from matplotlib import pyplot as plt   
            seaborn.displot( 
              data=np.array(probs), 
              kind="hist", 
              aspect=1.4,
              bins=100,
            )
            np.save(f'{self.args.save_path}/{self.train_logger._experiment_name}/prob_{self.args.ckpt_iterations}.npy', np.array(probs))
            plt.plot()
            plt.tight_layout()
            plt.xlabel("Summed probs")
            plt.ylabel("Amount")
            plt.savefig(f'{self.args.save_path}/{self.train_logger._experiment_name}/dist_{self.args.ckpt_iterations}.jpg',
                        dpi=400,bbox_inches='tight')
