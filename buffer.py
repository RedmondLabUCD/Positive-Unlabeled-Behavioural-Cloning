#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:20:34 2023

@author: qiang
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gc
import utils
import random
import math

class PUBuffer():
    def __init__(self,
                 args
                 ):
        self.raw_dataset = np.load(args.raw_dataset_path, allow_pickle=True).item()
        if args.pos_seed_dataset_path != None:
            self.pos_seed_dataset = np.load(args.pos_seed_dataset_path, allow_pickle=True).item()
        else:
            self.pos_seed_dataset = utils.get_positve_seed_by_reward(self.raw_dataset, args.pos_seed_rate)
        
        self.ground_truth = self.raw_dataset['labels']
        self.pos_seed_len = self.pos_seed_dataset['observations'].shape[0]
        self.raw_len = self.raw_dataset['observations'].shape[0]
        self.args = args
        self.torch_loader = None
        self.th_conf = None
        
        self.args.obs_dim = self.raw_dataset['observations'].shape[1]
        print(f'Observation dimension:{self.args.obs_dim}')
        self.args.action_dim = self.raw_dataset['actions'].shape[1]
        print(f'Action dimension:{self.args.action_dim}')
        self.args.max_action = self.raw_dataset['actions'].max()
        self.args.max_observation = self.raw_dataset['observations'].max()
    
    # Init the seed-positive dataset
    def set_seed_positive(self):
        self.torch_loader = []
        for i in range(self.args.models_in_ensemble):
            dataset = TorchDatasetHandler(
                self.sub_seed_positive(), self.raw_dataset, self.args, transform=None)
            self.torch_loader.append(DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=True))
            
    def sub_seed_positive(self):
        rand_idx = random.sample(range(0,self.pos_seed_len),int(self.pos_seed_len*0.95))
        sub_obs = self.pos_seed_dataset['observations'][rand_idx]
        sub_act = self.pos_seed_dataset['actions'][rand_idx]
        subset = {}
        subset['observations'] = sub_obs
        subset['actions'] = sub_act
        return subset
    
    def reset(self):
        del self.buffers, self.torch_loader
        gc.collect()
    
    # Update the positive dataset
    def update_pos(self):
        num_in_each = math.floor(self.relabel_num / self.args.models_in_ensemble)
        print(f'{self.relabel_num}/{self.raw_len} samples in the new dataset')
        self.buffers = []
        full_idx = 0
        sub_len = 0
        subset = {'observations': [],'actions':[]}
        for idx, lab in enumerate(self.relabel):
            if lab:
                subset['observations'].append(self.raw_dataset['observations'][idx])
                subset['actions'].append(self.raw_dataset['actions'][idx])
                sub_len += 1
                if sub_len >= num_in_each:
                    subset['observations'] = np.array(subset['observations'])
                    subset['actions'] = np.array(subset['actions'])
                    self.buffers.append(subset)
                    full_idx += 1
                    sub_len = 0
                    subset = {'observations': [],'actions':[]}
                if full_idx >= self.args.models_in_ensemble:
                    break
        self._init_torch_loaders()
        
    def init_torch_loader_to_train_policy(self):
        train_set = {'observations': [],'actions':[], 'timeouts':[], 'rewards':[]}
        for idx, lab in enumerate(self.relabel):
            if lab:
                train_set['observations'].append(self.raw_dataset['observations'][idx])
                train_set['actions'].append(self.raw_dataset['actions'][idx])
                train_set['timeouts'].append(self.raw_dataset['timeouts'][idx])
                train_set['rewards'].append(self.raw_dataset['rewards'][idx])
        train_set['observations'] = np.array(train_set['observations'])
        train_set['actions'] = np.array(train_set['actions'])
        train_set['timeouts'] = np.array(train_set['timeouts'])
        train_set['rewards'] = np.array(train_set['rewards'])
        print(f'{self.relabel_num}/{self.raw_len} samples in the new dataset')
        return train_set
    
    # Evaluate the accuracy of classfication
    def _eval_accracy(self):
        assert self.relabel.shape[0] == self.ground_truth.shape[0], "Error"
        total = self.ground_truth.shape[0]
        correct_num = 0
        for idx, label in enumerate(self.ground_truth):
            if label == self.relabel[idx]:
                correct_num += 1
        return correct_num / total
    
    def _init_torch_loaders(self):
        self.torch_loader = []
        
        for i in range(self.args.models_in_ensemble):
            dataset = TorchDatasetHandler(
                self.buffers[i], self.raw_dataset, self.args, transform=None)
            self.torch_loader.append(DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=True))
    
    # Use the classifier to filter the dataset
    def classifier_validate(self, classifiers, log_path=None):
        with torch.no_grad():
            temp_obs = []
            temp_actions = []
            probs = []
            count = 0
            self.relabel = []
            self.relabel_num = 0
            self.relabelled_idx = []

            for idx, timeout in enumerate(self.raw_dataset['timeouts']):
                
                temp_obs.append(self.raw_dataset['observations'][idx].tolist())
                temp_actions.append(self.raw_dataset['actions'][idx].tolist())
    
                if timeout:
                    if torch.cuda.is_available() and self.args.use_gpu:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                    else:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).to(
                            torch.float32).to(torch.device('cpu'))
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).to(
                            torch.float32).to(torch.device('cpu'))
                    
                    temp_prob = 0
                    temp_count = 0
                    for classifier in classifiers:
                        classifier.eval()
                        prob = classifier(temp_obs_tensor, temp_actions_tensor).cpu(
                        ).detach().numpy()[..., 0].mean()
                        temp_prob += prob
                        temp_count += 1
                    probs.append(temp_prob/temp_count)
                    temp_obs = []
                    temp_actions = []

            adap_th = utils.adap_probs(np.array(probs), 
                                       bins=self.args.th_bins, 
                                       fit_pow=self.args.th_fit_pow, 
                                       prob_th=self.args.th_high_bound, 
                                       plot=False, 
                                       save_plot_path = log_path)
                    
            self.th_conf = adap_th
                    
            for idx, timeout in enumerate(self.raw_dataset['timeouts']):
                
                temp_obs.append(self.raw_dataset['observations'][idx].tolist())
                temp_actions.append(self.raw_dataset['actions'][idx].tolist())
    
                if timeout:
                    if torch.cuda.is_available() and self.args.use_gpu:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                    else:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).to(
                            torch.float32).to(torch.device('cpu'))
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).to(
                            torch.float32).to(torch.device('cpu'))
                        
                    if self.args.ensemble_method == 'vote':
                        votes = 0
                        for classifier in classifiers:
                            classifier.eval()
                            prob = classifier(temp_obs_tensor, temp_actions_tensor).cpu(
                            ).detach().numpy()[..., 0].mean()
                            votes += float(prob >= self.th_conf)
                        if votes >= len(classifiers) / 2:
                            cond = True
                        else:
                            cond = False
                    
                    if self.args.ensemble_method == 'avg':
                        temp_prob = 0
                        cnt = 0
                        for classifier in classifiers:
                            classifier.eval()
                            prob = classifier(temp_obs_tensor, temp_actions_tensor).cpu(
                            ).detach().numpy()[..., 0].mean()
                            temp_prob += prob
                            cnt += 1
                        avg_prob = temp_prob / cnt
                        if avg_prob >= self.th_conf:
                            cond = True
                        else:
                            cond = False
                            
                    if cond:
                        self.relabel += [1.0] * len(temp_obs)
                        self.relabel_num += len(temp_obs)
                        self.relabelled_idx.append(count)
                    else:
                        self.relabel += [0.0] * len(temp_obs)
                        
                    temp_obs = []
                    temp_actions = []
                    count += 1
        
        print('Relabelling')
        self.relabel = np.array(self.relabel)
        acc = self._eval_accracy()
        assert self.relabel.shape[0] == self.raw_dataset['observations'].shape[0], "Error"
        
        return len(self.relabelled_idx), acc, count, probs
    
class TorchPolicyDatasetHandler(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset['actions'].shape[0]

    def __getitem__(self, index):
        observation = self.dataset['observations'][index]
        action = self.dataset['actions'][index]
        if self.transform:
            observation = self.transform(observation)
            action = self.transform(action)
        data = [observation, action]
        return data

class TorchDatasetHandler(Dataset):
    def __init__(self, filtered_dataset, raw_dataset, args=None, transform=None):
        self.args = args
        self.dataset = filtered_dataset
        self.transform = transform
        self.raw_dataset = raw_dataset
        self.raw_len = self.raw_dataset['observations'].shape[0]

    def __len__(self):
        return self.dataset['actions'].shape[0]

    def __getitem__(self, index):
        pos_observation = self.dataset['observations'][index]
        pos_action = self.dataset['actions'][index]

        if np.random.uniform(0, 1) <= 0.5:
            action = pos_action
            observation = pos_observation
            label = np.array([1, 0])
        else:
            label = np.array([0, 1])
            if self.args.negative_sampler == 'part':
                observation = pos_observation
                if np.random.uniform(0, 1) < 0.5:
                    action = np.random.uniform(-self.args.max_action, self.args.max_action, size=self.args.action_dim)
                else:
                    while True:
                        random_num = np.random.randint(0, self.raw_len)
                        if not (pos_action == self.raw_dataset["actions"][random_num]).all():
                            action = self.raw_dataset["actions"][random_num]
                            break
            elif self.args.negative_sampler == 'full':
                if np.random.uniform(0, 1) < 1/7:
                    observation = pos_observation
                    action = np.random.uniform(-self.args.max_action, self.args.max_action, size=self.args.action_dim)
                elif 1/7 <= np.random.uniform(0, 1) < 2/7:
                    observation = pos_observation
                    while True:
                        random_num = np.random.randint(0, self.raw_len)
                        if not (pos_action == self.raw_dataset["actions"][random_num]).all():
                            action = self.raw_dataset["actions"][random_num]
                            break
                elif 2/7 <= np.random.uniform(0, 1) < 3/7:
                    action = pos_action
                    while True:
                        random_num = np.random.randint(0, self.raw_len)
                        if not (pos_observation == self.raw_dataset["observations"][random_num]).all():
                            observation = self.raw_dataset["observations"][random_num]
                            break
                elif 3/7 <= np.random.uniform(0, 1) < 4/7:
                    random_num = np.random.randint(0, self.raw_len)
                    observation = self.raw_dataset["observations"][random_num]
                    action = np.random.uniform(-self.args.max_action, self.args.max_action, size=self.args.action_dim)
                elif 4/7 <= np.random.uniform(0, 1) < 5/7:
                    action = pos_action
                    observation = np.random.uniform(-self.args.max_observation, self.args.max_observation, size=self.args.obs_dim)
                elif 5/7 <= np.random.uniform(0, 1) < 6/7:
                    random_num = np.random.randint(0, self.raw_len)
                    action = self.raw_dataset["actions"][random_num]
                    observation = np.random.uniform(-self.args.max_observation, self.args.max_observation, size=self.args.obs_dim)
                else:
                    action = np.random.uniform(-self.args.max_action, self.args.max_action, size=self.args.action_dim)
                    observation = np.random.uniform(-self.args.max_observation, self.args.max_observation, size=self.args.obs_dim)

        if self.transform:
            observation = self.transform(observation)
            action = self.transform(action)
            label = self.transform(label)

        data = [observation, action, label]
        return data